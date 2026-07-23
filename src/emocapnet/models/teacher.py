"""Teacher: EmoCapNet v3 (ViT-base + GPT-2 with cross-attention), loaded frozen.

The layer names and shapes here intentionally mirror the original v3 training code
so a v3 ``best.pt`` loads cleanly. A dirty load raises rather than silently
distilling from a partly-random teacher.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model, ViTModel

from emocapnet.config import Config

log = logging.getLogger(__name__)

#: architecture-determining keys synced from the checkpoint's embedded config.
_ARCH_KEYS = ("prefix_len", "num_emotions", "vad_dim", "max_length")


class EmoCapNetV3(nn.Module):
    """ViT-base + GPT-2(+cross-attn) emotional captioner — identical to v3-round3."""

    def __init__(self, cfg: Config, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.vit = ViTModel.from_pretrained(cfg.teacher_vit)
        vit_dim = self.vit.config.hidden_size
        gpt2_cfg = GPT2Config.from_pretrained(cfg.teacher_gpt2)
        gpt2_cfg.add_cross_attention = True
        self.gpt2 = GPT2Model.from_pretrained(cfg.teacher_gpt2, config=gpt2_cfg)
        self.gpt2.resize_token_embeddings(vocab_size)
        hidden = gpt2_cfg.n_embd
        self.visual_prefix = nn.Sequential(
            nn.Linear(vit_dim, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden * cfg.prefix_len)
        )
        self.spatial_proj = nn.Linear(vit_dim, hidden)
        self.vad_mlp = nn.Sequential(
            nn.Linear(cfg.vad_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.cond_to_scale = nn.Linear(hidden, hidden)
        self.cond_to_shift = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.lm_head.weight = self.gpt2.wte.weight  # weight tying
        self.vad_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, cfg.vad_dim), nn.Sigmoid()
        )
        self.emotion_head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, cfg.num_emotions))
        self.img_proj = nn.Linear(vit_dim, 256)
        self.cap_proj = nn.Linear(hidden, 256)
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)

    def _apply_film(self, kv: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return kv * (1 + self.cond_to_scale(cond).unsqueeze(1)) + self.cond_to_shift(cond).unsqueeze(1)

    def forward(self, pixel_values, input_ids, attention_mask, vad):
        B = pixel_values.size(0)
        vo = self.vit(pixel_values=pixel_values)
        cls_feat = vo.last_hidden_state[:, 0]
        spatial = vo.last_hidden_state[:, 1:]
        cond = self.vad_mlp(vad)
        spatial_kv = self._apply_film(self.spatial_proj(spatial), cond)
        prefix = self.visual_prefix(cls_feat).view(B, self.cfg.prefix_len, -1) + cond.unsqueeze(1)
        tok_emb = self.gpt2.wte(input_ids)
        inp = torch.cat([prefix, tok_emb], dim=1)
        pmask = torch.ones(B, self.cfg.prefix_len, device=attention_mask.device, dtype=attention_mask.dtype)
        out = self.gpt2(
            inputs_embeds=inp,
            attention_mask=torch.cat([pmask, attention_mask], 1),
            encoder_hidden_states=spatial_kv,
            use_cache=False,
        )
        h = out.last_hidden_state[:, self.cfg.prefix_len :]
        logits = self.lm_head(h)
        pool = h.mean(1)
        return (
            logits,
            self.vad_head(pool),
            self.emotion_head(pool),
            F.normalize(self.img_proj(cls_feat), dim=-1),
            F.normalize(self.cap_proj(pool), dim=-1),
        )

    @torch.no_grad()
    def generate(self, pixel_values, vad, emotion_ids, eos_id, cfg: Config) -> torch.Tensor:
        """Greedy decode mirroring the student decoder (used for teacher eval rows)."""
        B = pixel_values.size(0)
        vo = self.vit(pixel_values=pixel_values)
        cls_feat = vo.last_hidden_state[:, 0]
        spatial = vo.last_hidden_state[:, 1:]
        cond = self.vad_mlp(vad)
        spatial_kv = self._apply_film(self.spatial_proj(spatial), cond)
        prefix = self.visual_prefix(cls_feat).view(B, cfg.prefix_len, -1) + cond.unsqueeze(1)
        ids = emotion_ids.view(B, 1).clone()
        done = torch.zeros(B, dtype=torch.bool, device=pixel_values.device)
        for step in range(cfg.hard_max_length):
            inp = torch.cat([prefix, self.gpt2.wte(ids)], 1)
            mask = torch.ones(B, cfg.prefix_len + ids.size(1), device=ids.device, dtype=torch.long)
            out = self.gpt2(
                inputs_embeds=inp, attention_mask=mask, encoder_hidden_states=spatial_kv, use_cache=False
            )
            logits = self.lm_head(out.last_hidden_state[:, -1])
            for b in range(B):
                for tok in set(ids[b].tolist()):
                    if logits[b, tok] > 0:
                        logits[b, tok] /= cfg.repetition_penalty
                    else:
                        logits[b, tok] *= cfg.repetition_penalty
            if step < cfg.min_length:
                logits[:, eos_id] = -1e9
            nxt = logits.argmax(-1, keepdim=True)
            nxt[done] = eos_id
            ids = torch.cat([ids, nxt], 1)
            done = done | (nxt.squeeze(1) == eos_id)
            if done.all():
                break
        return ids


def load_teacher(cfg: Config, vocab_size: int, device: torch.device) -> EmoCapNetV3 | None:
    """Load, verify, and freeze the teacher checkpoint. Returns None if not configured.

    Raises ``RuntimeError`` on a dirty load — distilling from a partly-random
    teacher is worse than not distilling at all.
    """
    import os

    if not cfg.teacher_ckpt or not os.path.exists(cfg.teacher_ckpt) or cfg.distill_alpha <= 0:
        log.info("Teacher disabled -> student trains on ground truth only")
        return None

    from emocapnet.utils import torch_load

    ckpt = torch_load(cfg.teacher_ckpt, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    embed_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(embed_cfg, dict):
        for k in _ARCH_KEYS:
            if k in embed_cfg and getattr(cfg, k) != embed_cfg[k]:
                log.warning("Syncing cfg.%s: %s -> %s (from checkpoint)", k, getattr(cfg, k), embed_cfg[k])
                setattr(cfg, k, embed_cfg[k])

    teacher = EmoCapNetV3(cfg, vocab_size=vocab_size)
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]  # tied to wte -> harmless
    if missing or unexpected:
        raise RuntimeError(
            "Teacher load incomplete: EmoCapNetV3 does not match this checkpoint "
            f"(missing={missing[:5]}, unexpected={unexpected[:5]}). Fix the arch/config, "
            "or set distill_alpha=0 to train on ground truth only."
        )
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in teacher.parameters())
    log.info("Teacher loaded: %.1fM params (frozen, clean load)", n_params / 1e6)
    return teacher
