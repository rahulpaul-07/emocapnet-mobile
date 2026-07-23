"""Student: EmoCapNet-Mobile (small vision encoder + DistilGPT-2 or scratch decoder).

Same conditioning machinery as the teacher (VAD-FiLM, visual prefix, ``<emo>``
token, four heads) on a far smaller backbone, plus a KV-cached greedy decoder
that is the honest on-device latency build.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model

from emocapnet.config import Config
from emocapnet.models.encoders import build_student_encoder

log = logging.getLogger(__name__)


def build_decoder(cfg: Config, vocab_size: int) -> tuple[GPT2Model, int, str]:
    """Return (decoder, hidden_dim, name). ``ultralight`` builds a 4L/512d scratch decoder."""
    if cfg.ultralight:
        gcfg = GPT2Config(
            n_layer=4,
            n_head=8,
            n_embd=512,
            n_positions=cfg.max_length + cfg.prefix_len + 8,
            vocab_size=vocab_size,
            add_cross_attention=True,
        )
        return GPT2Model(gcfg), gcfg.n_embd, "ultralight-4L-512d (scratch)"
    gcfg = GPT2Config.from_pretrained(cfg.student_decoder)
    gcfg.add_cross_attention = True
    dec = GPT2Model.from_pretrained(cfg.student_decoder, config=gcfg)
    return dec, gcfg.n_embd, cfg.student_decoder


class EmoCapNetMobile(nn.Module):
    """Mobile emotional captioner: encoder -> FiLM-conditioned cross-attn decoder."""

    def __init__(self, cfg: Config, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = build_student_encoder(cfg)
        vit_dim = self.encoder.out_dim
        self.gpt2, hidden, self.dec_name = build_decoder(cfg, vocab_size)
        self.gpt2.resize_token_embeddings(vocab_size)
        self.visual_prefix = nn.Sequential(
            nn.Linear(vit_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden * cfg.prefix_len)
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
        self.prefix_len = cfg.prefix_len

    # ------------------------------------------------------------------ layers
    def _film(self, kv: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return kv * (1 + self.cond_to_scale(cond).unsqueeze(1)) + self.cond_to_shift(cond).unsqueeze(1)

    def encode_image(self, pixel_values: torch.Tensor, vad: torch.Tensor):
        B = pixel_values.size(0)
        cls_feat, spatial = self.encoder(pixel_values)
        cond = self.vad_mlp(vad)
        spatial_kv = self._film(self.spatial_proj(spatial), cond)
        prefix = self.visual_prefix(cls_feat).view(B, self.prefix_len, -1) + cond.unsqueeze(1)
        return cls_feat, spatial_kv, prefix

    # ------------------------------------------------------------------ forward
    def forward(self, pixel_values, input_ids, attention_mask, vad):
        B = pixel_values.size(0)
        cls_feat, spatial_kv, prefix = self.encode_image(pixel_values, vad)
        inp = torch.cat([prefix, self.gpt2.wte(input_ids)], dim=1)
        pmask = torch.ones(B, self.prefix_len, device=attention_mask.device, dtype=attention_mask.dtype)
        out = self.gpt2(
            inputs_embeds=inp,
            attention_mask=torch.cat([pmask, attention_mask], 1),
            encoder_hidden_states=spatial_kv,
            use_cache=False,
        )
        h = out.last_hidden_state[:, self.prefix_len :]
        logits = self.lm_head(h)
        pool = h.mean(1)
        return (
            logits,
            self.vad_head(pool),
            self.emotion_head(pool),
            F.normalize(self.img_proj(cls_feat), dim=-1),
            F.normalize(self.cap_proj(pool), dim=-1),
        )

    # ------------------------------------------------------------------ decoding
    @torch.no_grad()
    def generate(self, pixel_values, vad, emotion_ids, eos_id: int, cfg: Config) -> torch.Tensor:
        """Greedy decode with KV cache — identical outputs to the no-cache version,
        3-5x faster; this is also the honest on-device latency build."""
        self.eval()
        B = pixel_values.size(0)
        _, spatial_kv, prefix = self.encode_image(pixel_values, vad)
        ids = emotion_ids.view(B, 1).clone()  # seed with <emo=...> token
        done = torch.zeros(B, dtype=torch.bool, device=pixel_values.device)
        past = None
        inp = torch.cat([prefix, self.gpt2.wte(ids)], dim=1)  # first pass: prefix + seed
        for step in range(cfg.hard_max_length):
            out = self.gpt2(
                inputs_embeds=inp, encoder_hidden_states=spatial_kv, past_key_values=past, use_cache=True
            )
            past = out.past_key_values
            logits = self.lm_head(out.last_hidden_state[:, -1])  # (B, V)
            # repetition penalty
            for b in range(B):
                for tok in set(ids[b].tolist()):
                    if logits[b, tok] > 0:
                        logits[b, tok] /= cfg.repetition_penalty
                    else:
                        logits[b, tok] *= cfg.repetition_penalty
            # no-repeat-ngram
            n = cfg.no_repeat_ngram_size
            if n > 0 and ids.size(1) >= n - 1:
                for b in range(B):
                    seq = ids[b].tolist()
                    prefix_gram = tuple(seq[-(n - 1) :])
                    banned = [
                        seq[i + n - 1]
                        for i in range(len(seq) - n + 1)
                        if tuple(seq[i : i + n - 1]) == prefix_gram
                    ]
                    for tok in banned:
                        logits[b, tok] = -1e9
            if step < cfg.min_length:
                logits[:, eos_id] = -1e9
            nxt = logits.argmax(-1, keepdim=True)
            nxt[done] = eos_id
            ids = torch.cat([ids, nxt], dim=1)
            done = done | (nxt.squeeze(1) == eos_id)
            if done.all():
                break
            inp = self.gpt2.wte(nxt)  # KV cache: feed only the new token
        return ids

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
