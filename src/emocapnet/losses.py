"""Distillation + multi-task losses.

``combined_loss`` = α·KD + (1−α)·hard-LM + VAD + emotion + contrastive + feature
alignment. KD is the KL divergence between the temperature-softened student and
teacher next-token distributions, computed only on real caption tokens (padding
ignored). With ``distill_alpha = 0`` this reduces exactly to the v3 objective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from emocapnet.config import Config


def lm_loss(logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.1) -> torch.Tensor:
    """Shifted cross-entropy over caption tokens (``-100`` = ignore)."""
    sl = logits[:, :-1].contiguous()
    lb = labels[:, 1:].contiguous()
    return F.cross_entropy(
        sl.view(-1, sl.size(-1)), lb.view(-1), ignore_index=-100, label_smoothing=label_smoothing
    )


def kd_logit_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Temperature-scaled KL(student ‖ teacher) on real caption tokens only."""
    s = student_logits[:, :-1].contiguous()
    t = teacher_logits[:, :-1].contiguous()
    lb = labels[:, 1:].contiguous()
    mask = lb != -100
    if mask.sum() == 0:
        return student_logits.new_zeros(())
    s, t = s[mask], t[mask]
    T = temperature
    return F.kl_div(F.log_softmax(s / T, -1), F.softmax(t / T, -1), reduction="batchmean") * (T * T)


def vad_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def emotion_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


def contrastive_loss(img_emb: torch.Tensor, cap_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE between image and caption embeddings."""
    scale = torch.clamp(logit_scale.exp(), max=100.0)
    labels = torch.arange(img_emb.size(0), device=img_emb.device)
    sim = scale * img_emb @ cap_emb.t()
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2


def combined_loss(
    student_out: tuple,
    batch: dict,
    teacher_out: tuple | None,
    cfg: Config,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full training objective. Returns (total, per-term scalars for logging)."""
    s_logits, s_vad, s_emo, s_img_emb, s_cap_emb = student_out
    labels, vad, emo = batch["labels"], batch["vad"], batch["emotion_idx"]

    hard_lm = lm_loss(s_logits, labels, cfg.label_smoothing)
    feat = s_logits.new_zeros(())  # grounding: image-feature distillation

    if teacher_out is not None and cfg.distill_alpha > 0:
        t_logits, t_vad, t_emo, t_img_emb, _ = teacher_out
        kd = kd_logit_loss(s_logits, t_logits, labels, cfg.distill_temp)
        lm_term = cfg.distill_alpha * kd + (1 - cfg.distill_alpha) * hard_lm
        if cfg.distill_vad_emo:
            vd = F.mse_loss(s_vad, t_vad.detach())  # mimic teacher VAD
            em = F.kl_div(F.log_softmax(s_emo, -1), F.softmax(t_emo.detach(), -1), reduction="batchmean")
        else:
            vd = vad_loss(s_vad, vad)
            em = emotion_loss(s_emo, emo)
        # Pull the student image embedding toward the teacher's (transfers what the
        # bigger ViT "sees" into the small encoder). Skipped silently on dim mismatch.
        if s_img_emb.shape[-1] == t_img_emb.shape[-1]:
            feat = (1 - F.cosine_similarity(s_img_emb, t_img_emb.detach(), dim=-1)).mean()
        kd_val = float(kd)
    else:
        lm_term = hard_lm
        vd = vad_loss(s_vad, vad)
        em = emotion_loss(s_emo, emo)
        kd_val = 0.0

    con = contrastive_loss(s_img_emb, s_cap_emb, logit_scale)
    total = (
        cfg.lm_weight * lm_term
        + cfg.vad_weight * vd
        + cfg.emotion_weight * em
        + cfg.contrastive_weight * con
        + cfg.feat_weight * feat
    )
    parts = {
        "lm": float(hard_lm),
        "kd": kd_val,
        "vad": float(vd),
        "emo": float(em),
        "con": float(con),
        "ft": float(feat),
    }
    return total, parts
