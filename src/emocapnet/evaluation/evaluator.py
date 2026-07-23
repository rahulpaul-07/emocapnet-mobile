"""Generation-based evaluation (BLEU/METEOR/ROUGE-L/CIDEr + VAD-MSE)."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from emocapnet.config import Config
from emocapnet.evaluation.metrics import cider_score, compute_caption_metrics
from emocapnet.tokenization import TokenizerBundle, strip_caption
from emocapnet.training.schedule import unwrap

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_captions(
    model,
    loader: DataLoader,
    cfg: Config,
    tok: TokenizerBundle,
    device: torch.device | str,
    use_student_pixels: bool = True,
    tag: str = "model",
    with_cider: bool = True,
) -> dict:
    """Generate one caption per test row (seeded with its ``<emo=...>`` token) and
    score against the reference. Per-emotion breakdown included, since emotion
    control is the whole point of the model."""
    model.eval()
    inner = unwrap(model)
    refs, hyps, emos = [], [], []
    for batch in tqdm(loader, desc=f"eval[{tag}]", leave=False):
        vad = batch["vad"].to(device)
        emo_ids = torch.tensor([tok.emo_token_ids[e] for e in batch["emotion"]], device=device)
        pv = (batch["pv_student"] if use_student_pixels else batch["pv_teacher"]).to(device)
        gen = inner.generate(pv, vad, emo_ids, tok.eos_id, cfg)
        for i in range(gen.size(0)):
            hyp = strip_caption(tok.tokenizer.decode(gen[i], skip_special_tokens=False))
            refs.append([batch["ref_caption"][i].split()])
            hyps.append(hyp.split())
            emos.append(batch["emotion"][i])
    metrics = compute_caption_metrics(refs, hyps, emos)
    if with_cider:
        metrics["cider"] = cider_score(hyps, refs)
    metrics["hyps"] = hyps
    metrics["refs"] = refs
    return metrics


@torch.no_grad()
def vad_mse(model, loader: DataLoader, device: torch.device | str, use_student_pixels: bool = True) -> float:
    """MSE of the VAD regression head against ground-truth VAD (teacher-forced)."""
    model.eval()
    inner = unwrap(model)
    se, n = 0.0, 0
    for batch in tqdm(loader, desc="vad-mse", leave=False):
        vad = batch["vad"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        pv = (batch["pv_student"] if use_student_pixels else batch["pv_teacher"]).to(device)
        out = inner(pv, ids, mask, vad)
        se += F.mse_loss(out[1], vad, reduction="sum").item()
        n += vad.numel()
    return se / max(n, 1)


def format_metrics(m: dict, tag: str) -> str:
    parts = [f"=== {tag} ==="]
    parts.append(
        f"BLEU-1 {m['bleu1']:.4f} | BLEU-4 {m['bleu4']:.4f} | METEOR {m['meteor']:.4f} | "
        f"ROUGE-L {m['rougeL']:.4f}"
        + (f" | CIDEr {m['cider']:.3f}" if "cider" in m else "")
        + (f" | VAD-MSE {m['vad_mse']:.5f}" if "vad_mse" in m else "")
    )
    if m.get("per_emotion"):
        parts.append("per-emotion BLEU-4: " + str({k: round(v, 4) for k, v in m["per_emotion"].items()}))
    return "\n".join(parts)
