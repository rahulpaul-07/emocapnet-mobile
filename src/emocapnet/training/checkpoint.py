"""Checkpoint save/load with embedded architecture config for safe reloads."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from emocapnet.config import Config
from emocapnet.training.schedule import unwrap
from emocapnet.utils import torch_load

log = logging.getLogger(__name__)


def checkpoint_payload(
    model: nn.Module, epoch: int, val_loss: float, history: list[dict], cfg: Config
) -> dict:
    inner = unwrap(model)
    return {
        "epoch": epoch,
        "model_state": inner.state_dict(),
        "val_loss": val_loss,
        "history": history,
        "dec_name": getattr(inner, "dec_name", "unknown"),
        "config": {
            "student_encoder_type": cfg.student_encoder_type,
            "student_vit": cfg.student_vit,
            "student_clip_model": cfg.student_clip_model,
            "student_decoder": cfg.student_decoder,
            "ultralight": cfg.ultralight,
            "num_emotions": cfg.num_emotions,
            "vad_dim": cfg.vad_dim,
            "max_length": cfg.max_length,
            "prefix_len": cfg.prefix_len,
        },
    }


def save_checkpoint(
    model: nn.Module, epoch: int, val_loss: float, history: list[dict], cfg: Config, is_best: bool
) -> None:
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, epoch, val_loss, history, cfg)
    torch.save(payload, ckpt_dir / "last.pt")
    if is_best:
        torch.save(payload, ckpt_dir / "best.pt")
        pd.DataFrame(history).to_csv(Path(cfg.work_dir) / "train_history.csv", index=False)
        log.info("★ new best val=%.4f @ epoch %d -> best.pt", val_loss, epoch)


def load_student_weights(model: nn.Module, path: str | Path, device: str | torch.device = "cpu") -> None:
    ckpt = torch_load(path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    unwrap(model).load_state_dict(state)
    log.info("Loaded student weights from %s", path)
