"""Student vision encoders behind a single interface.

Every encoder is an ``nn.Module`` whose ``forward(pixel_values)`` returns
``(pooled, spatial)`` where ``pooled`` is ``(B, D)`` and ``spatial`` is
``(B, N, D)`` patch/grid features, and which exposes ``out_dim``.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from emocapnet.config import Config

log = logging.getLogger(__name__)


class CLIPEncoder(nn.Module):
    """TinyCLIP vision tower — image-text aligned from the start, better grounding."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        from transformers import CLIPVisionModel

        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        self.out_dim = self.backbone.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(pixel_values=pixel_values)
        return out.pooler_output, out.last_hidden_state[:, 1:]  # drop CLS from spatial


class MobileViTEncoder(nn.Module):
    """apple/mobilevit-small — ~5.6M params, mobile-grade convolutional ViT."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        from transformers import MobileViTModel

        self.backbone = MobileViTModel.from_pretrained(model_name)
        self.out_dim = self.backbone.config.neck_hidden_sizes[-1]

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(pixel_values=pixel_values)
        spatial = out.last_hidden_state.flatten(2).transpose(1, 2)  # (B, HW, C)
        return out.pooler_output, spatial


class TinyConvEncoder(nn.Module):
    """Dependency-free conv encoder for tests and smoke runs (no downloads).

    Not intended for real accuracy — it exists so the full pipeline can run
    hermetically in CI.
    """

    def __init__(self, out_dim: int = 128, in_size: int = 64) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)  # -> (B, D, 4, 4) = 16 spatial tokens

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fm = self.pool(self.features(pixel_values))
        spatial = fm.flatten(2).transpose(1, 2)  # (B, 16, D)
        return spatial.mean(dim=1), spatial


def build_student_encoder(cfg: Config) -> nn.Module:
    kind = cfg.student_encoder_type
    if kind == "clip":
        try:
            return CLIPEncoder(cfg.student_clip_model)
        except Exception as e:  # download/auth failure -> graceful fallback
            log.warning("CLIP encoder unavailable (%r) -> falling back to MobileViT", e)
            cfg.student_encoder_type = "mobilevit"
            kind = "mobilevit"
    if kind == "mobilevit":
        return MobileViTEncoder(cfg.student_vit)
    if kind == "tiny":
        return TinyConvEncoder()
    raise ValueError(f"Unknown student_encoder_type: {kind!r}")
