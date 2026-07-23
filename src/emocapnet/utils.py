"""Small cross-version torch helpers."""

from __future__ import annotations

import torch


def torch_load(path, map_location="cpu"):
    """``torch.load`` that works on torch versions with and without ``weights_only``."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location=map_location)


def make_grad_scaler(device_type: str, enabled: bool):
    """GradScaler across the torch 2.3 API change (and older CUDA-only API)."""
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled and device_type == "cuda")
