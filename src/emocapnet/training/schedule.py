"""Optimizer schedule and freeze/unfreeze helpers."""

from __future__ import annotations

import math

import torch
from torch import nn


def unwrap(model: nn.Module) -> nn.Module:
    """Return the inner module if wrapped in DataParallel/DDP."""
    return model.module if hasattr(model, "module") else model


def set_requires_grad(params, flag: bool) -> None:
    for p in params:
        p.requires_grad = flag


def freeze_encoder(model: nn.Module) -> None:
    set_requires_grad(unwrap(model).encoder.parameters(), False)


def unfreeze_encoder(model: nn.Module) -> None:
    set_requires_grad(unwrap(model).encoder.parameters(), True)


def warmup_cosine(
    optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay, stepped per optimizer step."""
    total = max(1, total_epochs * steps_per_epoch)
    warm = warmup_epochs * steps_per_epoch

    def f(step: int) -> float:
        if step < warm:
            return step / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, [f] * len(optimizer.param_groups))
