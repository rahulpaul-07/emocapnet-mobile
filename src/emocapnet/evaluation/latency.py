"""CPU latency benchmark — the honest 'mobile' number."""

from __future__ import annotations

import copy
import time

import torch

from emocapnet.config import Config
from emocapnet.tokenization import TokenizerBundle
from emocapnet.training.schedule import unwrap


@torch.no_grad()
def cpu_latency_ms(
    model,
    pixel_values: torch.Tensor,
    vad: torch.Tensor,
    emotion_id: int,
    cfg: Config,
    tok: TokenizerBundle,
    repeats: int = 10,
    copy_to_cpu: bool = True,
) -> float:
    """Mean milliseconds per generated caption on CPU (greedy, KV-cache)."""
    m = copy.deepcopy(unwrap(model)).to("cpu").eval() if copy_to_cpu else unwrap(model).eval()
    pv = pixel_values.to("cpu")
    v = vad.to("cpu")
    ei = torch.tensor([emotion_id])
    m.generate(pv, v, ei, tok.eos_id, cfg)  # warmup
    t0 = time.time()
    for _ in range(repeats):
        m.generate(pv, v, ei, tok.eos_id, cfg)
    return (time.time() - t0) / repeats * 1000
