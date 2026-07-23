import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from emocapnet.compression.quantize import (
    compression_summary,
    convert_conv1d_to_linear,
    quantize_int8,
    state_size_mb,
    to_fp16,
)


def test_conv1d_to_linear_is_lossless():
    conv = Conv1D(nf=12, nx=8)  # y = x @ W + b, weight (nx, nf)
    wrapper = nn.Sequential(conv)
    x = torch.randn(3, 8)
    before = wrapper(x)
    convert_conv1d_to_linear(wrapper)
    assert isinstance(wrapper[0], nn.Linear)
    after = wrapper(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_int8_shrinks_and_still_runs(student):
    fp32 = state_size_mb(student)
    q = quantize_int8(student)
    int8 = state_size_mb(q)
    assert int8 < fp32 * 0.6, f"expected >40% size cut, got {fp32:.1f} -> {int8:.1f} MB"
    # quantized forward still works
    B, L = 1, 8
    pv = torch.randn(B, 3, 64, 64)
    ids = torch.randint(0, 100, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    out = q(pv, ids, mask, torch.rand(B, 3))
    assert torch.isfinite(out[0]).all()


def test_fp16_halves_size(student):
    fp32 = state_size_mb(student)
    fp16 = state_size_mb(to_fp16(student))
    assert fp16 < fp32 * 0.6


def test_compression_summary_keys(student):
    s = compression_summary(student)
    assert s["fp32_mb"] > 0
    assert s["student_params_M"] > 0
    assert s["int8_mb"] is None or s["int8_mb"] < s["fp32_mb"]
