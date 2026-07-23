"""INT8 dynamic quantization (plus an FP16 safety net).

Dynamic quantization converts Linear layers (the bulk of the weights) to INT8
with no calibration data — the standard first step for shipping a transformer to
CPU/mobile. HF GPT-2 uses ``Conv1D`` (weight ``(in, out)``, ``y = x@W + b``); we
convert it to a numerically identical ``nn.Linear`` first so dynamic quant can
compress it.

For an actual phone build the next hop is ExecuTorch or ONNX Runtime Mobile /
TFLite; dynamic INT8 here is the portable, framework-agnostic proof of the size
win.
"""

from __future__ import annotations

import copy
import io
import logging

import torch
from torch import nn

try:
    from torch.ao.quantization import (
        default_dynamic_qconfig,
        float_qparams_weight_only_qconfig,
        quantize_dynamic,
    )
except ImportError:  # older torch
    from torch.quantization import (  # type: ignore[no-redef]
        default_dynamic_qconfig,
        float_qparams_weight_only_qconfig,
        quantize_dynamic,
    )

from emocapnet.training.schedule import unwrap

log = logging.getLogger(__name__)


def state_size_mb(model_or_state) -> float:
    """Serialized state-dict size in MB."""
    obj = unwrap(model_or_state).state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getbuffer().nbytes / 1e6


def convert_conv1d_to_linear(module: nn.Module) -> nn.Module:
    """Recursively replace HF ``Conv1D`` with numerically identical ``nn.Linear``."""
    from transformers.pytorch_utils import Conv1D

    for name, child in list(module.named_children()):
        if isinstance(child, Conv1D):
            nx, nf = child.weight.shape
            lin = nn.Linear(nx, nf, bias=child.bias is not None)
            with torch.no_grad():
                lin.weight.copy_(child.weight.t().contiguous())
                if child.bias is not None:
                    lin.bias.copy_(child.bias)
            setattr(module, name, lin)
        else:
            convert_conv1d_to_linear(child)
    return module


def to_fp16(model: nn.Module) -> nn.Module:
    """FP16 copy: reliable ~2x size cut, near-zero accuracy loss, GPU-runnable."""
    return copy.deepcopy(unwrap(model)).to("cpu").half().eval()


def quantize_int8(model: nn.Module) -> nn.Module:
    """Dynamic INT8 copy (Linear + Embedding) of a CPU model."""
    cvt = convert_conv1d_to_linear(copy.deepcopy(unwrap(model)).to("cpu").eval())
    qspec = {
        nn.Linear: default_dynamic_qconfig,
        nn.Embedding: float_qparams_weight_only_qconfig,
    }
    return quantize_dynamic(cvt, qspec, dtype=torch.qint8).eval()


def compression_summary(student: nn.Module, teacher: nn.Module | None = None) -> dict:
    """Size/params table for FP32 / FP16 / INT8 builds of the student."""
    inner = unwrap(student).to("cpu").eval()
    fp32_mb = state_size_mb(inner)
    fp16_mb = state_size_mb(to_fp16(inner))
    int8_mb = None
    try:
        int8_mb = state_size_mb(quantize_int8(inner))
    except Exception as e:
        log.warning("INT8 quantization failed in this build: %r", e)
    params_m = sum(p.numel() for p in inner.parameters()) / 1e6
    out = {
        "student_params_M": round(params_m, 2),
        "fp32_mb": round(fp32_mb, 1),
        "fp16_mb": round(fp16_mb, 1),
        "int8_mb": round(int8_mb, 1) if int8_mb else None,
    }
    if teacher is not None:
        tp = sum(p.numel() for p in unwrap(teacher).parameters()) / 1e6
        out["teacher_params_M"] = round(tp, 2)
        out["param_compression_x"] = round(tp / params_m, 1)
        if int8_mb:
            out["disk_compression_x"] = round((tp * 4) / int8_mb, 1)
    return out
