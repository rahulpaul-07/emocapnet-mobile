"""ONNX export of the student model + onnxruntime greedy decoding.

The model is split into two graphs, the standard pattern for encoder-decoder
captioners:

* ``encoder.onnx``  — pixels + VAD  ->  visual prefix + FiLM-conditioned
  cross-attention memory (runs once per image).
* ``decoder.onnx``  — token ids + prefix + memory  ->  next-token logits
  (runs once per generated token, no KV cache; the cache-less graph keeps the
  export simple and portable — see the KV-cache note in the README).

``OnnxCaptioner`` reimplements the exact greedy loop of
``EmoCapNetMobile.generate`` (repetition penalty, no-repeat-ngram, min-length)
on top of onnxruntime, and ``verify_parity`` asserts token-for-token agreement
with the torch decoder.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from emocapnet.config import Config
from emocapnet.training.schedule import unwrap

log = logging.getLogger(__name__)

OPSET = 14


def _onnx_export(module: nn.Module, args: tuple, path: str, **kw) -> None:
    """Version-robust ``torch.onnx.export``.

    torch >= 2.6 defaults to the dynamo exporter (needs ``onnxscript``); we prefer
    the legacy TorchScript exporter when available for stable dynamic_axes
    semantics, falling back to the dynamo path if legacy is removed. Older torch
    has no ``dynamo`` kwarg at all. Either way, ``verify_parity`` is the arbiter.
    """
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        try:
            torch.onnx.export(module, args, path, dynamo=False, **kw)
            return
        except Exception as e:
            log.warning("Legacy ONNX exporter failed (%r) -> trying dynamo exporter", e)
        torch.onnx.export(module, args, path, dynamo=True, **kw)
    else:
        torch.onnx.export(module, args, path, **kw)


class _EncoderWrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor, vad: torch.Tensor):
        _, spatial_kv, prefix = self.model.encode_image(pixel_values, vad)
        return prefix, spatial_kv


class _DecoderWrapper(nn.Module):
    """Full-sequence decoder step: (ids, prefix, memory) -> logits for the last token."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, prefix: torch.Tensor, spatial_kv: torch.Tensor):
        m = self.model
        inp = torch.cat([prefix, m.gpt2.wte(input_ids)], dim=1)
        out = m.gpt2(inputs_embeds=inp, encoder_hidden_states=spatial_kv, use_cache=False)
        return m.lm_head(out.last_hidden_state[:, -1])


def export_onnx(model, cfg: Config, out_dir: str | Path, image_size: int, sample_vocab_id: int = 1) -> dict:
    """Export encoder/decoder ONNX graphs plus a small metadata file.

    Returns the metadata dict (paths + shapes).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inner = unwrap(model).to("cpu").eval()

    pv = torch.randn(1, 3, image_size, image_size)
    vad = torch.rand(1, 3)
    with torch.no_grad():
        _, spatial_kv, prefix = inner.encode_image(pv, vad)
    ids = torch.tensor([[sample_vocab_id, sample_vocab_id + 1]], dtype=torch.long)

    enc_path, dec_path = out / "encoder.onnx", out / "decoder.onnx"
    _onnx_export(
        _EncoderWrapper(inner),
        (pv, vad),
        str(enc_path),
        opset_version=OPSET,
        input_names=["pixel_values", "vad"],
        output_names=["prefix", "spatial_kv"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "vad": {0: "batch"},
            "prefix": {0: "batch"},
            "spatial_kv": {0: "batch"},
        },
    )
    _onnx_export(
        _DecoderWrapper(inner),
        (ids, prefix, spatial_kv),
        str(dec_path),
        opset_version=OPSET,
        input_names=["input_ids", "prefix", "spatial_kv"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "prefix": {0: "batch"},
            "spatial_kv": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    meta = {
        "encoder": enc_path.name,
        "decoder": dec_path.name,
        "image_size": image_size,
        "opset": OPSET,
        "prefix_len": cfg.prefix_len,
        "decoding": {
            "hard_max_length": cfg.hard_max_length,
            "min_length": cfg.min_length,
            "repetition_penalty": cfg.repetition_penalty,
            "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
        },
        "sizes_mb": {
            "encoder": round(enc_path.stat().st_size / 1e6, 1),
            "decoder": round(dec_path.stat().st_size / 1e6, 1),
        },
    }
    (out / "onnx_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info(
        "ONNX export: encoder %.1f MB, decoder %.1f MB -> %s",
        meta["sizes_mb"]["encoder"],
        meta["sizes_mb"]["decoder"],
        out,
    )
    return meta


class OnnxCaptioner:
    """Greedy decoding on onnxruntime, mirroring ``EmoCapNetMobile.generate`` exactly."""

    def __init__(self, onnx_dir: str | Path, cfg: Config) -> None:
        import onnxruntime as ort

        onnx_dir = Path(onnx_dir)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.enc = ort.InferenceSession(
            str(onnx_dir / "encoder.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.dec = ort.InferenceSession(
            str(onnx_dir / "decoder.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.cfg = cfg

    def generate(self, pixel_values: np.ndarray, vad: np.ndarray, emotion_id: int, eos_id: int) -> list[int]:
        cfg = self.cfg
        prefix, spatial_kv = self.enc.run(
            None, {"pixel_values": pixel_values.astype(np.float32), "vad": vad.astype(np.float32)}
        )
        ids = [int(emotion_id)]
        for step in range(cfg.hard_max_length):
            logits = self.dec.run(
                None,
                {
                    "input_ids": np.asarray([ids], dtype=np.int64),
                    "prefix": prefix,
                    "spatial_kv": spatial_kv,
                },
            )[0][0]
            # repetition penalty (identical to torch loop)
            for tok in set(ids):
                if logits[tok] > 0:
                    logits[tok] /= cfg.repetition_penalty
                else:
                    logits[tok] *= cfg.repetition_penalty
            # no-repeat-ngram
            n = cfg.no_repeat_ngram_size
            if n > 0 and len(ids) >= n - 1:
                prefix_gram = tuple(ids[-(n - 1) :])
                for i in range(len(ids) - n + 1):
                    if tuple(ids[i : i + n - 1]) == prefix_gram:
                        logits[ids[i + n - 1]] = -1e9
            if step < cfg.min_length:
                logits[eos_id] = -1e9
            nxt = int(np.argmax(logits))
            ids.append(nxt)
            if nxt == eos_id:
                break
        return ids


def verify_parity(
    model, captioner: OnnxCaptioner, cfg: Config, image_size: int, emotion_id: int, eos_id: int, seed: int = 0
) -> bool:
    """Assert ORT and torch greedy decoding agree token-for-token on a random input."""
    g = torch.Generator().manual_seed(seed)
    pv = torch.randn(1, 3, image_size, image_size, generator=g)
    vad = torch.rand(1, 3, generator=g)
    inner = unwrap(model).to("cpu").eval()
    torch_ids = inner.generate(pv, vad, torch.tensor([emotion_id]), eos_id, cfg)[0].tolist()
    ort_ids = captioner.generate(pv.numpy(), vad.numpy(), emotion_id, eos_id)
    if torch_ids != ort_ids:
        raise AssertionError(f"ONNX/torch divergence:\n  torch: {torch_ids}\n  onnx : {ort_ids}")
    log.info("ONNX parity verified: %d tokens identical", len(ort_ids))
    return True
