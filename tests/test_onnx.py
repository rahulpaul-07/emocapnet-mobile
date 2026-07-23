"""ONNX export + onnxruntime parity tests (tiny model)."""

from __future__ import annotations

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.mark.slow
def test_export_and_parity(student, tiny_cfg, tok, tmp_path):
    from emocapnet.export.onnx_export import OnnxCaptioner, export_onnx, verify_parity

    image_size = 64
    meta = export_onnx(student, tiny_cfg, tmp_path, image_size)
    assert (tmp_path / "encoder.onnx").exists()
    assert (tmp_path / "decoder.onnx").exists()
    assert meta["sizes_mb"]["decoder"] > 0

    captioner = OnnxCaptioner(tmp_path, tiny_cfg)
    # token-for-token agreement with the torch greedy decoder, several seeds/emotions
    for seed, emotion in [(0, "happiness"), (1, "fear"), (2, "factual")]:
        assert verify_parity(
            student, captioner, tiny_cfg, image_size, tok.emo_token_ids[emotion], tok.eos_id, seed=seed
        )
