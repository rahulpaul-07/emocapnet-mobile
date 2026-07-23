"""Shared fixtures: tiny config, synthetic dataset, tokenizer, and student model.

Everything is CPU-only. The only network access is the one-time GPT-2 tokenizer
download (~1 MB, cached by HF); tests are skipped cleanly if it is unreachable.
"""

from __future__ import annotations

import pytest

from emocapnet.config import Config, seed_everything
from emocapnet.data.synthetic import generate_synthetic_dataset


@pytest.fixture(scope="session")
def tok(tmp_path_factory):
    from emocapnet.tokenization import build_tokenizer, write_tiny_gpt2_tokenizer

    try:
        return build_tokenizer("gpt2")
    except Exception:  # offline environment -> hermetic byte-level tokenizer
        path = write_tiny_gpt2_tokenizer(tmp_path_factory.mktemp("tok"))
        return build_tokenizer(path)


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    csv_path, img_dir = generate_synthetic_dataset(out, n_images=16, captions_per_image=2, seed=0)
    return csv_path, img_dir


@pytest.fixture()
def tiny_cfg(synthetic_data, tmp_path) -> Config:
    csv_path, img_dir = synthetic_data
    return Config(
        captions_csv=csv_path,
        image_dirs=[img_dir],
        work_dir=str(tmp_path / "run"),
        student_encoder_type="tiny",
        ultralight=True,
        epochs=1,
        pretrain_epochs=0,
        batch_size=4,
        grad_accum=1,
        num_workers=0,
        max_length=24,
        prefix_len=4,
        hard_max_length=8,
        min_length=2,
        distill_alpha=0.0,
    )


@pytest.fixture()
def student(tiny_cfg, tok):
    from emocapnet.models.student import EmoCapNetMobile

    seed_everything(0)
    return EmoCapNetMobile(tiny_cfg, vocab_size=tok.vocab_size)
