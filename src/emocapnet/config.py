"""Configuration: a typed dataclass loadable from YAML with dot-key overrides.

Usage::

    cfg = load_config("configs/default.yaml", overrides=["train.epochs=2"])
"""

from __future__ import annotations

import dataclasses
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger(__name__)


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────
    captions_csv: str = "data/captions.csv"
    teacher_ckpt: str = ""  # path to trained EmoCapNet v3 best.pt; empty = no distillation
    image_dirs: list[str] = field(default_factory=lambda: ["data/images"])
    pretrain_captions_csv: str = ""  # optional Flickr30k-style (image, caption) file for Stage 0
    work_dir: str = "runs/default"

    # ── Teacher architecture (must match how the checkpoint was trained) ──
    teacher_vit: str = "google/vit-base-patch16-224-in21k"
    teacher_gpt2: str = "gpt2"

    # ── Student architecture ─────────────────────────────────────────
    # encoder: "clip" (TinyCLIP), "mobilevit", or "tiny" (dependency-free conv net for smoke tests)
    student_encoder_type: str = "clip"
    student_clip_model: str = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
    student_vit: str = "apple/mobilevit-small"
    # decoder: "distilgpt2" (pretrained) or "ultralight" (4-layer/512-dim scratch decoder)
    student_decoder: str = "distilgpt2"
    ultralight: bool = False
    tokenizer_name: str = "gpt2"

    # ── Shared task config ───────────────────────────────────────────
    num_emotions: int = 7
    vad_dim: int = 3
    max_length: int = 40
    prefix_len: int = 10

    # ── Stage 0: language pre-finetuning ─────────────────────────────
    pretrain_epochs: int = 3

    # ── Training ─────────────────────────────────────────────────────
    epochs: int = 12
    batch_size: int = 16
    grad_accum: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    num_workers: int = 4
    warmup_epochs: int = 1
    label_smoothing: float = 0.1
    unfreeze_vit_epoch: int = 1
    use_data_parallel: bool = False
    seed: int = 42

    # ── Distillation ─────────────────────────────────────────────────
    distill_alpha: float = 0.6
    distill_temp: float = 2.0
    distill_vad_emo: bool = True
    feat_weight: float = 0.5

    # ── Multi-task loss weights ──────────────────────────────────────
    lm_weight: float = 1.0
    vad_weight: float = 0.3
    emotion_weight: float = 0.15
    contrastive_weight: float = 0.15

    # ── Decoding ─────────────────────────────────────────────────────
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 3
    min_length: int = 6
    hard_max_length: int = 20

    # ── Derived (filled in __post_init__) ────────────────────────────
    ckpt_dir: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.ckpt_dir = str(Path(self.work_dir) / "checkpoints")

    # ---------------------------------------------------------------- helpers
    @property
    def image_size_fallback(self) -> int:
        return 224

    def ensure_dirs(self) -> None:
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        for f in dataclasses.fields(self):
            if not f.init:  # derived fields are recomputed, never serialized
                data.pop(f.name, None)
        return data

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


_FIELD_TYPES = {f.name: f.type for f in dataclasses.fields(Config) if f.init}


def _coerce(name: str, value: str):
    """Coerce a CLI override string to the field's declared type."""
    current = getattr(Config(), name, None)
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a :class:`Config` from a YAML file, then apply ``key=value`` overrides."""
    data: dict = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - set(_FIELD_TYPES)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        data.update(raw)
    cfg = Config(**data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in _FIELD_TYPES:
            raise ValueError(f"Unknown config key: {key}")
        setattr(cfg, key, _coerce(key, value))
    cfg.__post_init__()  # refresh derived paths after overrides
    return cfg


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + CUDA) for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
