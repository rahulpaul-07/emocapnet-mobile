"""Dataset, image preprocessing, and dataloader collation.

Each item returns two image tensors: one preprocessed for the teacher's ViT (224px)
and one for the student encoder. The teacher sees its expected input so its soft
targets stay faithful; if distillation is disabled the teacher tensor is unused.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from emocapnet.config import Config
from emocapnet.constants import emotion_token
from emocapnet.tokenization import TokenizerBundle

log = logging.getLogger(__name__)

ImageTransform = Callable[[Image.Image], torch.Tensor]


@dataclass
class Processors:
    """Callable preprocessors mapping a PIL image to a model-ready tensor."""

    student: ImageTransform
    teacher: ImageTransform | None
    student_size: int


def _hf_processor_to_transform(proc) -> ImageTransform:
    return lambda img: proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)


def _plain_transform(size: int) -> ImageTransform:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_processors(cfg: Config, need_teacher: bool) -> Processors:
    """Build the student (and optionally teacher) image preprocessors.

    ``student_encoder_type == "tiny"`` uses plain torchvision transforms and never
    touches the network — this is what tests and the smoke pipeline run on.
    """
    kind = cfg.student_encoder_type
    if kind == "tiny":
        size = 64
        student: ImageTransform = _plain_transform(size)
    elif kind == "clip":
        from transformers import CLIPImageProcessor

        proc = CLIPImageProcessor.from_pretrained(cfg.student_clip_model)
        sz = getattr(proc, "crop_size", None) or getattr(proc, "size", {})
        size = sz.get("height", sz.get("shortest_edge", 224)) if isinstance(sz, dict) else 224
        student = _hf_processor_to_transform(proc)
    elif kind == "mobilevit":
        from transformers import MobileViTImageProcessor

        proc = MobileViTImageProcessor.from_pretrained(cfg.student_vit)
        size = proc.crop_size.get("height", 256) if isinstance(proc.crop_size, dict) else 256
        student = _hf_processor_to_transform(proc)
    else:
        raise ValueError(f"Unknown student_encoder_type: {kind!r}")

    teacher: ImageTransform | None = None
    if need_teacher:
        from transformers import ViTImageProcessor

        teacher = _hf_processor_to_transform(ViTImageProcessor.from_pretrained(cfg.teacher_vit))
    log.info("Processors ready: student=%s (%dpx), teacher=%s", kind, size, bool(teacher))
    return Processors(student=student, teacher=teacher, student_size=size)


def build_image_index(image_dirs: list[str]) -> dict[str, str]:
    """Index every image path once (shared across DataLoader workers)."""
    index: dict[str, str] = {}
    for d in image_dirs:
        if not os.path.isdir(d):
            log.warning("Image directory missing: %s", d)
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                index[f] = os.path.join(d, f)
    log.info("Indexed %d image paths", len(index))
    return index


def build_train_augmentation(size: int) -> Callable[[Image.Image], Image.Image]:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ]
    )


class EmoVADDataset(Dataset):
    """Caption rows -> (student pixels, teacher pixels, token ids, VAD, emotion)."""

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        tok: TokenizerBundle,
        processors: Processors,
        image_index: dict[str, str],
        is_train: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.tok = tok
        self.proc = processors
        self.image_index = image_index
        self.is_train = is_train
        self.augment = build_train_augmentation(processors.student_size) if is_train else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, name: str) -> Image.Image:
        path = self.image_index.get(name)
        size = self.proc.student_size
        try:
            return Image.open(path).convert("RGB") if path else Image.new("RGB", (size, size), 128)
        except Exception:
            return Image.new("RGB", (size, size), 128)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img = self._load_image(row["image"])
        if self.augment is not None:
            img = self.augment(img)

        pv_student = self.proc.student(img)
        pv_teacher = self.proc.teacher(img) if self.proc.teacher is not None else pv_student

        emotion = row["emotion"]
        caption = str(row["caption"]).strip()
        full_text = f"{emotion_token(emotion)} {caption}"
        vad = torch.tensor(
            [float(row["valence"]), float(row["arousal"]), float(row["dominance"])],
            dtype=torch.float32,
        )
        enc = self.tok.tokenizer(
            full_text,
            max_length=self.cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        labels = ids.clone()
        labels[mask == 0] = -100
        return {
            "pv_student": pv_student,
            "pv_teacher": pv_teacher,
            "input_ids": ids,
            "attention_mask": mask,
            "labels": labels,
            "vad": vad,
            "emotion_idx": torch.tensor(int(row["emotion_idx"]), dtype=torch.long),
            "image_name": row["image"],
            "emotion": emotion,
            "ref_caption": caption,
        }


def collate(batch: list[dict]) -> dict:
    out: dict = {}
    for key in batch[0]:
        if key in ("image_name", "emotion", "ref_caption"):
            out[key] = [item[key] for item in batch]
        else:
            out[key] = torch.stack([item[key] for item in batch])
    return out


def make_loader(
    df: pd.DataFrame,
    cfg: Config,
    tok: TokenizerBundle,
    processors: Processors,
    image_index: dict[str, str],
    shuffle: bool,
    is_train: bool,
) -> DataLoader:
    ds = EmoVADDataset(df, cfg, tok, processors, image_index, is_train=is_train)
    kwargs: dict = {}
    if cfg.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        **kwargs,
    )
