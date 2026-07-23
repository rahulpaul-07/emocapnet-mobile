"""Synthetic dataset generator.

Creates a tiny, fully self-contained dataset (colored-shape images + templated
emotional captions with VAD scores) so the entire pipeline — training,
distillation plumbing, evaluation, quantization — can be exercised end-to-end
without any external data or network access. Used by unit tests, the CI smoke
job, and ``emocapnet smoke``.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from emocapnet.constants import EMOTION_LABELS
from emocapnet.constants import EMOTION_VAD as _EMO_VAD

_SUBJECTS = ["a red circle", "a blue square", "a green triangle", "a yellow star", "an orange dot"]
_VERBS = ["sits", "floats", "rests", "appears", "stands"]
_PLACES = ["on a gray field", "near the corner", "in the middle", "against the background", "by the edge"]
_EMO_PHRASES = {
    "factual": "in plain view",
    "happiness": "glowing with cheerful bright light",
    "sadness": "looking dim and forlorn",
    "anger": "burning with harsh sharp edges",
    "fear": "shrinking into the dark shadows",
    "surprise": "bursting suddenly into view",
    "disgust": "smeared with an unpleasant stain",
}

_COLORS = {
    "red": (200, 40, 40),
    "blue": (40, 60, 200),
    "green": (40, 160, 60),
    "yellow": (220, 200, 40),
    "orange": (230, 130, 30),
}


def _draw_image(rng: random.Random, size: int = 96) -> Image.Image:
    img = Image.new("RGB", (size, size), (rng.randint(90, 170),) * 3)
    draw = ImageDraw.Draw(img)
    color = rng.choice(list(_COLORS.values()))
    x, y = rng.randint(8, size - 40), rng.randint(8, size - 40)
    w = rng.randint(16, 36)
    shape = rng.choice(["ellipse", "rectangle", "triangle"])
    if shape == "ellipse":
        draw.ellipse([x, y, x + w, y + w], fill=color)
    elif shape == "rectangle":
        draw.rectangle([x, y, x + w, y + w], fill=color)
    else:
        draw.polygon([(x, y + w), (x + w // 2, y), (x + w, y + w)], fill=color)
    return img


def _caption(rng: random.Random, emotion: str) -> str:
    return f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_PLACES)} {_EMO_PHRASES[emotion]}"


def generate_synthetic_dataset(
    out_dir: str | Path, n_images: int = 40, captions_per_image: int = 2, seed: int = 0
) -> tuple[str, str]:
    """Write ``images/`` and ``captions.csv`` under *out_dir*.

    Returns ``(captions_csv_path, images_dir_path)``.
    """
    rng = random.Random(seed)
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_images):
        image_id = f"synthetic_{i:04d}"
        _draw_image(rng).save(img_dir / f"{image_id}.jpg")
        for _ in range(captions_per_image):
            emotion = rng.choice(EMOTION_LABELS)
            v, a, d = _EMO_VAD[emotion]
            jitter = lambda x: min(1.0, max(0.0, x + rng.uniform(-0.05, 0.05)))  # noqa: E731
            rows.append(
                {
                    "image_id": image_id,
                    "emotion": emotion,
                    "caption": _caption(rng, emotion),
                    "valence": round(jitter(v), 3),
                    "arousal": round(jitter(a), 3),
                    "dominance": round(jitter(d), 3),
                }
            )
    csv_path = out / "captions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(csv_path), str(img_dir)
