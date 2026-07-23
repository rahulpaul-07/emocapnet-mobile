"""VAD controllability: does a requested VAD actually steer the caption?

Measured with the *independent* NRC-VAD lexicon (human word ratings), falling
back to NLTK-VADER valence-only if the download fails. We deliberately do NOT
score with our own ``vad_head`` — that would be circular.
"""

from __future__ import annotations

import io
import logging
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from emocapnet.config import Config
from emocapnet.constants import EMOTION_LABELS
from emocapnet.tokenization import TokenizerBundle, strip_caption
from emocapnet.training.schedule import unwrap

log = logging.getLogger(__name__)

NRC_VAD_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip"


def load_nrc_vad_lexicon(timeout: int = 60) -> dict[str, tuple[float, float, float]]:
    """Download and parse the NRC-VAD lexicon. Returns {} on failure."""
    lex: dict[str, tuple[float, float, float]] = {}
    try:
        raw = urllib.request.urlopen(NRC_VAD_URL, timeout=timeout).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        fn = next(n for n in zf.namelist() if n.endswith("NRC-VAD-Lexicon.txt"))
        import contextlib

        for line in zf.read(fn).decode("utf-8").splitlines()[1:]:
            p = line.split("\t")
            if len(p) >= 4:
                with contextlib.suppress(ValueError):
                    lex[p[0]] = (float(p[1]), float(p[2]), float(p[3]))
        log.info("NRC-VAD lexicon loaded: %d words", len(lex))
    except Exception as e:
        log.warning("NRC-VAD download failed (%r) -> VADER fallback (valence only)", e)
    return lex


def text_vad(caption: str, lex: dict) -> tuple | None:
    """Read VAD back from the text with the independent probe."""
    if lex:
        vs = [lex[w] for w in re.findall(r"[a-z']+", caption.lower()) if w in lex]
        return tuple(float(np.mean([v[k] for v in vs])) for k in range(3)) if vs else None
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        nltk.download("vader_lexicon", quiet=True)
        sc = SentimentIntensityAnalyzer().polarity_scores(caption)["compound"]
        return ((sc + 1) / 2, None, None)
    except Exception:
        return None


@torch.no_grad()
def controllability_report(
    model,
    df: pd.DataFrame,
    image_index: dict[str, str],
    student_transform,
    cfg: Config,
    tok: TokenizerBundle,
    n_images: int = 30,
    lex: dict | None = None,
) -> dict[str, float]:
    """Pearson r between requested VAD and VAD read back from generated text.

    Generates one caption per (image, emotion) pair over ``n_images`` test images.
    """
    if lex is None:
        lex = load_nrc_vad_lexicon()
    emo_vad = df.groupby("emotion")[["valence", "arousal", "dominance"]].mean()
    inner = unwrap(model).eval()
    device = next(inner.parameters()).device

    req, prd = [], []
    for name in df["image"].drop_duplicates().head(n_images):
        path = image_index.get(name)
        if not path:
            continue
        img = Image.open(path).convert("RGB")
        pv = student_transform(img).unsqueeze(0).to(device)
        for e in EMOTION_LABELS:
            if e not in emo_vad.index:
                continue
            v = torch.tensor(emo_vad.loc[e].values, dtype=torch.float32).view(1, 3).to(device)
            gen = inner.generate(pv, v, torch.tensor([tok.emo_token_ids[e]], device=device), tok.eos_id, cfg)
            tv = text_vad(strip_caption(tok.tokenizer.decode(gen[0], skip_special_tokens=False)), lex)
            if tv is not None:
                req.append(emo_vad.loc[e].values)
                prd.append(tv)

    result: dict[str, float] = {}
    if not req:
        return result
    req_arr = np.array(req, dtype=float)
    prd_arr = np.array(prd, dtype=object)
    for k, dim in enumerate(["valence", "arousal", "dominance"]):
        col = np.array([row[k] if row[k] is not None else np.nan for row in prd_arr], dtype=float)
        mask = ~np.isnan(col)
        if mask.sum() >= 3 and np.std(req_arr[mask, k]) > 0 and np.std(col[mask]) > 0:
            result[dim] = float(np.corrcoef(req_arr[mask, k], col[mask])[0, 1])
    return result
