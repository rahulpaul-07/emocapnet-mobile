"""Loading, cleaning, and splitting the emotional-captions CSV.

Split logic is identical to EmoCapNet v3 (image-level 80/10/10 with fixed seed) so
test-set numbers stay comparable across model generations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from emocapnet.config import Config
from emocapnet.constants import EMOTION2IDX
from emocapnet.data.dataset import Processors, build_image_index, build_processors, make_loader
from emocapnet.tokenization import TokenizerBundle

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"image_id", "emotion", "caption", "valence", "arousal", "dominance"}


def load_captions(csv_path: str, min_words: int = 4, max_words: int = 30) -> pd.DataFrame:
    """Load and clean the captions CSV, adding ``image`` and ``emotion_idx`` columns."""
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    df["image"] = df["image_id"].astype(str).apply(lambda x: x if x.endswith(".jpg") else x + ".jpg")
    df["emotion_idx"] = df["emotion"].map(EMOTION2IDX)
    if df["emotion_idx"].isna().any():
        bad = df.loc[df["emotion_idx"].isna(), "emotion"].unique()
        raise ValueError(f"Unknown emotion labels in CSV: {list(bad)}")
    df["caption"] = df["caption"].astype(str).str.strip().str.rstrip(".").str.strip()
    df = df[df["caption"].str.split().str.len().between(min_words, max_words)].reset_index(drop=True)
    log.info("Loaded %d captions (%d unique images) from %s", len(df), df["image"].nunique(), csv_path)
    return df


def split_by_image(
    df: pd.DataFrame, test_frac: float = 0.10, val_frac: float = 0.10, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split at the *image* level so no image leaks across train/val/test."""
    unique_imgs = df["image"].drop_duplicates().reset_index(drop=True)
    train_imgs, test_imgs = train_test_split(unique_imgs, test_size=test_frac, random_state=seed)
    train_imgs, val_imgs = train_test_split(
        train_imgs, test_size=val_frac / (1 - test_frac), random_state=seed
    )
    parts = tuple(
        df[df["image"].isin(imgs)].reset_index(drop=True) for imgs in (train_imgs, val_imgs, test_imgs)
    )
    log.info("Split: train %d  val %d  test %d captions", *(len(p) for p in parts))
    return parts


class CaptionDataModule:
    """Owns the dataframes, processors, image index, and dataloaders."""

    def __init__(self, cfg: Config, tok: TokenizerBundle, need_teacher: bool) -> None:
        self.cfg = cfg
        self.tok = tok
        self.df = load_captions(cfg.captions_csv)
        self.train_df, self.val_df, self.test_df = split_by_image(self.df, seed=cfg.seed)
        self.processors: Processors = build_processors(cfg, need_teacher=need_teacher)
        self.image_index = build_image_index(cfg.image_dirs)
        self._save_splits()

    def _save_splits(self) -> None:
        out = Path(self.cfg.work_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, part in (("train", self.train_df), ("val", self.val_df), ("test", self.test_df)):
            part.to_csv(out / f"{name}_split.csv", index=False)

    def _loader(self, df: pd.DataFrame, shuffle: bool, is_train: bool) -> DataLoader:
        return make_loader(df, self.cfg, self.tok, self.processors, self.image_index, shuffle, is_train)

    def train_loader(self) -> DataLoader:
        return self._loader(self.train_df, shuffle=True, is_train=True)

    def val_loader(self) -> DataLoader:
        return self._loader(self.val_df, shuffle=False, is_train=False)

    def test_loader(self) -> DataLoader:
        return self._loader(self.test_df, shuffle=False, is_train=False)

    def pretrain_loader(self) -> DataLoader | None:
        """Stage-0 loader: plain factual captions for decoder fluency pre-finetuning."""
        df = build_pretrain_frame(self.cfg, self.df, self.image_index)
        if df is None or len(df) < 100:
            log.warning("Not enough usable pretrain captions -> Stage 0 will be skipped")
            return None
        return self._loader(df, shuffle=True, is_train=True)


def build_pretrain_frame(cfg: Config, df: pd.DataFrame, image_index: dict[str, str]) -> pd.DataFrame | None:
    """Assemble the Stage-0 factual-caption frame (external CSV if given, else own factuals)."""
    pcsv = cfg.pretrain_captions_csv
    if pcsv and Path(pcsv).exists():
        pre_df = _read_flexible_captions(pcsv)
    else:
        pre_df = df[df["emotion"] == "factual"][["image", "caption"]].copy()
        log.info("Stage 0 source: own factual captions (%d)", len(pre_df))
    pre_df["caption"] = pre_df["caption"].astype(str).str.strip().str.rstrip(".").str.strip()
    pre_df = pre_df[pre_df["caption"].str.split().str.len().between(4, 30)]
    pre_df = pre_df[pre_df["image"].isin(image_index.keys())].reset_index(drop=True)
    pre_df["emotion"] = "factual"
    pre_df["emotion_idx"] = EMOTION2IDX["factual"]
    pre_df[["valence", "arousal", "dominance"]] = 0.5
    log.info("Stage 0: %d usable captions", len(pre_df))
    return pre_df


def _read_flexible_captions(path: str) -> pd.DataFrame:
    """Read a Flickr30k-style caption file with unknown separator/header layout."""
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ("|" if "|" in first else ",")
    has_header = (
        any(k in first.lower() for k in ("caption", "comment", "image", "filename"))
        and ".jpg" not in first.lower()
    )
    raw = pd.read_csv(
        path, sep=sep, header=0 if has_header else None, engine="python", on_bad_lines="skip", dtype=str
    )
    raw.columns = [str(c).strip() for c in raw.columns]
    cols = {c.lower().strip(): c for c in raw.columns}
    icol = cols.get("image") or cols.get("image_name") or cols.get("filename") or raw.columns[0]
    ccol = cols.get("caption") or cols.get("comment") or cols.get("raw") or raw.columns[-1]
    out = raw[[icol, ccol]].rename(columns={icol: "image", ccol: "caption"}).copy()
    out["image"] = out["image"].astype(str).str.split("#").str[0].str.strip()
    out["image"] = out["image"].apply(lambda x: x if x.endswith(".jpg") else x + ".jpg")
    log.info("Stage 0 source: %s (%d raw captions)", path, len(out))
    return out
