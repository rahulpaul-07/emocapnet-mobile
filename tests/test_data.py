import pandas as pd
import pytest
import torch

from emocapnet.constants import EMOTION_LABELS
from emocapnet.data.datamodule import CaptionDataModule, load_captions, split_by_image
from emocapnet.data.synthetic import generate_synthetic_dataset


def test_synthetic_dataset_is_valid(tmp_path):
    csv_path, img_dir = generate_synthetic_dataset(tmp_path, n_images=6, captions_per_image=2)
    df = pd.read_csv(csv_path)
    assert len(df) == 12
    assert set(df["emotion"]).issubset(set(EMOTION_LABELS))
    assert df[["valence", "arousal", "dominance"]].apply(lambda c: c.between(0, 1).all()).all()


def test_load_captions_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"image_id": ["a"], "caption": ["hello world one two"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_captions(str(bad))


def test_split_has_no_image_leakage(synthetic_data):
    csv_path, _ = synthetic_data
    df = load_captions(csv_path)
    train, val, test = split_by_image(df, seed=1)
    assert not (set(train["image"]) & set(test["image"]))
    assert not (set(train["image"]) & set(val["image"]))
    assert not (set(val["image"]) & set(test["image"]))
    assert len(train) + len(val) + len(test) == len(df)


def test_split_is_deterministic(synthetic_data):
    csv_path, _ = synthetic_data
    df = load_captions(csv_path)
    a = split_by_image(df, seed=42)[2]["image"].tolist()
    b = split_by_image(df, seed=42)[2]["image"].tolist()
    assert a == b


def test_datamodule_batch_shapes(tiny_cfg, tok):
    dm = CaptionDataModule(tiny_cfg, tok, need_teacher=False)
    batch = next(iter(dm.train_loader()))
    B = batch["input_ids"].shape[0]
    assert batch["pv_student"].shape == (B, 3, 64, 64)
    assert batch["input_ids"].shape == (B, tiny_cfg.max_length)
    assert batch["labels"].shape == (B, tiny_cfg.max_length)
    assert batch["vad"].shape == (B, 3)
    assert batch["emotion_idx"].dtype == torch.long
    # padding is masked out of labels
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
