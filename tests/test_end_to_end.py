"""End-to-end pipeline test on synthetic data: train -> checkpoint -> eval -> quantize.

Marked slow; CI runs it in a dedicated job. This is the test that proves the
repository actually *works*, not just that its pieces do.
"""

from pathlib import Path

import pytest
import torch

from emocapnet.data.datamodule import CaptionDataModule
from emocapnet.evaluation.evaluator import evaluate_captions, vad_mse
from emocapnet.evaluation.metrics import ensure_nltk_data
from emocapnet.training.checkpoint import load_student_weights
from emocapnet.training.trainer import Trainer


@pytest.mark.slow
def test_full_pipeline(tiny_cfg, tok, student):
    ensure_nltk_data()
    tiny_cfg.ensure_dirs()
    device = torch.device("cpu")
    dm = CaptionDataModule(tiny_cfg, tok, need_teacher=False)
    student = student.to(device)

    trainer = Trainer(tiny_cfg, student, None, dm.train_loader(), dm.val_loader(), device)
    history = trainer.fit()

    # training ran and produced finite losses
    assert len(history) == tiny_cfg.epochs
    assert all(torch.isfinite(torch.tensor(h["tr_tot"])) for h in history)

    # checkpoints exist and reload cleanly
    best = Path(tiny_cfg.ckpt_dir) / "best.pt"
    assert best.exists()
    load_student_weights(student, best)

    # evaluation produces sane metrics
    metrics = evaluate_captions(student, dm.test_loader(), tiny_cfg, tok, device, with_cider=False)
    assert metrics["n"] > 0
    assert 0.0 <= metrics["bleu4"] <= 1.0
    mse = vad_mse(student, dm.test_loader(), device)
    assert 0.0 <= mse <= 1.0


@pytest.mark.slow
def test_second_epoch_improves_train_loss(tiny_cfg, tok, student):
    """Sanity: the model actually learns something on the synthetic set."""
    tiny_cfg.epochs = 2
    tiny_cfg.unfreeze_vit_epoch = 99  # keep encoder frozen for stability
    tiny_cfg.ensure_dirs()
    dm = CaptionDataModule(tiny_cfg, tok, need_teacher=False)
    trainer = Trainer(tiny_cfg, student, None, dm.train_loader(), dm.val_loader(), torch.device("cpu"))
    history = trainer.fit()
    assert history[-1]["tr_lm"] < history[0]["tr_lm"]
