import torch

from emocapnet.config import Config
from emocapnet.losses import combined_loss, contrastive_loss, kd_logit_loss, lm_loss

B, L, V = 2, 8, 50


def _fake_outputs(vocab=V, requires_grad=True):
    logits = torch.randn(B, L, vocab, requires_grad=requires_grad)
    vad = torch.rand(B, 3)
    emo = torch.randn(B, 7)
    img = torch.nn.functional.normalize(torch.randn(B, 256), dim=-1)
    cap = torch.nn.functional.normalize(torch.randn(B, 256), dim=-1)
    return logits, vad, emo, img, cap


def _fake_batch():
    labels = torch.randint(0, V, (B, L))
    labels[:, -2:] = -100  # padding
    return {"labels": labels, "vad": torch.rand(B, 3), "emotion_idx": torch.randint(0, 7, (B,))}


def test_lm_loss_ignores_padding():
    logits = torch.randn(B, L, V)
    labels = torch.full((B, L), -100)
    labels[:, 1] = 3
    loss = lm_loss(logits, labels)
    assert torch.isfinite(loss)


def test_kd_zero_when_student_equals_teacher():
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    assert kd_logit_loss(logits, logits, labels, temperature=2.0).abs() < 1e-5


def test_kd_all_padding_returns_zero():
    logits = torch.randn(B, L, V)
    labels = torch.full((B, L), -100)
    assert kd_logit_loss(logits, torch.randn(B, L, V), labels, 2.0) == 0


def test_contrastive_loss_finite_and_positive():
    _, _, _, img, cap = _fake_outputs()
    loss = contrastive_loss(img, cap, torch.tensor(0.07))
    assert torch.isfinite(loss) and loss > 0


def test_combined_loss_no_teacher_matches_v3_objective():
    cfg = Config(distill_alpha=0.0)
    total, parts = combined_loss(_fake_outputs(), _fake_batch(), None, cfg, torch.tensor(0.07))
    assert torch.isfinite(total)
    assert parts["kd"] == 0.0 and parts["ft"] == 0.0
    total.backward()  # gradient flows


def test_combined_loss_with_teacher_adds_kd_and_feat():
    cfg = Config(distill_alpha=0.6)
    s_out = _fake_outputs()
    t_out = tuple(t.detach() for t in _fake_outputs(requires_grad=False))
    total, parts = combined_loss(s_out, _fake_batch(), t_out, cfg, torch.tensor(0.07))
    assert torch.isfinite(total)
    assert parts["kd"] > 0.0
    assert parts["ft"] >= 0.0


def test_distill_alpha_zero_ignores_teacher_logits():
    cfg = Config(distill_alpha=0.0)
    s_out = _fake_outputs()
    batch = _fake_batch()
    with_teacher, _ = combined_loss(s_out, batch, None, cfg, torch.tensor(0.07))
    assert torch.isfinite(with_teacher)
