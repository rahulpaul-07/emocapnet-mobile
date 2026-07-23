import torch

from emocapnet.config import seed_everything
from emocapnet.models.encoders import TinyConvEncoder


def test_tiny_encoder_contract():
    enc = TinyConvEncoder(out_dim=128)
    pooled, spatial = enc(torch.randn(2, 3, 64, 64))
    assert pooled.shape == (2, 128)
    assert spatial.shape == (2, 16, 128)


def test_student_forward_shapes(student, tiny_cfg, tok):
    B, L = 2, tiny_cfg.max_length
    pv = torch.randn(B, 3, 64, 64)
    ids = torch.randint(0, tok.vocab_size, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    vad = torch.rand(B, 3)
    logits, vad_pred, emo_logits, img_emb, cap_emb = student(pv, ids, mask, vad)
    assert logits.shape == (B, L, tok.vocab_size)
    assert vad_pred.shape == (B, 3)
    assert (vad_pred >= 0).all() and (vad_pred <= 1).all()  # sigmoid head
    assert emo_logits.shape == (B, tiny_cfg.num_emotions)
    assert img_emb.shape == cap_emb.shape == (B, 256)
    assert torch.allclose(img_emb.norm(dim=-1), torch.ones(B), atol=1e-4)


def test_weight_tying(student):
    assert student.lm_head.weight.data_ptr() == student.gpt2.wte.weight.data_ptr()


def test_generate_terminates_and_seeds_emotion(student, tiny_cfg, tok):
    B = 2
    pv = torch.randn(B, 3, 64, 64)
    vad = torch.rand(B, 3)
    emo_ids = torch.tensor([tok.emo_token_ids["happiness"], tok.emo_token_ids["fear"]])
    out = student.generate(pv, vad, emo_ids, tok.eos_id, tiny_cfg)
    assert out.shape[0] == B
    assert out.shape[1] <= 1 + tiny_cfg.hard_max_length
    assert out[0, 0] == tok.emo_token_ids["happiness"]
    assert out[1, 0] == tok.emo_token_ids["fear"]


def test_generate_is_deterministic(student, tiny_cfg, tok):
    seed_everything(0)
    pv = torch.randn(1, 3, 64, 64)
    vad = torch.rand(1, 3)
    ei = torch.tensor([tok.emo_token_ids["factual"]])
    a = student.generate(pv, vad, ei, tok.eos_id, tiny_cfg)
    b = student.generate(pv, vad, ei, tok.eos_id, tiny_cfg)
    assert torch.equal(a, b)


def test_encoder_freeze_unfreeze(student):
    from emocapnet.training.schedule import freeze_encoder, unfreeze_encoder

    freeze_encoder(student)
    assert not any(p.requires_grad for p in student.encoder.parameters())
    assert any(p.requires_grad for p in student.gpt2.parameters())
    unfreeze_encoder(student)
    assert all(p.requires_grad for p in student.encoder.parameters())
