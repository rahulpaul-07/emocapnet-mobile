"""Tokenizer construction.

The student shares the teacher's vocabulary — same added ``<emo=...>`` tokens in the
same order — so distillation lines up the two models' logits token-for-token.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import GPT2Tokenizer

from emocapnet.constants import CAP_TOKEN, EMOTION_LABELS, emotion_token


@dataclass
class TokenizerBundle:
    tokenizer: GPT2Tokenizer
    vocab_size: int
    emo_token_ids: dict[str, int]
    eos_id: int


def build_tokenizer(name_or_path: str = "gpt2") -> TokenizerBundle:
    """Load a GPT-2 tokenizer and append emotion + caption special tokens.

    Token order is deterministic (``EMOTION_LABELS`` order, then ``<|cap|>``) so any
    two models built from this function agree on ids — a hard requirement for KD.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    emotion_tokens = [emotion_token(e) for e in EMOTION_LABELS]
    tokenizer.add_special_tokens({"additional_special_tokens": emotion_tokens + [CAP_TOKEN]})
    emo_token_ids = {e: tokenizer.convert_tokens_to_ids(emotion_token(e)) for e in EMOTION_LABELS}
    return TokenizerBundle(
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        emo_token_ids=emo_token_ids,
        eos_id=tokenizer.eos_token_id,
    )


def write_tiny_gpt2_tokenizer(out_dir) -> str:
    """Write a minimal byte-level GPT-2-format tokenizer (256 bytes + EOS, no merges).

    Character-level and useless for real captioning quality, but format-compatible
    with ``GPT2Tokenizer.from_pretrained`` — lets tests and the smoke pipeline run
    fully offline. Returns the directory path.
    """
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # GPT-2's bytes-to-unicode mapping (verbatim algorithm)
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    chars = [chr(c) for c in cs]
    vocab = {ch: i for i, ch in enumerate(chars)}
    vocab["<|endoftext|>"] = len(vocab)
    (out / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (out / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (out / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 1024, "tokenizer_class": "GPT2Tokenizer"})
    )
    return str(out)


def strip_caption(text: str) -> str:
    """Remove special tokens from decoded model output."""
    import re

    for e in EMOTION_LABELS:
        text = text.replace(emotion_token(e), " ")
    text = text.replace("<|endoftext|>", " ").replace(CAP_TOKEN, " ")
    return re.sub(r"\s+", " ", text).strip()
