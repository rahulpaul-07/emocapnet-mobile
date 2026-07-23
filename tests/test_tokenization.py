from emocapnet.constants import EMOTION_LABELS
from emocapnet.tokenization import strip_caption


def test_all_emotion_tokens_registered(tok):
    assert set(tok.emo_token_ids) == set(EMOTION_LABELS)
    assert len(set(tok.emo_token_ids.values())) == len(EMOTION_LABELS)  # unique ids


def test_token_ids_are_deterministic(tok):
    """KD requires two independently built tokenizers to agree token-for-token."""
    from emocapnet.tokenization import build_tokenizer

    tok2 = build_tokenizer(tok.tokenizer.name_or_path)  # rebuild from the same source
    assert tok2.emo_token_ids == tok.emo_token_ids
    assert tok2.vocab_size == tok.vocab_size


def test_strip_caption_removes_special_tokens():
    raw = "<emo=happiness> a dog runs <|cap|> in the park <|endoftext|>"
    assert strip_caption(raw) == "a dog runs in the park"


def test_roundtrip_encode_decode(tok):
    text = "<emo=sadness> a lonely bench sits in the rain"
    ids = tok.tokenizer(text)["input_ids"]
    assert ids[0] == tok.emo_token_ids["sadness"]
    assert strip_caption(tok.tokenizer.decode(ids)) == "a lonely bench sits in the rain"
