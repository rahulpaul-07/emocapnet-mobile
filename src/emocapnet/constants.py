"""Task-level constants shared by teacher, student, data, and evaluation code."""

EMOTION_LABELS: list[str] = [
    "factual",
    "happiness",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
]

EMOTION2IDX: dict[str, int] = {e: i for i, e in enumerate(EMOTION_LABELS)}

#: VAD (valence / arousal / dominance) midpoint used for factual captions.
NEUTRAL_VAD: tuple[float, float, float] = (0.5, 0.5, 0.5)

#: Canonical VAD prior per emotion — used as the default conditioning vector when
#: a caller requests an emotion without an explicit VAD (serving, synthetic data).
EMOTION_VAD: dict[str, tuple[float, float, float]] = {
    "factual": (0.50, 0.50, 0.50),
    "happiness": (0.85, 0.60, 0.65),
    "sadness": (0.20, 0.35, 0.30),
    "anger": (0.15, 0.80, 0.60),
    "fear": (0.15, 0.75, 0.25),
    "surprise": (0.60, 0.80, 0.45),
    "disgust": (0.20, 0.55, 0.45),
}

#: Special caption-start token appended to the tokenizer alongside emotion tokens.
CAP_TOKEN: str = "<|cap|>"


def emotion_token(emotion: str) -> str:
    """Return the special token string for an emotion, e.g. ``<emo=happiness>``."""
    return f"<emo={emotion}>"
