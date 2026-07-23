import math

from emocapnet.evaluation.metrics import cider_d, compute_caption_metrics, ensure_nltk_data

CORPUS = [
    "a red circle sits on a gray field",
    "a blue square floats near the corner",
    "a green triangle rests in the middle",
    "a yellow star appears by the edge",
]


def test_cider_perfect_match_scores_high():
    # distinct references so idf is non-degenerate
    hyps = [c.split() for c in CORPUS]
    refs = [[h] for h in hyps]
    score = cider_d(hyps, refs)
    assert score > 0
    assert math.isfinite(score)


def test_cider_disjoint_scores_zero():
    hyps = [["completely", "different", "words", "here"]]
    refs = [[["nothing", "shared", "at", "all", "whatsoever"]]]
    assert cider_d(hyps, refs) == 0.0


def test_cider_ranks_match_above_mismatch():
    refs = [[c.split()] for c in CORPUS]
    good_hyps = [c.split() for c in CORPUS]  # perfect
    bad_hyps = [["random", "unrelated", "tokens", "entirely"]] * len(CORPUS)
    assert cider_d(good_hyps, refs) > cider_d(bad_hyps, refs)


def test_caption_metrics_shapes():
    ensure_nltk_data()
    refs = [[r.split()] for r in ["a dog runs fast", "a cat sleeps well"]]
    hyps = [h.split() for h in ["a dog runs fast", "a bird flies away"]]
    m = compute_caption_metrics(refs, hyps, emotions=["happiness", "sadness"])
    for key in ("bleu1", "bleu4", "rougeL"):
        assert 0.0 <= m[key] <= 1.0
    assert math.isnan(m["meteor"]) or 0.0 <= m["meteor"] <= 1.0  # nan iff wordnet offline
    assert m["n"] == 2
    assert set(m["per_emotion"]) <= {"happiness", "sadness"}


def test_perfect_hypothesis_beats_wrong_one():
    ensure_nltk_data()
    refs = [[["a", "red", "circle", "sits", "on", "the", "field"]]]
    perfect = compute_caption_metrics(refs, [refs[0][0]])
    wrong = compute_caption_metrics(refs, [["totally", "unrelated", "caption", "text", "here"]])
    assert perfect["bleu4"] > wrong["bleu4"]
    assert perfect["rougeL"] > wrong["rougeL"]
