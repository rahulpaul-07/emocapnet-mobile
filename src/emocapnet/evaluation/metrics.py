"""Caption metrics: BLEU / METEOR / ROUGE-L, per-emotion BLEU-4, and CIDEr-D.

``cider_d`` is a pure-python implementation (n=1..4 tf-idf cosine with the
CIDEr-D length penalty). When ``pycocoevalcap`` is installed the reference
scorer is preferred.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict

import numpy as np

from emocapnet.constants import EMOTION_LABELS

log = logging.getLogger(__name__)


def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def cider_d(hyps: list, refs_list: list, N: int = 4, sigma: float = 6.0) -> float:
    """CIDEr-D over tokenized hypotheses and per-item reference lists."""
    hyps = [h if isinstance(h, list) else h.split() for h in hyps]
    refs_list = [[r if isinstance(r, list) else r.split() for r in rs] for rs in refs_list]
    df: list[dict] = [defaultdict(int) for _ in range(N)]
    for refs in refs_list:
        for n in range(N):
            seen: set = set()
            for r in refs:
                seen |= set(_ngrams(r, n + 1).keys())
            for g in seen:
                df[n][g] += 1
    M = len(refs_list)
    log_M = math.log(max(M, 1))

    def tfidf(toks: list[str], n: int) -> dict:
        cnt = _ngrams(toks, n + 1)
        L = max(len(toks) - n, 1)
        return {g: (c / L) * (log_M - math.log(max(df[n][g], 1))) for g, c in cnt.items()}

    scores = []
    for h, refs in zip(hyps, refs_list):
        s = 0.0
        for n in range(N):
            vh = tfidf(h, n)
            acc = 0.0
            for r in refs:
                vr = tfidf(r, n)
                num = sum(min(vh[g], vr.get(g, 0)) * vr.get(g, 0) for g in vh)
                na = math.sqrt(sum(v * v for v in vh.values()))
                nb = math.sqrt(sum(v * v for v in vr.values()))
                cos = num / (na * nb) if na > 0 and nb > 0 else 0.0
                acc += cos * math.exp(-((len(h) - len(r)) ** 2) / (2 * sigma**2))
            s += acc / len(refs)
        scores.append(10.0 * s / N)
    return float(np.mean(scores)) if scores else 0.0


def cider_score(hyps: list, refs_list: list) -> float:
    """Prefer the pycocoevalcap reference scorer; fall back to the local CIDEr-D."""
    try:
        from pycocoevalcap.cider.cider import Cider

        gts = {str(i): [" ".join(r) for r in rs] for i, rs in enumerate(refs_list)}
        res = {str(i): [" ".join(h)] for i, h in enumerate(hyps)}
        score, _ = Cider().compute_score(gts, res)
        return float(score)
    except Exception as e:
        log.info("pycocoevalcap unavailable (%r) -> local CIDEr-D", e)
        return cider_d(hyps, refs_list)


def compute_caption_metrics(refs: list, hyps: list, emotions: list[str] | None = None) -> dict:
    """BLEU-1/4, METEOR, ROUGE-L (+ per-emotion BLEU-4 when emotions are given).

    ``refs``: list of reference lists (each reference is a token list).
    ``hyps``: list of hypothesis token lists.
    """
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    smooth = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    try:
        meteor = float(np.mean([meteor_score(r, h) for r, h in zip(refs, hyps)]))
    except LookupError:  # wordnet unavailable (offline env); other metrics still valid
        log.warning("METEOR skipped: NLTK wordnet corpus not available")
        meteor = float("nan")

    metrics = {
        "bleu1": corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "bleu4": corpus_bleu(refs, hyps, weights=(0.25,) * 4, smoothing_function=smooth),
        "meteor": meteor,
        "rougeL": float(
            np.mean([rouge.score(" ".join(r[0]), " ".join(h))["rougeL"].fmeasure for r, h in zip(refs, hyps)])
        ),
        "n": len(hyps),
    }
    per: dict[str, float] = {}
    if emotions:
        for e in EMOTION_LABELS:
            idx = [i for i, x in enumerate(emotions) if x == e]
            if idx:
                per[e] = corpus_bleu(
                    [refs[i] for i in idx],
                    [hyps[i] for i in idx],
                    weights=(0.25,) * 4,
                    smoothing_function=smooth,
                )
    metrics["per_emotion"] = per
    return metrics


def ensure_nltk_data() -> None:
    """Download the small NLTK corpora METEOR needs (idempotent)."""
    import nltk

    for pkg in ("punkt", "punkt_tab", "wordnet"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            log.warning("nltk download %s failed: %r", pkg, e)
