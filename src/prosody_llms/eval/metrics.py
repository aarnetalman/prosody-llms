"""Evaluation metrics.

**Primary (2019) metric:** token-level accuracy over words whose gold label is not ``NA``
(punctuation). A predicted ``NA``, ``<pad>`` or any other non-label on a scored word
counts as an error. This is unchanged from 2019.

**Additional metrics** (reported alongside, never instead): per-class precision, recall
and F1, and macro-F1 over the gold label set.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Sequence

from prosody_llms.data.corpus import LABEL_NA


def scored_pairs(gold: Iterable[str], pred: Iterable[str]) -> list[tuple[str, str]]:
    gold, pred = list(gold), list(pred)
    if len(gold) != len(pred):
        raise ValueError(f"length mismatch: {len(gold)} gold vs {len(pred)} predicted")
    return [(g, p) for g, p in zip(gold, pred) if g != LABEL_NA]


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> tuple[float, int]:
    """Return (accuracy in percent, number of scored words). The 2019 metric."""
    pairs = scored_pairs(gold, pred)
    if not pairs:
        raise ValueError("no scored words")
    correct = sum(g == p for g, p in pairs)
    return 100.0 * correct / len(pairs), len(pairs)


def confusion(gold: Sequence[str], pred: Sequence[str]) -> dict[str, dict[str, int]]:
    """Confusion counts as {gold label: {predicted label: count}} over scored words."""
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for g, p in scored_pairs(gold, pred):
        out[g][p] += 1
    return {g: dict(sorted(c.items())) for g, c in sorted(out.items())}


def per_class(gold: Sequence[str], pred: Sequence[str]) -> dict[str, dict[str, float]]:
    """Precision, recall and F1 (percent) and support for every gold label."""
    pairs = scored_pairs(gold, pred)
    labels = sorted({g for g, _ in pairs})
    out = {}
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in pairs)
        fp = sum(g != lab and p == lab for g, p in pairs)
        fn = sum(g == lab and p != lab for g, p in pairs)
        prec = 100.0 * tp / (tp + fp) if tp + fp else 0.0
        rec = 100.0 * tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    pc = per_class(gold, pred)
    return sum(v["f1"] for v in pc.values()) / len(pc)


def summary(gold: Sequence[str], pred: Sequence[str]) -> dict:
    """All metrics for one prediction set. ``accuracy`` is the primary 2019 metric."""
    acc, n = accuracy(gold, pred)
    return {"accuracy": acc, "scored_words": n, "macro_f1": macro_f1(gold, pred),
            "per_class": per_class(gold, pred), "confusion": confusion(gold, pred)}
