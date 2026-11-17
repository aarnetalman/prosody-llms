"""Uncertainty and significance for word-level accuracy.

Words in one utterance are not independent (same speaker, same sentence), so the
bootstrap resamples **utterances**, not words.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

from prosody_llms.data.corpus import LABEL_NA


def _per_utterance_counts(gold: Sequence[Sequence[str]], preds: Sequence[Sequence[Sequence[str]]]):
    """For each utterance: number of scored words and correct counts for each system."""
    rows = []
    for u, g in enumerate(gold):
        idx = [i for i, lab in enumerate(g) if lab != LABEL_NA]
        rows.append((len(idx), [sum(p[u][i] == g[i] for i in idx) for p in preds]))
    return rows


def bootstrap_accuracy(
    gold: Sequence[Sequence[str]], *systems: Sequence[Sequence[str]],
    n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> dict:
    """Utterance-level bootstrap.

    ``gold`` and each system are lists of per-utterance label lists. Returns the
    accuracy and a percentile confidence interval for each system, and, for two
    systems, for the difference (system 1 - system 2) with a two-sided p-value.
    """
    if not systems:
        raise ValueError("at least one system is needed")
    for s in systems:
        if len(s) != len(gold) or any(len(a) != len(b) for a, b in zip(s, gold)):
            raise ValueError("every system needs one prediction per gold token")
    rows = _per_utterance_counts(gold, systems)
    n_words = sum(r[0] for r in rows)
    point = [100.0 * sum(r[1][k] for r in rows) / n_words for k in range(len(systems))]

    rng = random.Random(seed)
    samples: list[list[float]] = [[] for _ in systems]
    for _ in range(n_resamples):
        pick = [rows[rng.randrange(len(rows))] for _ in rows]
        n = sum(r[0] for r in pick) or 1
        for k in range(len(systems)):
            samples[k].append(100.0 * sum(r[1][k] for r in pick) / n)

    def ci(xs):
        xs = sorted(xs)
        lo = xs[int(math.floor(alpha / 2 * (len(xs) - 1)))]
        hi = xs[int(math.ceil((1 - alpha / 2) * (len(xs) - 1)))]
        return lo, hi

    out = {"accuracy": point, "ci": [ci(s) for s in samples], "n_utterances": len(rows),
           "scored_words": n_words, "n_resamples": n_resamples, "alpha": alpha}
    if len(systems) == 2:
        diffs = [a - b for a, b in zip(samples[0], samples[1])]
        d = point[0] - point[1]
        # Two-sided p-value: how often the resampled difference crosses zero.
        if d >= 0:
            p = 2 * sum(x <= 0 for x in diffs) / n_resamples
        else:
            p = 2 * sum(x >= 0 for x in diffs) / n_resamples
        out["difference"] = {"value": d, "ci": ci(diffs), "p_value": min(1.0, p)}
    return out


def mcnemar(gold: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str]) -> dict:
    """McNemar's test on scored words (exact binomial for small counts, else chi-square
    with continuity correction). Note that it treats words as independent."""
    b = c = 0
    for g, x, y in zip(gold, pred_a, pred_b):
        if g == LABEL_NA:
            continue
        if x == g and y != g:
            b += 1
        elif x != g and y == g:
            c += 1
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if n <= 50:
        k = min(b, c)
        p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        return {"b": b, "c": c, "statistic": float(k), "p_value": min(1.0, 2 * p), "method": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(stat / 2))  # chi-square with 1 degree of freedom
    return {"b": b, "c": c, "statistic": stat, "p_value": p, "method": "chi2"}


def mean_sd(values: Sequence[float]) -> dict:
    values = list(values)
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values), "n": len(values)}
