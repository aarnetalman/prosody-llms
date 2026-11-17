"""Fixed, reproducible subsets of the corpus used by the analyses.

None of these change the official splits: they are views on them, used for

* the prompting grid (a fixed dev subset, so the test set is never tuned on),
* the text-only ceiling (the same text read by different speakers),
* the leakage breakdown (test texts that also occur in training or dev),
* discourse context (the preceding utterances of the same chapter).
"""

from __future__ import annotations

import collections
import hashlib
from collections.abc import Iterable, Sequence

from prosody_llms.data.corpus import LABEL_NA, Utterance

# LibriVox announcements read by many speakers ("This is a LibriVox recording ...").
BOILERPLATE_WORDS = frozenset({"librivox", "domain"})


def utterance_key(utt_id: str, seed: int = 0) -> str:
    """A stable pseudo-random key: the same on every machine and Python version."""
    return hashlib.sha256(f"{seed}:{utt_id}".encode()).hexdigest()


def fixed_subset(utts: Sequence[Utterance], n: int, seed: int = 0) -> list[Utterance]:
    """``n`` utterances chosen by hash of their id, returned in corpus order.

    Adding or removing other utterances does not change which of the remaining ones are
    chosen, unlike ``random.sample``.
    """
    if n >= len(utts):
        return list(utts)
    chosen = set(sorted((u.utt_id for u in utts), key=lambda i: utterance_key(i, seed))[:n])
    return [u for u in utts if u.utt_id in chosen]


def text_key(u: Utterance) -> tuple[str, ...]:
    """The utterance text, lower-cased, as a tuple of tokens (punctuation included)."""
    return tuple(w.lower() for w in u.words)


def is_boilerplate(u: Utterance) -> bool:
    return any(w.lower() in BOILERPLATE_WORDS for w in u.words)


def scored_words(u: Utterance) -> int:
    return sum(t.prominence != LABEL_NA for t in u.tokens)


def multi_reader_groups(
    splits: dict[str, Sequence[Utterance]], min_scored_words: int = 5
) -> list[list[tuple[str, Utterance]]]:
    """Groups of utterances with identical text read by at least two different speakers.

    Each group is a list of ``(split name, utterance)``. Utterances with fewer than
    ``min_scored_words`` scored words are ignored.
    """
    by_text: dict[tuple[str, ...], list[tuple[str, Utterance]]] = collections.defaultdict(list)
    for name, utts in splits.items():
        for u in utts:
            if scored_words(u) >= min_scored_words:
                by_text[text_key(u)].append((name, u))
    return [g for g in by_text.values() if len({u.speaker for _, u in g}) > 1]


def cross_speaker_agreement(
    groups: Iterable[list[tuple[str, Utterance]]], n_classes: int
) -> tuple[float, int]:
    """Pairwise label agreement (percent, word pairs) between different speakers reading
    the same text. Pairs from the same speaker are skipped."""
    agree = total = 0
    for g in groups:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i][1], g[j][1]
                if a.speaker == b.speaker:
                    continue
                for x, y in zip(a.labels(n_classes), b.labels(n_classes)):
                    if x == LABEL_NA:
                        continue
                    total += 1
                    agree += x == y
    if not total:
        raise ValueError("no cross-speaker word pairs")
    return 100.0 * agree / total, total


def overlapping_texts(
    target: Sequence[Utterance], others: Iterable[Sequence[Utterance]],
    min_scored_words: int = 5, include_boilerplate: bool = False,
) -> set[str]:
    """Ids of ``target`` utterances whose exact text also occurs in any of ``others``."""
    seen = {text_key(u) for utts in others for u in utts}
    return {u.utt_id for u in target
            if text_key(u) in seen and scored_words(u) >= min_scored_words
            and (include_boilerplate or not is_boilerplate(u))}


def chapter_position(utt_id: str) -> tuple[str, tuple[int, ...]]:
    """``speaker_chapter_paragraph_sentence`` -> (``speaker_chapter``, (paragraph, sentence))."""
    parts = utt_id.split("_")
    if len(parts) < 4:
        raise ValueError(f"unexpected utterance id {utt_id!r}")
    return "_".join(parts[:2]), tuple(int(p) for p in parts[2:])


def preceding_context(utts: Sequence[Utterance], k: int) -> dict[str, list[Utterance]]:
    """For every utterance, up to ``k`` preceding utterances of the same chapter (and
    reader), oldest first. Only utterances present in ``utts`` can serve as context;
    LibriTTS removed some sentences, so the context can have gaps."""
    chapters: dict[str, list[tuple[tuple[int, ...], Utterance]]] = collections.defaultdict(list)
    for u in utts:
        chapter, pos = chapter_position(u.utt_id)
        chapters[chapter].append((pos, u))
    out = {}
    for items in chapters.values():
        items.sort(key=lambda x: x[0])
        ordered = [u for _, u in items]
        for i, u in enumerate(ordered):
            out[u.utt_id] = ordered[max(0, i - k):i] if k > 0 else []
    return out
