"""Word timings for corpus tokens from Montreal Forced Aligner TextGrids.

The public LibriTTS alignments (``kan-bayashi/LibriTTSLabel``) have a ``words`` and a
``phones`` tier per utterance. Their words are lower-cased and without punctuation,
abbreviations are spelled out (``mr`` -> ``mister``), unknown words are ``<unk>``, and
the tokenisation can differ from the corpus (``pshaw`` against ``p`` + ``shaw``).

The mapping first matches the two word sequences (normalised: lower case, letters,
digits and inner apostrophes; known abbreviations expanded) with ``difflib``. Inside a
stretch that differs:

* if the characters agree (a different split into words), every corpus word gets the
  span of the aligned words covering its characters; corpus words sharing one aligned
  word are marked ``shared``;
* if the aligned side is ``<unk>`` words or has the same number of words, they are
  paired one to one (``mrs`` -> ``<unk>``);
* otherwise the corpus words stay without a timing.

Punctuation tokens never get a timing (they are never scored).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from prosody_llms.audio.base import WordTiming
from prosody_llms.data.corpus import Utterance

SILENCES = {"", "sil", "sp", "spn", "<eps>"}
_KEEP = re.compile(r"[^a-z0-9']")


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    text: str


def read_textgrid(path: str | Path) -> dict[str, list[Interval]]:
    """Interval tiers of a long-format Praat TextGrid: {tier name: intervals}."""
    tiers: dict[str, list[Interval]] = {}
    name, cur, xmin, xmax = None, None, None, None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("name ="):
            name = line.split("=", 1)[1].strip().strip('"')
            cur = tiers.setdefault(name, [])
        elif line.startswith("intervals ["):
            xmin = xmax = None
        elif line.startswith("xmin =") and cur is not None:
            xmin = float(line.split("=", 1)[1])
        elif line.startswith("xmax =") and cur is not None:
            xmax = float(line.split("=", 1)[1])
        elif line.startswith("text =") and cur is not None and xmin is not None:
            text = line.split("=", 1)[1].strip()
            if len(text) >= 2 and text[0] == text[-1] == '"':
                text = text[1:-1].replace('""', '"')
            cur.append(Interval(xmin, xmax, text))
            xmin = xmax = None
    return tiers


ABBREVIATIONS = {"mr": "mister", "dr": "doctor", "st": "saint"}
UNK = "<unk>"


def norm(word: str) -> str:
    w = _KEEP.sub("", word.lower().replace("\u2019", "'")).strip("'")
    return ABBREVIATIONS.get(w, w)


@dataclass
class UtteranceTimings:
    timings: list[WordTiming]   # one per corpus token (punctuation: no timing)
    shared: list[bool]          # word shares its aligned interval with a neighbour
    exact: bool                 # the character strings matched exactly
    phones: list[Interval]
    duration: float             # end of the last interval (seconds)

    def mask(self) -> list[bool]:
        return [t.aligned for t in self.timings]


def _greedy(co: list[str], tg: list[str]) -> tuple[dict[int, list[int]], int, int]:
    """Pairs words from the start while they agree: ``<unk>`` takes one corpus word,
    otherwise words are joined on either side until the characters agree. Returns the
    pairs and how many corpus and aligned words were used."""
    out, i, j = {}, 0, 0
    while i < len(co) and j < len(tg):
        if tg[j] == UNK:
            out[i] = [j]
            i, j = i + 1, j + 1
            continue
        i2, j2, cs, ts = i + 1, j + 1, co[i], tg[j]
        while cs != ts:
            if ts.startswith(cs) and i2 < len(co):
                cs += co[i2]
                i2 += 1
            elif cs.startswith(ts) and j2 < len(tg) and tg[j2] != UNK:
                ts += tg[j2]
                j2 += 1
            else:
                return out, i, j
        for k in range(i, i2):
            out[k] = list(range(j, j2))
        i, j = i2, j2
    return out, i, j


def _pair_block(co: list[str], tg: list[str]) -> dict[int, list[int]]:
    """Corpus word -> aligned words inside a stretch where the sequences differ: pairs
    from the start, then from the end; what is left in the middle stays unaligned unless
    both sides have the same number of words."""
    left, i, j = _greedy(co, tg)
    rco, rtg = [w[::-1] for w in co[i:][::-1]], [w if w == UNK else w[::-1] for w in tg[j:][::-1]]
    right, ri, rj = _greedy(rco, rtg)
    out = dict(left)
    n_co, n_tg = len(co), len(tg)
    for k, own in right.items():
        out[n_co - 1 - k] = sorted(n_tg - 1 - x for x in own)
    mid_co = range(i, n_co - ri)
    mid_tg = range(j, n_tg - rj)
    if len(mid_co) == len(mid_tg):
        for a, b in zip(mid_co, mid_tg):
            out[a] = [b]
    return out


def map_tokens(utt: Utterance, words: list[Interval], phones: list[Interval] | None = None
               ) -> UtteranceTimings:
    aligned = [w for w in words if w.text.lower() not in SILENCES
               and (w.text.lower() == UNK or norm(w.text))]
    tg = [UNK if w.text.lower() == UNK else norm(w.text) for w in aligned]
    tok_idx = [i for i, t in enumerate(utt.tokens) if not t.is_punct and norm(t.word)]
    co = [norm(utt.tokens[i].word) for i in tok_idx]
    owners: dict[int, list[int]] = {}  # token index -> aligned word indices
    exact = co == tg

    sm = difflib.SequenceMatcher(None, co, tg, autojunk=False)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            for k in range(a1 - a0):
                owners[tok_idx[a0 + k]] = [b0 + k]
        elif op == "replace":
            for k, own in _pair_block(co[a0:a1], tg[b0:b1]).items():
                owners[tok_idx[a0 + k]] = [b0 + j for j in own]
    count: dict[int, int] = {}
    for own in owners.values():
        for j in own:
            count[j] = count.get(j, 0) + 1
    timings, shared = [], []
    for i, t in enumerate(utt.tokens):
        own = owners.get(i)
        if not own:
            timings.append(WordTiming(t.word, None, None))
            shared.append(False)
            continue
        timings.append(WordTiming(t.word, aligned[own[0]].start, aligned[own[-1]].end))
        shared.append(any(count[j] > 1 for j in own))
    duration = max((w.end for w in words), default=0.0)
    return UtteranceTimings(timings, shared, exact, phones or [], duration)


def load_timings(textgrid_root: str | Path, subset: str, utt: Utterance) -> UtteranceTimings | None:
    """Timings for one utterance, or None if the aligner produced no TextGrid for it."""
    speaker, chapter = utt.utt_id.split("_")[:2]
    path = Path(textgrid_root) / subset / speaker / chapter / f"{utt.utt_id}.TextGrid"
    if not path.exists():
        return None
    tiers = read_textgrid(path)
    return map_tokens(utt, tiers.get("words", []), tiers.get("phones", []))


def coverage(utts: list[Utterance], timings: list[UtteranceTimings | None]) -> dict:
    """Share of scored words with a timing, utterances without a TextGrid, etc."""
    words = aligned = shared = 0
    missing_utts = inexact = 0
    for u, t in zip(utts, timings):
        n = sum(not tok.is_punct for tok in u.tokens)
        words += n
        if t is None:
            missing_utts += 1
            continue
        inexact += not t.exact
        aligned += sum(m for m, tok in zip(t.mask(), u.tokens) if not tok.is_punct)
        shared += sum(s for s, m in zip(t.shared, t.mask()) if m)
    return {"utterances": len(utts), "without_textgrid": missing_utts,
            "inexact_text_match": inexact, "words": words, "aligned_words": aligned,
            "word_coverage": round(aligned / words, 5) if words else None,
            "shared_interval_words": shared}
