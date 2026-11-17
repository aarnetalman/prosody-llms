"""Reader for the prominence corpus of Talman et al. (2019).

File format: one token per line with five tab-separated fields
(word, discrete prominence, discrete boundary, real prominence, real boundary);
each utterance starts with a line ``<file>\\t<LibriTTS utterance id>.txt``.
Punctuation has the label ``NA`` and is never scored.

``legacy=True`` reproduces the 2019 loader (``prosody_dataset.load_dataset``) exactly:

* ``fraction`` slices the *raw lines* of the file (including ``<file>`` lines) before
  parsing, as ``--fraction_of_train_data`` did;
* the last utterance of the (sliced) file is dropped unless the slice happens to end on
  a ``<file>`` line, because the 2019 loader only flushed a sentence when it saw the
  next ``<file>`` line.

``legacy=False`` (the default) keeps every utterance.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path

LABEL_NA = "NA"
FILE_MARKER = "<file>"
SPLITS = ("train", "dev", "test")
TRAIN_SETS = ("train_100", "train_360")


@dataclass(frozen=True)
class Token:
    word: str
    prominence: str  # "0" | "1" | "2" | "NA"
    boundary: str  # "0" | "1" | "2" | "NA"
    prominence_value: float | None
    boundary_value: float | None

    @property
    def is_punct(self) -> bool:
        return self.prominence == LABEL_NA


@dataclass
class Utterance:
    utt_id: str  # LibriTTS utterance id, e.g. "1272_128104_000003_000000"
    tokens: list[Token] = field(default_factory=list)

    @property
    def speaker(self) -> str:
        return self.utt_id.split("_")[0]

    @property
    def words(self) -> list[str]:
        return [t.word for t in self.tokens]

    def labels(self, n_classes: int = 3) -> list[str]:
        return [collapse_label(t.prominence, n_classes) for t in self.tokens]


def collapse_label(label: str, n_classes: int) -> str:
    """Map a 3-way label to the requested task. 2-way merges 1 and 2 into "1"."""
    if n_classes == 3:
        return label
    if n_classes == 2:
        return "1" if label == "2" else label
    raise ValueError(f"n_classes must be 2 or 3, got {n_classes}")


def _value(s: str) -> float | None:
    return None if s == LABEL_NA else float(s)


def _parse_token(line: str, path: Path, lineno: int) -> Token:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 5:
        raise ValueError(f"{path}:{lineno}: expected 5 tab-separated fields, got {len(parts)}")
    w, p, b, pv, bv = parts
    return Token(w, p, b, _value(pv), _value(bv))


def read_split(path: str | Path, fraction: float = 1.0, legacy: bool = False) -> list[Utterance]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if fraction < 1:
        lines = lines[: int(round(len(lines) * fraction))]

    utts: list[Utterance] = []
    cur: Utterance | None = None
    for i, line in enumerate(lines):
        if line.startswith(FILE_MARKER + "\t"):
            if cur is not None:
                utts.append(cur)
            utt_id = line.rstrip("\n").split("\t")[1]
            cur = Utterance(utt_id.removesuffix(".txt"))
            continue
        if not line.strip():
            continue
        if cur is None:
            raise ValueError(f"{path}:{i + 1}: token before the first {FILE_MARKER} line")
        cur.tokens.append(_parse_token(line, path, i + 1))
    if cur is not None and not legacy:
        utts.append(cur)
    # A slice that ends on a <file> line leaves an empty utterance; it has nothing to
    # score and the 2019 loader never kept it either.
    return [u for u in utts if u.tokens]


def load_splits(
    data_dir: str | Path,
    train_set: str = "train_360",
    fraction_of_train_data: float = 1.0,
    legacy: bool = False,
) -> dict[str, list[Utterance]]:
    if train_set not in TRAIN_SETS:
        raise ValueError(f"train_set must be one of {TRAIN_SETS}")
    data_dir = Path(data_dir)
    return {
        "train": read_split(data_dir / f"{train_set}.txt", fraction_of_train_data, legacy),
        "dev": read_split(data_dir / "dev.txt", 1.0, legacy),
        "test": read_split(data_dir / "test.txt", 1.0, legacy),
    }


def corpus_stats(utts: list[Utterance]) -> dict:
    labels = collections.Counter(t.prominence for u in utts for t in u.tokens)
    return {
        "utterances": len(utts),
        "tokens": sum(len(u.tokens) for u in utts),
        "scored_words": sum(v for k, v in labels.items() if k != LABEL_NA),
        "labels": dict(sorted(labels.items())),
        "speakers": len({u.speaker for u in utts}),
    }
