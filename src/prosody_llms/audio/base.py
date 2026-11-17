"""Word timings, LibriTTS paths and the speech feature extractor interface."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from prosody_llms.data.corpus import Utterance

# Corpus split -> LibriTTS subset.
LIBRITTS_SUBSETS = {"train_100": "train-clean-100", "train_360": "train-clean-360",
                    "dev": "dev-clean", "test": "test-clean"}


@dataclass(frozen=True)
class WordTiming:
    """Start and end (seconds) of one corpus word in its utterance's audio.

    ``start``/``end`` are None for words without a reliable alignment; they are excluded
    from both text and audio scores in the text-versus-audio comparison.
    """

    word: str
    start: float | None
    end: float | None

    @property
    def aligned(self) -> bool:
        return self.start is not None and self.end is not None and self.end > self.start


def libritts_path(root: str | Path, split: str, utt_id: str) -> Path:
    """``<root>/LibriTTS/<subset>/<speaker>/<chapter>/<utt_id>.wav``."""
    speaker, chapter = utt_id.split("_")[:2]
    return Path(root) / "LibriTTS" / LIBRITTS_SUBSETS[split] / speaker / chapter / f"{utt_id}.wav"


class AudioFeatureExtractor(Protocol):
    name: str

    @property
    def n_layers(self) -> int: ...

    @property
    def dim(self) -> int: ...

    def extract(self, utts: Sequence[Utterance],
                timings: Sequence[Sequence[WordTiming]]) -> Iterator[np.ndarray]:
        """Yield one ``(layers, words, dim)`` array per utterance; words without a
        timing get NaN rows."""
