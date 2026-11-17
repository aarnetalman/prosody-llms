"""Probe interface and label handling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from prosody_llms.data.corpus import LABEL_NA, Utterance


@dataclass(frozen=True)
class LabelSpace:
    """Scored labels of a task (``NA`` is never a class: punctuation is not scored)."""

    n_classes: int

    @property
    def labels(self) -> tuple[str, ...]:
        return ("0", "1") if self.n_classes == 2 else ("0", "1", "2")

    def encode(self, utt: Utterance) -> np.ndarray:
        """Class index per token, -1 for punctuation."""
        index = {lab: i for i, lab in enumerate(self.labels)}
        return np.array([index.get(lab, -1) if lab != LABEL_NA else -1
                         for lab in utt.labels(self.n_classes)], dtype=np.int64)

    def decode(self, idx: Sequence[int], utt: Utterance) -> list[str]:
        """Labels for every token; punctuation gets ``NA`` (it is not scored)."""
        return [LABEL_NA if t.is_punct else self.labels[i] for i, t in zip(idx, utt.tokens)]


class Probe(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray:
        """``features``: (layers, words, dim) -> class indices (layers, words)."""
