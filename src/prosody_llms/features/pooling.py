"""Pool token-level hidden states into one vector per word."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

POOLINGS = ("first", "last", "mean")


def pool_words(hidden, word_tokens: Sequence[Sequence[int]], how: str = "first"):
    """``hidden``: array or tensor of shape (layers, tokens, dim).

    Returns shape (layers, words, dim) of the same type. ``first`` is the 2019 choice for
    encoders; for left-to-right decoders ``last`` is the token that has seen the whole word.
    """
    if how not in POOLINGS:
        raise ValueError(f"how must be one of {POOLINGS}")
    if any(len(t) == 0 for t in word_tokens):
        raise ValueError("every word needs at least one token")
    if how == "first":
        idx = [t[0] for t in word_tokens]
        return hidden[:, idx]
    if how == "last":
        idx = [t[-1] for t in word_tokens]
        return hidden[:, idx]
    try:
        import torch

        if isinstance(hidden, torch.Tensor):
            return torch.stack([hidden[:, list(t)].mean(dim=1) for t in word_tokens], dim=1)
    except ImportError:
        pass
    return np.stack([np.asarray(hidden)[:, list(t)].mean(axis=1) for t in word_tokens], axis=1)
