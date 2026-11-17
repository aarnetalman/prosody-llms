"""Helpers specific to the 2019 reproduction. General utilities live in
``prosody_llms.eval.io`` and ``prosody_llms.utils.runinfo`` and are re-exported here."""

from __future__ import annotations

from prosody_llms.data.corpus import Utterance
from prosody_llms.eval.io import evaluate as evaluate_predictions
from prosody_llms.eval.io import write_predictions
from prosody_llms.utils.runinfo import environment_info, git_commit, write_json

PAD = "<pad>"

__all__ = ["PAD", "environment_info", "evaluate_predictions", "git_commit", "tag_vocab",
           "write_json", "write_predictions"]


def tag_vocab(splits: dict[str, list[Utterance]], n_classes: int) -> list[str]:
    """``["<pad>"] + labels`` as in 2019, but in sorted order.

    The 2019 code used ``list(set(...))``, whose order depends on Python's hash
    randomisation and so differed between processes. Sorting makes it stable;
    this is one of the documented deviations.
    """
    labels = {lab for utts in splits.values() for u in utts for lab in u.labels(n_classes)}
    return [PAD] + sorted(labels)
