"""Standard formats for predictions and results, shared by all experiments.

* **Predictions** (``*.tsv``): the 2019 output format ``word<TAB>gold<TAB>prediction``, one
  block per utterance, each preceded by ``<file><TAB><utterance id>`` and followed by a
  blank line.
* **Records** (``records.jsonl``): one JSON object per (experiment, model, condition,
  layer, task, split) with its metrics; ``records_to_csv`` flattens them into a tidy
  table for analysis and plots.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from prosody_llms.data.corpus import LABEL_NA, Utterance
from prosody_llms.eval.metrics import summary


def write_predictions(path: Path, utts: Sequence[Utterance], preds, n_classes: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for u, ps in zip(utts, preds):
            if len(ps) != len(u.tokens):
                raise ValueError(f"{u.utt_id}: {len(ps)} predictions for {len(u.tokens)} tokens")
            f.write(f"<file>\t{u.utt_id}\n")
            for w, g, p in zip(u.words, u.labels(n_classes), ps):
                f.write(f"{w}\t{g}\t{p}\n")
            f.write("\n")


def read_predictions(path: Path) -> dict[str, list[tuple[str, str, str]]]:
    """``{utterance id: [(word, gold, prediction), ...]}`` in file order."""
    out: dict[str, list[tuple[str, str, str]]] = {}
    cur = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("<file>\t"):
                cur = line.split("\t", 1)[1]
                out[cur] = []
            elif line:
                if cur is None:
                    raise ValueError(f"{path}: token before the first <file> line")
                w, g, p = line.split("\t")
                out[cur].append((w, g, p))
    return out


def evaluate(utts: Sequence[Utterance], preds: Sequence[Sequence[str]], n_classes: int) -> dict:
    """Metrics on all utterances (``full``) and without the last one (``legacy_2019``,
    the subset the 2019 numbers were computed on). ``legacy_2019`` is left out when that
    subset has no scored words (e.g. a one-utterance check)."""
    if len(utts) != len(preds):
        raise ValueError("one prediction list per utterance expected")

    def score(k: int) -> dict:
        gold = [lab for u in utts[:k] for lab in u.labels(n_classes)]
        pred = [p for ps in preds[:k] for p in ps]
        return {**summary(gold, pred), "utterances": k}

    out = {"full": score(len(utts))}
    if any(lab != LABEL_NA for u in utts[:-1] for lab in u.labels(n_classes)):
        out["legacy_2019"] = score(len(utts) - 1)
    return out


RECORD_KEYS = ("experiment", "model", "condition", "layer", "task", "split")


def make_record(*, experiment: str, model: str, task: int, split: str, metrics: dict,
                condition: dict | None = None, layer: int | None = None, **extra) -> dict:
    return {"experiment": experiment, "model": model, "condition": condition or {},
            "layer": layer, "task": task, "split": split, "metrics": metrics, **extra}


def append_records(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            missing = [k for k in RECORD_KEYS if k not in r]
            if missing:
                raise ValueError(f"record is missing {missing}")
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def records_to_csv(records: Sequence[dict], path: Path,
                   metrics: Sequence[str] = ("accuracy", "macro_f1", "scored_words")) -> None:
    """One row per record: identifying columns, condition fields (``cond.<name>``), and
    the requested scalar metrics (taken from ``metrics`` or ``metrics['full']``)."""
    cond_keys = sorted({k for r in records for k in r["condition"]})
    rows = []
    for r in records:
        m = r["metrics"].get("full", r["metrics"])
        rows.append({**{k: r[k] for k in RECORD_KEYS if k != "condition"},
                     **{f"cond.{k}": r["condition"].get(k) for k in cond_keys},
                     **{k: m.get(k) for k in metrics}})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else list(RECORD_KEYS))
        w.writeheader()
        w.writerows(rows)
