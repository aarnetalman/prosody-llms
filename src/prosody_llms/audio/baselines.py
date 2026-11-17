"""Duration-only baseline: prominence from the word timings alone.

    python -m prosody_llms.audio.baselines

Features per aligned word, from the public alignments only (no audio):

* log duration, and log duration minus the utterance's mean;
* number of phones and log duration per phone;
* log pause before and after the word (silence between aligned words);
* relative position in the utterance, and whether it is the last word.

A linear discriminant (shared covariance, small shrinkage, standardised features) is
trained on ``train_100`` and scored on dev and test in the shared record format: ``full``
counts words without a timing as non-prominent ("0"), ``aligned`` scores only words with
a timing. Duration is one of the three signals the labels were made from, so this row
says how much of prominence is timing alone.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from prosody_llms.audio.base import LIBRITTS_SUBSETS
from prosody_llms.audio.download import default_audio_dir, textgrid_root
from prosody_llms.audio.reconstruct import mask_utterances
from prosody_llms.audio.timings import SILENCES, UtteranceTimings, load_timings
from prosody_llms.data.corpus import LABEL_NA, read_split
from prosody_llms.data.download import default_data_dir
from prosody_llms.eval.io import evaluate, make_record
from prosody_llms.utils.runinfo import environment_info, write_json

log = logging.getLogger("prosody_llms.audio.baselines")
FEATURES = ("log_dur", "log_dur_rel", "phones", "log_dur_per_phone", "log_pause_before",
            "log_pause_after", "position", "is_last")


def word_features(t: UtteranceTimings) -> np.ndarray:
    """(tokens, features); NaN rows for tokens without a timing."""
    spans = [(x.start, x.end) if x.aligned else None for x in t.timings]
    present = [s for s in spans if s]
    out = np.full((len(spans), len(FEATURES)), np.nan)
    if not present:
        return out
    mean_log = float(np.mean([np.log(e - s) for s, e in present]))
    phones = [p for p in t.phones if p.text.lower() not in SILENCES]
    order = sorted({s for s in present})
    for i, sp in enumerate(spans):
        if sp is None:
            continue
        s, e = sp
        k = order.index(sp)
        prev_end = order[k - 1][1] if k > 0 else 0.0
        next_start = order[k + 1][0] if k + 1 < len(order) else t.duration
        n_ph = sum(1 for p in phones if p.start >= s - 1e-6 and p.end <= e + 1e-6) or 1
        d = max(e - s, 1e-3)
        out[i] = [np.log(d), np.log(d) - mean_log, n_ph, np.log(d / n_ph),
                  np.log1p(max(s - prev_end, 0.0) * 1000), np.log1p(max(next_start - e, 0.0) * 1000),
                  k / max(len(order) - 1, 1), float(k == len(order) - 1)]
    return out


def split_features(split: str, args):
    utts = read_split(args.data_dir / f"{split}.txt")
    if args.max_utterances:
        utts = utts[: args.max_utterances]
    tg = textgrid_root(args.audio_dir)
    feats, timings = [], []
    for u in utts:
        t = load_timings(tg, LIBRITTS_SUBSETS[split], u)
        timings.append(t)
        feats.append(word_features(t) if t else np.full((len(u.tokens), len(FEATURES)), np.nan))
    return utts, feats


class LDA:
    def __init__(self, shrinkage: float = 1e-3):
        self.shrinkage = shrinkage

    def fit(self, x: np.ndarray, y: np.ndarray) -> LDA:
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        z = (x - self.mu) / self.sd
        self.classes = np.unique(y)
        means = np.stack([z[y == c].mean(0) for c in self.classes])
        cov = sum(np.cov(z[y == c].T, bias=True) * (y == c).sum() for c in self.classes) / len(y)
        cov = (1 - self.shrinkage) * cov + self.shrinkage * np.eye(len(cov))
        prec = np.linalg.inv(cov)
        priors = np.array([(y == c).mean() for c in self.classes])
        self.w = means @ prec
        self.b = -0.5 * np.einsum("ij,ij->i", self.w, means) + np.log(priors)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mu) / self.sd
        return self.classes[np.argmax(z @ self.w.T + self.b, axis=1)]


def labelled(utts, feats, task: int):
    xs, ys = [], []
    for u, f in zip(utts, feats):
        for tok, lab, row in zip(u.tokens, u.labels(task), f):
            if lab != LABEL_NA and not np.isnan(row).any():
                xs.append(row)
                ys.append(int(lab))
    return np.array(xs), np.array(ys)


def predict(model: LDA, utts, feats) -> list[list[str]]:
    out = []
    for u, f in zip(utts, feats):
        ok = ~np.isnan(f).any(axis=1)
        p = np.zeros(len(u.tokens), dtype=int)
        if ok.any():
            p[ok] = model.predict(f[ok])
        out.append([LABEL_NA if tok.is_punct else str(v) for tok, v in zip(u.tokens, p)])
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-set", default="train_100", choices=["train_100", "dev"])
    p.add_argument("--max-utterances", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--audio-dir", type=Path, default=default_audio_dir())
    p.add_argument("--out-root", type=Path,
                   default=Path(os.environ.get("PROSODY_RUNS", "runs")) / "audio" / "baselines" / "duration")
    args = p.parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    train = split_features(args.train_set, args)
    evals = {s: split_features(s, args) for s in ("dev", "test")}
    records, summary = [], {}
    for task in (3, 2):
        x, y = labelled(*train, task)
        model = LDA().fit(x, y)
        for split, (utts, feats) in evals.items():
            preds = predict(model, utts, feats)
            from prosody_llms.probing.run import write_predictions_gz

            write_predictions_gz(args.out_root / f"predictions_{split}_{task}way.tsv.gz", utts, preds, task)
            masks = [list(~np.isnan(f).any(axis=1)) for f in feats]
            m = evaluate(utts, preds, task)
            m["aligned"] = evaluate(mask_utterances(utts, masks), preds, task)["full"]
            records.append(make_record(experiment="audio_baseline", model="duration", task=task, split=split,
                                       condition={"features": list(FEATURES), "train_set": args.train_set,
                                                  "classifier": "lda"}, metrics=m))
            summary[f"{split}/{task}way"] = {"full": m["full"]["accuracy"], "aligned": m["aligned"]["accuracy"]}
            log.info("%s %d-way: %.2f (all words), %.2f (aligned words)", split, task,
                     m["full"]["accuracy"], m["aligned"]["accuracy"])
    with open(args.out_root / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    write_json(args.out_root / "result.json", {"experiment": "audio_baseline", "model": "duration",
                                               "train_set": args.train_set, "train_words": int(len(x)),
                                               "accuracy": summary, "environment": environment_info()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
