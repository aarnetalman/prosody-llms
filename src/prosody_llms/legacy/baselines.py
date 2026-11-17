"""Non-neural baselines from Table 3 of the 2019 paper.

* ``majority``: the most frequent training label (non-punctuation) for every word.
* ``word_majority_2019``: port of the 2019 ``WordMajority`` model. Statistics are keyed
  by the BERT-uncased id of the word's *first word piece* (so words sharing a first piece
  share statistics), counted over all training tokens including punctuation. Ties go to
  the lowest label index and unseen pieces get label index 1. In 2019 the index order was
  hash-dependent; with the sorted order used here, index 1 is "0" (non-prominent).
* ``word_majority``: majority label per lower-cased word form, unseen words get the
  global majority label (a cleaner variant, reported separately).
* ``random_uniform`` / ``random_prior``: a random label per word, uniformly or by the
  training label distribution. The 2019 paper does not say how its random baseline was
  drawn, so both are reported.
"""

from __future__ import annotations

import argparse
import collections
import logging
import random
import sys
from pathlib import Path

from prosody_llms.data.corpus import LABEL_NA, Utterance, load_splits
from prosody_llms.data.download import default_data_dir
from prosody_llms.legacy.common import PAD, environment_info, evaluate_predictions, tag_vocab, write_json

log = logging.getLogger("prosody_llms.legacy.baselines")

BASELINES = ("majority", "word_majority_2019", "word_majority", "random_uniform", "random_prior")


def label_counts(train: list[Utterance], n_classes: int) -> collections.Counter:
    return collections.Counter(lab for u in train for lab in u.labels(n_classes) if lab != LABEL_NA)


def majority(train, utts, n_classes):
    top = label_counts(train, n_classes).most_common(1)[0][0]
    return [[top] * len(u.tokens) for u in utts]


def word_majority(train, utts, n_classes):
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for u in train:
        for w, lab in zip(u.words, u.labels(n_classes)):
            counts[w.lower()][lab] += 1
    top = label_counts(train, n_classes).most_common(1)[0][0]

    def best(w):
        c = counts.get(w.lower())
        if not c:
            return top
        return max(sorted(c), key=lambda k: c[k])  # ties -> smallest label

    return [[best(w) for w in u.words] for u in utts]


def word_majority_2019(train, utts, n_classes, tokenizer, tags):
    tag_to_index = {t: i for i, t in enumerate(tags)}

    def piece_ids(w: str) -> list[int]:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w) or [tokenizer.unk_token])

    def first_piece(w: str) -> int:
        return piece_ids(w)[0]

    # As in 2019, every word piece is counted: the first with the word's label, the rest
    # with <pad>. Pieces such as "'" in "don't" therefore collect <pad> counts and can
    # predict <pad> when they start a word, which is scored as an error.
    counts: dict[int, list[int]] = collections.defaultdict(lambda: [0] * len(tags))
    for u in train:
        for w, lab in zip(u.words, u.labels(n_classes)):
            ids = piece_ids(w)
            counts[ids[0]][tag_to_index[lab]] += 1
            for i in ids[1:]:
                counts[i][tag_to_index[PAD]] += 1

    def best(w):
        c = counts.get(first_piece(w))
        if c is None:
            return tags[1]
        return tags[max(range(len(c)), key=lambda i: (c[i], -i))]  # ties -> lowest index

    return [[best(w) for w in u.words] for u in utts]


def random_labels(train, utts, n_classes, seed, prior: bool):
    counts = label_counts(train, n_classes)
    labels = sorted(counts)
    weights = [counts[k] for k in labels] if prior else None
    rng = random.Random(seed)
    return [rng.choices(labels, weights=weights, k=len(u.tokens)) for u in utts]


def run(args) -> dict:
    splits = load_splits(args.data_dir, args.train_set, 1.0, legacy=args.protocol == "legacy")
    full = load_splits(args.data_dir, args.train_set, 1.0, legacy=False)
    train, dev, test = splits["train"], full["dev"], full["test"]
    tokenizer = None
    if "word_majority_2019" in args.baselines:
        from transformers import AutoTokenizer

        from prosody_llms.legacy.train import BERT_MODEL, BERT_REVISION
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, revision=BERT_REVISION,
                                                  do_lower_case=True)
    results = {"experiment": "repro_2019_baselines", "train_set": args.train_set,
               "protocol": args.protocol, "seed": args.seed, "results": {},
               "environment": environment_info()}
    for n_classes in (2, 3):
        tags = tag_vocab(full, n_classes)
        for name in args.baselines:
            out = {}
            for split_name, utts in (("dev", dev), ("test", test)):
                if name == "majority":
                    preds = majority(train, utts, n_classes)
                elif name == "word_majority":
                    preds = word_majority(train, utts, n_classes)
                elif name == "word_majority_2019":
                    preds = word_majority_2019(train, utts, n_classes, tokenizer, tags)
                else:
                    preds = random_labels(train, utts, n_classes, args.seed,
                                          prior=name == "random_prior")
                out[split_name] = evaluate_predictions(utts, preds, n_classes)
            results["results"][f"{name}/{n_classes}way"] = out
            log.info("%-20s %d-way test %.2f%% (full) %.2f%% (2019 subset)", name, n_classes,
                     out["test"]["full"]["accuracy"], out["test"]["legacy_2019"]["accuracy"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, results)
    return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--train-set", choices=["train_100", "train_360"], default="train_360")
    p.add_argument("--protocol", choices=["legacy", "full"], default="legacy")
    p.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, default=Path("results/repro_2019/baselines.json"))
    run(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
