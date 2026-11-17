"""Faithful port of the 2019 BERT prominence classifier (``main.py --model BertUncased``).

Kept from 2019 (all of these affect results):

* each word is word-piece tokenised on its own; ``[CLS]``/``[SEP]`` wrap the utterance;
  the label sits on the first word piece, other pieces get ``<pad>`` (ignored by the loss);
* a single linear layer on BERT's last hidden layer, full fine-tuning;
* punctuation (``NA``) is a training class; only ``<pad>`` is ignored by the loss;
* padding uses token id 0 and **no attention mask** is passed to BERT
  (``--attention-mask`` turns one on, as a non-legacy option);
* Adam, lr 5e-5, no weight decay, no warm-up or schedule, batch 32, 2 epochs;
* plain ("eager") attention, as in 2019; it is also the only BERT attention kernel
  whose backward pass is deterministic on GPU;
* the test score comes from the final-epoch model (the 2019 best-dev logic was a no-op);
* predictions are the argmax over all output units, including ``<pad>``;
* with ``--protocol legacy`` the training data is read with the 2019 loader
  (``--fraction-of-train-data`` slices raw lines; the last utterance is dropped).

Deviations: label order is sorted (2019: hash-dependent), runs use deterministic
kernels, and dev/test are scored both on all utterances (``full``) and without the last
utterance of each file (``legacy_2019``, the subset the 2019 numbers were computed on).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from prosody_llms.data.corpus import Utterance, load_splits, read_split
from prosody_llms.data.download import FILES, default_data_dir, sha256
from prosody_llms.legacy.common import (
    PAD,
    environment_info,
    evaluate_predictions,
    tag_vocab,
    write_json,
    write_predictions,
)
from prosody_llms.utils.seed import set_seed

log = logging.getLogger("prosody_llms.legacy.train")

BERT_MODEL = "google-bert/bert-base-uncased"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--out-dir", type=Path, required=True, help="run directory (created)")
    p.add_argument("--n-classes", type=int, choices=[2, 3], required=True)
    p.add_argument("--train-set", choices=["train_100", "train_360"], default="train_360")
    p.add_argument("--fraction-of-train-data", type=float, default=1.0)
    p.add_argument("--protocol", choices=["legacy", "full"], default="legacy",
                   help="how training data is read (dev/test are always scored both ways)")
    p.add_argument("--model", default=BERT_MODEL)
    p.add_argument("--name", default=None,
                   help="registry name of another encoder (fine-tuning runs; default: the 2019 BERT)")
    p.add_argument("--revision", default=BERT_REVISION)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--attn-implementation", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2"],
                   help="eager (default) is deterministic on GPU and matches 2019's BERT; "
                        "sdpa is faster but its backward pass is non-deterministic")
    p.add_argument("--attention-mask", action="store_true",
                   help="pass an attention mask (not done in 2019)")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--resume", action="store_true", help="continue from the last finished epoch")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="save no resume checkpoint (for many large runs at once on a tight quota)")
    p.add_argument("--keep-checkpoint", action="store_true",
                   help="keep checkpoint.pt (about 1.3 GB) after the run finishes")
    p.add_argument("--max-train-steps", type=int, default=None, help="debugging only")
    return p


# ----------------------------------------------------------------------------- data


def needs_word_space(tokenizer) -> bool:
    """True for tokenizers that mark the start of a word with a space (byte-level BPE as
    in ModernBERT or RoBERTa): words tokenised one at a time then get the leading space
    they have in running text. False for WordPiece (BERT) and SentencePiece (XLM-R),
    which tokenise a lone word the same either way, so 2019 behaviour is unchanged."""
    return tokenizer.tokenize("house") != tokenizer.tokenize(" house")


def encode(utt: Utterance, tokenizer, tag_to_index: dict[str, int], n_classes: int,
           stats: dict | None = None, word_space: bool = False):
    """Return (token ids, label ids, first-piece mask) for one utterance, as in 2019."""
    cls, sep = tokenizer.cls_token, tokenizer.sep_token
    words = [cls] + utt.words + [sep]
    tags = [PAD] + utt.labels(n_classes) + [PAD]
    ids, ys, heads = [], [], []
    for w, t in zip(words, tags):
        pieces = [w] if w in (cls, sep) else tokenizer.tokenize(" " + w if word_space else w)
        if not pieces:  # never happens on this corpus with BERT; guard for other tokenisers
            pieces = [tokenizer.unk_token]
            if stats is not None:
                stats["empty_words"] = stats.get("empty_words", 0) + 1
        ids.extend(tokenizer.convert_tokens_to_ids(pieces))
        ys.extend([tag_to_index[t]] + [tag_to_index[PAD]] * (len(pieces) - 1))
        heads.extend([1] + [0] * (len(pieces) - 1))
    return ids, ys, heads


def collate(batch, pad_id: int):
    import torch

    maxlen = max(len(ids) for ids, _, _ in batch)
    x = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    y = torch.zeros((len(batch), maxlen), dtype=torch.long)
    mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, (ids, ys, _) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids)
        y[i, : len(ys)] = torch.tensor(ys)
        mask[i, : len(ids)] = 1
    heads = [h for _, _, h in batch]
    return x, y, mask, heads


# ----------------------------------------------------------------------------- model


def build_model(model_name: str, revision: str | None, n_labels: int,
                attn_implementation: str = "eager"):
    import torch.nn as nn
    from transformers import AutoModel

    class BertTagger(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name, revision=revision,
                                                  attn_implementation=attn_implementation)
            self.fc = nn.Linear(self.bert.config.hidden_size, n_labels)

        def forward(self, x, attention_mask=None):
            enc = self.bert(input_ids=x, attention_mask=attention_mask).last_hidden_state
            return self.fc(enc)

    return BertTagger()


def predict(model, encoded, utts, index_to_tag, device, batch_size, use_mask,
            pad_id: int = 0) -> list[list[str]]:
    import torch

    model.eval()
    out: list[list[str]] = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            chunk = encoded[start : start + batch_size]
            x, _, mask, heads = collate(chunk, pad_id)
            x, mask = x.to(device), mask.to(device)
            y_hat = model(x, mask if use_mask else None).argmax(-1).cpu().tolist()
            for row, head in zip(y_hat, heads):
                firsts = [index_to_tag[h] for h, keep in zip(row, head) if keep]
                out.append(firsts[1:-1])  # drop [CLS] and [SEP]
    for u, p in zip(utts, out):
        assert len(p) == len(u.tokens), (u.utt_id, len(p), len(u.tokens))
    return out


# ----------------------------------------------------------------------------- main


def run(args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoTokenizer

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    legacy = args.protocol == "legacy"
    train = load_splits(args.data_dir, args.train_set, args.fraction_of_train_data, legacy)["train"]
    dev = read_split(args.data_dir / "dev.txt")
    test = read_split(args.data_dir / "test.txt")
    tags = tag_vocab({"train": train, "dev": dev, "test": test}, args.n_classes)
    tag_to_index = {t: i for i, t in enumerate(tags)}
    index_to_tag = dict(enumerate(tags))
    log.info("train %d / dev %d / test %d utterances; labels %s",
             len(train), len(dev), len(test), tags)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, do_lower_case=True)
    stats: dict = {}
    word_space = needs_word_space(tokenizer)
    # 2019: padding with id 0 and no attention mask. With --attention-mask (other
    # encoders) the model's own padding token is used; the 2019 setting is unchanged.
    pad_id = tokenizer.pad_token_id if args.attention_mask and tokenizer.pad_token_id is not None else 0
    enc_train = [encode(u, tokenizer, tag_to_index, args.n_classes, stats, word_space) for u in train]
    enc_dev = [encode(u, tokenizer, tag_to_index, args.n_classes, stats, word_space) for u in dev]
    enc_test = [encode(u, tokenizer, tag_to_index, args.n_classes, stats, word_space) for u in test]
    if stats:
        log.warning("tokenisation: %s", stats)

    model = build_model(args.model, args.revision, len(tags),
                        args.attn_implementation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tag_to_index[PAD])

    ckpt_path = args.out_dir / "checkpoint.pt"
    history: list[dict] = []
    start_epoch = 0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["rng_torch"])
        if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])
        start_epoch, history = ckpt["epoch"], ckpt["history"]
        log.info("resumed after epoch %d", start_epoch)

    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(start_epoch):  # replay the shuffles of finished epochs
        torch.randperm(len(enc_train), generator=generator)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = torch.randperm(len(enc_train), generator=generator).tolist()
        n_steps = (len(order) + args.batch_size - 1) // args.batch_size
        for step in range(n_steps):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break
            batch = [enc_train[i] for i in order[step * args.batch_size : (step + 1) * args.batch_size]]
            x, y, mask, _ = collate(batch, pad_id)
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(x, mask if args.attention_mask else None)
            loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))
            loss.backward()
            optimizer.step()
            if step % args.log_every == 0 or step + 1 == n_steps:
                log.info("epoch %d step %d/%d loss %.4f (%.0fs)",
                         epoch + 1, step + 1, n_steps, loss.item(), time.time() - t0)
        dev_preds = predict(model, enc_dev, dev, index_to_tag, device, args.eval_batch_size,
                            args.attention_mask, pad_id)
        dev_scores = evaluate_predictions(dev, dev_preds, args.n_classes)
        history.append({"epoch": epoch + 1,
                        "dev_accuracy_full": dev_scores["full"]["accuracy"],
                        "dev_accuracy_legacy_2019": dev_scores["legacy_2019"]["accuracy"],
                        "elapsed_s": round(time.time() - t0, 1)})
        log.info("epoch %d dev accuracy %.2f%%", epoch + 1, dev_scores["full"]["accuracy"])
        if args.no_checkpoint:
            continue
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                    "epoch": epoch + 1, "history": history}, ckpt_path)

    test_preds = predict(model, enc_test, test, index_to_tag, device, args.eval_batch_size,
                         args.attention_mask, pad_id)
    if start_epoch >= args.epochs:  # resumed after training had already finished
        dev_preds = predict(model, enc_dev, dev, index_to_tag, device, args.eval_batch_size,
                            args.attention_mask, pad_id)
    write_predictions(args.out_dir / "predictions_test.tsv", test, test_preds, args.n_classes)
    write_predictions(args.out_dir / "predictions_dev.tsv", dev, dev_preds, args.n_classes)

    files = [f"{args.train_set}.txt", "dev.txt", "test.txt"]
    result = {
        "experiment": "repro_2019" if args.name is None else "finetune",
        "model": args.name or "bert-base-uncased",
        "tokenisation": {"word_start_space": word_space, "pad_id": pad_id},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "labels": tags,
        "data": {"train_utterances": len(train),
                 "sha256_ok": {f: sha256(args.data_dir / f) == FILES[f][0] for f in files}},
        "history": history,
        "dev": evaluate_predictions(dev, dev_preds, args.n_classes),
        "test": evaluate_predictions(test, test_preds, args.n_classes),
        "train_time_s": round(time.time() - t0, 1),
        "environment": environment_info(),
    }
    write_json(args.out_dir / "result.json", result)
    if ckpt_path.exists() and not args.keep_checkpoint:
        ckpt_path.unlink()
    log.info("test accuracy %.2f%% (full), %.2f%% (2019 subset)",
             result["test"]["full"]["accuracy"], result["test"]["legacy_2019"]["accuracy"])
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if (args.out_dir / "result.json").exists():
        log.info("%s already has result.json; nothing to do", args.out_dir)
        return 0
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
