"""Stronger probes on the best layer of a frozen model: logistic regression, MLP, BiLSTM.

    python -m prosody_llms.probing.nonlinear --model muse-glimmer-30b --context echo

The layer is the best layer of the linear (LDA) run for the same model and condition,
chosen on dev separately for each task. Its word vectors for train, dev and test are
extracted once and kept in CPU memory as bfloat16 (float16 would overflow on the very
large activations some LLMs have); nothing is written to disk. The model is then freed
and three probes are trained on the GPU, each with several seeds:

* ``logistic``: softmax regression (linear, trained discriminatively);
* ``mlp``: one hidden layer (non-linear, one word at a time);
* ``bilstm``: a bidirectional LSTM over the frozen word vectors of the whole utterance
  (non-linear and contextual; for left-to-right models it adds the right-hand context).

Features are standardised with training-set statistics. Training stops early on dev
accuracy (all words); the best epoch is evaluated on test. Punctuation is in the input
sequence for the BiLSTM but never in the loss or the scores.

Outputs in ``<out-root>/<model>/<condition>/``: ``records.jsonl`` (one record per probe,
seed, task and split), ``summary.md``/``summary.csv`` (mean and sd over seeds), seed-0
test predictions (``*.tsv.gz``) and ``result.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from prosody_llms.config import get_model
from prosody_llms.data.corpus import read_split
from prosody_llms.data.download import default_data_dir
from prosody_llms.data.subsets import fixed_subset
from prosody_llms.eval.io import append_records, evaluate, make_record, read_records, records_to_csv
from prosody_llms.features.alignment import CONTEXTS, STYLES
from prosody_llms.features.extractor import HFTextExtractor, length_batches
from prosody_llms.features.pooling import POOLINGS
from prosody_llms.probes.base import LabelSpace
from prosody_llms.probing.run import AUDIO_KINDS, condition_name, write_predictions_gz
from prosody_llms.utils.runinfo import environment_info, write_json
from prosody_llms.utils.seed import set_seed

log = logging.getLogger("prosody_llms.probing.nonlinear")
PROBES = ("logistic", "mlp", "bilstm")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--context", choices=CONTEXTS, default="plain")
    p.add_argument("--pooling", choices=POOLINGS, default=None)
    p.add_argument("--style", choices=STYLES, default="natural")
    p.add_argument("--layers", default=None,
                   help="'<2-way layer>,<3-way layer>' (default: best LDA layers of the linear run)")
    p.add_argument("--probes", nargs="+", choices=PROBES, default=list(PROBES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--train-set", choices=["train_100", "train_360"], default=None,
                   help="default: train_360 for text models, train_100 for speech models")
    p.add_argument("--audio-dir", type=Path, default=None)
    p.add_argument("--kind", choices=["speech", "speech_text"], default=None)
    p.add_argument("--part", choices=["llm", "encoder"], default="llm")
    p.add_argument("--train-exemplars", type=int, default=None,
                   help="train only on the K utterances used as prompt examples")
    p.add_argument("--train-utterances", type=int, default=None)
    runs = Path(os.environ.get("PROSODY_RUNS", "runs"))
    p.add_argument("--linear-root", type=Path, default=runs / "probing")
    p.add_argument("--out-root", type=Path, default=runs / "probing_nonlinear")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=None)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--random-features", type=int, default=None, metavar="DIM",
                   help="control: instead of a model, a fixed random DIM-dimensional vector per word type "
                        "(use with --model random-words-DIM --layers 0,0)")
    return p


# ----------------------------------------------------------------------------- features


def extract_layers(ext, utts, layers: list[int], max_tokens: int):
    """``{layer: bfloat16 CPU tensor (tokens, dim)}`` in corpus order, plus utterance offsets."""

    offsets = np.cumsum([0] + [len(u.tokens) for u in utts])
    out = {layer: torch.empty(int(offsets[-1]), ext.dim, dtype=torch.bfloat16) for layer in layers}
    aligned = [ext.align(u) for u in utts]
    batches = length_batches([len(a.input_ids) for a in aligned], max_tokens)
    for b, idx in enumerate(batches):
        for i, h in zip(idx, ext.forward([aligned[i] for i in idx])):
            for layer in layers:
                out[layer][offsets[i]:offsets[i + 1]] = h[layer].to(torch.bfloat16).cpu()
        if b % 200 == 0:
            log.info("extracted batch %d/%d", b + 1, len(batches))
    return out, offsets


# ----------------------------------------------------------------------------- probes


class BiLSTMProbe(nn.Module):
    """Projection, one bidirectional LSTM layer over the utterance, linear output."""

    def __init__(self, dim: int, n_classes: int, proj: int = 256, hidden: int = 256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, proj), nn.Dropout(0.1))
        self.lstm = nn.LSTM(proj, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x, lengths):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        packed = pack_padded_sequence(self.proj(x), lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        return self.out(h)


def make_probe(kind: str, dim: int, n_classes: int) -> nn.Module:
    if kind == "logistic":
        return nn.Linear(dim, n_classes)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, n_classes))
    if kind == "bilstm":
        return BiLSTMProbe(dim, n_classes)
    raise ValueError(kind)


class ProbeData:
    """Standardised features and labels for one split, on the training device."""

    def __init__(self, feats, offsets, labels, mean, std, device):
        import torch

        self.x = feats  # bfloat16, (tokens, dim), on `device`
        self.offsets = offsets
        self.y = torch.as_tensor(labels, device=device)
        self.mean, self.std = mean, std

    def rows(self, idx):
        import torch

        return ((self.x[idx].to(torch.float32) - self.mean) / self.std)

    def sequences(self, utt_idx):
        """Padded (batch, max_len, dim) features, labels (-1 = pad/punctuation), lengths."""
        import torch

        lens = [int(self.offsets[i + 1] - self.offsets[i]) for i in utt_idx]
        maxlen = max(lens)
        x = torch.zeros(len(utt_idx), maxlen, self.x.shape[1], device=self.x.device)
        y = torch.full((len(utt_idx), maxlen), -1, dtype=torch.long, device=self.x.device)
        for j, i in enumerate(utt_idx):
            s, e = int(self.offsets[i]), int(self.offsets[i + 1])
            x[j, : e - s] = self.rows(slice(s, e))
            y[j, : e - s] = self.y[s:e]
        return x, y, torch.tensor(lens)


def predict(kind, model, data: ProbeData, batch_words=65536, batch_utts=256):
    """Class indices for every token of the split (punctuation included, ignored later)."""

    model.eval()
    out = torch.empty(data.x.shape[0], dtype=torch.long, device=data.x.device)
    with torch.no_grad():
        if kind == "bilstm":
            n = len(data.offsets) - 1
            for s in range(0, n, batch_utts):
                idx = list(range(s, min(n, s + batch_utts)))
                x, _, lens = data.sequences(idx)
                p = model(x, lens).argmax(-1)
                for j, i in enumerate(idx):
                    a, b = int(data.offsets[i]), int(data.offsets[i + 1])
                    out[a:b] = p[j, : b - a]
        else:
            for s in range(0, data.x.shape[0], batch_words):
                out[s:s + batch_words] = model(data.rows(slice(s, s + batch_words))).argmax(-1)
    return out


def accuracy(pred, data: ProbeData) -> float:
    mask = data.y >= 0
    return float((pred[mask] == data.y[mask]).float().mean() * 100)


def train_probe(kind, train: ProbeData, dev: ProbeData, n_classes, seed, epochs, patience):

    set_seed(seed)
    device = train.x.device
    model = make_probe(kind, train.x.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    scored = torch.nonzero(train.y >= 0).squeeze(1)
    n_utts = len(train.offsets) - 1
    for epoch in range(epochs):
        model.train()
        if kind == "bilstm":
            order = torch.randperm(n_utts, generator=gen).tolist()
            for s in range(0, n_utts, 64):
                x, y, lens = train.sequences(order[s:s + 64])
                logits = model(x, lens)
                loss = loss_fn(logits.reshape(-1, n_classes), y.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
        else:
            order = scored[torch.randperm(len(scored), generator=gen).to(device)]
            for s in range(0, len(order), 4096):
                idx = order[s:s + 4096]
                loss = loss_fn(model(train.rows(idx)), train.y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        acc = accuracy(predict(kind, model, dev), dev)
        history.append(round(acc, 3))
        if acc > best:
            best, best_epoch, bad = acc, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "dev_accuracy": best, "dev_history": history}


# ----------------------------------------------------------------------------- main


def best_layers(linear_dir: Path) -> dict[int, int]:
    res = json.loads((linear_dir / "result.json").read_text(encoding="utf-8"))
    return {s["task"]: s["best_layer"] for s in res["summary"] if s["probe"] == "lda"}


def run(args) -> dict:

    set_seed(0)
    try:
        spec = get_model(args.model)
        model_id, revision, kind, name = spec.id, spec.revision, spec.kind, spec.name
    except KeyError:
        if not Path(args.model).exists() and not args.random_features:
            raise
        spec = None
        model_id, revision, kind, name = args.model, args.revision, args.kind or "custom", Path(args.model).name
    audio = kind in AUDIO_KINDS
    if args.train_set is None:
        args.train_set = "train_100" if audio else "train_360"
    if args.pooling is None:
        args.pooling = "mean" if audio else "last" if kind == "decoder" else "first"
    cond = condition_name(args, kind)
    out_dir = args.out_root / name / cond
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "result.json").exists():
        log.info("already finished (%s)", out_dir)
        return json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    if args.layers:
        l2, l3 = (int(x) for x in args.layers.split(","))
        layers = {2: l2, 3: l3}
    else:
        layers = best_layers(args.linear_root / name / cond)
    log.info("%s %s: layers %s", name, cond, layers)

    train = read_split(args.data_dir / f"{args.train_set}.txt")
    if args.train_utterances:
        train = fixed_subset(train, args.train_utterances, seed=0)
    if args.train_exemplars:  # exactly the training utterances used as prompt examples
        from prosody_llms.prompting.prompts import select_exemplars

        train = select_exemplars(train, args.train_exemplars)
    dev = read_split(args.data_dir / "dev.txt")
    test = read_split(args.data_dir / "test.txt")

    t0 = time.time()
    masks = None
    if audio:
        from types import SimpleNamespace

        from prosody_llms.audio.download import default_audio_dir
        from prosody_llms.audio.extractors import make_audio_extractor

        ext = make_audio_extractor(spec or SimpleNamespace(id=model_id, revision=revision, kind=kind, name=name),
                                   args.audio_dir or default_audio_dir(), args.dtype, args.device, part=args.part)
        ext.register({args.train_set: train, "dev": dev, "test": test})
        masks = {s: np.concatenate([np.array(ext.word_mask(u), dtype=bool) for u in utts])
                 for s, utts in (("train", train), ("dev", dev), ("test", test))}
    elif args.random_features:  # control: fixed random vectors per word type, no model
        from prosody_llms.features.extractor import RandomWordExtractor

        ext = RandomWordExtractor(args.random_features, device=args.device, name=name)
    else:
        ext = HFTextExtractor(model_id, revision, style=args.style, pooling=args.pooling,
                              context=args.context, device=args.device, dtype=args.dtype, name=name)
        if args.context.startswith("prev"):
            ext.register_discourse([read_split(args.data_dir / f"{s}.txt") for s in (args.train_set, "dev", "test")])
    needed = sorted(set(layers.values()))
    feats = {s: extract_layers(ext, u, needed, args.max_tokens)
             for s, u in (("train", train), ("dev", dev), ("test", test))}
    device = ext.device
    meta = {"model_class": ext.model_class, "n_layers": ext.n_layers, "dim": ext.dim,
            "oom_splits": getattr(ext, "oom_splits", 0)}
    del ext
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t1 = time.time()

    records_path = out_dir / "records.jsonl"
    records_path.unlink(missing_ok=True)
    runs_info = []
    for task in (2, 3):
        layer = layers[task]
        space = LabelSpace(task)
        x_train = feats["train"][0][layer]
        mean = x_train.float().mean(0).to(device)
        std = x_train.float().std(0).clamp_min(1e-6).to(device)
        data = {}
        for split, utts in (("train", train), ("dev", dev), ("test", test)):
            x, offsets = feats[split][0][layer].to(device), feats[split][1]
            labels = np.concatenate([space.encode(u) for u in utts])
            if masks is not None:  # words without a timing: no features, not trained or selected on
                labels[~masks[split]] = -1
            data[split] = ProbeData(x, offsets, labels, mean, std, device)
        for probe in args.probes:
            for seed in args.seeds:
                t = time.time()
                model, info = train_probe(probe, data["train"], data["dev"], task, seed,
                                          args.epochs, args.patience)
                recs = []
                for split, utts in (("dev", dev), ("test", test)):
                    flat = predict(probe, model, data[split]).cpu().numpy()
                    if masks is not None:
                        flat[~masks[split]] = 0  # no timing: "0", as in the linear runs
                    off = data[split].offsets
                    preds = [space.decode(flat[off[i]:off[i + 1]].tolist(), u)
                             for i, u in enumerate(utts)]
                    metrics = evaluate(utts, preds, task)
                    if masks is not None:
                        from prosody_llms.audio.reconstruct import mask_utterances

                        m = [list(masks[split][off[i]:off[i + 1]]) for i in range(len(utts))]
                        metrics["aligned"] = evaluate(mask_utterances(utts, m), preds, task)["full"]
                    recs.append(make_record(
                        experiment="probing_nonlinear", model=name, task=task, split=split,
                        layer=layer, metrics=metrics,
                        condition=({"input": "audio", "pooling": "mean", "train_set": args.train_set,
                                    "probe": probe, "seed": seed} if audio else
                                   {"context": args.context, "pooling": args.pooling,
                                    "style": args.style, "probe": probe, "seed": seed})))
                    if split == "test" and seed == args.seeds[0]:
                        write_predictions_gz(out_dir / f"predictions_test_{task}way_{probe}.tsv.gz",
                                             utts, preds, task)
                append_records(records_path, recs)
                runs_info.append({"task": task, "probe": probe, "seed": seed, "layer": layer,
                                  **info, "time_s": round(time.time() - t, 1)})
                log.info("%d-way %-8s seed %d layer %d: dev %.2f (epoch %d), test %.2f", task,
                         probe, seed, layer, info["dev_accuracy"], info["best_epoch"],
                         recs[1]["metrics"]["legacy_2019"]["accuracy"])

    recs = read_records(records_path)
    records_to_csv(recs, out_dir / "summary.csv")
    lines = [f"# Stronger probes: {name} ({cond})", "",
             f"Seeds {args.seeds}; mean ± sd over seeds; test on the 2019 subset.", "",
             "| task | probe | layer | dev | test (2019 subset) | test macro-F1 | epochs |",
             "|---|---|---|---|---|---|---|"]
    summary = []
    for task in (2, 3):
        for probe in args.probes:
            rs = [r for r in recs if r["task"] == task and r["condition"]["probe"] == probe]
            dev_acc = [r["metrics"]["full"]["accuracy"] for r in rs if r["split"] == "dev"]
            test_acc = [r["metrics"]["legacy_2019"]["accuracy"] for r in rs if r["split"] == "test"]
            f1 = [r["metrics"]["full"]["macro_f1"] for r in rs if r["split"] == "test"]
            ep = [x["best_epoch"] for x in runs_info if x["task"] == task and x["probe"] == probe]
            sd = statistics.stdev(test_acc) if len(test_acc) > 1 else 0.0
            row = {"task": task, "probe": probe, "layer": layers[task],
                   "dev": statistics.mean(dev_acc), "test_2019": statistics.mean(test_acc),
                   "test_2019_sd": sd, "test_macro_f1": statistics.mean(f1), "epochs": ep}
            summary.append(row)
            lines.append(f"| {task}-way | {probe} | {layers[task]} | {row['dev']:.2f} | "
                         f"{row['test_2019']:.2f} ± {sd:.2f} | {row['test_macro_f1']:.1f} | {ep} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"experiment": "probing_nonlinear", "model": name, "model_id": model_id,
              "revision": revision, "kind": kind, "condition": cond, "layers": layers, **meta,
              "summary": summary, "runs": runs_info, "train_utterances": len(train),
              "time_s": {"extract": round(t1 - t0, 1), "total": round(time.time() - t0, 1)},
              "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "environment": environment_info()}
    if torch.cuda.is_available():
        result["environment"]["max_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    write_json(out_dir / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
