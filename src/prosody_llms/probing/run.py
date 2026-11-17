"""Probe every layer of a frozen model for prosodic prominence.

    python -m prosody_llms.probing.run --model qwen3.5-0.8b-base [--context echo] [--resume]

1. **Pass 1 (training data).** One forward pass over the training split. Per-layer
   statistics are accumulated on the GPU in float64 for the 3-way labels; the 2-way
   statistics follow from them exactly. The state is checkpointed every
   ``--checkpoint-minutes`` and the pass resumes from the last checkpoint.
2. **Solve.** Ridge and LDA probes for every layer and every regularisation value.
3. **Pass 2 (dev and test).** One forward pass. For every layer, probe family and task,
   the regularisation value is chosen on dev; test is scored with that choice only.

Outputs in ``<out-root>/<model>/<condition>/``: ``records.jsonl`` (one record per layer,
family, task and split with all metrics), ``summary.md``/``summary.csv``, the best layer's
dev and test predictions (``*.tsv.gz``), selected test predictions for every layer
(``predictions_layers.npz``) and ``result.json`` (configuration, environment, timings).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from prosody_llms.config import get_model
from prosody_llms.data.corpus import read_split
from prosody_llms.data.download import default_data_dir
from prosody_llms.data.subsets import fixed_subset
from prosody_llms.eval.io import append_records, evaluate, make_record, records_to_csv
from prosody_llms.features.alignment import CONTEXTS, STYLES
from prosody_llms.features.extractor import HFTextExtractor, length_batches
from prosody_llms.features.pooling import POOLINGS
from prosody_llms.probes.base import LabelSpace
from prosody_llms.probes.linear import FAMILIES, LinearStats, solve
from prosody_llms.utils.runinfo import environment_info, write_json
from prosody_llms.utils.seed import set_seed

log = logging.getLogger("prosody_llms.probing")
TASKS = (2, 3)
EXIT_INTERRUPTED = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="registry name (configs/models.toml) or a path")
    p.add_argument("--revision", default=None, help="only for models outside the registry")
    p.add_argument("--context", choices=CONTEXTS, default="plain")
    p.add_argument("--pooling", choices=POOLINGS, default=None,
                   help="default: first for encoders, last for decoders")
    p.add_argument("--style", choices=STYLES, default="natural")
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--train-set", choices=["train_100", "train_360"], default=None,
                   help="default: train_360 for text models, train_100 for speech models")
    p.add_argument("--audio-dir", type=Path, default=None,
                   help="speech models: LibriTTS audio and alignments (default $PROSODY_AUDIO)")
    p.add_argument("--kind", choices=["speech", "speech_text"], default=None,
                   help="only for local model paths outside the registry")
    p.add_argument("--part", choices=["llm", "encoder"], default="llm",
                   help="speech-text models: the language model at the audio positions, or the audio encoder")
    p.add_argument("--train-exemplars", type=int, default=None,
                   help="train only on the K utterances used as prompt examples (as in prosody_llms.prompting.local)")
    p.add_argument("--train-utterances", type=int, default=None,
                   help="use a fixed subset of this many training utterances")
    p.add_argument("--out-root", type=Path,
                   default=Path(os.environ.get("PROSODY_RUNS", "runs")) / "probing")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=None)
    p.add_argument("--max-tokens", type=int, default=4096, help="padded tokens per batch")
    p.add_argument("--stats-device", choices=["auto", "gpu", "cpu"], default="auto",
                   help="where the pass-1 statistics live; auto puts them in CPU memory when "
                        "they would take more than half of the free GPU memory")
    p.add_argument("--checkpoint-minutes", type=float, default=30.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-stats", action="store_true",
                   help="keep the pass-1 statistics (large) after the run")
    p.add_argument("--max-batches", type=int, default=None,
                   help="stop pass 1 after this many batches (testing resume)")
    return p


AUDIO_KINDS = ("speech", "speech_text")


def condition_name(args, kind: str) -> str:
    if kind in AUDIO_KINDS:
        name = "audio-mean" + (f"-{args.part}" if kind == "speech_text" else "")
        return name + ("" if args.train_set == "train_100" else f"-{args.train_set}") + \
            (f"-n{args.train_utterances}" if args.train_utterances else "")
    name = f"{args.context}-{args.pooling}-{args.style}"
    if args.train_set != "train_360":
        name += f"-{args.train_set}"
    if args.train_utterances:
        name += f"-n{args.train_utterances}"
    if getattr(args, "train_exemplars", None):
        name += f"-ex{args.train_exemplars}"
    return name


def _labels(utts, n_classes=3):
    space = LabelSpace(n_classes)
    return [space.encode(u) for u in utts]


def stats_device(ext, choice: str):
    """GPU unless the statistics (float64 Gram matrices for every layer) would take more
    than half of the GPU memory left after loading the model."""
    import torch

    if ext.device.type != "cuda" or choice == "cpu":
        return torch.device("cpu")
    if choice == "gpu":
        return ext.device
    need = ext.n_layers * (ext.dim * ext.dim + 5 * ext.dim) * 8
    free, _ = torch.cuda.mem_get_info(ext.device)
    device = ext.device if need < 0.5 * free else torch.device("cpu")
    log.info("statistics need %.1f GB, %.1f GB GPU memory free -> %s", need / 1e9, free / 1e9,
             device)
    return device


def pass_one(ext, train, out_dir: Path, args) -> LinearStats | None:
    """Accumulate training statistics; returns None if stopped by ``--max-batches``."""
    import torch

    state_path = out_dir / "pass1_state.pt"
    args.stats_device_used = str(stats_device(ext, args.stats_device))
    stats = LinearStats(ext.n_layers, ext.dim, 3, device=args.stats_device_used)
    labels = _labels(train)
    if hasattr(ext, "word_mask"):  # speech: words without a timing are not trained on
        for lab, u in zip(labels, train):
            lab[~np.array(ext.word_mask(u))] = -1
    aligned = [ext.align(u) for u in train]
    batches = length_batches([len(a.input_ids) for a in aligned], args.max_tokens)
    start = 0
    if state_path.exists():
        if not args.resume:
            raise SystemExit(f"{state_path} exists; pass --resume to continue it")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["n_batches"] != len(batches):
            raise SystemExit("batching differs from the checkpoint (different --max-tokens?)")
        stats.load_state_dict(state["stats"])
        start = state["next_batch"]
        log.info("resumed pass 1 at batch %d/%d", start, len(batches))
        if state.get("done"):
            return stats

    def save(next_batch: int, done: bool = False) -> None:
        tmp = state_path.with_suffix(".part")
        torch.save({"stats": stats.state_dict(), "next_batch": next_batch,
                    "n_batches": len(batches), "done": done}, tmp)
        tmp.replace(state_path)

    t0 = last_save = time.time()
    saved_any = start > 0
    for b in range(start, len(batches)):
        if args.max_batches is not None and b - start >= args.max_batches:
            save(b)
            log.info("stopped after %d batches (--max-batches)", args.max_batches)
            return None
        idx = batches[b]
        for i, h in zip(idx, ext.forward([aligned[i] for i in idx])):
            stats.update(h, labels[i])
        if time.time() - last_save > args.checkpoint_minutes * 60:
            save(b + 1)
            last_save, saved_any = time.time(), True
        if b % 50 == 0 or b + 1 == len(batches):
            log.info("pass 1 batch %d/%d, %d words (%.0fs)", b + 1, len(batches), stats.n,
                     time.time() - t0)
    # The final state is only written when a run was long enough to checkpoint at all (or
    # with --keep-stats): it lets a run that fails later restart without redoing pass 1,
    # but for large models it takes several GB of quota.
    if saved_any or args.keep_stats:
        save(len(batches), done=True)
    return stats


def pass_two(ext, splits: dict, grids: dict, max_tokens: int) -> dict:
    """Predictions for every split, task, family, layer and grid value:
    ``{split: {(task, family): int8 array (layers, grid, tokens)}}``."""
    import torch

    out = {}
    for split, utts in splits.items():
        offsets = np.cumsum([0] + [len(u.tokens) for u in utts])
        n_tokens = int(offsets[-1])
        preds = {key: np.zeros((ext.n_layers, len(g.grid), n_tokens), dtype=np.int8)
                 for key, g in grids.items()}
        aligned = [ext.align(u) for u in utts]
        for b, idx in enumerate(length_batches([len(a.input_ids) for a in aligned],
                                               max_tokens)):
            hs = ext.forward([aligned[i] for i in idx])
            for layer in range(ext.n_layers):
                x = torch.cat([h[layer] for h in hs])
                for key, g in grids.items():
                    p = g.predict_layer(layer, x).to(torch.int8).cpu().numpy()
                    col = 0
                    for i, h in zip(idx, hs):
                        w = h.shape[1]
                        preds[key][layer, :, offsets[i]:offsets[i] + w] = p[:, col:col + w]
                        col += w
            if b % 50 == 0:
                log.info("pass 2 %s batch %d", split, b + 1)
        out[split] = (preds, offsets)
    return out


def split_predictions(flat: np.ndarray, offsets, utts, task: int) -> list[list[str]]:
    space = LabelSpace(task)
    return [space.decode(flat[offsets[i]:offsets[i + 1]].tolist(), u) for i, u in enumerate(utts)]


def write_predictions_gz(path: Path, utts, preds, task: int) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for u, ps in zip(utts, preds):
            f.write(f"<file>\t{u.utt_id}\n")
            for w, g, p in zip(u.words, u.labels(task), ps):
                f.write(f"{w}\t{g}\t{p}\n")
            f.write("\n")


def run(args) -> dict:
    import torch

    set_seed(0)
    spec = None
    try:
        spec = get_model(args.model)
        model_id, revision, kind = spec.id, spec.revision, spec.kind
    except KeyError:
        if not Path(args.model).exists():
            raise
        model_id, revision, kind = args.model, args.revision, args.kind or "custom"
    audio = kind in AUDIO_KINDS
    if args.train_set is None:
        args.train_set = "train_100" if audio else "train_360"
    if args.pooling is None:
        args.pooling = "mean" if audio else "last" if kind == "decoder" else "first"
    name = spec.name if spec else Path(args.model).name
    out_dir = args.out_root / name / condition_name(args, kind)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output: %s", out_dir)
    if (out_dir / "result.json").exists():
        log.info("already finished (result.json exists); nothing to do")
        return json.loads((out_dir / "result.json").read_text(encoding="utf-8"))

    train = read_split(args.data_dir / f"{args.train_set}.txt")
    if args.train_utterances:
        train = fixed_subset(train, args.train_utterances, seed=0)
    if args.train_exemplars:  # exactly the training utterances used as prompt examples
        from prosody_llms.prompting.prompts import select_exemplars

        train = select_exemplars(train, args.train_exemplars)
    dev = read_split(args.data_dir / "dev.txt")
    test = read_split(args.data_dir / "test.txt")

    t0 = time.time()
    if audio:
        from prosody_llms.audio.download import default_audio_dir
        from prosody_llms.audio.extractors import make_audio_extractor

        from types import SimpleNamespace

        ext = make_audio_extractor(spec or SimpleNamespace(id=model_id, revision=revision, kind=kind,
                                                           name=Path(model_id).name),
                                   args.audio_dir or default_audio_dir(), args.dtype,
                                   args.device, part=args.part)
        ext.register({args.train_set: train, "dev": dev, "test": test})
    else:
        ext = HFTextExtractor(model_id, revision, style=args.style, pooling=args.pooling,
                              context=args.context, device=args.device, dtype=args.dtype,
                              name=name)
        if args.context.startswith("prev"):
            ext.register_discourse([read_split(args.data_dir / f"{s}.txt") for s in (args.train_set, "dev", "test")])
    log.info("%s (%s): %d layers x %d", name, ext.model_class, ext.n_layers, ext.dim)

    stats = pass_one(ext, train, out_dir, args)
    if stats is None:
        return {"interrupted": True}
    t1 = time.time()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # leave room for the solver's workspace
    grids = {}
    for task in TASKS:
        st = stats.collapse(task)
        for family in args.families:
            grids[(task, family)] = solve(st, family, device=ext.device)
    t2 = time.time()

    splits = {"dev": dev, "test": test}
    preds = pass_two(ext, splits, grids, args.max_tokens)
    masks = None
    if hasattr(ext, "word_mask"):
        # Words without a timing have no audio features: they count as "0" (not
        # prominent), as unparsed words in prompting; "aligned" metrics leave them out.
        masks = {split: [ext.word_mask(u) for u in utts] for split, utts in splits.items()}
        for split, (flat, _) in preds.items():
            keep = np.concatenate([np.array(m, dtype=bool) for m in masks[split]])
            for arr in flat.values():
                arr[:, :, ~keep] = 0
    t3 = time.time()

    records_path = out_dir / "records.jsonl"
    records_path.unlink(missing_ok=True)
    condition = {"context": args.context, "pooling": args.pooling, "style": args.style,
                 "train_set": args.train_set, "train_utterances": args.train_utterances}
    if audio:
        condition = {"input": "audio", "pooling": "mean", "train_set": args.train_set,
                     "train_utterances": args.train_utterances}
        if kind == "speech_text":
            condition["part"] = args.part

    def scored(utts, split, labels, task):
        metrics = evaluate(utts, labels, task)
        if masks is not None:
            from prosody_llms.audio.reconstruct import mask_utterances

            metrics["aligned"] = evaluate(mask_utterances(utts, masks[split]), labels, task)["full"]
        return metrics
    dev_gold = np.concatenate(_labels(dev, 3))
    selected = {}
    summary_rows = []
    for (task, family), grid in grids.items():
        gold = dev_gold if task == 3 else np.where(dev_gold == 2, 1, dev_gold)
        mask = gold >= 0
        dev_flat, dev_off = preds["dev"][0][(task, family)], preds["dev"][1]
        test_flat, test_off = preds["test"][0][(task, family)], preds["test"][1]
        dev_acc = (dev_flat[:, :, mask] == gold[mask]).mean(axis=2) * 100  # (layers, grid)
        best_g = dev_acc.argmax(axis=1)
        selected[(task, family)] = np.stack([test_flat[l, g] for l, g in enumerate(best_g)])
        recs = []
        for layer, g in enumerate(best_g):
            cond = {**condition, "probe": family, "hyper": grid.grid[g]}
            for split, utts, flat, off in (("dev", dev, dev_flat, dev_off),
                                           ("test", test, test_flat, test_off)):
                metrics = scored(utts, split, split_predictions(flat[layer, g], off, utts, task), task)
                recs.append(make_record(experiment="probing", model=name, task=task,
                                        split=split, layer=layer, condition=cond,
                                        metrics=metrics))
        append_records(records_path, recs)
        best_layer = int(dev_acc.max(axis=1).argmax())
        g = int(best_g[best_layer])
        for split, utts, flat, off in (("dev", dev, dev_flat, dev_off),
                                       ("test", test, test_flat, test_off)):
            write_predictions_gz(out_dir / f"predictions_{split}_{task}way_{family}_best.tsv.gz",
                                 utts, split_predictions(flat[best_layer, g], off, utts, task), task)
        test_rec = next(r for r in recs if r["layer"] == best_layer and r["split"] == "test")
        summary_rows.append({
            "task": task, "probe": family, "best_layer": best_layer, "layers": ext.n_layers,
            "hyper": grid.grid[g], "dev": float(dev_acc[best_layer, g]),
            "test_2019": test_rec["metrics"]["legacy_2019"]["accuracy"],
            "test_full": test_rec["metrics"]["full"]["accuracy"],
            "test_macro_f1": test_rec["metrics"]["full"]["macro_f1"],
            "test_aligned": test_rec["metrics"].get("aligned", {}).get("accuracy")})
    np.savez_compressed(out_dir / "predictions_layers.npz",
                        **{f"{t}way_{f}": a for (t, f), a in selected.items()},
                        offsets=preds["test"][1])

    from prosody_llms.eval.io import read_records

    records_to_csv(read_records(records_path), out_dir / "summary.csv")
    lines = [f"# Probing {name} ({condition_name(args, kind)})", "",
             "Best layer chosen on dev; test accuracy on the 2019 subset and the full file.", "",
             "| task | probe | best layer | dev | test (2019 subset) | test (full) | test macro-F1 |",
             "|---|---|---|---|---|---|---|"]
    for r in summary_rows:
        lines.append(f"| {r['task']}-way | {r['probe']} | {r['best_layer']} / {r['layers'] - 1} | "
                     f"{r['dev']:.2f} | {r['test_2019']:.2f} | {r['test_full']:.2f} | "
                     f"{r['test_macro_f1']:.2f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "experiment": "probing", "model": name, "model_id": model_id, "revision": revision,
        "model_class": ext.model_class, "key_mapping": ext.key_mapping, "kind": kind, "n_layers": ext.n_layers, "dim": ext.dim,
        "condition": condition, "families": args.families,
        "grids": {f: list(grids[(3, f)].grid) for f in args.families},
        "train_words": stats.n, "train_utterances": len(train), "summary": summary_rows,
        "alignment": ext.alignment_stats,
        "oom_splits": getattr(ext, "oom_splits", 0),
        "time_s": {"load_and_pass1": round(t1 - t0, 1), "solve": round(t2 - t1, 1),
                   "pass2": round(t3 - t2, 1), "total": round(time.time() - t0, 1)},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "environment": environment_info(),
    }
    if torch.cuda.is_available():
        result["environment"]["max_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    write_json(out_dir / "result.json", result)
    if not args.keep_stats:
        (out_dir / "pass1_state.pt").unlink(missing_ok=True)
    for r in summary_rows:
        log.info("%d-way %-5s best layer %d: dev %.2f, test %.2f (2019 subset)", r["task"],
                 r["probe"], r["best_layer"], r["dev"], r["test_2019"])
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args(argv)
    result = run(args)
    return EXIT_INTERRUPTED if result.get("interrupted") else 0


if __name__ == "__main__":
    sys.exit(main())
