"""Overview of probing runs: best layers, confidence intervals, paired comparisons.

    python -m prosody_llms.probing.report results/probing --data-dir data

Reads every full run (``<model>/<context>-<pooling>-<style>/``, no training subset) and
writes, next to the runs:

* ``overview.md``: per run and task, the best layer (chosen on dev) with test accuracy and
  a 95% utterance-bootstrap interval on the 2019 test subset, macro-F1 and class-1 recall
  for LDA and ridge; paired differences (echo vs plain, every model vs frozen BERT, the
  best model vs the others); reference rows (fine-tuned BERT and the 2019 baselines, when
  their result files are present);
* ``overview.csv``: the same per-run numbers;
* ``layer_curves.csv``: dev and test accuracy for every model, context, task, probe and
  layer (for plots).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

from prosody_llms.data.corpus import read_split
from prosody_llms.data.download import default_data_dir
from prosody_llms.eval.io import read_records
from prosody_llms.eval.stats import bootstrap_accuracy

FULL_RUN = re.compile(r"^(plain|echo)-(first|last|mean)-(natural|spaced)$")
REFERENCE_MODEL = "bert-base-uncased"


def find_runs(root: Path) -> list[Path]:
    return sorted(d for d in root.glob("*/*") if d.is_dir() and FULL_RUN.match(d.name)
                  and (d / "result.json").exists())


def read_predictions_gz(path: Path, utts) -> list[list[str]]:
    by_id: dict[str, list[str]] = {}
    cur = None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("<file>\t"):
                cur = line.split("\t", 1)[1]
                by_id[cur] = []
            elif line:
                by_id[cur].append(line.split("\t")[2])
    return [by_id[u.utt_id] for u in utts]


def run_label(d: Path) -> str:
    return f"{d.parent.name} ({d.name.split('-')[0]})"


def summarise(runs: list[Path], test, n_resamples: int, seed: int):
    rows, preds = [], {}
    for d in runs:
        res = json.loads((d / "result.json").read_text(encoding="utf-8"))
        recs = read_records(d / "records.jsonl")
        best = {(s["task"], s["probe"]): s for s in res["summary"]}
        for task in (2, 3):
            gold = [u.labels(task) for u in test]
            p = read_predictions_gz(d / f"predictions_test_{task}way_lda_best.tsv.gz", test)
            preds[(d, task)] = p
            b = bootstrap_accuracy(gold, p, n_resamples=n_resamples, seed=seed)

            def recall(probe):
                layer = best[(task, probe)]["best_layer"]
                r = next(x for x in recs if x["task"] == task and x["split"] == "test"
                         and x["condition"]["probe"] == probe and x["layer"] == layer)
                return r["metrics"]["full"]["per_class"]["1"]["recall"]

            lda = best[(task, "lda")]
            rows.append({
                "model": res["model"], "context": res["condition"]["context"],
                "pooling": res["condition"]["pooling"], "kind": res["kind"],
                "layers": res["n_layers"] - 1, "task": task, "best_layer": lda["best_layer"],
                "dev": round(lda["dev"], 2), "test_2019": round(b["accuracy"][0], 2),
                "ci_low": round(b["ci"][0][0], 2), "ci_high": round(b["ci"][0][1], 2),
                "test_full": round(lda["test_full"], 2), "macro_f1": round(lda["test_macro_f1"], 2),
                "ridge_test_2019": round(best[(task, "ridge")]["test_2019"], 2),
                "class1_recall_lda": round(recall("lda"), 1),
                "class1_recall_ridge": round(recall("ridge"), 1), "run": str(d)})
    return rows, preds


def comparisons(runs, preds, test, rows, n_resamples, seed):
    by_name = {(d.parent.name, d.name.split("-")[0]): d for d in runs}
    pairs = []
    for (model, ctx), d in by_name.items():  # echo vs plain
        if ctx == "echo" and (model, "plain") in by_name:
            pairs.append((d, by_name[(model, "plain")]))
    ref = [d for (m, c), d in by_name.items() if m == REFERENCE_MODEL]
    out = []
    for task in (2, 3):
        task_rows = [r for r in rows if r["task"] == task]
        best = max(task_rows, key=lambda r: r["test_2019"])
        best_run = Path(best["run"])
        task_pairs = list(pairs)
        if ref:
            task_pairs += [(d, ref[0]) for d in runs if d != ref[0]]
        task_pairs += [(best_run, d) for d in runs if d != best_run]
        seen = set()
        gold = [u.labels(task) for u in test]
        for a, b in task_pairs:
            if (a, b) in seen:
                continue
            seen.add((a, b))
            diff = bootstrap_accuracy(gold, preds[(a, task)], preds[(b, task)],
                                      n_resamples=n_resamples, seed=seed)["difference"]
            out.append({"task": task, "a": run_label(a), "b": run_label(b),
                        "difference": round(diff["value"], 2), "ci_low": round(diff["ci"][0], 2),
                        "ci_high": round(diff["ci"][1], 2), "p_value": round(diff["p_value"], 3)})
    return out


def references(results_root: Path) -> list[str]:
    lines = []
    repro = results_root / "repro_2019" / "summary.csv"
    if repro.exists():
        with open(repro, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["train_set"] == "train_360" and float(r["fraction_of_train_data"]) == 1.0:
                    lines.append(f"| fine-tuned BERT-base, {r['seeds']} seeds | "
                                 f"{r['n_classes']}-way | {float(r['test_2019_mean']):.2f} "
                                 f"± {float(r['test_2019_sd']):.2f} |")
    base = results_root / "repro_2019" / "baselines.json"
    if base.exists():
        res = json.loads(base.read_text(encoding="utf-8"))["results"]
        for key in ("word_majority_2019", "majority"):
            for task in (2, 3):
                acc = res[f"{key}/{task}way"]["test"]["legacy_2019"]["accuracy"]
                lines.append(f"| {key.replace('_', ' ')} baseline | {task}-way | {acc:.2f} |")
    return lines


def layer_curves(runs: list[Path], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "context", "task", "probe", "layer", "dev", "test_2019", "test_full",
                    "macro_f1"])
        for d in runs:
            res = json.loads((d / "result.json").read_text(encoding="utf-8"))
            recs = read_records(d / "records.jsonl")
            dev = {(r["task"], r["condition"]["probe"], r["layer"]): r["metrics"]["full"]["accuracy"]
                   for r in recs if r["split"] == "dev"}
            for r in recs:
                if r["split"] != "test":
                    continue
                key = (r["task"], r["condition"]["probe"], r["layer"])
                m = r["metrics"]
                w.writerow([res["model"], res["condition"]["context"], *key, round(dev[key], 3),
                            round(m["legacy_2019"]["accuracy"], 3), round(m["full"]["accuracy"], 3),
                            round(m["full"]["macro_f1"], 3)])


SIZE_SERIES = {  # family -> [(model, parameters in billions)]
    "Qwen3.5 base (+ Qwen3.8-27B)": [("qwen3.5-0.8b-base", 0.8), ("qwen3.5-2b-base", 2),
                                     ("qwen3.5-4b-base", 4), ("qwen3.5-9b-base", 9),
                                     ("qwen3.8-27b", 27)],
    "OLMo 3 base": [("olmo-3-7b", 7), ("olmo-3-32b", 32)],
    "BERT uncased": [("bert-base-uncased", 0.11), ("bert-large-uncased", 0.34)],
    "ModernBERT": [("modernbert-base", 0.15), ("modernbert-large", 0.4)],
    "XLM-R": [("xlm-roberta-base", 0.28), ("xlm-roberta-large", 0.56)],
}


def size_section(rows) -> list[str]:
    best = {}
    for r in rows:  # decoders: echo if available; encoders: plain
        key = (r["model"], r["task"])
        if key not in best or r["context"] == "echo":
            best[key] = r
    lines = ["## Size series", "",
             "LDA, best layer on dev; decoders with echo input. Test accuracy on the 2019 subset.", "",
             "| family | model | parameters | 2-way | 3-way |", "|---|---|---|---|---|"]
    for family, members in SIZE_SERIES.items():
        for model, size in members:
            if (model, 2) in best and (model, 3) in best:
                lines.append(f"| {family} | {model} | {size:g}B | {best[(model, 2)]['test_2019']:.2f} | "
                             f"{best[(model, 3)]['test_2019']:.2f} |")
    return lines + [""]


def nonlinear_section(nl_root: Path, rows, test, n_resamples, seed) -> list[str]:
    """Stronger probes (mean ± sd over seeds) next to LDA of the same condition, and paired
    bootstrap differences of the seed-0 predictions against frozen BERT-base."""
    runs = sorted(nl_root.glob("*/*/result.json"))
    if not runs:
        return []
    lda = {(r["model"], r["context"], r["task"]): r for r in rows}
    lines = ["## Stronger probes", "",
             "Logistic regression, MLP (one hidden layer) and BiLSTM over the frozen word vectors "
             "of the best LDA layer (per task); mean ± sd over seeds; test on the 2019 subset.", ""]
    preds = {}
    for task in (2, 3):
        lines += [f"### {task}-way", "", "| model | input | layer | LDA | logistic | MLP | BiLSTM |",
                  "|---|---|---|---|---|---|---|"]
        for f in runs:
            res = json.loads(f.read_text(encoding="utf-8"))
            ctx = res["condition"].split("-")[0]
            summ = {x["probe"]: x for x in res["summary"] if x["task"] == task}
            base = lda.get((res["model"], ctx, task))
            cells = [f"{summ[p]['test_2019']:.2f} ± {summ[p]['test_2019_sd']:.2f}" if p in summ else "-"
                     for p in ("logistic", "mlp", "bilstm")]
            lines.append(f"| {res['model']} | {ctx} | {res['layers'][str(task)]} | "
                         f"{base['test_2019']:.2f} | " + " | ".join(cells) + " |" if base else "")
            for probe in ("mlp", "bilstm"):
                path = f.parent / f"predictions_test_{task}way_{probe}.tsv.gz"
                if path.exists():
                    preds[(res["model"], ctx, task, probe)] = read_predictions_gz(path, test)
        lines.append("")
        ref = preds.get((REFERENCE_MODEL, "plain", task, "bilstm"))
        if ref is not None:
            gold = [u.labels(task) for u in test]
            lines += ["Paired differences against frozen BERT-base + BiLSTM (seed 0):", "",
                      "| model | probe | difference [95% CI] | p |", "|---|---|---|---|"]
            for (model, ctx, t, probe), p in sorted(preds.items()):
                if t != task or (model, probe) == (REFERENCE_MODEL, "bilstm"):
                    continue
                d = bootstrap_accuracy(gold, p, ref, n_resamples=n_resamples, seed=seed)["difference"]
                lines.append(f"| {model} ({ctx}) | {probe} | {d['value']:+.2f} "
                             f"[{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] | {d['p_value']:.3f} |")
            lines.append("")
    return lines


def write_markdown(path: Path, rows, comps, refs, n_resamples, extra=()) -> None:
    lines = ["# Probing overview", "",
             "Frozen models, linear probes trained on all of `train_360`. Best layer chosen on dev "
             "(LDA probe); test accuracy on the 2019 test subset (4,821 utterances, 90,004 words) "
             f"with a 95% utterance-bootstrap interval ({n_resamples} resamples). Class-1 recall: "
             "the middle class (\"somewhat prominent\") in the 3-way task.", ""]
    for task in (2, 3):
        lines += [f"## {task}-way", "",
                  "| model | input | best layer | dev | test (2019 subset) [95% CI] | macro-F1 | "
                  "ridge test | class-1 recall LDA / ridge |", "|---|---|---|---|---|---|---|---|"]
        for r in sorted((r for r in rows if r["task"] == task), key=lambda r: -r["test_2019"]):
            lines.append(f"| {r['model']} | {r['context']} | {r['best_layer']} / {r['layers']} | "
                         f"{r['dev']:.2f} | {r['test_2019']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | "
                         f"{r['macro_f1']:.1f} | {r['ridge_test_2019']:.2f} | "
                         f"{r['class1_recall_lda']:.1f} / {r['class1_recall_ridge']:.1f} |")
        lines += ["", "Paired differences (a − b, utterance bootstrap):", "",
                  "| a | b | difference [95% CI] | p |", "|---|---|---|---|"]
        for c in (c for c in comps if c["task"] == task):
            lines.append(f"| {c['a']} | {c['b']} | {c['difference']:+.2f} "
                         f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}] | {c['p_value']:.3f} |")
        lines.append("")
    lines += list(extra)
    if refs:
        lines += ["## Reference rows", "", "| system | task | test (2019 subset) |", "|---|---|---|",
                  *refs, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="folder with <model>/<condition>/ runs")
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    runs = find_runs(args.root)
    if not runs:
        print(f"no full probing runs under {args.root}", file=sys.stderr)
        return 1
    test = read_split(args.data_dir / "test.txt")[:-1]  # the 2019 subset
    rows, preds = summarise(runs, test, args.resamples, args.seed)
    comps = comparisons(runs, preds, test, rows, args.resamples, args.seed)
    refs = references(args.root.parent)
    with open(args.root / "overview.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    layer_curves(runs, args.root / "layer_curves.csv")
    extra = size_section(rows) + nonlinear_section(args.root.parent / "probing_nonlinear", rows,
                                                   test, args.resamples, args.seed)
    write_markdown(args.root / "overview.md", rows, comps, refs, args.resamples, extra)
    print(f"{len(runs)} runs -> {args.root}/overview.md, overview.csv, layer_curves.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
