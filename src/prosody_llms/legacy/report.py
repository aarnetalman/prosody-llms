"""Aggregate 2019-reproduction runs into a table (mean, sd, min, max over seeds).

Reads every ``result.json`` under the given run directories, groups by configuration
(everything except the seed and paths) and writes Markdown to stdout and a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

PAPER_2019 = {  # Table 3 (train_360, full training data) and the 10% learning-curve points
    ("train_360", 1.0, 2): 83.2, ("train_360", 1.0, 3): 68.6,
    ("train_360", 0.1, 2): 82.6, ("train_360", 0.1, 3): 67.1,
}
GROUP_KEYS = ("n_classes", "train_set", "fraction_of_train_data", "protocol", "attention_mask",
              "attn_implementation", "epochs", "batch_size", "learning_rate")
# Runs made before --attn-implementation existed used the transformers default (sdpa).
CONFIG_DEFAULTS = {"attn_implementation": "sdpa"}


def collect(dirs: list[Path]) -> list[dict]:
    rows = []
    for d in dirs:
        for path in sorted(d.rglob("result.json")):
            r = json.loads(path.read_text(encoding="utf-8"))
            if r.get("experiment") != "repro_2019":
                continue
            cfg = r["config"]
            rows.append({**{k: cfg.get(k, CONFIG_DEFAULTS.get(k)) for k in GROUP_KEYS},
                         "seed": cfg["seed"],
                         "test_full": r["test"]["full"]["accuracy"],
                         "test_2019": r["test"]["legacy_2019"]["accuracy"],
                         "dev_full": r["dev"]["full"]["accuracy"],
                         "device": r["environment"].get("device"), "path": str(path)})
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in GROUP_KEYS), []).append(r)
    out = []
    for key, rs in sorted(groups.items()):
        cfg = dict(zip(GROUP_KEYS, key))
        row = {**cfg, "seeds": len(rs), "paper_2019": PAPER_2019.get(
            (cfg["train_set"], cfg["fraction_of_train_data"], cfg["n_classes"]))}
        for m in ("test_2019", "test_full", "dev_full"):
            vals = [r[m] for r in rs]
            row[f"{m}_mean"] = statistics.mean(vals)
            row[f"{m}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            row[f"{m}_min"], row[f"{m}_max"] = min(vals), max(vals)
        out.append(row)
    return out


def to_markdown(summary: list[dict]) -> str:
    head = ("| task | train | fraction | protocol | seeds | test (2019 subset) mean ± sd [min, max] "
            "| test (full) mean ± sd | dev mean | 2019 paper |")
    lines = [head, "|" + "---|" * 9]
    for r in summary:
        paper = f"{r['paper_2019']:.1f}" if r["paper_2019"] is not None else "-"
        lines.append(
            f"| {r['n_classes']}-way | {r['train_set']} | {r['fraction_of_train_data']:g} | "
            f"{r['protocol']}{' +mask' if r['attention_mask'] else ''}, "
            f"{r['attn_implementation']} | {r['seeds']} | "
            f"{r['test_2019_mean']:.2f} ± {r['test_2019_sd']:.2f} "
            f"[{r['test_2019_min']:.2f}, {r['test_2019_max']:.2f}] | "
            f"{r['test_full_mean']:.2f} ± {r['test_full_sd']:.2f} | {r['dev_full_mean']:.2f} | {paper} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args(argv)
    summary = summarise(collect(args.run_dirs))
    if not summary:
        print("no repro_2019 result.json files found", file=sys.stderr)
        return 1
    print(to_markdown(summary))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
