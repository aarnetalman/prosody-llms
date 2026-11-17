#!/usr/bin/env python3
"""Copy the small result files of finished runs from the runs folder into ``results/``.

    python scripts/collect_results.py                  # every experiment under $PROSODY_RUNS
    python scripts/collect_results.py probing icl      # selected experiments
    python scripts/collect_results.py --dry-run
    python scripts/collect_results.py --force          # also replace files already in results/

The analysis (``python -m prosody_llms.analysis``) reads only ``results/``. Copied are
``*.json``, ``*.jsonl``, ``*.csv``, ``*.md``, ``*.txt`` and ``*.gz`` files of at most 1 MB
from runs that have finished (their folder has a ``result.json``). Checkpoints, feature
caches, the prompt cache and model files stay in the runs folder. Summaries that are
rewritten as experiments progress (the prompting sweep, the audio comparison and the
reconstruction) are always replaced.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = (".json", ".jsonl", ".csv", ".md", ".txt", ".gz")
MAX_BYTES = 1_000_000
SKIP_DIRS = {"hf_cache", "logs", "smoke"}
ALWAYS = {
    "prompting": ["sweep_summary.md", "sweep_selection.json", "costs_summary.txt"],
    "audio": ["comparison/comparison.md", "comparison/comparison.csv", "comparison/common_words.json",
              "comparison/common_words.tsv.gz", "comparison/result.json", "reconstruction/summary.md",
              "reconstruction/result.json", "reconstruction/dev/result.json",
              "reconstruction/dev/records.jsonl", "reconstruction/test/result.json",
              "reconstruction/test/records.jsonl"],
}


def unfinished_dirs(src: Path) -> set[Path]:
    """Run folders that have files but no result.json yet (still running or interrupted)."""
    out = set()
    for d in src.rglob("*"):
        if d.is_dir() and not set(d.relative_to(src).parts) & SKIP_DIRS:
            if not (d / "result.json").exists() and any(f.is_file() for f in d.iterdir()):
                out.add(d)
    return out


def collect(exp: str, runs: Path, results: Path, force: bool, dry: bool) -> int:
    src, dst = runs / exp, results / exp
    if not src.is_dir():
        print(f"missing: {src}", file=sys.stderr)
        return 0
    copied = 0
    skip = unfinished_dirs(src)
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if not f.is_file() or set(rel.parts) & SKIP_DIRS or ".partial" in f.name:
            continue
        if not f.name.endswith(SUFFIXES) or f.stat().st_size > MAX_BYTES:
            continue
        if f.parent == src and f.name in ("summary.md", "summary.csv"):
            continue  # regenerated below for repro_2019
        if any(d == f.parent or d in f.parents for d in skip):
            continue
        target = dst / rel
        replace = force or str(rel) in ALWAYS.get(exp, [])
        if target.exists() and not replace:
            continue
        copied += 1
        if not dry:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
    print(f"{exp}: {copied} file(s) {'would be ' if dry else ''}copied")
    if exp == "repro_2019" and not dry and dst.is_dir():
        with open(dst / "summary.md", "w", encoding="utf-8") as out:
            subprocess.run([sys.executable, "-m", "prosody_llms.legacy.report", str(dst),
                            "--csv", str(dst / "summary.csv")], stdout=out, check=False)
    return copied


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="*")
    p.add_argument("--runs", type=Path, default=Path(os.environ.get("PROSODY_RUNS", "runs")))
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    exps = args.experiments or sorted(d.name for d in args.runs.iterdir()
                                      if d.is_dir() and d.name not in SKIP_DIRS)
    total = sum(collect(e, args.runs, args.results, args.force, args.dry_run) for e in exps)
    print(f"{total} file(s) in total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
