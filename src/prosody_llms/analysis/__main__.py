"""Run the whole analysis:

    python -m prosody_llms.analysis [--results results] [--data-dir data]

Reads only ``results/`` and the corpus; writes ``results/analysis/``: the tables of the
paper (``tables/``), the data behind its figures (``figures/``), every number quoted in
the text (``numbers.csv``) and the error analysis. Parts whose inputs do not exist are
skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prosody_llms.analysis import errors, readings, tables, visuals
from prosody_llms.data.download import default_data_dir


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.1f}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = p.parse_args(argv)
    extra = errors.write_all(args.results, args.data_dir)
    extra.update(readings.write_all(args.results, args.data_dir))
    vis = None
    if (args.results / "audio" / "comparison" / "comparison.csv").exists():
        vis = visuals.write_all(args.results, args.data_dir)
        extra.update(vis.pop("numbers"))
    info = tables.write_all(args.results, {k: fmt(v) for k, v in extra.items()})
    print(f"{info['rows']} result rows, {info['table_rows']} rows in the main table, "
          f"{info['numbers']} named numbers, figure data: {', '.join(info['figure_files'])}")
    if vis:
        print(f"visuals: {vis}")
    print(f"written: {args.results / 'analysis'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
