"""Where the analysis writes, and a CSV writer.

Everything goes under ``results/analysis/``: ``tables/`` holds the tables of the paper,
``figures/`` the data behind its figures, and ``numbers.csv`` every number quoted in the
text, by name.
"""

from __future__ import annotations

import csv
from pathlib import Path


def tables_dir(root: Path) -> Path:
    return root / "analysis" / "tables"


def figures_dir(root: Path) -> Path:
    return root / "analysis" / "figures"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def rnd(x, nd: int = 2):
    """Round for a CSV cell; empty for a missing value."""
    return "" if x is None or x == "" else round(float(x), nd)
