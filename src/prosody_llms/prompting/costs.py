"""Spending so far, from the cost log.

    python -m prosody_llms.prompting.costs
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from prosody_llms.prompting.cache import CostLog, default_root
from prosody_llms.prompting.run import load_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=default_root())
    args = p.parse_args(argv)
    entries = CostLog(args.out_root / "costs.jsonl").entries()
    by_run: dict[tuple, list] = collections.defaultdict(lambda: [0, 0, 0, 0.0])
    for e in entries:
        row = by_run[(e.get("model"), e.get("run"))]
        row[0] += 1
        row[1] += int(e.get("input_tokens", 0))
        row[2] += int(e.get("output_tokens", 0))
        row[3] += float(e.get("cost_usd", 0.0))
    print(f"{'model':24} {'run':50} {'calls':>7} {'in tok':>10} {'out tok':>9} {'USD':>8}")
    for (model, run), (n, i, o, c) in sorted(by_run.items()):
        print(f"{model:24} {run:50} {n:7d} {i:10d} {o:9d} {c:8.2f}")
    total = sum(r[3] for r in by_run.values())
    cap = load_config()["budget"]["total_usd"]
    print(f"total: ${total:.2f} of ${cap:.2f} ({len(entries)} calls). Estimated from list prices; "
          "check against your provider's billing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
