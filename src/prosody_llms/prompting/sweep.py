"""The whole prompting sweep in one command, stage by stage.

    python -m prosody_llms.prompting.sweep                                  # plan and cost estimate
    python -m prosody_llms.prompting.sweep --confirm-spend --max-usd 165    # run everything
    python -m prosody_llms.prompting.sweep --confirm-spend --max-usd 165 --until stage1

Stages (each runs for every model before the next stage starts, so the cheap stages finish
for all models first):

1. ``check``: one item per model, to test the connection (a few cents).
2. ``stage1``: 3 instruction variants x 3 formats, 3-way, 0 shots, on the dev subset.
3. ``shots``: the best stage-1 setting with 1, 5 and 10 shots.
4. ``twoway``: the best setting so far (variant, format, shots) as a native 2-way prompt.
5. ``test``: the best 3-way setting on the full test set.
6. ``reasoning``: GPT-5.5 on 200 test utterances with GPT-4o's selected setting.

"Best" is the highest 3-way word accuracy on the dev subset; ties go to the earlier
setting (variants and formats in their listed order, fewer shots first). Selections are
recomputed from the result files every time, so the sweep can be stopped and restarted:
cached answers are never paid for twice. A stage whose results are incomplete (a budget
stop, skipped failures) stops the sweep before anything is selected from it; rerunning the
same command sends only what is missing.

Spending: without ``--confirm-spend`` nothing is sent. ``--max-usd`` is the budget for this
invocation of the whole sweep; every stage also gets its own budget (its worst-case
estimate plus 10%, within what is left), and the project cap in configs/prompting.toml
still applies to every request.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from prosody_llms.data.corpus import read_split
from prosody_llms.data.download import default_data_dir
from prosody_llms.prompting import run as prun
from prosody_llms.prompting.cache import CostLog, ResponseCache, default_root
from prosody_llms.prompting.prompts import FORMATS, VARIANTS, PromptSpec

log = logging.getLogger("prosody_llms.prompting.sweep")

MODELS = ("gpt-4o", "deepseek-v4-pro", "mistral-large-3")
REASONING_MODEL = "gpt-5.5"
REASONING_FOLLOWS = "gpt-4o"  # GPT-5.5 uses the setting selected for GPT-4o
STAGES = ("check", "stage1", "shots", "twoway", "test", "reasoning")
REASONING_EXTRA_TOKENS = 1000  # expected hidden reasoning tokens per request (estimate only)


class SweepIncomplete(RuntimeError):
    """A stage has missing answers, so nothing can be selected from it yet."""


@dataclass
class Choice:
    variant: str
    fmt: str
    shots: int = 0

    def spec(self, task: int = 3) -> PromptSpec:
        return PromptSpec(self.variant, self.fmt, task, self.shots)


# ----------------------------------------------------------------------------- selection


def stage1_specs() -> list[PromptSpec]:
    return [PromptSpec(v, f, 3, 0) for v in VARIANTS for f in FORMATS]


def read_result(root: Path, model: str, split: str, spec: PromptSpec, n_utts: int) -> dict:
    path = prun.run_dir(root, model, split, spec) / "result.json"
    if not path.exists():
        raise SweepIncomplete(f"{model} {split} {spec.name}: no result yet")
    res = json.loads(path.read_text(encoding="utf-8"))
    cov = res["coverage"]
    if cov["utterances"] != n_utts or cov["missing_responses"]:
        raise SweepIncomplete(f"{model} {split} {spec.name}: {cov['missing_responses']} of "
                              f"{cov['utterances']} answers missing (expected {n_utts} utterances); "
                              "rerun the same command to send only what is missing")
    return res


def accuracy3(res: dict) -> float:
    acc = res["accuracy"]
    return float(acc.get("3", acc.get(3)))


def best(root: Path, model: str, split: str, specs: list[PromptSpec], n_utts: int) -> PromptSpec:
    """The spec with the highest 3-way accuracy; the first one on ties."""
    scored = [(accuracy3(read_result(root, model, split, s, n_utts)), -i, s) for i, s in enumerate(specs)]
    return max(scored, key=lambda t: (t[0], t[1]))[2]


def select_setting(root: Path, model: str, split: str, n_utts: int) -> Choice:
    """Best stage-1 setting, then the best number of shots for it (0 is stage 1)."""
    s1 = best(root, model, split, stage1_specs(), n_utts)
    shots = [PromptSpec(s1.variant, s1.fmt, 3, k) for k in (0, *prun.SHOTS_GRID)]
    s2 = best(root, model, split, shots, n_utts)
    return Choice(s2.variant, s2.fmt, s2.shots)


# ----------------------------------------------------------------------------- stages


@dataclass
class Step:
    stage: str
    model: str
    argv: list[str]
    candidates: list[PromptSpec]  # for the cost estimate when the setting is not known yet
    split: str
    groups: list[list[PromptSpec]] | None = None  # alternatives priced as a whole (unknown setting)


def steps_for(stage: str, models: list[str], root: Path, args, n_dev: int) -> list[Step]:
    dev, test = args.dev_split, args.test_split
    out = []
    worst_cases = [PromptSpec(v, f, 3, max(prun.SHOTS_GRID)) for v in VARIANTS for f in FORMATS]

    def known(model):
        try:
            return select_setting(root, model, dev, n_dev)
        except SweepIncomplete:
            return None

    for model in models:
        if stage == "check":
            out.append(Step(stage, model, ["--split", dev, "--max-examples", "1"],
                            [PromptSpec()], dev))
        elif stage == "stage1":
            out.append(Step(stage, model, ["--split", dev, "--grid", "stage1"], stage1_specs(), dev))
        elif stage == "shots":
            try:
                s1 = best(root, model, dev, stage1_specs(), n_dev)
                cands = [PromptSpec(s1.variant, s1.fmt, 3, k) for k in prun.SHOTS_GRID]
                argv = ["--variant", s1.variant, "--format", s1.fmt]
            except SweepIncomplete:
                groups = [[PromptSpec(v, f, 3, k) for k in prun.SHOTS_GRID] for v in VARIANTS for f in FORMATS]
                out.append(Step(stage, model, None, [s for g in groups for s in g], dev, groups))
                continue
            out.append(Step(stage, model, ["--split", dev, "--grid", "shots", *argv], cands, dev))
        elif stage in ("twoway", "test"):
            c = known(model)
            task, split, grid = (2, dev, ["--grid", "twoway"]) if stage == "twoway" else (3, test, [])
            if c:
                argv = ["--split", split, *grid, "--variant", c.variant, "--format", c.fmt,
                        "--shots", str(c.shots)]
                cands = [c.spec(task)]
            else:
                argv, cands = None, [PromptSpec(s.variant, s.fmt, task, s.shots) for s in worst_cases]
            out.append(Step(stage, model, argv, cands, split))
    if stage == "reasoning" and args.reasoning_model not in args.skip:
        c = known(args.reasoning_follows) if args.reasoning_follows in models else None
        argv = (["--split", args.reasoning_split, "--variant", c.variant, "--format", c.fmt,
                 "--shots", str(c.shots)] if c else None)
        out.append(Step(stage, args.reasoning_model, argv, [c.spec()] if c else worst_cases,
                        args.reasoning_split))
    return out


def estimate(step: Step, utts, train, cache, cfg) -> dict:
    """Cost of a step; when its setting is not known yet, of its most expensive candidate."""
    m, _ = prun.model_entry(step.model, cfg)
    reasoning = m.get("reasoning") == "always"
    if step.stage == "check":
        utts = utts[:1]
    groups = [step.candidates] if step.argv else (step.groups or [[c] for c in step.candidates])
    worst = None
    for specs in groups:
        p = prun.plan(step.model, prun.build_requests(step.model, utts, specs, train, cfg), cache, cfg)
        exp = (prun.cost_usd(m, p["est_input_tokens"],
                             p["est_output_tokens"] + REASONING_EXTRA_TOKENS * p["to_send"])
               if reasoning else p["est_usd_expected"])
        row = {"requests": p["requests"], "to_send": p["to_send"], "expected": round(exp, 2),
               "upper": p["est_usd_upper_bound"]}
        if worst is None or row["upper"] > worst["upper"]:
            worst = row
    worst["setting"] = "selected" if step.argv else "not chosen yet (most expensive candidate)"
    return worst


# ----------------------------------------------------------------------------- summary


def write_summary(root: Path, models: list[str], args, n_dev: int, n_test: int, n_reason: int) -> str:
    rows, sel = [], {}
    for model in dict.fromkeys([*models, args.reasoning_model]):
        for split, n in ((args.dev_split, n_dev), (args.test_split, n_test), (args.reasoning_split, n_reason)):
            d = root / model / split
            if not d.is_dir():
                continue
            for r in sorted(d.glob("*/result.json")):
                res = json.loads(r.read_text(encoding="utf-8"))
                acc, cov = res["accuracy"], res["coverage"]
                if cov["utterances"] != n:
                    continue  # e.g. the one-item connection check
                rows.append((model, split, r.parent.name, acc.get("3"), acc.get("2"),
                             res.get("auc_prominent"), cov.get("word_coverage"), cov["missing_responses"]))
        if model in models:
            try:
                sel[model] = select_setting(root, model, args.dev_split, n_dev).__dict__
            except SweepIncomplete:
                pass

    def f(x, nd=2):
        return "" if x is None else f"{x:.{nd}f}"

    lines = ["# Prompting sweep", "",
             "Accuracy on scored words (punctuation excluded). 2-way from a 3-way prompt is derived; "
             "`2way` prompts are native. Selection: best 3-way accuracy on the dev subset.", "",
             "## Selected settings", ""]
    lines += [f"- {m}: {c['variant']} / {c['fmt']} / {c['shots']} shots" for m, c in sel.items()] or ["- none yet"]
    lines += ["", "## Results", "", "| model | split | prompt | 3-way | 2-way | AUC | coverage | missing |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {m} | {s} | {p} | {f(a3)} | {f(a2)} | {f(auc, 3)} | {f(c, 4)} | {miss} |"
              for m, s, p, a3, a2, auc, c, miss in rows]
    total = CostLog(root / "costs.jsonl").total()
    lines += ["", f"Estimated spending so far: ${total:.2f} (list prices; check your provider's billing).", ""]
    text = "\n".join(lines)
    (root / "sweep_summary.md").write_text(text, encoding="utf-8")
    (root / "sweep_selection.json").write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")
    return text


# ----------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--skip", nargs="*", default=[], help="models to leave out (e.g. gpt-5.5)")
    p.add_argument("--until", choices=STAGES, default=STAGES[-1], help="last stage to run")
    p.add_argument("--confirm-spend", action="store_true")
    p.add_argument("--max-usd", type=float, default=None, help="budget for this whole invocation")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--dev-split", default="dev500")
    p.add_argument("--test-split", default="test")
    p.add_argument("--reasoning-split", default="test200")
    p.add_argument("--reasoning-model", default=REASONING_MODEL)
    p.add_argument("--reasoning-follows", default=REASONING_FOLLOWS,
                   help="model whose selected setting the reasoning model uses")
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--out-root", type=Path, default=default_root())
    return p


def sweep(args, client_for=None) -> dict:
    """Runs (or, without --confirm-spend, plans) the stages up to --until. ``client_for``
    maps a model name to a client (tests); by default the runner makes the real client."""
    cfg = prun.load_config()
    models = [m for m in args.models if m not in args.skip]
    root = args.out_root
    stages = STAGES[: STAGES.index(args.until) + 1]
    splits = {s: prun.load_split(args.data_dir, s)
              for s in (args.dev_split, args.test_split, args.reasoning_split)}
    n_dev, n_test, n_reason = (len(splits[args.dev_split]), len(splits[args.test_split]),
                               len(splits[args.reasoning_split]))
    train = read_split(args.data_dir / "train_360.txt")
    cache = ResponseCache(root / "cache.sqlite")
    costs = CostLog(root / "costs.jsonl")
    cap = float(cfg["budget"]["total_usd"])

    if not args.confirm_spend:
        rows, exp, upper = [], 0.0, 0.0
        for stage in stages:
            for st in steps_for(stage, models, root, args, n_dev):
                e = estimate(st, splits[st.split], train, cache, cfg)
                rows.append({"stage": stage, "model": st.model, "split": st.split, **e})
                exp += e["expected"]
                upper += e["upper"]
        print(f"{'stage':10} {'model':24} {'split':8} {'requests':>8} {'to send':>8} "
              f"{'expected':>9} {'upper':>9}  setting")
        for r in rows:
            print(f"{r['stage']:10} {r['model']:24} {r['split']:8} {r['requests']:8d} {r['to_send']:8d} "
                  f"{r['expected']:9.2f} {r['upper']:9.2f}  {r['setting']}")
        spent = costs.total()
        print(f"TOTAL to send: expected ${exp:.2f}, at most ${upper:.2f}. Spent so far ${spent:.2f} "
              f"of the ${cap:.2f} project cap. DRY RUN: nothing was sent.")
        return {"dry_run": True, "rows": rows, "expected": round(exp, 2), "upper": round(upper, 2)}

    if args.max_usd is None:
        raise prun.SpendNotConfirmed("--confirm-spend needs --max-usd (the budget for the whole sweep)")
    start = costs.total()
    for stage in stages:
        for st in steps_for(stage, models, root, args, n_dev):
            if st.argv is None:  # cannot happen after a complete previous stage; be safe
                raise SweepIncomplete(f"{stage} for {st.model}: earlier stage not complete")
            left = args.max_usd - (costs.total() - start)
            e = estimate(st, splits[st.split], train, cache, cfg)
            if e["to_send"] == 0:
                log.info("%s %s: everything cached", stage, st.model)
            elif e["upper"] > left:
                log.warning("%s %s may cost up to $%.2f; $%.2f of this sweep's budget is left. "
                            "Stopping before it.", stage, st.model, e["upper"], left)
                raise SweepIncomplete(f"sweep budget: {stage} for {st.model} needs up to ${e['upper']:.2f}, "
                                      f"${left:.2f} left")
            budget = min(left, round(e["upper"] * 1.1 + 0.05, 2))
            log.info("%s %s: %d to send, expected $%.2f, at most $%.2f (stage budget $%.2f)",
                     stage, st.model, e["to_send"], e["expected"], e["upper"], budget)
            run_args = prun.build_parser().parse_args(
                ["--model", st.model, *st.argv, "--confirm-spend", "--max-usd", str(budget),
                 "--concurrency", str(args.concurrency), "--data-dir", str(args.data_dir),
                 "--out-root", str(root)])
            prun.run(run_args, client=client_for(st.model) if client_for else None,
                     cache=cache, costs=costs)
        # Check that the stage is complete before anything is selected from it.
        for st in steps_for(stage, models, root, args, n_dev):
            n = {args.dev_split: n_dev, args.test_split: n_test, args.reasoning_split: n_reason}[st.split]
            if stage == "check":
                continue
            for spec in (st.candidates if st.argv else []):
                read_result(root, st.model, st.split, spec, n)
        text = write_summary(root, models, args, n_dev, n_test, n_reason)
        log.info("stage %s complete; spent $%.2f in this sweep, $%.2f in total of $%.2f",
                 stage, costs.total() - start, costs.total(), cap)
    print(text)
    return {"dry_run": False, "spent": round(costs.total() - start, 4)}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        sweep(build_parser().parse_args(argv))
    except prun.SpendNotConfirmed as e:
        print(f"NOT SENT: {e}", file=sys.stderr)
        return 2
    except SweepIncomplete as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
