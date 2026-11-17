"""Collect every committed result under ``results/`` into one tidy list of rows.

Each row is one system on the test set for one task:

    {"family", "system", "model", "condition", "modality", "task", "test_2019",
     "test_full", "macro_f1", "sd", "n", "dev", "note"}

``test_2019`` is accuracy on the 2019 subset (the last utterance of the file left out,
as in the 2019 paper), ``test_full`` on all 4,822 utterances. ``sd`` and ``n`` are the
standard deviation and number of seeds where there are several. Only files that exist
are read, so this works at any stage of the project.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

AUDIO_CONDITION = "audio-"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _acc(metrics: dict, key: str) -> float | None:
    m = metrics.get(key)
    return None if m is None else m.get("accuracy")


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def row(family, system, model, condition, modality, task, test_2019, test_full, macro_f1=None,
        sd=None, n=1, dev=None, note="") -> dict:
    return {"family": family, "system": system, "model": model, "condition": condition,
            "modality": modality, "task": task, "test_2019": test_2019, "test_full": test_full,
            "macro_f1": macro_f1, "sd": sd, "n": n, "dev": dev, "note": note}


# ----------------------------------------------------------------------------- sources


def baselines(root: Path) -> list[dict]:
    path = root / "repro_2019" / "baselines.json"
    if not path.exists():
        return []
    out = []
    names = {"majority": "Majority class", "word_majority": "Word majority"}
    for key, res in _json(path)["results"].items():
        name, task = key.split("/")
        if name not in names:
            continue
        t = res["test"]
        out.append(row("baseline", names[name], name, "", "text", int(task[0]),
                       _acc(t, "legacy_2019"), _acc(t, "full")))
    return out


def finetuned(root: Path) -> list[dict]:
    """The 2019 BERT reproduction and the fine-tuned encoders: mean and sd over seeds."""
    out = []
    groups: dict[tuple, list[dict]] = {}
    for p in (root / "repro_2019" / "train_360-1.0").glob("*way/seed*/result.json"):
        groups.setdefault(("bert-base-uncased", int(p.parent.parent.name[0]), "2019 recipe"), []).append(_json(p))
    for p in (root / "finetune").glob("*/*way/seed*/result.json"):
        groups.setdefault((p.parent.parent.parent.name, int(p.parent.parent.name[0]), "mask, lr 2e-5"), []).append(_json(p))
    # A run whose dev accuracy is within 2 points of the majority-class baseline did not
    # train (it predicts one class); it is left out of the mean and reported in the note.
    majority_dev = {}
    bpath = root / "repro_2019" / "baselines.json"
    if bpath.exists():
        for key, res in _json(bpath)["results"].items():
            name, task = key.split("/")
            if name == "majority":
                majority_dev[int(task[0])] = res["dev"]["full"]["accuracy"]
    for (model, task, cond), all_runs in sorted(groups.items()):
        floor = majority_dev.get(task, 0.0) + 2.0
        runs = [r for r in all_runs if r["dev"]["full"]["accuracy"] > floor]
        failed = len(all_runs) - len(runs)
        if not runs:
            continue
        t19, sd = _mean_sd([r["test"]["legacy_2019"]["accuracy"] for r in runs])
        tf, _ = _mean_sd([r["test"]["full"]["accuracy"] for r in runs])
        f1, _ = _mean_sd([r["test"]["full"].get("macro_f1") for r in runs])
        dev, _ = _mean_sd([r["dev"]["full"]["accuracy"] for r in runs])
        out.append(row("finetune", f"{model} (fine-tuned)", model, cond, "text", task, t19, tf, f1, sd,
                       len(runs), dev, note=(f"{failed} of {len(all_runs)} seeds did not train (dev at the "
                                            f"majority-class level), left out" if failed else "")))
    return out


def linear_probes(root: Path) -> list[dict]:
    """Best layer (chosen on dev) of the LDA probe, every model and condition."""
    out = []
    for p in sorted((root / "probing").glob("*/*/result.json")):
        res = _json(p)
        cond = p.parent.name
        if re.search(r"-(n|ex)\d+$", cond) or cond.startswith("prev"):
            continue  # training-size curves and discourse-context runs: separate analyses
        modality = "audio" if cond.startswith(AUDIO_CONDITION) else "text"
        for s in res.get("summary", []):
            if s["probe"] != "lda":
                continue
            out.append(row("probe_linear", f"{res['model']} ({cond}) LDA", res["model"], cond, modality,
                           s["task"], s.get("test_2019"), s.get("test_full"), s.get("test_macro_f1"),
                           dev=s.get("dev"), note=f"layer {s['best_layer']}/{s['layers'] - 1}"))
    return out


def nonlinear_probes(root: Path) -> list[dict]:
    out = []
    for p in sorted((root / "probing_nonlinear").glob("*/*/records.jsonl")):
        cond = p.parent.name
        if cond.startswith("prev") or re.search(r"-(n|ex)\d+$", cond):
            continue  # discourse-context runs and reduced training sets: separate analyses
        modality = "audio" if cond.startswith(AUDIO_CONDITION) else "text"
        by: dict[tuple, list[dict]] = {}
        dev: dict[tuple, list[float]] = {}
        for r in _records(p):
            key = (r["condition"]["probe"], r["task"])
            if r["split"] == "test":
                by.setdefault(key, []).append(r)
            elif r["split"] == "dev":
                dev.setdefault(key, []).append(_acc(r["metrics"], "full"))
        for (probe, task), rs in sorted(by.items()):
            t19, sd = _mean_sd([_acc(r["metrics"], "legacy_2019") for r in rs])
            tf, _ = _mean_sd([_acc(r["metrics"], "full") for r in rs])
            f1, _ = _mean_sd([r["metrics"]["full"].get("macro_f1") for r in rs])
            model = rs[0]["model"]
            out.append(row("probe_nonlinear", f"{model} ({cond}) {probe}", model, cond, modality, task,
                           t19, tf, f1, sd, len(rs), dev=_mean_sd(dev.get((probe, task), []))[0],
                           note=f"layer {rs[0]['layer']}"))
    return out


def prompting(root: Path) -> list[dict]:
    """Prompting of hosted models, many-shot prompting and audio prompting on test splits."""
    out = []
    for exp, fam in (("prompting", "prompting"), ("icl", "icl"), ("audio_prompting", "audio_prompting")):
        for p in sorted((root / exp).glob("*/test*/*/records.jsonl")):
            split, prompt = p.parent.parent.name, p.parent.name
            modality = "audio" if fam == "audio_prompting" and not prompt.endswith("no_audio") else "text"
            for r in _records(p):
                derived = r["condition"].get("derived", False)
                out.append(row(fam, f"{r['model']} {prompt}" + (" (derived 2-way)" if derived else ""),
                               r["model"], prompt, modality, r["task"], _acc(r["metrics"], "legacy_2019"),
                               _acc(r["metrics"], "full"), r["metrics"]["full"].get("macro_f1"),
                               note=split if split != "test" else ""))
    return out


def audio_references(root: Path) -> list[dict]:
    out = []
    for sub, fam, name in (("audio/reconstruction/test", "reference", "Labelling procedure (toolkit)"),
                           ("audio/baselines/duration", "audio_baseline", "Duration only")):
        p = root / sub / "records.jsonl"
        if not p.exists():
            continue
        for r in _records(p):
            if r["split"] != "test":
                continue
            out.append(row(fam, name, r["model"], "", "audio", r["task"], _acc(r["metrics"], "legacy_2019"),
                           _acc(r["metrics"], "full"), r["metrics"]["full"].get("macro_f1")))
    return out


def collect(root: Path) -> list[dict]:
    return (baselines(root) + finetuned(root) + linear_probes(root) + nonlinear_probes(root)
            + prompting(root) + audio_references(root))


def comparison(root: Path) -> list[dict]:
    """The text-against-audio comparison on identical words, if present."""
    import csv

    p = root / "audio" / "comparison" / "comparison.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def icl_curve(root: Path, model: str = "qwen3.8-27b", split: str = "dev500") -> list[dict]:
    out = []
    for p in (root / "icl" / model / split).glob("*/result.json"):
        r = _json(p)
        out.append({"shots": r["prompt"]["shots"], "prefix_tokens": r["prefix_tokens"],
                    "acc3": r["accuracy"]["3"], "acc2": r["accuracy"]["2"], "auc": r.get("auc_prominent")})
    return sorted(out, key=lambda r: r["shots"])
