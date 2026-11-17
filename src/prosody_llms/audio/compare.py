"""Text against audio on identical words: the headline comparison.

    python -m prosody_llms.audio.compare

Collects every test prediction file under ``$PROSODY_RUNS`` and scores all systems on
the same words of the test set: words with a timing in the public alignments that
start inside the first 30 s (the input window of Whisper and Qwen2-Audio). Files that
were written with words outside that set (text models, prompting) are simply restricted
to it; nothing is re-run.

Systems found (3-way files also give derived 2-way scores):

* ``probing/<model>/<condition>/predictions_test_<k>way_lda_best.tsv.gz`` (linear, best layer);
* ``probing_nonlinear/<model>/<condition>/predictions_test_<k>way_<probe>.tsv.gz`` (seed 0);
* ``audio/reconstruction/test/predictions_test_<k>way.tsv.gz`` (the labelling tool);
* ``audio/baselines/duration/predictions_test_<k>way.tsv.gz``;
* ``prompting/<model>/test/<prompt>/predictions.tsv.gz`` and
  ``icl/<model>/test/<prompt>/predictions.tsv.gz``;
* ``audio_prompting/<model>/test/<variant>-<audio|no_audio>/predictions.tsv.gz`` (the
  ``no_audio`` control counts as text).

For every task: accuracy on all words (as reported elsewhere) and on the common words,
an utterance-level bootstrap 95% interval on the common words, and McNemar's test of
each system against the best text system on the common words. Written to
``$PROSODY_RUNS/audio/comparison/`` (``comparison.md``, ``comparison.csv``,
``common_words.json``, and ``common_words.tsv.gz``: per utterance, one 0/1 per token).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

from prosody_llms.audio.download import default_audio_dir, textgrid_root
from prosody_llms.audio.timings import load_timings
from prosody_llms.data.corpus import LABEL_NA, read_split
from prosody_llms.data.download import default_data_dir
from prosody_llms.eval.stats import bootstrap_accuracy, mcnemar

WINDOW_S = 30.0
AUDIO_CONDITION = "audio-"


def read_predictions_gz(path: Path) -> dict[str, list[tuple[str, str, str]]]:
    out, cur = {}, None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("<file>\t"):
                cur = line.split("\t", 1)[1]
                out[cur] = []
            elif line:
                w, g, p = line.split("\t")
                out[cur].append((w, g, p))
    return out


def collapse(lab: str) -> str:
    return "1" if lab == "2" else lab


def find_systems(runs: Path) -> list[dict]:
    """[{name, family, modality, task, path, derived}] for every test prediction file."""
    out = []

    def add(name, family, modality, task, path, derived=False):
        out.append({"name": name, "family": family, "modality": modality, "task": task,
                    "path": path, "derived": derived})

    for task in (3, 2):
        for p in sorted((runs / "probing").glob(f"*/*/predictions_test_{task}way_lda_best.tsv.gz")):
            if p.parent.name.startswith("prev"):
                continue  # discourse-context runs are a separate comparison, not text systems here
            mod = "audio" if p.parent.name.startswith(AUDIO_CONDITION) else "text"
            add(f"{p.parent.parent.name} ({p.parent.name}) LDA", "probe_linear", mod, task, p)
        for p in sorted((runs / "probing_nonlinear").glob(f"*/*/predictions_test_{task}way_*.tsv.gz")):
            if p.parent.name.startswith("prev"):
                continue
            probe = p.name.split("_")[-1].split(".")[0]
            mod = "audio" if p.parent.name.startswith(AUDIO_CONDITION) else "text"
            add(f"{p.parent.parent.name} ({p.parent.name}) {probe}", "probe_nonlinear", mod, task, p)
        for sub, name, fam in (("audio/reconstruction/test", "wavelet toolkit (labelling procedure)", "reference"),
                               ("audio/baselines/duration", "duration only", "baseline")):
            p = runs / sub / f"predictions_test_{task}way.tsv.gz"
            if p.exists():
                add(name, fam, "audio", task, p)
    for exp, fam in (("prompting", "prompting"), ("icl", "icl")):
        for p in sorted((runs / exp).glob("*/test/*/predictions.tsv.gz")):
            prompt = p.parent.name
            native = 2 if "-2way-" in prompt else 3
            add(f"{p.parent.parent.parent.name} {prompt}", fam, "text", native, p)
            if native == 3:
                add(f"{p.parent.parent.parent.name} {prompt}", fam, "text", 2, p, derived=True)
    for p in sorted((runs / "audio_prompting").glob("*/test/*/predictions.tsv.gz")):
        mod = "text" if p.parent.name.endswith("no_audio") else "audio"
        for task in (3, 2):
            add(f"{p.parent.parent.parent.name} {p.parent.name}", "audio_prompting", mod, task, p,
                derived=task == 2)
    return out


def common_words(utts, audio_dir: Path) -> dict[str, list[bool]]:
    root = textgrid_root(audio_dir)
    out = {}
    for u in utts:
        t = load_timings(root, "test-clean", u)
        if t is None:
            out[u.utt_id] = [False] * len(u.tokens)
            continue
        out[u.utt_id] = [m and not tok.is_punct and w.start < WINDOW_S
                         for m, tok, w in zip(t.mask(), u.tokens, t.timings)]
    return out


def labels_for(system: dict, utts, task: int):
    preds = read_predictions_gz(system["path"])
    out = []
    for u in utts:
        rows = preds.get(u.utt_id)
        if rows is None or len(rows) != len(u.tokens):
            raise ValueError(f"{system['path']}: {u.utt_id} missing or different length")
        labs = [p for _, _, p in rows]
        out.append([collapse(x) for x in labs] if task == 2 else labs)
    return out


def compare(runs: Path, data_dir: Path, audio_dir: Path) -> list[dict]:
    utts = read_split(data_dir / "test.txt")
    mask = common_words(utts, audio_dir)
    rows = []
    for task in (3, 2):
        gold_all = [u.labels(task) for u in utts]
        gold_common = [[g if m else LABEL_NA for g, m in zip(gs, mask[u.utt_id])] for u, gs in zip(utts, gold_all)]
        systems = [s for s in find_systems(runs) if s["task"] == task]
        preds = {}
        for s in systems:
            try:
                preds[id(s)] = labels_for(s, utts, task)
            except ValueError as e:
                print(f"skipped: {e}", file=sys.stderr)
        systems = [s for s in systems if id(s) in preds]
        if not systems:
            continue
        boot = bootstrap_accuracy(gold_common, *[preds[id(s)] for s in systems])
        acc_all = bootstrap_accuracy(gold_all, *[preds[id(s)] for s in systems], n_resamples=1)
        results = []
        for k, s in enumerate(systems):
            results.append({"task": task, "system": s["name"], "family": s["family"], "modality": s["modality"],
                            "derived_2way": s["derived"], "accuracy_all_words": round(acc_all["accuracy"][k], 2),
                            "accuracy_common": round(boot["accuracy"][k], 2), "ci_low": round(boot["ci"][k][0], 2),
                            "ci_high": round(boot["ci"][k][1], 2), "path": str(s["path"].relative_to(runs))})
        text = [r for r in results if r["modality"] == "text"]
        if text:
            best = max(text, key=lambda r: r["accuracy_common"])
            bi = results.index(best)
            flat_gold = [g for gs in gold_common for g in gs]
            ref = [p for ps in preds[id(systems[bi])] for p in ps]
            for k, r in enumerate(results):
                r["vs_best_text"] = best["system"]
                if k != bi:
                    t = mcnemar(flat_gold, [p for ps in preds[id(systems[k])] for p in ps], ref)
                    r["mcnemar_p"] = t["p_value"]
                    r["diff_vs_best_text"] = round(r["accuracy_common"] - best["accuracy_common"], 2)
        rows += sorted(results, key=lambda r: -r["accuracy_common"])
    n_common = sum(sum(m) for m in mask.values())
    n_words = sum(not t.is_punct for u in utts for t in u.tokens)
    return rows, {"common_words": n_common, "test_words": n_words, "mask": mask}


def write(rows, info, out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    mask = info.pop("mask", None)
    if mask is not None:  # which test tokens are compared, for the analysis
        with gzip.open(out / "common_words.tsv.gz", "wt", encoding="utf-8") as f:
            for utt_id, m in mask.items():
                f.write(f"{utt_id}\t{''.join('1' if x else '0' for x in m)}\n")
    (out / "common_words.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    lines = ["# Text against audio on identical words", "",
             f"Common words: {info['common_words']} of {info['test_words']} scored test words "
             f"(aligned, starting within {WINDOW_S:.0f} s). 95% intervals: utterance-level bootstrap. "
             "McNemar: against the best text system on the common words.", ""]
    for task in (3, 2):
        sub = [r for r in rows if r["task"] == task]
        if not sub:
            continue
        lines += [f"## {task}-way", "", "| system | modality | family | common words | 95% CI | all words | vs best text | p |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in sub:
            name = r["system"] + (" (derived)" if r["derived_2way"] else "")
            diff = r.get("diff_vs_best_text")
            p = r.get("mcnemar_p")
            lines.append(f"| {name} | {r['modality']} | {r['family']} | {r['accuracy_common']:.2f} | "
                         f"{r['ci_low']:.2f}-{r['ci_high']:.2f} | {r['accuracy_all_words']:.2f} | "
                         f"{'' if diff is None else f'{diff:+.2f}'} | {'' if p is None else f'{p:.2g}'} |")
        lines.append("")
    text = "\n".join(lines)
    (out / "comparison.md").write_text(text, encoding="utf-8")
    (out / "result.json").write_text(json.dumps({"experiment": "comparison", **info, "systems": len(rows)},
                                                indent=2) + "\n", encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runs_default = Path(os.environ.get("PROSODY_RUNS", "runs"))
    p.add_argument("--runs", type=Path, default=runs_default)
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--audio-dir", type=Path, default=default_audio_dir())
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    rows, info = compare(args.runs, args.data_dir, args.audio_dir)
    print(write(rows, info, args.out or args.runs / "audio" / "comparison"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
