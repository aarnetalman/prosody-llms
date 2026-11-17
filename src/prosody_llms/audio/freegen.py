"""Does Qwen2-Audio-Instruct follow the answer format? A check on a few test sentences.

    python -m prosody_llms.audio.freegen --model qwen2-audio-7b-instruct --n 20

The prompted scores of the instruction-tuned model are read from its label probabilities
with the answer written for it (``prosody_llms.audio.prompt``). This only makes sense if the
model would write an answer of that form by itself. Here the model gets the same prompts,
with and without the recording, and generates freely (greedy). For every sentence we keep
the raw answer, parse it as in the text prompting experiment, and compare it with the gold labels and with the
labels read from the probabilities.

Outputs in ``$PROSODY_RUNS/audio_prompting/<model>/test<N>/<variant>-freegen-<audio|no_audio>/``:
``answers.jsonl`` (one line per sentence), ``summary.md`` and ``result.json``. There is
no ``records.jsonl``, so the analysis does not count these as results.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from prosody_llms.audio.base import libritts_path
from prosody_llms.audio.download import default_audio_dir
from prosody_llms.audio.extractors import SAMPLE_RATE, read_wav, resample
from prosody_llms.audio.prompt import AudioPrompter, conversation
from prosody_llms.data.corpus import LABEL_NA
from prosody_llms.data.download import default_data_dir
from prosody_llms.prompting.parse import parse_lines, parse_pairs
from prosody_llms.prompting.prompts import VARIANTS, PromptSpec, gold_answer, word_items
from prosody_llms.prompting.run import load_split
from prosody_llms.utils.runinfo import environment_info, write_json

log = logging.getLogger("prosody_llms.audio.freegen")


def generate(prompter: AudioPrompter, conv: list[dict], audio, max_new_tokens: int) -> str:
    import torch

    text = prompter.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    kw = {"text": text, "return_tensors": "pt"}
    if audio is not None:
        kw.update(audio=audio, sampling_rate=SAMPLE_RATE)
    inp = {k: v.to(prompter.device) for k, v in prompter.processor(**kw).items()}
    if "input_features" in inp:
        inp["input_features"] = inp["input_features"].to(next(prompter.model.parameters()).dtype)
    with torch.no_grad():
        out = prompter.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return prompter.processor.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)


def classify_answer(answer: str, utt) -> tuple[str, list[tuple[int, str]]]:
    """What a free answer is: ``labels`` (word-label pairs, numbered lines or one line),
    ``enumeration`` (the numbered words repeated without labels) or ``other``; for ``labels``,
    the (token index, label) of every corpus word that received one, matched in order."""
    import re

    from prosody_llms.prompting.parse import _norm, clean

    text = clean(answer).replace("\n", " ")
    items = word_items(utt)
    if re.match(r"^\s*1\.?\s+\S+\s+2\.?\s+\S+\s+3\.?\s", text + " "):
        return "enumeration", []  # "1 The 2 little 3 knot ...": numbers, no labels
    pairs = [(w, lab) for w, lab in re.findall(r"(\S+)\s+([012])(?=\s|$)", text) if not re.fullmatch(r"\d+\.?", w)]
    if len(pairs) < 2:
        return "other", []
    out, pos = [], 0
    for w, lab in pairs:
        for j in range(pos, min(pos + 4, len(items))):
            if _norm(items[j][1]) == _norm(w):
                out.append((items[j][0], lab))
                pos = j + 1
                break
    return "labels", out


def parse_free(answer: str, utt) -> tuple:
    """The requested line format if it covers more words, else "word label" pairs in order."""
    a, b = parse_lines(answer, utt, 3), parse_pairs(answer, utt, 3)
    return (a, "lines") if a.coverage >= b.coverage else (b, "pairs")


def compare(utt, free: list[str], scored: list[str]) -> dict:
    """Word-level agreement of the free answer and the scored labels with gold."""
    gold = utt.labels(3)
    idx = [i for i, _ in word_items(utt) if gold[i] != LABEL_NA]
    n = len(idx)
    return {"words": n,
            "free_correct": sum(free[i] == gold[i] for i in idx),
            "scored_correct": sum(scored[i] == gold[i] for i in idx),
            "free_vs_scored": sum(free[i] == scored[i] for i in idx),
            "free_dist": {lab: sum(free[i] == lab for i in idx) for lab in "012"},
            "scored_dist": {lab: sum(scored[i] == lab for i in idx) for lab in "012"},
            "gold_dist": {lab: sum(gold[i] == lab for i in idx) for lab in "012"}}


def summary_md(name: str, cond: str, rows: list[dict]) -> str:
    tot = {k: sum(r["compare"][k] for r in rows) for k in ("words", "free_correct", "scored_correct", "free_vs_scored")}
    dist = {k: {lab: sum(r["compare"][k][lab] for r in rows) for lab in "012"}
            for k in ("free_dist", "scored_dist", "gold_dist")}
    full = sum(r["coverage"] == 1.0 for r in rows)
    pct = lambda a, b: f"{100.0 * a / b:.1f}" if b else "--"  # noqa: E731
    lines = [f"# {name}, {cond}: free answers on {len(rows)} test sentences", "",
             f"- Answers that label every word in the expected format: {full} of {len(rows)}",
             f"- Mean share of words labelled: {pct(sum(r['coverage'] for r in rows), len(rows))}%",
             f"- Accuracy of the free answers (unlabelled words counted as 0): {pct(tot['free_correct'], tot['words'])}%",
             f"- Accuracy of the labels read from the probabilities: {pct(tot['scored_correct'], tot['words'])}%",
             f"- Free answers and probability labels agree on {pct(tot['free_vs_scored'], tot['words'])}% of the words", "",
             "| labels | 0 | 1 | 2 |", "|---|---|---|---|"]
    for k, lab in (("gold_dist", "gold"), ("free_dist", "free answers"), ("scored_dist", "from probabilities")):
        lines.append(f"| {lab} | " + " | ".join(pct(dist[k][x], tot["words"]) for x in "012") + " |")
    lines += ["", "## Answers", ""]
    for r in rows:
        lines += [f"### {r['utt_id']} (coverage {r['coverage']:.2f})", "", "Expected:", "", "```",
                  r["expected"].strip(), "```", "", "Model:", "", "```", r["answer"].strip() or "(empty)", "```", ""]
    return "\n".join(lines) + "\n"


def rescore(out_root: Path, name: str, split: str, utts) -> dict:
    """Parse saved answers again (both formats) and recompute the comparison; writes
    ``rescored.json`` and ``summary.md`` next to the answers. No model is needed."""
    by_id = {u.utt_id: u for u in utts}
    out = {}
    for d in sorted((out_root / name / split).glob("*-freegen-*")):
        f = d / "answers.jsonl" if (d / "answers.jsonl").exists() else d / "answers.partial.jsonl"
        if not f.exists():
            continue
        rows = []
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            u = by_id[r["utt_id"]]
            parsed, fmt = parse_free(r["answer"], u)
            r.update(coverage=parsed.coverage, format=fmt, free_labels=parsed.labels,
                     compare=compare(u, parsed.labels, r["scored_labels"]))
            rows.append(r)
        kinds, given, right, labelled = {}, {"0": 0, "1": 0, "2": 0}, 0, 0
        for r in rows:
            u = by_id[r["utt_id"]]
            kind, labs = classify_answer(r["answer"], u)
            r["kind"] = kind
            kinds[kind] = kinds.get(kind, 0) + 1
            g = u.labels(3)
            for i, lab in labs:
                if g[i] != LABEL_NA:
                    labelled += 1
                    right += lab == g[i]
                    given[lab] += 1
        res = {"condition": d.name, "utterances": len(rows), "kinds": kinds, "labelled_words": labelled,
               "labels_given": given, "accuracy_on_labelled_3": 100.0 * right / labelled if labelled else None,
               "gold_nonprominent_share": 100.0 * sum(r["compare"]["gold_dist"]["0"] for r in rows)
               / sum(r["compare"]["words"] for r in rows)}
        write_json(d / "rescored.json", res)
        (d / "summary.md").write_text(summary_md(name, d.name, rows), encoding="utf-8")
        out[d.name] = res
        print(json.dumps(res, indent=1))
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen2-audio-7b-instruct", help="registry name or a local path")
    p.add_argument("--revision", default=None)
    p.add_argument("--n", type=int, default=20, help="number of test sentences (the fixed test<N> subset)")
    p.add_argument("--variant", choices=VARIANTS, default="linguistic")
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--audio-dir", type=Path, default=default_audio_dir())
    p.add_argument("--out-root", type=Path,
                   default=Path(os.environ.get("PROSODY_RUNS", "runs")) / "audio_prompting")
    p.add_argument("--rescore", action="store_true",
                   help="only parse the saved answers again (no model needed)")
    p.add_argument("--tokens-per-word", type=int, default=6,
                   help="generation budget per word (a correct answer line takes about 4-5 tokens)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    from prosody_llms.config import get_model

    if args.rescore:
        try:
            name = get_model(args.model).name
        except KeyError:
            name = Path(args.model).name
        rescore(args.out_root, name, f"test{args.n}", load_split(args.data_dir, f"test{args.n}"))
        return 0
    try:
        m = get_model(args.model)
        model_id, revision, name = m.id, m.revision, m.name
    except KeyError:
        model_id, revision, name = args.model, args.revision, Path(args.model).name
    prompter = AudioPrompter(model_id, revision, args.dtype, args.device)
    split = f"test{args.n}"
    utts = load_split(args.data_dir, split)
    spec = PromptSpec(args.variant, "lines", 3, 0)
    for with_audio in (True, False):
        cond = f"{args.variant}-freegen-{'audio' if with_audio else 'no_audio'}"
        out = args.out_root / name / split / cond
        if (out / "result.json").exists():
            log.info("%s already finished", out)
            continue
        out.mkdir(parents=True, exist_ok=True)
        partial = out / "answers.partial.jsonl"  # written as we go, so a stopped job can resume
        rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()] \
            if partial.exists() else []
        done = {r["utt_id"] for r in rows}
        if done:
            log.info("%s: resuming after %d sentences", cond, len(done))
        t0 = time.time()
        for u in utts:
            if u.utt_id in done:
                continue
            audio = None
            if with_audio:
                x, sr = read_wav(libritts_path(args.audio_dir, "test", u.utt_id))
                audio = resample(x, sr)
            conv = conversation(u, spec, with_audio)
            answer = generate(prompter, conv, audio,
                              max_new_tokens=args.tokens_per_word * len(word_items(u)) + 24)
            parsed, fmt = parse_free(answer, u)
            cache, _ = prompter.prompt_state(conv, audio)
            scored, _ = prompter.scorer.score_utterance(cache, "", u, task=3)
            row = {"utt_id": u.utt_id, "answer": answer, "expected": gold_answer(u, spec),
                   "coverage": parsed.coverage, "format": fmt, "parse_stats": parsed.stats,
                   "free_labels": parsed.labels, "scored_labels": scored, "gold": u.labels(3),
                   "compare": compare(u, parsed.labels, scored)}
            rows.append(row)
            with open(partial, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.info("%s %s (%d/%d): coverage %.2f; answer starts %r", cond, u.utt_id, len(rows), len(utts),
                     parsed.coverage, answer[:100])
        with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        md = summary_md(name, cond, rows)
        (out / "summary.md").write_text(md, encoding="utf-8")
        words = sum(r["compare"]["words"] for r in rows)
        write_json(out / "result.json", {
            "experiment": "audio_prompting_freegen", "model": name, "split": split, "audio": with_audio,
            "variant": args.variant, "utterances": len(rows),
            "full_coverage": sum(r["coverage"] == 1.0 for r in rows),
            "mean_coverage": sum(r["coverage"] for r in rows) / len(rows),
            "free_accuracy_3": 100.0 * sum(r["compare"]["free_correct"] for r in rows) / words,
            "scored_accuracy_3": 100.0 * sum(r["compare"]["scored_correct"] for r in rows) / words,
            "free_vs_scored": 100.0 * sum(r["compare"]["free_vs_scored"] for r in rows) / words,
            "time_s": round(time.time() - t0, 1), "environment": environment_info()})
        print(md.split("## Answers")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
