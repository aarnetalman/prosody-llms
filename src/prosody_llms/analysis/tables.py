"""Results tables, the numbers quoted in the paper and the data behind its figures.

Writes, from ``results/`` only, under ``results/analysis/``:

* ``all_results.csv``: every system, task and test score (tidy);
* ``main_table.md`` and ``tables/main_table.csv``;
* ``tables/comparison.csv``: text against audio on identical words;
* ``tables/paired.csv``: probing against prompting the same model;
* ``tables/models.csv``: the models with their pinned revisions;
* ``numbers.csv``: every number quoted in the paper, by name;
* ``figures/*.csv``: layer curves, model size, number of examples.

Test accuracy is on the 2019 subset (as in the 2019 paper) unless stated otherwise.
Systems are chosen on dev where a choice is made (best layer, best text and audio
model); never on test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from prosody_llms.analysis import collect as C
from prosody_llms.analysis.output import figures_dir, rnd, tables_dir, write_csv

PAPER_2019 = {2: 83.2, 3: 68.6}  # Talman et al. (2019), BERT-base, Table 3

NAMES = {
    "bert-base-uncased": "BERT-base", "bert-base-cased": "BERT-base cased",
    "bert-large-uncased": "BERT-large", "modernbert-base": "ModernBERT-base",
    "modernbert-large": "ModernBERT-large", "xlm-roberta-base": "XLM-R base",
    "xlm-roberta-large": "XLM-R large", "muse-glimmer-30b": "Muse Glimmer 30B",
    "qwen3.8-27b": "Qwen3.8 27B", "olmo-3-7b": "OLMo 3 7B", "olmo-3-32b": "OLMo 3 32B",
    "qwen3.5-0.8b-base": "Qwen3.5 0.8B", "qwen3.5-2b-base": "Qwen3.5 2B",
    "qwen3.5-4b-base": "Qwen3.5 4B", "qwen3.5-9b-base": "Qwen3.5 9B",
    "wavlm-large": "WavLM-large", "hubert-large": "HuBERT-large",
    "whisper-large-v3": "Whisper large-v3 (encoder)", "qwen2-audio-7b": "Qwen2-Audio 7B",
    "qwen2-audio-7b-instruct": "Qwen2-Audio 7B Instruct", "gpt-4o": "GPT-4o",
    "deepseek-v4-pro": "DeepSeek-V4-Pro", "mistral-large-3": "Mistral Large 3",
    "gpt-5.5": "GPT-5.5",
}


def display(model: str, cond: str = "") -> str:
    name = NAMES.get(model, model)
    if cond.startswith("audio-mean-"):
        name += {"llm": " (LM)", "encoder": " (encoder)"}.get(cond.rsplit("-", 1)[-1], "")
    elif cond.startswith("echo"):
        name += ", echo"
    return name


def fmt_md(x, sd=None, nd=2) -> str:
    if x is None:
        return ""
    return f"{x:.{nd}f}" + (f" ± {sd:.{nd}f}" if sd else "")


def by_key(rows):
    out = {}
    for r in rows:
        out.setdefault((r["family"], r["system"]), {})[r["task"]] = r
    return out


def best(rows, family, modality, task, key="dev"):
    """The row of a family and modality with the highest dev score for a task (test as a
    fallback only where no dev score exists, e.g. prompting, where settings were already
    chosen on dev)."""
    cands = [r for r in rows if r["family"] == family and r["modality"] == modality and r["task"] == task]
    if not cands:
        return None
    k = key if all(r.get(key) is not None for r in cands) else "test_2019"
    return max(cands, key=lambda r: (r[k] if r[k] is not None else -1))


# ----------------------------------------------------------------------------- main table


def main_table_rows(rows) -> list[tuple[str, str, dict, dict]]:
    """(group, label, row 2-way, row 3-way) in table order."""
    k = by_key(rows)
    out = []

    def add(group, label, family, system):
        pair = k.get((family, system))
        if pair:
            out.append((group, label, pair.get(2), pair.get(3)))

    add("Baselines", "Majority class", "baseline", "Majority class")
    add("Baselines", "Word majority (2019)", "baseline", "Word majority")
    add("Fine-tuned", "BERT-base (2019 recipe)", "finetune", "bert-base-uncased (fine-tuned)")
    for m, name in (("bert-large-uncased", "BERT-large"), ("modernbert-large", "ModernBERT-large"),
                    ("xlm-roberta-large", "XLM-R large")):
        add("Fine-tuned", name, "finetune", f"{m} (fine-tuned)")
    add("Frozen, linear probe", "BERT-base", "probe_linear", "bert-base-uncased (plain-first-natural) LDA")
    for fam, group in (("probe_linear", "Frozen, linear probe"), ("probe_nonlinear", "Frozen, BiLSTM classifier")):
        for modality in ("text", "audio"):
            cand = [r for r in rows if r["family"] == fam and r["modality"] == modality
                    and (fam == "probe_linear" or r["system"].endswith(" bilstm"))
                    and "train_100" not in r["condition"]]
            b2 = best(cand, fam, modality, 2)
            b3 = best(cand, fam, modality, 3)
            if b3:
                same = lambda a, t: next((r for r in cand if r["model"] == a["model"]  # noqa: E731
                                          and r["condition"] == a["condition"] and r["task"] == t), None)
                if b2 and (b2["model"], b2["condition"]) != (b3["model"], b3["condition"]):
                    # different systems are best on the two tasks: one row each, both tasks shown
                    out.append((group, f"best {modality} (3-way): {display(b3['model'], b3['condition'])}",
                                same(b3, 2), b3))
                    out.append((group, f"best {modality} (2-way): {display(b2['model'], b2['condition'])}",
                                b2, same(b2, 3)))
                else:
                    out.append((group, f"best {modality}: {display(b3['model'], b3['condition'])}", b2, b3))
    for m, name in (("gpt-4o", "GPT-4o"), ("deepseek-v4-pro", "DeepSeek-V4-Pro"),
                    ("mistral-large-3", "Mistral Large 3"), ("gpt-5.5", "GPT-5.5 (200 test utterances)")):
        r3 = next((r for r in rows if r["family"] == "prompting" and r["model"] == m and r["task"] == 3), None)
        r2 = next((r for r in rows if r["family"] == "prompting" and r["model"] == m and r["task"] == 2
                   and "derived" in r["system"]), None)
        if r3:
            out.append(("Prompting", name, r2, r3))
    r3 = next((r for r in rows if r["family"] == "icl" and r["task"] == 3 and r["condition"].endswith("-k500")), None)
    if r3:
        r2 = next((r for r in rows if r["family"] == "icl" and r["task"] == 2 and r["condition"].endswith("-k500")), None)
        out.append(("Prompting", f"{display(r3['model'])}, {r3['condition'].split('-k')[-1]} examples", r2, r3))
    for cond, label in (("linguistic-audio", "Qwen2-Audio, with recording"),
                        ("linguistic-no_audio", "Qwen2-Audio, text only")):
        r3 = next((r for r in rows if r["family"] == "audio_prompting" and r["condition"] == cond and r["task"] == 3), None)
        r2 = next((r for r in rows if r["family"] == "audio_prompting" and r["condition"] == cond and r["task"] == 2), None)
        if r3:
            out.append(("Prompting", label, r2, r3))
    add("Audio references", "Duration only", "audio_baseline", "Duration only")
    add("Audio references", "Labelling procedure", "reference", "Labelling procedure (toolkit)")
    return out


def main_table_md(trows) -> str:
    lines = ["# Main results (test, 2019 subset)", "",
             "Accuracy on scored words; mean ± sd over seeds where there are several. Best layers and "
             "best models chosen on dev.", "",
             "| group | system | modality | 2-way | 3-way | note |", "|---|---|---|---|---|---|"]
    for group, label, r2, r3 in trows:
        ref = r3 or r2
        notes = [(r2 or {}).get("note", ""), (r3 or {}).get("note", "")]  # same order as the columns
        note = "; ".join(x for i, x in enumerate(notes) if x and x not in notes[:i])
        lines.append(f"| {group} | {label} | {ref['modality']} | {fmt_md(r2 and r2['test_2019'], r2 and r2['sd'])} | "
                     f"{fmt_md(r3 and r3['test_2019'], r3 and r3['sd'])} | {note} |")
    return "\n".join(lines) + "\n"


def dev_scores(rows) -> dict:
    """(system, task) -> development accuracy, for choosing systems in the comparison."""
    return {(r["system"], int(r["task"])): float(r["dev"]) for r in rows if r.get("dev") not in (None, "")}


def pick(cands, dev: dict, task_of=lambda r: int(r["task"])):
    """The candidate with the best development score; candidates without one (prompting,
    whose settings were already chosen on dev) only if none has a score."""
    scored = [r for r in cands if (r["system"], task_of(r)) in dev]
    if scored:
        return max(scored, key=lambda r: dev[(r["system"], task_of(r))])
    return max(cands, key=lambda r: float(r["accuracy_common"]))


def mcnemar_vs(root: Path, comp: list[dict], ref: dict, others: list[dict]) -> dict:
    """McNemar p-values of ``others`` against ``ref`` on the common words, recomputed from
    the prediction files (comparison rows of one task; gold labels from the files)."""
    import gzip

    from prosody_llms.analysis.errors import common_mask
    from prosody_llms.eval.stats import mcnemar

    mask = common_mask(root)

    def flat(r):
        out_g, out_p = [], []
        with gzip.open(root / r["path"], "rt", encoding="utf-8") as f:
            utt = None
            i = 0
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("<file>"):
                    utt, i = line.split("\t")[1].removesuffix(".txt"), 0
                    continue
                if not line:
                    continue
                _, g, pr = line.split("\t")[:3]
                keep = mask is None or (utt in mask and i < len(mask[utt]) and mask[utt][i])
                i += 1
                if int(r["task"]) == 2 and str(r.get("derived_2way")) == "True":
                    g, pr = ("1" if g == "2" else g), ("1" if pr == "2" else pr)
                out_g.append(g if keep else "NA")
                out_p.append(pr)
        return out_g, out_p

    if not (root / ref["path"]).exists():
        return {}
    gold, rp = flat(ref)
    out = {}
    for r in others:
        if (root / r["path"]).exists():
            _, op = flat(r)
            if len(op) == len(rp):
                out[r["system"]] = mcnemar(gold, op, rp)["p_value"]
    return out


# ----------------------------------------------------------------------------- comparison


PROBE_NAMES = {"LDA": "linear", "bilstm": "BiLSTM", "mlp": "MLP", "logistic": "logistic"}


def system_label(name: str) -> str:
    """'whisper-large-v3 (audio-mean) bilstm' -> 'Whisper large-v3 (encoder), BiLSTM classifier'."""
    m = re.match(r"^(\S+) \(([^)]*)\) (\S+)$", name)
    if m:
        model, cond, probe = m.groups()
        extra = ", train_100" if "train_100" in cond else ""
        kind = "probe" if probe == "LDA" else "classifier"
        return f"{display(model, cond)}, {PROBE_NAMES.get(probe, probe)} {kind}{extra}"
    m = re.match(r"^(\S+) (\S+)$", name)
    if m and m.group(1) in NAMES:
        return f"{display(m.group(1))}, prompted"
    return {"wavelet toolkit (labelling procedure)": "Labelling procedure", "duration only": "Duration only"}.get(name, name)


def comparison_rows(comp: list[dict], dev: dict | None = None, root: Path | None = None) -> list[dict]:
    """One row per kind of system on the common words (the best of each kind on dev, by
    3-way where there is a 3-way row), with McNemar p-values against the best text system."""
    if not comp:
        return []
    by_sys: dict[str, dict] = {}
    for r in comp:
        by_sys.setdefault(r["system"], {})[int(r["task"])] = r
    kinds = [
        ("reference", lambda r: r["family"] == "reference"),
        ("speech BiLSTM", lambda r: r["modality"] == "audio" and r["system"].endswith(" bilstm") and "llm" not in r["system"]),
        ("speech linear", lambda r: r["modality"] == "audio" and r["system"].endswith(" LDA") and "llm" not in r["system"]),
        ("speech LLM", lambda r: r["modality"] == "audio" and "audio-mean-llm" in r["system"] and r["system"].endswith(" bilstm")),
        ("duration", lambda r: r["family"] == "baseline"),
        ("text BiLSTM", lambda r: r["modality"] == "text" and r["system"].endswith(" bilstm") and "train_100" not in r["system"]),
        ("text BiLSTM, train_100", lambda r: r["modality"] == "text" and r["system"].endswith(" bilstm") and "train_100" in r["system"]),
        ("text linear", lambda r: r["modality"] == "text" and r["system"].endswith(" LDA") and "train_100" not in r["system"]
         and not re.search(r"-n\d+\)", r["system"])),
        ("prompting", lambda r: r["family"] in ("prompting", "icl")),
        ("audio prompting", lambda r: r["family"] == "audio_prompting" and r["modality"] == "audio"),
    ]
    chosen = []
    for kind, pred in kinds:
        cands = [(name, d) for name, d in by_sys.items() if pred(d.get(3) or d.get(2))]
        if cands:  # the best of each kind on dev (3-way where it exists)
            rep = {id(d.get(3) or d.get(2)): (name, d) for name, d in cands}
            best = pick([d.get(3) or d.get(2) for _, d in cands], dev or {})
            chosen.append((kind, *rep[id(best)]))
    # Significance against the best text system chosen on dev, recomputed from the predictions.
    pvals: dict = {}
    if root is not None:
        for t in (2, 3):
            text = [r for r in comp if int(r["task"]) == t and r["modality"] == "text"]
            if text:
                ref = pick(text, dev or {})
                rows_t = [d[t] for _, _, d in chosen if d.get(t) and d[t]["system"] != ref["system"]]
                pvals[t] = {"ref": ref["system"], **mcnemar_vs(root, comp, ref, rows_t)}
    out = []
    for kind, name, d in chosen:
        r = d.get(3) or d.get(2)
        row = {"kind": kind, "system": system_label(name), "modality": r["modality"]}
        for t in (2, 3):
            x = d.get(t)
            row[f"acc{t}"] = rnd(x and x["accuracy_common"])
            if not x:
                pv = None
            elif t in pvals:
                pv = None if x["system"] == pvals[t]["ref"] else pvals[t].get(x["system"])
            else:
                pv = float(x["mcnemar_p"]) if x.get("mcnemar_p") else None
            row[f"p_vs_best_text{t}"] = "" if pv is None else f"{pv:.3g}"
        x = d.get(3)
        row["ci3_low"] = rnd(x and x.get("ci_low"))
        row["ci3_high"] = rnd(x and x.get("ci_high"))
        out.append(row)
    return out


# ----------------------------------------------------------------------------- numbers

def numbers(rows, comp, root: Path, extra: dict | None = None) -> dict[str, str]:
    """Named numbers used in the paper text."""
    n: dict[str, str] = {}
    k = by_key(rows)
    dev = dev_scores(rows)
    def put(key, value, nd=1):
        if value is not None:
            n[key] = f"{value:.{nd}f}" if isinstance(value, float) else str(value)

    # The linear probe on the embedding layer (layer 0, no context) against the best layer:
    # what the contextual layers add beyond the identity of the word.
    for model, cond, tag in (("qwen3.8-27b", "echo-last-natural", "qwen"), ("bert-base-uncased", "plain-first-natural", "bert")):
        curve = layer_curve_rows(root, model, cond)
        if curve:
            for t in (2, 3):
                accs = [float(r[f"acc{t}"]) for r in curve]
                put(f"emb-{tag}-{t}", accs[0])

    for t in (2, 3):
        put(f"paper2019-{t}", PAPER_2019[t])
        r = k.get(("finetune", "bert-base-uncased (fine-tuned)"), {}).get(t)
        if r:
            put(f"repro-{t}", r["test_2019"], 2)
            put(f"repro-sd-{t}", r["sd"], 2)
        for name, fam in (("majority", "Majority class"), ("wordmaj", "Word majority")):
            r = k.get(("baseline", fam), {}).get(t)
            if r:
                put(f"{name}-{t}", r["test_2019"])
        r = k.get(("probe_linear", "bert-base-uncased (plain-first-natural) LDA"), {}).get(t)
        if r:
            put(f"bertprobe-{t}", r["test_2019"])
        for fam, tag in (("probe_linear", "lin"), ("probe_nonlinear", "bilstm")):
            for modality in ("text", "audio"):
                cand = [r for r in rows if r["family"] == fam and r["modality"] == modality
                        and (fam == "probe_linear" or r["system"].endswith(" bilstm"))
                        and "train_100" not in r["condition"]]
                b = best(cand, fam, modality, t)
                if b:
                    put(f"{tag}-{modality}-{t}", b["test_2019"])
                    n[f"{tag}-{modality}-model-{t}"] = display(b["model"], b["condition"])
        prompts = [r for r in rows if r["family"] == "prompting" and r["task"] == t and r["note"] != "test200"
                   and (t == 3 or "derived" in r["system"])]
        if prompts:
            b = max(prompts, key=lambda r: r["test_2019"])
            put(f"prompt-best-{t}", b["test_2019"])
            n[f"prompt-best-model-{t}"] = display(b["model"])
        r = next((r for r in rows if r["family"] == "icl" and r["task"] == t and r["condition"].endswith("-k500")), None)
        if r:
            put(f"icl-test-{t}", r["test_2019"])
        r = next((r for r in rows if r["family"] == "icl" and r["task"] == t and r["condition"].endswith("-k0")), None)
        if r:  # zero-shot on the test set, if run
            put(f"icl-test-k0-{t}", r["test_2019"])
        for cond, tag in (("linguistic-audio", "ap-audio"), ("linguistic-no_audio", "ap-text")):
            r = next((r for r in rows if r["family"] == "audio_prompting" and r["condition"] == cond and r["task"] == t), None)
            if r:
                put(f"{tag}-{t}", r["test_2019"])
        for fam, tag in (("audio_baseline", "duration"), ("reference", "toolkit")):
            r = next((r for r in rows if r["family"] == fam and r["task"] == t), None)
            if r:
                put(f"{tag}-{t}", r["test_2019"])
        for ft in ("bert-large-uncased", "modernbert-large", "xlm-roberta-large"):
            r = k.get(("finetune", f"{ft} (fine-tuned)"), {}).get(t)
            if r:
                put(f"ft-{ft}-{t}", r["test_2019"])
                put(f"ft-{ft}-sd-{t}", r["sd"], 2)
    # Text against audio on identical words.
    for t in (2, 3):
        sub = [r for r in comp if int(r["task"]) == t]
        for modality in ("text", "audio"):
            cand = [r for r in sub if r["modality"] == modality and r["family"] not in ("reference", "baseline")]
            if cand:
                b = pick(cand, dev)
                put(f"common-{modality}-{t}", float(b["accuracy_common"]))
                n[f"common-{modality}-system-{t}"] = b["system"]
        if any(r["modality"] == "audio" for r in sub) and any(r["modality"] == "text" for r in sub):
            put(f"gap-{t}", float(n[f"common-audio-{t}"]) - float(n[f"common-text-{t}"]))
    # Best text system overall (fine-tuned or probed, test 2019 subset), the gap from the
    # best speech probe to it, the gap to the labelling procedure on the common words, and
    # the loss from Qwen2-Audio's encoder to its language model.
    for t in (2, 3):
        text = [r for r in rows if r["modality"] == "text" and r["task"] == t
                and r["family"] in ("finetune", "probe_linear", "probe_nonlinear")
                and "train_100" not in r["condition"] and r["test_2019"] is not None]
        speech = [r for r in rows if r["modality"] == "audio" and r["task"] == t
                  and r["family"] in ("probe_linear", "probe_nonlinear") and r["test_2019"] is not None]
        if text:
            bt = max(text, key=lambda r: (r["dev"] if r.get("dev") is not None else -1))  # chosen on dev
            put(f"text-best-{t}", bt["test_2019"])
            n[f"text-best-model-{t}"] = display(bt["model"]) + (" (fine-tuned)" if bt["family"] == "finetune" else "")
            if speech:
                bs = max(speech, key=lambda r: (r["dev"] if r.get("dev") is not None else -1))
                put(f"gap-best-{t}", bs["test_2019"] - bt["test_2019"])
        ref = next((r for r in comp if int(r["task"]) == t and r["family"] == "reference"), None)
        best_audio = [r for r in comp if int(r["task"]) == t and r["modality"] == "audio"
                      and r["family"] not in ("reference", "baseline")]
        if ref and best_audio:
            put(f"toolkit-gap-{t}", float(ref["accuracy_common"]) - float(pick(best_audio, dev)["accuracy_common"]))
        enc = next((r for r in rows if r["family"] == "probe_nonlinear" and r["model"] == "qwen2-audio-7b"
                    and r["condition"] == "audio-mean-encoder" and r["task"] == t and r["system"].endswith("bilstm")), None)
        lm = next((r for r in rows if r["family"] == "probe_nonlinear" and r["model"] == "qwen2-audio-7b"
                   and r["condition"] == "audio-mean-llm" and r["task"] == t and r["system"].endswith("bilstm")), None)
        if enc and lm:
            put(f"qa-loss-{t}", enc["test_2019"] - lm["test_2019"])
    info = root / "audio" / "comparison" / "common_words.json"
    if info.exists():
        d = json.loads(info.read_text(encoding="utf-8"))
        put("common-words", d["common_words"])
        put("common-share", 100.0 * d["common_words"] / d["test_words"])
    rec = root / "audio" / "reconstruction" / "result.json"
    if rec.exists():
        d = json.loads(rec.read_text(encoding="utf-8"))["splits"].get("test", {})
        if d:
            put("recon-3", d["accuracy_aligned"]["3"])
            put("recon-2", d["accuracy_aligned"]["2"])
            put("recon-pearson", d["values"]["pearson"], 3)
            put("recon-spearman", d["values"]["spearman"], 3)
            put("recon-coverage", 100.0 * d["values"]["coverage"])
    # Each speech model (and two text references): linear and BiLSTM probe, best layer.
    probe_sets = [("wavlm", "wavlm-large", "audio-mean"), ("hubert", "hubert-large", "audio-mean"),
                  ("whisper", "whisper-large-v3", "audio-mean"), ("qaenc", "qwen2-audio-7b", "audio-mean-encoder"),
                  ("qalm", "qwen2-audio-7b", "audio-mean-llm"), ("bert", "bert-base-uncased", "plain-first-natural"),
                  ("glimmer", "muse-glimmer-30b", "echo-last-natural"), ("qwen", "qwen3.8-27b", "echo-last-natural"),
                  ("bert100", "bert-base-uncased", "plain-first-natural-train_100"),
                  ("glimmer100", "muse-glimmer-30b", "echo-last-natural-train_100")]
    for tag, model, cond in probe_sets:
        for t in (2, 3):
            lin = next((r for r in rows if r["family"] == "probe_linear" and r["model"] == model
                        and r["condition"] == cond and r["task"] == t), None)
            bil = next((r for r in rows if r["family"] == "probe_nonlinear" and r["model"] == model
                        and r["condition"] == cond and r["task"] == t and r["system"].endswith("bilstm")), None)
            if lin:
                put(f"p-{tag}-lin-{t}", lin["test_2019"])
                m = re.match(r"layer (\d+)/(\d+)", lin["note"] or "")
                if m and t == 3:
                    n[f"p-{tag}-layer"] = f"{m.group(1)}/{m.group(2)}"
                    put(f"p-{tag}-depth", 100.0 * int(m.group(1)) / int(m.group(2)), 0)
            if bil:
                put(f"p-{tag}-bilstm-{t}", bil["test_2019"])
    # Echo against plain input for the decoders (LDA, test), and the largest difference
    # between the 2019 subset and the full test file over all rows.
    for t in (2, 3):
        gains = []
        for r in rows:
            if r["family"] == "probe_linear" and r["condition"] == "echo-last-natural" and r["task"] == t:
                plain = next((x for x in rows if x["family"] == "probe_linear" and x["model"] == r["model"]
                              and x["condition"] == "plain-last-natural" and x["task"] == t), None)
                if plain:
                    gains.append(r["test_2019"] - plain["test_2019"])
        if gains:
            put(f"echo-gain-min-{t}", min(gains))
            put(f"echo-gain-max-{t}", max(gains))
    diffs = [abs(r["test_2019"] - r["test_full"]) for r in rows if r["test_2019"] is not None and r["test_full"] is not None]
    if diffs:
        put("legacy-full-maxdiff", max(diffs), 2)
    n["encoders-probed"] = str(len({r["model"] for r in rows if r["family"] == "probe_linear" and r["modality"] == "text"
                                    and r["condition"].endswith("first-natural")}))
    n["decoders-probed"] = str(len({r["model"] for r in rows if r["family"] == "probe_linear" and r["modality"] == "text"
                                    and r["condition"].endswith("last-natural")}))
    for model, b in SIZES.items():
        tag = re.sub(r"[^0-9.]", "", str(b))
        for t in (2, 3):
            r = next((r for r in rows if r["family"] == "probe_linear" and r["model"] == model and r["task"] == t
                      and r["condition"].startswith("echo")), None)
            if r:
                put(f"size-{tag}-{t}", r["test_2019"])
    curve = C.icl_curve(root)
    for p in curve:
        put(f"icl-dev-{p['shots']}-3", p["acc3"])
        put(f"icl-dev-{p['shots']}-2", p["acc2"])
    # Label profile (3-way, test): predicted against gold class shares and per-class
    # recall, for the best prompted model and the best linear text probe.
    best_prompt = max((r for r in rows if r["family"] == "prompting" and r["task"] == 3 and r["note"] != "test200"),
                      key=lambda r: r["test_2019"], default=None)
    best_lin = best([r for r in rows if r["family"] == "probe_linear" and r["modality"] == "text"
                     and "train_100" not in r["condition"]], "probe_linear", "text", 3)
    for tag, r in (("prompt", best_prompt), ("probe", best_lin)):
        prof = label_profile(root, r)
        for k, v in (prof or {}).items():
            put(f"profile-{tag}-{k}", v)
    cost = root / "prompting" / "costs_summary.txt"
    if cost.exists():
        m = re.search(r"total: \$([0-9.]+)", cost.read_text(encoding="utf-8"))
        if m:
            put("azure-cost", float(m.group(1)), 2)
    n.update(extra or {})
    return n


def label_profile(root: Path, r: dict | None) -> dict | None:
    """Gold and predicted class shares and per-class recall (3-way, full test file) for a
    prompting run or a linear probe (its best layer)."""
    if r is None:
        return None
    if r["family"] == "prompting":
        cands = list((root / "prompting" / r["model"]).glob(f"test/{r['condition']}/records.jsonl"))
        layer = None
    else:
        cands = [root / "probing" / r["model"] / r["condition"] / "records.jsonl"]
        layer = int(r["note"].split()[1].split("/")[0])
    for path in cands:
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            if rec["task"] != 3 or rec.get("split", "test") != "test":
                continue
            if layer is not None and (rec["layer"] != layer or rec["condition"].get("probe") != "lda"):
                continue
            m = rec["metrics"].get("full") or {}
            conf = m.get("confusion")
            if not conf or "per_class" not in m:
                continue
            gold = {g: sum(row.values()) for g, row in conf.items()}
            pred: dict[str, int] = {}
            for row in conf.values():
                for p_, n_ in row.items():
                    pred[p_] = pred.get(p_, 0) + n_
            tot = sum(gold.values())
            out = {}
            for c in ("0", "1", "2"):
                out[f"gold-{c}"] = 100.0 * gold.get(c, 0) / tot
                out[f"pred-{c}"] = 100.0 * pred.get(c, 0) / tot
                out[f"recall-{c}"] = m["per_class"][c]["recall"]
            return out
    return None


def params_label(n) -> str:
    """110106428 -> '110M', 27781427952 -> '27.8B'."""
    if not n:
        return "--"
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{round(n / 1e6):d}M"


def models_rows() -> list[dict]:
    """The models with their Hugging Face identifiers and pinned revisions."""
    from prosody_llms.config import load_registry

    groups = [("Encoders", ["bert-base-uncased", "bert-base-cased", "bert-large-uncased", "modernbert-base",
                            "modernbert-large", "xlm-roberta-base", "xlm-roberta-large"]),
              ("Decoder LLMs", ["qwen3.5-0.8b-base", "qwen3.5-2b-base", "qwen3.5-4b-base", "qwen3.5-9b-base",
                                "qwen3.8-27b", "olmo-3-7b", "olmo-3-32b", "muse-glimmer-30b"]),
              ("Speech", ["wavlm-large", "hubert-large", "whisper-large-v3", "qwen2-audio-7b",
                          "qwen2-audio-7b-instruct"])]
    reg = load_registry()
    return [{"group": group, "model": display(name), "hf_id": m.id, "params": params_label(m.extra.get("params")),
             "revision": m.revision} for group, names in groups for name in names if (m := reg.get(name))]


# ----------------------------------------------------------------------------- figure data


def layer_curve_rows(root: Path, model: str, cond: str) -> list[dict]:
    """Test accuracy (2019 subset) per layer of the LDA probe, with relative depth."""
    p = root / "probing" / model / cond / "records.jsonl"
    if not p.exists():
        return []
    acc: dict[tuple[int, int], float] = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "test" and r["condition"]["probe"] == "lda":
            acc[(r["layer"], r["task"])] = r["metrics"]["legacy_2019"]["accuracy"]
    layers = sorted({layer for layer, _ in acc})
    last = max(layers) or 1
    return [{"layer": layer, "depth": round(layer / last, 4), "acc2": round(acc.get((layer, 2), float("nan")), 3),
             "acc3": round(acc.get((layer, 3), float("nan")), 3)} for layer in layers]


SIZES = {"qwen3.5-0.8b-base": 0.8, "qwen3.5-2b-base": 2, "qwen3.5-4b-base": 4, "qwen3.5-9b-base": 9,
         "qwen3.8-27b": 27}

LAYER_SERIES = [("bert-base-uncased", "plain-first-natural"), ("muse-glimmer-30b", "echo-last-natural"),
                ("qwen3.8-27b", "echo-last-natural"), ("wavlm-large", "audio-mean"),
                ("hubert-large", "audio-mean"), ("whisper-large-v3", "audio-mean"),
                ("qwen2-audio-7b", "audio-mean-llm"), ("qwen2-audio-7b", "audio-mean-encoder")]


def figure_data(rows, root: Path, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for model, cond in LAYER_SERIES:
        r = layer_curve_rows(root, model, cond)
        if r:
            suffix = {"audio-mean-llm": "", "audio-mean-encoder": "_encoder"}.get(cond, "")
            name = f"layers_{re.sub(r'[^a-z0-9]+', '_', model)}{suffix}.csv"
            write_csv(out / name, r, ["layer", "depth", "acc2", "acc3"])
            written.append(name)
    size = []
    for model, b in SIZES.items():
        for fam, tag in (("probe_linear", "lda"), ("probe_nonlinear", "bilstm")):
            d = {t: next((r for r in rows if r["family"] == fam and r["model"] == model and r["task"] == t
                          and r["condition"].startswith("echo") and (fam == "probe_linear" or r["system"].endswith("bilstm"))),
                         None) for t in (2, 3)}
            if d[3]:
                size.append({"params_b": b, "probe": tag, "acc2": d[2] and round(d[2]["test_2019"], 3),
                             "acc3": round(d[3]["test_2019"], 3)})
    for tag in ("lda", "bilstm"):
        pts = [s for s in size if s["probe"] == tag]
        if pts:
            write_csv(out / f"size_{tag}.csv", pts, ["params_b", "acc2", "acc3"])
            written.append(f"size_{tag}.csv")
    curve = C.icl_curve(root)
    if curve:
        write_csv(out / "icl_curve.csv", [{**p, "shots_plot": max(p["shots"], 1)} for p in curve],
                  ["shots", "shots_plot", "prefix_tokens", "acc2", "acc3", "auc"])
        written.append("icl_curve.csv")
    return written


# ----------------------------------------------------------------------------- probed against prompted


PAIRS = [
    ("Qwen3.8 27B", "text", [
        ("linear probe", lambda r: r["family"] == "probe_linear" and r["model"] == "qwen3.8-27b"
         and r["condition"] == "echo-last-natural"),
        ("BiLSTM classifier", lambda r: r["family"] == "probe_nonlinear" and r["model"] == "qwen3.8-27b"
         and r["condition"] == "echo-last-natural" and r["system"].endswith("bilstm")),
        ("prompted, no examples", lambda r: r["family"] == "icl" and r["model"] == "qwen3.8-27b"
         and r["condition"].endswith("-k0")),
        ("prompted, 500 examples", lambda r: r["family"] == "icl" and r["model"] == "qwen3.8-27b"
         and r["condition"].endswith("-k500")),
    ]),
    ("Qwen2-Audio 7B", "audio", [
        ("linear probe (LM)", lambda r: r["family"] == "probe_linear" and r["model"] == "qwen2-audio-7b"
         and r["condition"] == "audio-mean-llm"),
        ("BiLSTM classifier (LM)", lambda r: r["family"] == "probe_nonlinear" and r["model"] == "qwen2-audio-7b"
         and r["condition"] == "audio-mean-llm" and r["system"].endswith("bilstm")),
        ("prompted (Instruct)", lambda r: r["family"] == "audio_prompting" and r["condition"] == "linguistic-audio"),
    ]),
]


def paired_rows(rows):
    out = []
    for model, modality, items in PAIRS:
        for label, pred in items:
            r = {t: next((x for x in rows if x["task"] == t and pred(x)), None) for t in (2, 3)}
            if r[2] is None and r[3] is None:
                continue  # not run (yet)
            out.append((model, modality, label, r[2], r[3]))
    return out


def main_table_csv(trows) -> list[dict]:
    return [{"group": g, "system": label, "modality": (r3 or r2)["modality"],
             "acc2": rnd(r2 and r2["test_2019"]), "sd2": rnd(r2 and r2["sd"]),
             "acc3": rnd(r3 and r3["test_2019"]), "sd3": rnd(r3 and r3["sd"])} for g, label, r2, r3 in trows]


def paired_csv(prows) -> list[dict]:
    return [{"model": model, "modality": modality, "access": label,
             "acc2": rnd(r2 and r2["test_2019"]), "sd2": rnd(r2 and r2["sd"]),
             "acc3": rnd(r3 and r3["test_2019"]), "sd3": rnd(r3 and r3["sd"])}
            for model, modality, label, r2, r3 in prows]


def write_all(root: Path, extra_numbers: dict | None = None) -> dict:
    rows = C.collect(root)
    comp = C.comparison(root)
    out, tab = root / "analysis", tables_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["family", "system", "model", "condition", "modality", "task", "test_2019", "test_full",
              "macro_f1", "sd", "n", "dev", "note"]
    write_csv(out / "all_results.csv", rows, fields)
    trows = main_table_rows(rows)
    (out / "main_table.md").write_text(main_table_md(trows), encoding="utf-8")
    write_csv(tab / "main_table.csv", main_table_csv(trows),
              ["group", "system", "modality", "acc2", "sd2", "acc3", "sd3"])
    write_csv(tab / "comparison.csv", comparison_rows(comp, dev_scores(rows), root),
              ["kind", "system", "modality", "acc2", "acc3", "p_vs_best_text2", "p_vs_best_text3", "ci3_low", "ci3_high"])
    write_csv(tab / "paired.csv", paired_csv(paired_rows(rows)),
              ["model", "modality", "access", "acc2", "sd2", "acc3", "sd3"])
    write_csv(tab / "models.csv", models_rows(), ["group", "model", "hf_id", "params", "revision"])
    nums = numbers(rows, comp, root, extra_numbers)
    write_csv(out / "numbers.csv", [{"key": k, "value": nums[k]} for k in sorted(nums)], ["key", "value"])
    figs = figure_data(rows, root, figures_dir(root))
    return {"rows": len(rows), "table_rows": len(trows), "numbers": len(nums), "figure_files": figs}
