"""Two analyses of what the models predict, beyond accuracy.

* ``reader_tracking``: whose reading does a model predict? Some test texts are also read
  by another speaker in the development set. A text model sees the same input for both
  readings, so it can only predict a reading of the text; a speech model hears one of them.
  For every such test reading we count how often a model's labels agree with the reader
  it was evaluated on and with the other reader. The development readings were not used
  for training, so no model has seen their labels.
* ``prompt_profile``: what do prompted LLMs say? For gold labels, probes, prompted LLMs
  and a speech model: how often the highest level is given to content words, function
  words and the last word of the utterance, and the accuracy on each. The same profile
  for the three instruction variants on the development subset shows whether a pattern
  comes from our instructions.

Writes ``paper/generated/reader_table.tex`` and ``paper/generated/profile_table.tex`` and
returns numbers for ``numbers.tex``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from prosody_llms.analysis.errors import FUNCTION_WORDS, read_predictions_gz, reader_groups
from prosody_llms.data.corpus import LABEL_NA, read_split

READER_SYSTEMS = [  # (label, kind, tag, path under results/)
    ("Whisper large-v3", "speech", "whisper",
     "probing_nonlinear/whisper-large-v3/audio-mean/predictions_test_3way_bilstm.tsv.gz"),
    ("WavLM-large", "speech", "wavlm", "probing_nonlinear/wavlm-large/audio-mean/predictions_test_3way_bilstm.tsv.gz"),
    ("ModernBERT-large", "text", "modernbert",
     "probing_nonlinear/modernbert-large/plain-first-natural/predictions_test_3way_bilstm.tsv.gz"),
    ("Qwen3.8 27B", "text", "qwen",
     "probing_nonlinear/qwen3.8-27b/echo-last-natural/predictions_test_3way_bilstm.tsv.gz"),
    ("Qwen3.8 27B, 500 examples", "prompted", "qwen-icl", "icl/qwen3.8-27b/test/linguistic-k500/predictions.tsv.gz"),
    ("DeepSeek-V4-Pro, 5 examples", "prompted", "deepseek",
     "prompting/deepseek-v4-pro/test/linguistic-lines-3way-k5/predictions.tsv.gz"),
]

PATHS = {  # prediction files (test set) by system tag
    "whisper": "probing_nonlinear/whisper-large-v3/audio-mean/predictions_test_3way_bilstm.tsv.gz",
    "modernbert": "probing_nonlinear/modernbert-large/plain-first-natural/predictions_test_3way_bilstm.tsv.gz",
    "qwen": "probing_nonlinear/qwen3.8-27b/echo-last-natural/predictions_test_3way_bilstm.tsv.gz",
    "qwen-icl": "icl/qwen3.8-27b/test/linguistic-k500/predictions.tsv.gz",
    "deepseek": "prompting/deepseek-v4-pro/test/linguistic-lines-3way-k5/predictions.tsv.gz",
    "gpt4o": "prompting/gpt-4o/test/linguistic-lines-3way-k5/predictions.tsv.gz",
}

PROFILE_SYSTEMS = [  # (label, tag, path under results/); gold first
    ("Gold labels", "gold", None),
    ("ModernBERT-large, BiLSTM classifier", "modernbert", PATHS["modernbert"]),
    ("Qwen3.8 27B, BiLSTM classifier", "qwen", PATHS["qwen"]),
    ("Qwen3.8 27B, prompted (500 ex.)", "qwen-icl", PATHS["qwen-icl"]),
    ("DeepSeek-V4-Pro, prompted (5 ex.)", "deepseek", PATHS["deepseek"]),
    ("GPT-4o, prompted (5 ex.)", "gpt4o", PATHS["gpt4o"]),
    ("Whisper large-v3, BiLSTM classifier", "whisper", PATHS["whisper"]),
]
DEV_MODELS = (("gpt4o", "gpt-4o"), ("deepseek", "deepseek-v4-pro"), ("mistral", "mistral-large-3"))
VARIANTS = ("minimal", "linguistic", "narrator")


def _scored(u) -> list[int]:
    g = u.labels(3)
    return [i for i, t in enumerate(u.tokens) if not t.is_punct and g[i] != LABEL_NA]


def reader_tracking(root: Path, data_dir: Path) -> tuple[list[dict], dict]:
    splits = {s: read_split(data_dir / f"{s}.txt") for s in ("train_360", "dev", "test")}
    systems = [(lab, kind, tag, read_predictions_gz(root / p)) for lab, kind, tag, p in READER_SYSTEMS
               if (root / p).exists()]
    if not systems:
        return [], {}
    counts = {tag: [0, 0] for _, _, tag, _ in systems}
    words = readers_agree = 0
    test_ids = set()
    for g in reader_groups(splits):
        tests = [u for s, u in g if s == "test"]
        devs = [u for s, u in g if s == "dev"]
        for u in tests:
            for v in devs:
                if v.speaker == u.speaker:
                    continue
                test_ids.add(u.utt_id)
                gu, gv = u.labels(3), v.labels(3)
                for i in _scored(u):
                    if gv[i] == LABEL_NA:
                        continue
                    words += 1
                    readers_agree += gu[i] == gv[i]
                    for _, _, tag, preds in systems:
                        p = preds[u.utt_id][i]
                        counts[tag][0] += p == gu[i]
                        counts[tag][1] += p == gv[i]
    if not words:
        return [], {}
    rows, nums = [], {"rt-words": words, "rt-utterances": len(test_ids), "rt-readers": 100.0 * readers_agree / words}
    for lab, kind, tag, _ in systems:
        own, other = (100.0 * c / words for c in counts[tag])
        rows.append({"system": lab, "kind": kind, "own": own, "other": other, "diff": own - other})
        nums.update({f"rt-{tag}-own": own, f"rt-{tag}-other": other, f"rt-{tag}-diff": own - other})
    return rows, nums


def _profile(utts, preds) -> dict:
    c = Counter()
    for u in utts:
        if preds is not None and u.utt_id not in preds:
            continue
        idx = _scored(u)
        if not idx:
            continue
        g = u.labels(3)
        p = g if preds is None else preds[u.utt_id]
        for i in idx:
            groups = ["function" if u.tokens[i].word.lower() in FUNCTION_WORDS else "content"]
            if i == idx[-1]:
                groups.append("last")
            for grp in groups:
                c[grp] += 1
                c[grp, "top"] += p[i] == "2"
                c[grp, "acc"] += p[i] == g[i]
    return {f"{grp}-{k}": 100.0 * c[grp, k] / c[grp] for grp in ("content", "function", "last") for k in ("top", "acc")
            if c[grp]}


def prompt_profile(root: Path, data_dir: Path) -> tuple[list[dict], dict]:
    test = read_split(data_dir / "test.txt")
    rows, nums = [], {}
    for lab, tag, path in PROFILE_SYSTEMS:
        if path is not None and not (root / path).exists():
            continue
        prof = _profile(test, None if path is None else read_predictions_gz(root / path))
        rows.append({"system": lab, "tag": tag, **prof})
        nums.update({f"pp-{tag}-{k}": v for k, v in prof.items()})
    for tag in ("qwen-icl", "deepseek", "gpt4o"):  # loss against the probe on the same model or the LLM probe
        for grp in ("content", "function"):
            if f"pp-{tag}-{grp}-acc" in nums and f"pp-qwen-{grp}-acc" in nums:
                nums[f"pp-{tag}-{grp}-gap"] = nums[f"pp-qwen-{grp}-acc"] - nums[f"pp-{tag}-{grp}-acc"]
    from prosody_llms.prompting.run import load_split

    dev = load_split(data_dir, "dev500")
    tops = []
    for tag, model in DEV_MODELS:
        for v in VARIANTS:
            p = root / "prompting" / model / "dev500" / f"{v}-lines-3way-k0" / "predictions.tsv.gz"
            if p.exists():
                prof = _profile(dev, read_predictions_gz(p))
                nums[f"pp-dev-{tag}-{v}-last-top"] = prof["last-top"]
                tops.append(prof["last-top"])
    if tops:
        nums["pp-dev-gold-last-top"] = _profile(dev, None)["last-top"]
        minimal = [nums[k] for k in nums if k.startswith("pp-dev-") and k.endswith("-minimal-last-top")]
        if minimal:
            nums["pp-dev-minimal-last-top-min"] = min(minimal)
            nums["pp-dev-minimal-last-top-max"] = max(minimal)
    return rows, nums


def reader_tex(rows: list[dict], nums: dict) -> str:
    if not rows:
        return "\\begin{tabular}{l}\\textit{(results pending)}\\end{tabular}\n"
    kinds = {"speech": "\\textsc{a}", "text": "\\textsc{t}", "prompted": "\\textsc{p}"}
    out = ["% Generated by python -m prosody_llms.analysis; do not edit by hand.",
           "\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
           " & System & Own reader & Other reader & $\\Delta$ \\\\", "\\midrule"]
    for r in rows:
        out.append(f"{kinds[r['kind']]} & {r['system']} & {r['own']:.1f} & {r['other']:.1f} & {r['diff']:+.1f} \\\\")
    out += ["\\midrule", f" & The two readers & \\multicolumn{{2}}{{c}}{{{nums['rt-readers']:.1f}}} & \\\\",
            "\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def profile_tex(rows: list[dict]) -> str:
    if not rows:
        return "\\begin{tabular}{l}\\textit{(results pending)}\\end{tabular}\n"
    out = ["% Generated by python -m prosody_llms.analysis; do not edit by hand.",
           "\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
           " & \\multicolumn{3}{c}{Labelled 2 (\\%)} & \\multicolumn{2}{c}{Accuracy (\\%)} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(l){5-6}",
           "System & content & function & last & content & function \\\\", "\\midrule"]
    for r in rows:
        acc = ("--", "--") if r["tag"] == "gold" else (f"{r['content-acc']:.1f}", f"{r['function-acc']:.1f}")
        out.append(f"{r['system']} & {r['content-top']:.1f} & {r['function-top']:.1f} & {r['last-top']:.1f} & "
                   f"{acc[0]} & {acc[1]} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def compute_summary(root: Path) -> dict:
    """GPU time of the runs reported in the paper, from the run records (``time_s``)."""
    import json

    def seconds(d):
        t = d.get("time_s", d.get("train_time_s"))
        if isinstance(t, dict):
            t = t.get("total", sum(v for v in t.values() if isinstance(v, (int, float))))
        return t if isinstance(t, (int, float)) else 0.0

    pats = {"probe": ["probing/*/*/result.json", "probing_nonlinear/*/*/result.json"],
            "finetune": ["finetune/*/*/seed*/result.json", "repro_2019/*/*/seed*/result.json"],
            "prompt": ["icl/*/*/*/result.json", "audio_prompting/*/*/*/result.json"]}
    out, total, longest, runs = {}, 0.0, 0.0, 0
    for key, globs in pats.items():
        hours = 0.0
        for g in globs:
            for p in root.glob(g):
                d = json.loads(p.read_text(encoding="utf-8"))
                if str((d.get("environment") or {}).get("device", "cpu")) in ("", "cpu"):
                    continue
                t = seconds(d)
                hours += t / 3600
                longest = max(longest, t / 3600)
                runs += 1
        out[f"gpu-hours-{key}"] = hours
        total += hours
    out.update({"gpu-hours-total": total, "gpu-hours-longest": longest, "gpu-runs": runs})
    return out


def write_all(root: Path, data_dir: Path, paper_dir: Path) -> dict:
    rt_rows, rt_nums = reader_tracking(root, data_dir)
    pp_rows, pp_nums = prompt_profile(root, data_dir)
    gen = paper_dir / "generated"
    (gen / "reader_table.tex").write_text(reader_tex(rt_rows, rt_nums), encoding="utf-8")
    (gen / "profile_table.tex").write_text(profile_tex(pp_rows), encoding="utf-8")
    ex_tex, _ = two_reader_example(root, data_dir)
    (gen / "two_readers.tex").write_text(ex_tex, encoding="utf-8")
    lex_rows, lex_nums = lexical_default(root, data_dir)
    (gen / "lexical_table.tex").write_text(lexical_tex(lex_rows), encoding="utf-8")
    extra = {**lex_nums, **rethreshold(root, data_dir), **learning_curve(root, gen), **discourse_context(root), **probe_controls(root), **format_check(root), **matched_curve(root, data_dir, gen)}
    cl_rows, cl_nums = class_results(root, data_dir)
    (gen / "class_table.tex").write_text(class_tex(cl_rows), encoding="utf-8")
    (gen / "behaviour_table.tex").write_text(behaviour_tex(pp_rows, cl_rows), encoding="utf-8")
    (gen / "confusion.tex").write_text(confusion_tex(cl_rows), encoding="utf-8")
    return {**rt_nums, **pp_nums, **cl_nums, **extra, **compute_summary(root)}


# ----------------------------------------------------------------------------- per-class results

CLASS_SYSTEMS = [  # (label, tag, kind, path under results/)
    ("ModernBERT-large, BiLSTM classifier", "modernbert", "t", PATHS["modernbert"]),
    ("Qwen3.8 27B, BiLSTM classifier", "qwen", "t", PATHS["qwen"]),
    ("Qwen3.8 27B, prompted (500 ex.)", "qwen-icl", "p", PATHS["qwen-icl"]),
    ("DeepSeek-V4-Pro, prompted (5 ex.)", "deepseek", "p", PATHS["deepseek"]),
    ("GPT-4o, prompted (5 ex.)", "gpt4o", "p", PATHS["gpt4o"]),
    ("Whisper large-v3, BiLSTM classifier", "whisper", "a", PATHS["whisper"]),
]
CONFUSION_TAGS = (("qwen", "Qwen3.8 27B, probe"), ("qwen-icl", "Qwen3.8 27B, prompted"), ("whisper", "Whisper, probe"))


def class_results(root: Path, data_dir: Path) -> tuple[list[dict], dict]:
    """Macro-F1, per-class recall and row-normalised confusion matrices (3-way, the 2019
    test subset: all test utterances but the last)."""
    from prosody_llms.eval.metrics import confusion, macro_f1, per_class

    test = read_split(data_dir / "test.txt")[:-1]
    gold = [lab for u in test for lab in u.labels(3)]
    rows, nums = [], {}
    for label, tag, kind, path in CLASS_SYSTEMS:
        if not (root / path).exists():
            continue
        preds = read_predictions_gz(root / path)
        pred = [lab for u in test for lab in preds[u.utt_id]]
        pc = per_class(gold, pred)
        conf = confusion(gold, pred)
        norm = {g: {p: 100.0 * conf[g].get(p, 0) / sum(conf[g].values()) for p in "012"} for g in "012"}
        row = {"system": label, "tag": tag, "kind": kind, "macro_f1": macro_f1(gold, pred),
               **{f"recall-{k}": pc[k]["recall"] for k in "012"},
               **{f"precision-{k}": pc[k]["precision"] for k in "012"}, "confusion": norm}
        rows.append(row)
        nums[f"cls-{tag}-f1"] = row["macro_f1"]
        for k in "012":
            nums[f"cls-{tag}-recall-{k}"] = pc[k]["recall"]
            for p in "012":
                nums[f"cls-{tag}-conf-{k}{p}"] = norm[k][p]
    if "cls-qwen-f1" in nums and "cls-qwen-icl-f1" in nums:
        nums["icl-probe-f1-gap"] = nums["cls-qwen-f1"] - nums["cls-qwen-icl-f1"]
    return rows, nums


def class_tex(rows: list[dict]) -> str:
    if not rows:
        return "\\begin{tabular}{l}\\textit{(results pending)}\\end{tabular}\n"
    kinds = {"t": "\\textsc{t}", "p": "\\textsc{p}", "a": "\\textsc{a}"}
    out = ["% Generated by python -m prosody_llms.analysis; do not edit by hand.",
           "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
           " & & & \\multicolumn{3}{c}{Recall (\\%)} \\\\", "\\cmidrule(l){4-6}",
           " & System & Macro-F1 & 0 & 1 & 2 \\\\", "\\midrule"]
    for r in rows:
        out.append(f"{kinds[r['kind']]} & {r['system']} & {r['macro_f1']:.1f} & {r['recall-0']:.1f} & "
                   f"{r['recall-1']:.1f} & {r['recall-2']:.1f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def confusion_tex(rows: list[dict], cell: float = 0.62) -> str:
    """Row-normalised confusion matrices side by side, as a plain TikZ picture."""
    by_tag = {r["tag"]: r for r in rows}
    mats = [(by_tag[t], title) for t, title in CONFUSION_TAGS if t in by_tag]
    if not mats:
        return "\\textit{(results pending)}\n"
    out = ["% Generated by python -m prosody_llms.analysis; do not edit by hand.",
           "\\begin{tikzpicture}[x=%.2fcm, y=-%.2fcm, font=\\footnotesize]" % (cell, cell)]
    for k, (r, title) in enumerate(mats):
        x0 = k * 4.6
        out.append(f"\\node[anchor=south] at ({x0 + 1.5},-0.15) {{{title}}};")
        for i, g in enumerate("012"):
            if k == 0:
                out.append(f"\\node[anchor=east] at ({x0 - 0.05},{i + 0.5}) {{{g}}};")
            for j, p in enumerate("012"):
                v = r["confusion"][g][p]
                rr, gg, bb = _shade_pct(v)
                txt = "white" if (rr + gg + bb) / 3 < 140 else "black"
                out.append(f"\\fill[fill={{rgb,255:red,{rr};green,{gg};blue,{bb}}}] ({x0 + j},{i}) rectangle ++(1,1);"
                           f"\\node[text={txt}] at ({x0 + j + 0.5},{i + 0.5}) {{{v:.0f}}};")
        for j, p in enumerate("012"):
            out.append(f"\\node at ({x0 + j + 0.5},3.35) {{{p}}};")
        out.append(f"\\node at ({x0 + 1.5},3.95) {{predicted}};")
    out.append("\\node[rotate=90] at (-0.75,1.5) {gold};")
    out.append("\\end{tikzpicture}")
    return "\n".join(out) + "\n"


def _shade_pct(v: float) -> tuple[int, int, int]:
    ramp = [(240, 246, 253), (134, 182, 239), (42, 120, 214), (16, 66, 129)]
    t = min(max(v / 100.0, 0.0), 1.0) * (len(ramp) - 1)
    i = min(int(t), len(ramp) - 2)
    f = t - i
    return tuple(round(a + (b - a) * f) for a, b in zip(ramp[i], ramp[i + 1]))


# ----------------------------------------------------------------------------- two readers, one text

def two_reader_example(root: Path, data_dir: Path) -> tuple[str, dict]:
    """One test sentence that another speaker also reads in the development set, typeset
    by prominence: the test reader, the speech model hearing that reader, the other reader,
    and the text model (whose prediction is the same for both readings). Chosen as the
    sentence of 6-14 words where the speech model's preference for its own reader is
    largest (ties broken by utterance id)."""
    from prosody_llms.analysis.visuals import _tex

    splits = {s: read_split(data_dir / f"{s}.txt") for s in ("train_360", "dev", "test")}
    sp, tp = (root / PATHS["whisper"]), (root / PATHS["modernbert"])
    if not (sp.exists() and tp.exists()):
        return "\\textit{(results pending)}\n", {}
    speech, text = read_predictions_gz(sp), read_predictions_gz(tp)
    best = None
    for g in reader_groups(splits):
        for u in (u for s, u in g if s == "test"):
            for v in (v for s, v in g if s == "dev" and v.speaker != u.speaker):
                idx = [i for i in _scored(u) if v.labels(3)[i] != LABEL_NA]
                if not 6 <= len(idx) <= 14:
                    continue
                gu, gv, ps = u.labels(3), v.labels(3), speech[u.utt_id]
                differ = sum(gu[i] != gv[i] for i in idx)
                pref = sum(ps[i] == gu[i] for i in idx) - sum(ps[i] == gv[i] for i in idx)
                key = (pref, differ, u.utt_id)
                if differ >= 3 and (best is None or key > best[0]):
                    best = (key, u, v)
    if best is None:
        return "\\textit{(no example)}\n", {}
    _, u, v = best
    style = {"0": "\\textcolor{gray}{%s}", "1": "\\textbf{%s}", "2": "\\textbf{\\large %s}"}

    def row(labels):
        parts = []
        for t, lab in zip(u.tokens, labels):
            w = _tex(t.word)
            if t.is_punct:
                if parts:
                    parts[-1] += w
                continue
            parts.append(style.get(lab, "%s") % w)
        return " ".join(parts)

    rows = [("Speaker A", u.labels(3)), ("Speech model, hearing A", speech[u.utt_id]),
            ("Speaker B", v.labels(3)), ("Text model", text[u.utt_id])]
    out = [f"% Generated by python -m prosody_llms.analysis; do not edit by hand. {_tex(u.utt_id)} / {_tex(v.utt_id)}",
           "\\begin{tabular}{@{}ll@{}}"]
    out += [f"{name} & {row(labs)} \\\\" for name, labs in rows]
    out.append("\\end{tabular}")
    return "\n".join(out) + "\n", {"ex2-utt": u.utt_id.replace("_", "-"), "ex2-other": v.utt_id.replace("_", "-")}


# ----------------------------------------------------------------------------- the lexical default

LEXICAL_SYSTEMS = [  # (label, tag, kind, path under results/)
    ("ModernBERT-large, BiLSTM classifier", "modernbert", "t", PATHS["modernbert"]),
    ("Qwen3.8 27B, BiLSTM classifier", "qwen", "t", PATHS["qwen"]),
    ("Qwen3.8 27B, prompted (500 ex.)", "qwen-icl", "p", PATHS["qwen-icl"]),
    ("DeepSeek-V4-Pro, prompted (5 ex.)", "deepseek", "p", PATHS["deepseek"]),
    ("GPT-4o, prompted (5 ex.)", "gpt4o", "p", PATHS["gpt4o"]),
]


def word_majority(data_dir: Path):
    """Each training word's most frequent 3-way label and its share of prominent tokens."""
    import collections

    counts = collections.defaultdict(collections.Counter)
    for u in read_split(data_dir / "train_360.txt"):
        for t, lab in zip(u.tokens, u.labels(3)):
            if lab != LABEL_NA:
                counts[t.word.lower()][lab] += 1
    total = collections.Counter()
    for c in counts.values():
        total.update(c)
    prior = (total["1"] + total["2"]) / sum(total.values())
    label = {w: c.most_common(1)[0][0] for w, c in counts.items()}
    share = {w: (c["1"] + c["2"]) / sum(c.values()) for w, c in counts.items()}
    return (lambda w: label.get(w.lower(), "0")), (lambda w: share.get(w.lower(), prior))


def lexical_default(root: Path, data_dir: Path) -> tuple[list[dict], dict]:
    """How far each system follows the word-majority lookup on the test set, and how often it
    is right where it departs from it."""
    lookup, _ = word_majority(data_dir)
    test = read_split(data_dir / "test.txt")[:-1]
    rows, nums = [], {}
    for label, tag, kind, path in LEXICAL_SYSTEMS:
        if not (root / path).exists():
            continue
        preds = read_predictions_gz(root / path)
        n = agree = dep = dep_right = dep_wm_right = 0
        for u in test:
            g, p = u.labels(3), preds[u.utt_id]
            for i, t in enumerate(u.tokens):
                if g[i] == LABEL_NA:
                    continue
                n += 1
                w = lookup(t.word)
                if p[i] == w:
                    agree += 1
                else:
                    dep += 1
                    dep_right += p[i] == g[i]
                    dep_wm_right += w == g[i]
        r = {"system": label, "tag": tag, "kind": kind, "agree": 100.0 * agree / n, "departs": 100.0 * dep / n,
             "dep_right": 100.0 * dep_right / dep, "dep_wm_right": 100.0 * dep_wm_right / dep}
        rows.append(r)
        nums.update({f"lex-{tag}-{k}": r[k] for k in ("agree", "departs", "dep_right", "dep_wm_right")})
    return rows, nums


def lexical_tex(rows: list[dict]) -> str:
    if not rows:
        return "\\begin{tabular}{l}\\textit{(results pending)}\\end{tabular}\n"
    kinds = {"t": "\\textsc{t}", "p": "\\textsc{p}"}
    out = ["% Generated by python -m prosody_llms.analysis; do not edit by hand.",
           "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
           " & & & & \\multicolumn{2}{c}{Right where it departs} \\\\", "\\cmidrule(l){5-6}",
           " & System & Same as lookup & Departs & system & lookup \\\\", "\\midrule"]
    for r in rows:
        out.append(f"{kinds[r['kind']]} & {r['system']} & {r['agree']:.1f} & {r['departs']:.1f} & "
                   f"{r['dep_right']:.1f} & {r['dep_wm_right']:.1f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def rethreshold(root: Path, data_dir: Path) -> dict:
    """Whether better cut-offs on the prompted model's own label probabilities help: on the
    500 development utterances, cut-offs on the expected level P(1) + 2 P(2) are chosen on one
    half and scored on the other (two folds). Also the AUC of P(prominent) against that of the
    word-majority share of prominent tokens."""
    import gzip
    import json

    import numpy as np

    from prosody_llms.prompting.prompts import word_items
    from prosody_llms.prompting.run import load_split, roc_auc

    dev = load_split(data_dir, "dev500")
    lookup, share = word_majority(data_dir)
    nums = {}
    gold_all, lex_score, lex_pred = [], [], []
    for u in dev:
        g = u.labels(3)
        for n, (i, w) in enumerate(word_items(u), start=1):
            if g[i] != LABEL_NA:
                gold_all.append(int(g[i]))
                lex_score.append(share(w))
                lex_pred.append(int(lookup(w)))
    gold_all = np.array(gold_all)
    nums["rt-dev-lex-auc"] = f"{roc_auc(np.array(lex_score), gold_all > 0):.3f}"
    nums["rt-dev-wordmaj"] = 100.0 * float((np.array(lex_pred) == gold_all).mean())
    grid = np.linspace(0, 2, 81)
    for k in ("k0", "k500"):
        p = root / "icl" / "qwen3.8-27b" / "dev500" / f"linguistic-{k}" / "probabilities.json.gz"
        if not p.exists():
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            probs = json.load(f)
        fold, gold, argmax, ev, prom = [], [], [], [], []
        for ui, u in enumerate(dev):
            g = u.labels(3)
            for n, (i, _) in enumerate(word_items(u), start=1):
                if g[i] == LABEL_NA:
                    continue
                d = probs[u.utt_id][str(n)]
                fold.append(ui % 2)
                gold.append(int(g[i]))
                argmax.append(int(np.argmax(d)))
                ev.append(d[1] + 2 * d[2])
                prom.append(d[1] + d[2])
        fold, gold, argmax, ev, prom = map(np.array, (fold, gold, argmax, ev, prom))
        right = 0
        for f in (0, 1):
            m = fold == f
            best = max(((float((((ev[m] >= a).astype(int) + (ev[m] >= b)) == gold[m]).mean()), a, b)
                        for a in grid for b in grid if b > a))
            _, a, b = best
            o = ~m
            right += int((((ev[o] >= a).astype(int) + (ev[o] >= b)) == gold[o]).sum())
        nums[f"rt-dev-{k}-argmax"] = 100.0 * float((argmax == gold).mean())
        nums[f"rt-dev-{k}-cv"] = 100.0 * right / len(gold)
        nums[f"rt-dev-{k}-auc"] = f"{roc_auc(prom, gold > 0):.3f}"
    return nums


def learning_curve(root: Path, gen: Path) -> dict:
    """3-way test accuracy of BERT-base with growing training data: the linear probe on 500 and
    5,000 utterances of train_360, on train_100 and on train_360, the BiLSTM probe on the last
    two, and fine-tuning on 10% and all of train_360."""
    import csv
    import json

    nums, series = {}, {"linear": [], "bilstm": [], "finetune": []}
    base = root / "probing" / "bert-base-uncased"
    sizes = {"plain-first-natural-n500": 500, "plain-first-natural-n5000": 5000}
    for cond, n in sizes.items():
        p = base / cond / "result.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            v = next((s["test_2019"] for s in d["summary"] if s["task"] == 3 and s["probe"] == "lda"), None)
            if v is not None:
                series["linear"].append((n, v))
    rows = list(csv.DictReader(open(root / "analysis" / "all_results.csv", encoding="utf-8")))
    sizes = {"train_100": 33042, "train_360": 116263}
    for r in rows:
        if r["model"] != "bert-base-uncased" or r["task"] != "3" or not r["test_2019"]:
            continue
        n = sizes["train_100"] if "train_100" in r["condition"] else sizes["train_360"]
        if r["family"] == "probe_linear" and r["condition"].startswith("plain-first-natural"):
            series["linear"].append((n, float(r["test_2019"])))
        elif r["family"] == "probe_nonlinear" and r["system"].endswith("bilstm") and r["condition"].startswith("plain"):
            series["bilstm"].append((n, float(r["test_2019"])))
    summ = root / "repro_2019" / "summary.csv"
    if summ.exists():
        for r in csv.DictReader(open(summ, encoding="utf-8")):
            if r["n_classes"] == "3" and r["train_set"] == "train_360":
                frac = float(r["fraction_of_train_data"])
                acc = float(r["test_2019_mean"])
                series["finetune"].append((round(frac * sizes["train_360"]), acc))
    for name, pts in series.items():
        pts = sorted(set(pts))
        with open(gen / f"learning_{name}.csv", "w", encoding="utf-8") as f:
            f.write("utterances,acc3\n" + "".join(f"{n},{a:.3f}\n" for n, a in pts))
        for n, a in pts:
            nums[f"lc-{name}-{n}"] = a
    return nums


# ----------------------------------------------------------------------------- discourse context

DISCOURSE_MODELS = (("bert-base-uncased", "bert"), ("modernbert-large", "modernbert"))


def discourse_context(root: Path) -> dict:
    """Sentence-only probes against probes that also see the preceding sentences (prev1,
    prev3): best-layer linear probe (chosen on dev) and BiLSTM probe (mean over seeds), test
    accuracy on the 2019 subset. Numbers ``ctx-<model>-<context>-<probe>-<task>`` and the
    gain over the sentence-only run, ``ctx-<model>-<context>-<probe>-gain-<task>``."""
    import json

    nums = {}
    for model, tag in DISCOURSE_MODELS:
        for ctx in ("plain", "prev1", "prev3"):
            lin = root / "probing" / model / f"{ctx}-first-natural" / "result.json"
            if lin.exists():
                for s in json.loads(lin.read_text(encoding="utf-8")).get("summary", []):
                    if s["probe"] == "lda" and s.get("test_2019") is not None:
                        nums[f"ctx-{tag}-{ctx}-lin-{s['task']}"] = s["test_2019"]
            rec = root / "probing_nonlinear" / model / f"{ctx}-first-natural" / "records.jsonl"
            if rec.exists():
                accs: dict[int, list[float]] = {}
                for line in rec.read_text(encoding="utf-8").splitlines():
                    r = json.loads(line)
                    if r["split"] == "test" and r["condition"].get("probe") == "bilstm":
                        m = r["metrics"]
                        acc = (m.get("legacy_2019") or m["full"])["accuracy"]
                        accs.setdefault(r["task"], []).append(acc)
                for task, v in accs.items():
                    nums[f"ctx-{tag}-{ctx}-bilstm-{task}"] = sum(v) / len(v)
        for ctx in ("prev1", "prev3"):
            for probe in ("lin", "bilstm"):
                for task in (2, 3):
                    a, b = nums.get(f"ctx-{tag}-{ctx}-{probe}-{task}"), nums.get(f"ctx-{tag}-plain-{probe}-{task}")
                    if a is not None and b is not None:
                        nums[f"ctx-{tag}-{ctx}-{probe}-gain-{task}"] = (a - b) + 0.0 if abs(a - b) >= 0.05 else 0.0
    gains = [abs(v) for k, v in nums.items() if k.endswith("-gain-3")]
    if gains:
        nums["ctx-max-gain-3"] = max(max(gains), 0.1)  # reported as "at most": never below the rounding
    return nums


def behaviour_tex(pp_rows: list[dict], cl_rows: list[dict]) -> str:
    """One table for how each system labels: macro-F1 and recall per class, and the share of
    content words, function words and last words given the highest level."""
    if not pp_rows or not cl_rows:
        return "\\begin{tabular}{l}\\textit{(results pending)}\\end{tabular}\n"
    cl = {r["tag"]: r for r in cl_rows}
    pp = {r["tag"]: r for r in pp_rows}
    order = [("gold", "Gold labels", "")] + [(r["tag"], r["system"], r["kind"]) for r in cl_rows]
    kinds = {"t": "\\textsc{t}", "p": "\\textsc{p}", "a": "\\textsc{a}", "": ""}
    out = ["% Generated by python -m prosody_llms.analysis; do not edit by hand.",
           "\\begin{tabular}{@{}llrrrrrrr@{}}", "\\toprule",
           " & & & \\multicolumn{3}{c}{Recall (\\%)} & \\multicolumn{3}{c}{Labelled 2 (\\%)} \\\\",
           "\\cmidrule(lr){4-6}\\cmidrule(l){7-9}",
           " & System & Macro-F1 & 0 & 1 & 2 & content & function & last \\\\", "\\midrule"]
    for tag, name, kind in order:
        c, p = cl.get(tag), pp.get(tag)
        left = ("-- & -- & -- & --" if c is None else
                f"{c['macro_f1']:.1f} & {c['recall-0']:.1f} & {c['recall-1']:.1f} & {c['recall-2']:.1f}")
        right = ("-- & -- & --" if p is None else
                 f"{p['content-top']:.1f} & {p['function-top']:.1f} & {p['last-top']:.1f}")
        out.append(f"{kinds[kind]} & {name} & {left} & {right} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def probe_controls(root: Path) -> dict:
    """The two probe controls: Qwen3.8 27B probed after training on exactly the 500 prompt
    examples (``ctl-matched-<lin|bilstm>-<task>``), and a BiLSTM on BERT-base's embedding layer
    (``ctl-layer0-bilstm-<task>``); test accuracy on the 2019 subset."""
    import json

    def bilstm(rec: Path, nums: dict, key: str):
        if rec.exists():
            accs: dict[int, list[float]] = {}
            for line in rec.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r["split"] == "test" and r["condition"].get("probe") == "bilstm":
                    m = r["metrics"]
                    accs.setdefault(r["task"], []).append((m.get("legacy_2019") or m["full"])["accuracy"])
            for task, v in accs.items():
                nums[f"{key}-{task}"] = sum(v) / len(v)

    nums: dict = {}
    lin = root / "probing" / "qwen3.8-27b" / "echo-last-natural-ex500" / "result.json"
    if lin.exists():
        for s in json.loads(lin.read_text(encoding="utf-8")).get("summary", []):
            if s["probe"] == "lda" and s.get("test_2019") is not None:
                nums[f"ctl-matched-lin-{s['task']}"] = s["test_2019"]
    bilstm(root / "probing_nonlinear" / "qwen3.8-27b" / "echo-last-natural-ex500" / "records.jsonl", nums,
           "ctl-matched-bilstm")
    bilstm(root / "probing_layer0" / "bert-base-uncased" / "plain-first-natural" / "records.jsonl", nums,
           "ctl-layer0-bilstm")
    bilstm(root / "probing_layer0" / "qwen3.8-27b" / "echo-last-natural" / "records.jsonl", nums,
           "ctl-qwen-layer0-bilstm")
    bilstm(root / "probing_control" / "random-words-1024" / "plain-first-natural" / "records.jsonl", nums,
           "ctl-random-bilstm")
    return nums


def format_check(root: Path) -> dict:
    """Qwen2-Audio-Instruct answering freely (prosody_llms.audio.freegen, re-scored): how many
    answers give labels, only repeat the numbered words, or neither, and how many of the
    labelled words it calls non-prominent. ``fg-<audio|text>-...``."""
    import json

    nums = {}
    base = root / "audio_prompting" / "qwen2-audio-7b-instruct"
    for d in sorted(base.glob("test*/linguistic-freegen-*")):
        f = d / "rescored.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        tag = "text" if d.name.endswith("no_audio") else "audio"
        nums[f"fg-{tag}-n"] = r["utterances"]
        for k in ("labels", "enumeration", "other"):
            nums[f"fg-{tag}-{k}"] = r["kinds"].get(k, 0)
        nums[f"fg-{tag}-labelled"] = r["labelled_words"]
        nums[f"fg-{tag}-zero"] = r["labels_given"]["0"]
    return nums


def matched_curve(root: Path, data_dir: Path, gen: Path) -> dict:
    """Prompting against probing with the same labelled data, on the 500 development
    utterances used for the prompting curve: Qwen3.8 27B prompted with k examples, the linear
    probe on the same model trained on exactly those k utterances (best layer chosen on dev),
    and a word-majority lookup built from the same k utterances. Writes
    ``matched_curve.csv``; numbers ``mc-<k>-<probe|icl|lookup>-<task>`` and ``mc-all-probe-<task>``
    (the probe trained on all of train_360)."""
    import collections

    from prosody_llms.prompting.prompts import select_exemplars
    from prosody_llms.prompting.run import load_split

    dev = load_split(data_dir, "dev500")
    ids = {u.utt_id for u in dev}

    def score(path: Path, task: int):
        if not path.exists():
            return None
        preds = read_predictions_gz(path)
        n = right = 0
        for u in dev:
            if u.utt_id not in preds:
                return None
            g = u.labels(task)
            p = preds[u.utt_id]
            if task == 2 and "icl" in str(path):
                p = ["1" if x == "2" else x for x in p]
            for a, b in zip(g, p):
                if a != LABEL_NA:
                    n += 1
                    right += a == b
        return 100.0 * right / n

    train = read_split(data_dir / "train_360.txt")
    nums, rows = {}, []
    for k in (0, 10, 25, 100, 250, 500):
        row = {"shots": k}
        icl = root / "icl" / "qwen3.8-27b" / "dev500" / f"linguistic-k{k}" / "predictions.tsv.gz"
        for task in (3, 2):
            v = score(icl, task)
            if v is not None:
                nums[f"mc-{k}-icl-{task}"] = v
                if task == 3:
                    row["icl"] = v
            if k:
                pr = root / "probing" / "qwen3.8-27b" / f"echo-last-natural-ex{k}" / f"predictions_dev_{task}way_lda_best.tsv.gz"
                v = score(pr, task)
                if v is not None:
                    nums[f"mc-{k}-probe-{task}"] = v
                    if task == 3:
                        row["probe"] = v
        if k:  # word-majority lookup from the same k utterances
            cnt = collections.defaultdict(collections.Counter)
            for u in select_exemplars(train, k):
                for t, lab in zip(u.tokens, u.labels(3)):
                    if lab != LABEL_NA:
                        cnt[t.word.lower()][lab] += 1
            default = collections.Counter(lab for c in cnt.values() for lab in c.elements()).most_common(1)[0][0]
            n = right = 0
            for u in dev:
                for t, lab in zip(u.tokens, u.labels(3)):
                    if lab != LABEL_NA:
                        c = cnt.get(t.word.lower())
                        n += 1
                        right += (c.most_common(1)[0][0] if c else default) == lab
            nums[f"mc-{k}-lookup-3"] = row["lookup"] = 100.0 * right / n
        rows.append(row)
    for task in (3, 2):
        v = score(root / "probing" / "qwen3.8-27b" / "echo-last-natural" / f"predictions_dev_{task}way_lda_best.tsv.gz", task)
        if v is not None:
            nums[f"mc-all-probe-{task}"] = v
    with open(gen / "matched_curve.csv", "w", encoding="utf-8") as f:
        f.write("shots,shots_plot,icl,probe,lookup\n")
        for r in rows:
            f.write(f"{r['shots']},{max(r['shots'], 1)},{r.get('icl', 'nan')},{r.get('probe', 'nan')},{r.get('lookup', 'nan')}\n")
    del ids
    return nums
