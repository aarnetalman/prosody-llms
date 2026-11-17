"""Data for the visual summaries (the results page and paper figures).

* ``agreement``: how often two systems give a word the same 3-way label, on the words
  the text-against-audio comparison uses, for gold labels and a set of text, speech and
  prompted systems;
* ``accuracy``: every system's 3-way and 2-way test accuracy, grouped by how the model
  is used (prompted, probed, fine-tuned) and by input (text, speech);
* ``sentences``: a few test sentences with the gold, best text and best speech labels
  per word, chosen by a fixed hash among those where the two models differ on several
  words.

Writes ``results/analysis/visuals.json`` and, for the paper, ``paper/generated/agreement.csv``,
``agreement_heatmap.tex`` (the lower triangle as a TikZ picture) and ``example_sentence.tex``
(the shortest chosen sentence, typeset by its labels).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from prosody_llms.analysis.errors import common_mask, read_predictions_gz
from prosody_llms.data.corpus import LABEL_NA, read_split

SYSTEMS = [  # (label, group, path under results/)
    ("Gold labels", "gold", None),
    ("Whisper large-v3", "speech", "probing_nonlinear/whisper-large-v3/audio-mean/predictions_test_3way_bilstm.tsv.gz"),
    ("HuBERT-large", "speech", "probing_nonlinear/hubert-large/audio-mean/predictions_test_3way_bilstm.tsv.gz"),
    ("WavLM-large", "speech", "probing_nonlinear/wavlm-large/audio-mean/predictions_test_3way_bilstm.tsv.gz"),
    ("Qwen2-Audio encoder", "speech",
     "probing_nonlinear/qwen2-audio-7b/audio-mean-encoder/predictions_test_3way_bilstm.tsv.gz"),
    ("Qwen2-Audio LM", "speech", "probing_nonlinear/qwen2-audio-7b/audio-mean-llm/predictions_test_3way_bilstm.tsv.gz"),
    ("Duration only", "speech", "audio/baselines/duration/predictions_test_3way.tsv.gz"),
    ("ModernBERT-large", "text", "probing_nonlinear/modernbert-large/plain-first-natural/predictions_test_3way_bilstm.tsv.gz"),
    ("BERT-base", "text", "probing_nonlinear/bert-base-uncased/plain-first-natural/predictions_test_3way_bilstm.tsv.gz"),
    ("Qwen3.5 0.8B", "text", "probing_nonlinear/qwen3.5-0.8b-base/echo-last-natural/predictions_test_3way_bilstm.tsv.gz"),
    ("Qwen3.8 27B", "text", "probing_nonlinear/qwen3.8-27b/echo-last-natural/predictions_test_3way_bilstm.tsv.gz"),
    ("Muse Glimmer 30B", "text", "probing_nonlinear/muse-glimmer-30b/echo-last-natural/predictions_test_3way_bilstm.tsv.gz"),
    ("DeepSeek-V4-Pro (prompted)", "prompted", "prompting/deepseek-v4-pro/test/linguistic-lines-3way-k5/predictions.tsv.gz"),
    ("GPT-4o (prompted)", "prompted", "prompting/gpt-4o/test/linguistic-lines-3way-k5/predictions.tsv.gz"),
    ("Mistral Large 3 (prompted)", "prompted", "prompting/mistral-large-3/test/linguistic-lines-3way-k0/predictions.tsv.gz"),
]

GROUPS = {  # (family, modality) -> group shown in the accuracy view
    ("prompting", "text"): "Prompted LLM", ("icl", "text"): "Prompted LLM",
    ("audio_prompting", "audio"): "Prompted speech LLM", ("audio_prompting", "text"): "Prompted speech LLM",
    ("probe_linear", "text"): "Text, linear probe", ("probe_nonlinear", "text"): "Text, non-linear probe",
    ("finetune", "text"): "Text, fine-tuned", ("probe_linear", "audio"): "Speech, linear probe",
    ("probe_nonlinear", "audio"): "Speech, non-linear probe", ("audio_baseline", "audio"): "Reference",
    ("reference", "audio"): "Reference", ("baseline", "text"): "Reference",
}


def agreement(root: Path, data_dir: Path) -> dict:
    test = read_split(data_dir / "test.txt")
    mask = common_mask(root)
    systems = []
    for label, group, path in SYSTEMS:
        if path is None:
            preds = {u.utt_id: u.labels(3) for u in test}
        elif (root / path).exists():
            preds = read_predictions_gz(root / path)
        else:
            continue
        systems.append((label, group, preds))
    n = len(systems)
    agree = [[0] * n for _ in range(n)]
    total = 0
    for u in test:
        m = mask[u.utt_id] if mask else [not t.is_punct for t in u.tokens]
        gold = u.labels(3)
        labs = [s[2][u.utt_id] for s in systems]
        for i in range(len(u.tokens)):
            if not m[i] or gold[i] == LABEL_NA:
                continue
            total += 1
            for a in range(n):
                for b in range(a, n):
                    if labs[a][i] == labs[b][i]:
                        agree[a][b] += 1
    matrix = [[round(100.0 * agree[min(a, b)][max(a, b)] / total, 2) for b in range(n)] for a in range(n)]
    return {"labels": [s[0] for s in systems], "groups": [s[1] for s in systems], "matrix": matrix,
            "words": total}


def accuracy(root: Path) -> list[dict]:
    path = root / "analysis" / "all_results.csv"
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["test_2019"] or r["note"] == "test200" or "train_100" in r["condition"]:
                continue
            if r["family"] == "probe_nonlinear" and not r["system"].endswith("bilstm"):
                continue  # one non-linear probe per model: the BiLSTM
            group = GROUPS.get((r["family"], r["modality"]))
            if group is None:
                continue
            from prosody_llms.analysis.tables import display

            name = r["system"] if r["family"] in ("baseline", "reference", "audio_baseline") else display(r["model"], r["condition"])
            if r["family"] in ("prompting", "icl", "audio_prompting"):
                name += " (" + r["condition"].replace("linguistic-", "").replace("lines-3way-", "") + ")"
            out.append({"system": name, "group": group, "task": int(r["task"]), "accuracy": round(float(r["test_2019"]), 2),
                        "sd": round(float(r["sd"]), 2) if r["sd"] else None,
                        "derived": "derived" in r["system"]})
    return out


def sentences(root: Path, data_dir: Path, k: int = 4) -> list[dict]:
    p = root / "analysis" / "breakdown.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["task"] == "3")
    text_sys, audio_sys = row["text_system"], row["audio_system"]
    comp = list(csv.DictReader(open(root / "audio" / "comparison" / "comparison.csv", encoding="utf-8")))
    path = {r["system"]: r["path"] for r in comp}
    tp, ap = read_predictions_gz(root / path[text_sys]), read_predictions_gz(root / path[audio_sys])
    cands = []
    for u in read_split(data_dir / "test.txt"):
        words = [i for i, t in enumerate(u.tokens) if not t.is_punct]
        if not 8 <= len(words) <= 16:
            continue
        g = u.labels(3)
        text_wrong_audio_right = sum(tp[u.utt_id][i] != g[i] and ap[u.utt_id][i] == g[i] for i in words)
        if text_wrong_audio_right >= 3:
            cands.append((hashlib.sha256(u.utt_id.encode()).hexdigest(), u))
    out = []
    for _, u in sorted(cands)[:k]:
        out.append({"utt_id": u.utt_id, "words": [
            {"word": t.word, "punct": t.is_punct, "gold": g, "text": x, "speech": y}
            for t, g, x, y in zip(u.tokens, u.labels(3), tp[u.utt_id], ap[u.utt_id])],
            "text_system": text_sys, "speech_system": audio_sys})
    return out


def write_all(root: Path, data_dir: Path, paper_dir: Path) -> dict:
    agr = agreement(root, data_dir)
    data = {"agreement": agr, "accuracy": accuracy(root), "sentences": sentences(root, data_dir)}
    (root / "analysis" / "visuals.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    with open(paper_dir / "generated" / "agreement.csv", "w", encoding="utf-8") as f:
        f.write("x,y,agree\n")
        for a, row in enumerate(agr["matrix"]):
            for b, v in enumerate(row):
                f.write(f"{b},{a},{v}\n")
    (paper_dir / "generated" / "agreement_heatmap.tex").write_text(heatmap_tex(agr), encoding="utf-8")
    (paper_dir / "generated" / "example_sentence.tex").write_text(example_tex(data["sentences"]), encoding="utf-8")
    return {"systems": len(agr["labels"]), "accuracy_rows": len(data["accuracy"]),
            "sentences": len(data["sentences"]), "numbers": agreement_numbers(agr)}


# ----------------------------------------------------------------------------- paper figures

SHORT = {"Gold labels": "Gold", "Qwen2-Audio encoder": "Qwen2-Audio enc.", "Qwen2-Audio LM": "Qwen2-Audio LM",
         "DeepSeek-V4-Pro (prompted)": "DeepSeek-V4-Pro", "GPT-4o (prompted)": "GPT-4o",
         "Mistral Large 3 (prompted)": "Mistral Large 3"}
RAMP = [(240, 246, 253), (134, 182, 239), (42, 120, 214), (16, 66, 129)]  # light to dark blue
LO, HI = 55.0, 95.0


def _shade(v: float) -> tuple[int, int, int]:
    t = min(max((v - LO) / (HI - LO), 0.0), 1.0) * (len(RAMP) - 1)
    i = min(int(t), len(RAMP) - 2)
    f = t - i
    return tuple(round(a + (b - a) * f) for a, b in zip(RAMP[i], RAMP[i + 1]))


def _tex(s: str) -> str:
    return s.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%").replace("$", "\\$").replace("#", "\\#")


def heatmap_tex(agr: dict, cell: float = 0.42) -> str:
    """The lower triangle of the agreement matrix as a plain TikZ picture."""
    labels = [SHORT.get(x, x) for x in agr["labels"]]
    groups, m, n = agr["groups"], agr["matrix"], len(agr["labels"])
    out = ["% Generated by python -m prosody_llms.analysis; do not edit by hand.",
           "\\begin{tikzpicture}[x=%.2fcm, y=-%.2fcm, font=\\scriptsize]" % (cell, cell)]
    for a in range(1, n):
        out.append(f"\\node[anchor=east] at (-0.1, {a - 1 + 0.5}) {{{_tex(labels[a])}}};")
        for b in range(a):
            v = m[a][b]
            r, g, bl = _shade(v)
            txt = "white" if (r + g + bl) / 3 < 140 else "black"
            out.append(f"\\fill[fill={{rgb,255:red,{r};green,{g};blue,{bl}}}] ({b},{a - 1}) rectangle ++(1,1);"
                       f"\\node[text={txt}, font=\\tiny] at ({b + 0.5},{a - 0.5}) {{{v:.0f}}};")
    for b in range(n - 1):
        out.append(f"\\node[anchor=north east, rotate=60, inner sep=1pt] at ({b + 0.6}, {n - 1 + 0.1}) {{{_tex(labels[b])}}};")
    for i in range(1, n):  # lines between gold, speech, text and prompted systems
        if groups[i] != groups[i - 1]:
            out.append(f"\\draw[white, line width=1.2pt] ({i},{i - 1}) -- ({i},{n - 1});"
                       f"\\draw[white, line width=1.2pt] (0,{i - 1}) -- ({i},{i - 1});")
    out.append("\\end{tikzpicture}")
    return "\n".join(out) + "\n"


def agreement_numbers(agr: dict) -> dict:
    """Mean agreement within and between kinds of system (speech excludes the duration baseline)."""
    lab, grp, m = agr["labels"], agr["groups"], agr["matrix"]
    idx = {g: [i for i, x in enumerate(grp) if x == g and lab[i] != "Duration only"] for g in set(grp)}

    def mean(a, b):
        vals = [m[i][j] for i in idx.get(a, []) for j in idx.get(b, []) if i != j]
        return sum(vals) / len(vals) if vals else None

    out = {"vis-words": agr["words"], "vis-systems": len(lab)}
    for key, a, b in [("text-text", "text", "text"), ("speech-speech", "speech", "speech"),
                      ("text-speech", "text", "speech"), ("gold-text", "gold", "text"),
                      ("gold-speech", "gold", "speech"), ("gold-prompted", "gold", "prompted"),
                      ("text-prompted", "text", "prompted"), ("prompted-prompted", "prompted", "prompted")]:
        v = mean(a, b)
        if v is not None:
            out[f"agr-{key}"] = v
    return out


def example_tex(sents: list[dict]) -> str:
    """The shortest of the chosen sentences, each row typeset by its labels."""
    if not sents:
        return "\\textit{(results pending)}\n"
    s = min(sents, key=lambda x: (sum(not w["punct"] for w in x["words"]), x["utt_id"]))
    style = {"0": "\\textcolor{gray}{%s}", "1": "\\textbf{%s}", "2": "\\textbf{\\large %s}"}

    def row(key):
        parts = []
        for w in s["words"]:
            word = _tex(w["word"])
            if w["punct"]:
                if parts:
                    parts[-1] += word
                continue
            parts.append(style.get(w[key], "%s") % word)
        return " ".join(parts)

    rows = [("Reader", "gold"), ("Text", "text"), ("Speech", "speech")]
    out = ["% Generated by python -m prosody_llms.analysis; do not edit by hand. Utterance " + _tex(s["utt_id"]),
           "\\begin{tabular}{@{}ll@{}}"]
    out += [f"{name} & {row(key)} \\\\" for name, key in rows]
    out.append("\\end{tabular}")
    return "\n".join(out) + "\n"
