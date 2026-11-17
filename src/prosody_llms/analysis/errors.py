"""Error analysis and data checks for the paper, from committed predictions.

* ``agreement``: label agreement between different speakers reading the same text (all
  splits): how far any text-only system can go in principle.
* ``overlap``: test utterances whose text also occurs in train_360 or dev ,
  and each text system's accuracy on them and on the rest. The test set is unchanged.
* ``breakdown``: on the words the text-against-audio comparison uses, the best text and
  the best audio system (chosen by the comparison, which ranks on test; the breakdown
  describes them, it selects nothing) compared by word class (function / content),
  training frequency, position in the utterance, givenness (the word occurred earlier
  in the same chapter) and gold label.
* ``examples``: utterances where the audio system is right and the text system wrong on
  several words, picked by a fixed hash (not by hand).

Writes ``results/analysis/error_analysis.md``, ``breakdown.csv``, ``overlap.csv`` and
``paper/generated/breakdown_table.tex``; returns numbers for ``numbers.tex``.
"""

from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import math
from pathlib import Path

from prosody_llms.data.corpus import LABEL_NA, read_split
from prosody_llms.data.subsets import (
    chapter_position,
    cross_speaker_agreement,
    is_boilerplate,
    multi_reader_groups,
    overlapping_texts,
)


def reader_groups(splits) -> list:
    """Texts read by two or more speakers, without the LibriVox announcements ("This is a
    LibriVox recording.", read by up to 20 speakers), which are formulaic, not reading."""
    return [g for g in multi_reader_groups(splits) if not is_boilerplate(g[0][1])]

# Closed-class English words (articles, pronouns, prepositions, conjunctions, auxiliaries,
# modals, particles); everything else counts as a content word.
FUNCTION_WORDS = set("""
a an the this that these those my your his her its our their mine yours hers ours theirs
i me you he him she it we us they them myself yourself himself herself itself ourselves
themselves who whom whose which what whoever whatever one ones
of in on at by for with from to into onto upon about above across after against along
among around before behind below beneath beside besides between beyond but down during
except inside like near off out outside over past since than through throughout till
toward towards under underneath until unto up via within without
and or nor so yet either neither both whether if though although because unless while
whereas as when whenever where wherever then
be am is are was were been being have has had having do does did doing
will would shall should can could may might must ought
not no n't 's 'll 're 've 'd 'm there here
all any each every some such own same other another much many more most few less least
very too also just only even quite rather
""".split())

FREQ_BINS = ((0, 0, "unseen"), (1, 10, "1-10"), (11, 100, "11-100"), (101, 1000, "101-1k"),
             (1001, 10 ** 9, ">1k"))


def read_predictions_gz(path: Path) -> dict[str, list[str]]:
    out, cur = {}, None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("<file>\t"):
                cur = line.split("\t", 1)[1]
                out[cur] = []
            elif line:
                out[cur].append(line.split("\t")[2])
    return out


def collapse(x: str) -> str:
    return "1" if x == "2" else x


# ----------------------------------------------------------------------------- agreement and overlap


def agreement(data_dir: Path) -> dict:
    splits = {s: read_split(data_dir / f"{s}.txt") for s in ("train_360", "dev", "test")}
    groups = reader_groups(splits)
    out = {"groups": len(groups)}
    for t in (2, 3):
        try:
            acc, pairs = cross_speaker_agreement(groups, t)
        except ValueError:  # no text read by two speakers (small test corpora)
            return out
        out[f"agreement-{t}"] = acc
        out["agreement-pairs"] = pairs
    return out


def reader_ceiling(root: Path, data_dir: Path) -> dict:
    """An estimate of the best possible text-only accuracy, set by variation between
    readers. For texts read by at least three speakers, each reading is predicted from the
    *other* speakers' readings: their mean prominence score per word, cut with the
    corpus thresholds (the same scale the labels come from; no ties). A text-only model
    sees only the text, so it cannot do better on a reading than the best predictor
    built from other readings of the same text, up to the noise of a finite number of
    readers. On the test readings in these groups, the best text systems are scored on
    the same words for comparison."""
    from prosody_llms.audio.reconstruct import discretise

    splits = {s: read_split(data_dir / f"{s}.txt") for s in ("train_360", "dev", "test")}
    groups = [g for g in reader_groups(splits) if len({u.speaker for _, u in g}) >= 3]
    hits = {2: [0, 0], 3: [0, 0]}
    test_ids: dict[str, list[str]] = {}  # test utterance -> 3-way ceiling prediction
    readings = 0
    for g in groups:
        for split, u in g:
            others = [o for _, o in g if o.speaker != u.speaker]
            if len({o.speaker for o in others}) < 2:
                continue
            readings += 1
            pred = []
            for i, tok in enumerate(u.tokens):
                if tok.is_punct:
                    pred.append(LABEL_NA)
                    continue
                vals = [o.tokens[i].prominence_value for o in others if o.tokens[i].prominence_value is not None]
                pred.append(discretise(sum(vals) / len(vals)) if vals else "0")
            for t in (3, 2):
                for gl, p in zip(u.labels(t), pred):
                    if gl == LABEL_NA:
                        continue
                    hits[t][0] += gl == (collapse(p) if t == 2 else p)
                    hits[t][1] += 1
            if split == "test":
                test_ids[u.utt_id] = pred
    out = {"ceiling-groups": len(groups), "ceiling-readings": readings, "ceiling-words": hits[3][1]}
    if not hits[3][1]:
        return out
    for t in (3, 2):
        out[f"ceiling-{t}"] = 100.0 * hits[t][0] / hits[t][1]
    out["ceiling-test-utterances"] = len(test_ids)
    if not test_ids:
        return out
    test = {u.utt_id: u for u in splits["test"]}
    systems = {"ceiling": None}
    for name, path, _ in text_systems(root):
        systems[name] = read_predictions_gz(path)
    scores: dict[tuple[str, int], list[int]] = collections.defaultdict(lambda: [0, 0])
    for uid, pred in test_ids.items():
        u = test[uid]
        for name, preds in systems.items():
            p = pred if preds is None else preds.get(uid)
            if p is None:
                continue
            for t in (3, 2):
                for gl, x in zip(u.labels(t), p):
                    if gl == LABEL_NA:
                        continue
                    scores[(name, t)][0] += gl == (collapse(x) if t == 2 else x)
                    scores[(name, t)][1] += 1
    for t in (3, 2):
        c = scores[("ceiling", t)]
        out[f"ceiling-test-{t}"] = 100.0 * c[0] / c[1]
        out[f"ceiling-test-words"] = c[1]
        best = max(((n, 100.0 * a / b) for (n, tt), (a, b) in scores.items() if tt == t and n != "ceiling" and b),
                   key=lambda x: x[1], default=None)
        if best:
            out[f"ceiling-test-besttext-{t}"] = best[1]
            out[f"ceiling-test-besttext-system-{t}"] = best[0]
    return out


def text_systems(root: Path) -> list[tuple[str, Path, bool]]:
    """(name, 3-way prediction file, collapse for 2-way) for the main text systems."""
    out = []
    for cond_dir in sorted((root / "probing").glob("*/*")):
        if cond_dir.name.startswith(("audio-", "prev")):
            continue  # speech models and discourse-context runs are not part of the consensus
        p = cond_dir / "predictions_test_3way_lda_best.tsv.gz"
        if p.exists() and not cond_dir.name.split("natural")[-1].startswith(("-n", "-ex")):
            out.append((f"{cond_dir.parent.name} ({cond_dir.name}) LDA", p, False))
    for p in sorted((root / "probing_nonlinear").glob("*/*/predictions_test_3way_bilstm.tsv.gz")):
        if not p.parent.name.startswith(("audio-", "prev")) and "-ex" not in p.parent.name \
                and not p.parent.name.split("natural")[-1].startswith("-n"):
            out.append((f"{p.parent.parent.name} ({p.parent.name}) BiLSTM", p, False))
    for exp in ("prompting", "icl"):
        for p in sorted((root / exp).glob("*/test/*/predictions.tsv.gz")):
            if "-2way-" not in p.parent.name:
                out.append((f"{p.parent.parent.parent.name} {p.parent.name}", p, False))
    return out


def consensus(root: Path, data_dir: Path) -> tuple[dict, dict[str, list[str]]]:
    """How much the text models agree with each other, against how much they agree with
    the gold labels, on the whole test set (3-way and 2-way). Uses every probe (best LDA
    layer of each model and condition, and the BiLSTM probes). If different models agree
    with each other much more than with the gold labels, they predict the same reading,
    and the errors they share are where the gold reading departs from it.

    Returns numbers and, per test utterance, a consensus tag per token for the 3-way task:
    ``all-right``, ``all-wrong`` (every model wrong) or ``split``."""
    test = read_split(data_dir / "test.txt")
    systems = [(n, read_predictions_gz(p)) for n, p, _ in text_systems(root)
               if n.endswith(("LDA", "BiLSTM"))]
    systems = [(n, p) for n, p in systems if all(u.utt_id in p for u in test)]
    if len(systems) < 2:
        return {}, {}
    out: dict = {"consensus-systems": len(systems)}
    tags: dict[str, list[str]] = {}
    for t in (3, 2):
        n_words = acc_sum = 0
        pair_agree = pair_total = 0
        unanimous = unanimous_wrong = any_wrong = 0
        for u in test:
            gold = u.labels(t)
            preds = [[collapse(x) if t == 2 else x for x in p[u.utt_id]] for _, p in systems]
            utt_tags = []
            for i, g in enumerate(gold):
                if g == LABEL_NA:
                    utt_tags.append(None)
                    continue
                labs = [p[i] for p in preds]
                n_words += 1
                right = sum(x == g for x in labs)
                acc_sum += right / len(labs)
                counts = collections.Counter(labs)
                pair_agree += sum(c * (c - 1) for c in counts.values())
                pair_total += len(labs) * (len(labs) - 1)
                same = len(counts) == 1
                unanimous += same
                any_wrong += right < len(labs)
                unanimous_wrong += same and labs[0] != g
                utt_tags.append("all-right" if right == len(labs) else "all-wrong" if right == 0 else "split")
            if t == 3:
                tags[u.utt_id] = utt_tags
        out[f"consensus-acc-{t}"] = 100.0 * acc_sum / n_words
        out[f"consensus-mutual-{t}"] = 100.0 * pair_agree / pair_total
        out[f"consensus-unanimous-{t}"] = 100.0 * unanimous / n_words
        out[f"consensus-unanimous-wrong-{t}"] = 100.0 * unanimous_wrong / n_words
        # Share of an average model's errors on which every model makes the same mistake.
        out[f"consensus-shared-errors-{t}"] = 100.0 * unanimous_wrong / (n_words - acc_sum)
    return out, tags


def overlap(root: Path, data_dir: Path) -> tuple[list[dict], dict]:
    test = read_split(data_dir / "test.txt")
    others = [read_split(data_dir / f"{s}.txt") for s in ("train_360", "dev")]
    ids = overlapping_texts(test, others)
    rows = []
    for name, path, _ in text_systems(root):
        preds = read_predictions_gz(path)
        for t in (3, 2):
            c = {True: [0, 0], False: [0, 0]}
            for u in test:
                p = preds.get(u.utt_id)
                if p is None:
                    break
                for g, x in zip(u.labels(t), p):
                    if g == LABEL_NA:
                        continue
                    x = collapse(x) if t == 2 else x
                    k = u.utt_id in ids
                    c[k][0] += g == x
                    c[k][1] += 1
            else:
                if c[True][1]:
                    rows.append({"system": name, "task": t, "overlap_acc": 100 * c[True][0] / c[True][1],
                                 "rest_acc": 100 * c[False][0] / c[False][1], "overlap_words": c[True][1],
                                 "diff": 100 * c[True][0] / c[True][1] - 100 * c[False][0] / c[False][1]})
    words = sum(1 for u in test if u.utt_id in ids for t in u.tokens if not t.is_punct)
    return rows, {"overlap-utterances": len(ids), "overlap-words": words}


# ----------------------------------------------------------------------------- breakdown


def common_mask(root: Path) -> dict[str, list[bool]] | None:
    p = root / "audio" / "comparison" / "common_words.tsv.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return {a: [c == "1" for c in b] for a, b in (line.rstrip("\n").split("\t") for line in f)}


def best_pair(root: Path, task: int) -> tuple[dict, dict] | None:
    p = root / "audio" / "comparison" / "comparison.csv"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["task"]) == task and r["family"] not in ("reference",)]
    text = [r for r in rows if r["modality"] == "text"]
    audio = [r for r in rows if r["modality"] == "audio"]
    if not text or not audio:
        return None
    key = lambda r: float(r["accuracy_common"])  # noqa: E731
    return max(text, key=key), max(audio, key=key)


def word_features(test, train_counts) -> dict[str, list[dict]]:
    """Per test utterance, per token: features for the breakdown."""
    seen_in_chapter: dict[str, set] = collections.defaultdict(set)
    order = sorted(test, key=lambda u: chapter_position(u.utt_id))
    out = {}
    for u in order:
        chapter, _ = chapter_position(u.utt_id)
        scored = [i for i, t in enumerate(u.tokens) if not t.is_punct]
        feats = []
        for i, tok in enumerate(u.tokens):
            w = tok.word.lower()
            if tok.is_punct:
                feats.append(None)
                continue
            n = train_counts.get(w, 0)
            freq = next(name for lo, hi, name in FREQ_BINS if lo <= n <= hi)
            pos = "final" if i == scored[-1] else "initial" if i == scored[0] else "medial"
            func = w in FUNCTION_WORDS
            given = "-" if func else ("given" if w in seen_in_chapter[chapter] else "new")
            feats.append({"class": "function" if func else "content", "frequency": freq,
                          "position": pos, "givenness": given})
        for tok in u.tokens:
            if not tok.is_punct:
                seen_in_chapter[chapter].add(tok.word.lower())
        out[u.utt_id] = feats
    return out


def breakdown(root: Path, data_dir: Path, consensus_tags: dict | None = None) -> tuple[list[dict], list[dict], dict]:
    mask = common_mask(root)
    if mask is None:
        return [], [], {}
    test = read_split(data_dir / "test.txt")
    counts = collections.Counter(t.word.lower() for u in read_split(data_dir / "train_360.txt")
                                 for t in u.tokens if not t.is_punct)
    feats = word_features(test, counts)
    rows, examples, nums = [], [], {}
    for task in (3, 2):
        pair = best_pair(root, task)
        if pair is None:
            continue
        text_row, audio_row = pair
        tp = read_predictions_gz(root / text_row["path"])
        ap = read_predictions_gz(root / audio_row["path"])
        cells: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
        for u in test:
            m, g = mask[u.utt_id], u.labels(task)
            t_lab, a_lab = tp[u.utt_id], ap[u.utt_id]
            n_ar_tw = 0
            for i, f in enumerate(feats[u.utt_id]):
                if f is None or not m[i] or g[i] == LABEL_NA:
                    continue
                x = collapse(t_lab[i]) if task == 2 else t_lab[i]
                y = collapse(a_lab[i]) if task == 2 else a_lab[i]
                extra = []
                if consensus_tags and task == 3 and consensus_tags.get(u.utt_id):
                    extra.append(("text consensus", consensus_tags[u.utt_id][i]))
                for dim, val in (*f.items(), *extra, ("gold", g[i]), ("all", "all")):
                    c = cells[(dim, val)]
                    c[0] += 1
                    c[1] += x == g[i]
                    c[2] += y == g[i]
                    c[3] += y == g[i] and x != g[i]
                n_ar_tw += y == g[i] and x != g[i]
            if task == 3 and n_ar_tw >= 3:
                examples.append((hashlib.sha256(u.utt_id.encode()).hexdigest(), u, t_lab, a_lab))
        for (dim, val), (n, tc, ac, artw) in sorted(cells.items()):
            rows.append({"task": task, "dimension": dim, "value": val, "words": n,
                         "text_acc": 100 * tc / n, "audio_acc": 100 * ac / n, "diff": 100 * (ac - tc) / n,
                         "audio_right_text_wrong": 100 * artw / n,
                         "text_system": text_row["system"], "audio_system": audio_row["system"]})
        for (dim, val), (n, tc, ac, _) in cells.items():
            if dim in ("class", "position", "givenness", "text consensus"):
                nums[f"bd-{task}-{val}-diff"] = 100 * (ac - tc) / n
                nums[f"bd-{task}-{val}-text"] = 100 * tc / n
                nums[f"bd-{task}-{val}-audio"] = 100 * ac / n
    examples.sort(key=lambda e: e[0])
    ex_rows = []
    for _, u, t_lab, a_lab in examples[:5]:
        words = []
        for tok, gl, x, y in zip(u.tokens, u.labels(3), t_lab, a_lab):
            if tok.is_punct:
                words.append(tok.word)
            else:
                words.append(f"{tok.word}[{gl}|{x}|{y}]")
        ex_rows.append({"utt_id": u.utt_id, "text": " ".join(words)})
    return rows, ex_rows, nums


def breakdown_tex(rows: list[dict]) -> str:
    keep = [("class", "content"), ("class", "function"), ("position", "initial"), ("position", "medial"),
            ("position", "final"), ("givenness", "new"), ("givenness", "given"), ("frequency", "unseen"),
            ("frequency", "1-10"), ("frequency", ">1k"), ("text consensus", "all-wrong"),
            ("text consensus", "split"), ("text consensus", "all-right"), ("all", "all")]
    labels = {"content": "content words", "function": "function words", "initial": "first word",
              "medial": "medial", "final": "last word", "new": "new (content)", "given": "given (content)",
              "unseen": "unseen in training", "1-10": "seen 1--10 times", ">1k": "seen $>$1,000 times",
              "all-wrong": "every text model wrong", "split": "text models disagree",
              "all-right": "every text model right", "all": "all common words"}
    r3 = {(r["dimension"], r["value"]): r for r in rows if r["task"] == 3}
    if not r3:
        return "% No breakdown yet (needs the audio comparison).\n\\begin{tabular}{p{0.95\\columnwidth}}\\centering\\textit{results pending}\\end{tabular}\n"
    lines = ["% Generated by python -m prosody_llms.analysis; do not edit by hand.",
             "\\begin{tabular}{@{}lrccc@{}}", "\\toprule",
             "Words & $n$ & Text & Audio & $\\Delta$ \\\\", "\\midrule"]
    for key in keep:
        r = r3.get(key)
        if r:
            sep = "\\midrule\n" if key == ("all", "all") else ""
            lines.append(f"{sep}{labels[key[1]]} & {r['words']:,} & {r['text_acc']:.1f} & {r['audio_acc']:.1f} & "
                         f"{r['diff']:+.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def corpus_values(data_dir: Path) -> dict:
    """The corpus's real-valued prominence: share of exact zeros and of negative values
    (evidence that it stores max(0, score))."""
    n = zero = neg = 0
    for s in ("train_360", "dev", "test"):
        for u in read_split(data_dir / f"{s}.txt"):
            for t in u.tokens:
                if t.is_punct or t.prominence_value is None:
                    continue
                n += 1
                zero += t.prominence_value == 0.0
                neg += t.prominence_value < 0
    return {"corpus-zero-share": 100.0 * zero / n, "corpus-negative": neg, "corpus-words": n,
            "train-utterances": len(read_split(data_dir / "train_360.txt"))}


SPLITS = (  # (file, LibriTTS subset, use in the paper)
    ("train_360", "train-clean-360", "training (text)"),
    ("train_100", "train-clean-100", "training (speech; matched text)"),
    ("dev", "dev-clean", "model, layer and prompt selection"),
    ("test", "test-clean", "evaluation"),
)


def corpus_table(data_dir: Path) -> tuple[str, dict]:
    """Size, speakers and 3-way label distribution of each split, as a LaTeX table."""
    rows, nums, speakers = [], {}, {}
    for name, subset, use in SPLITS:
        p = data_dir / f"{name}.txt"
        if not p.exists():
            continue
        utts = read_split(p)
        counts = {"0": 0, "1": 0, "2": 0}
        for u in utts:
            for lab in u.labels(3):
                if lab != LABEL_NA:
                    counts[lab] += 1
        words = sum(counts.values())
        speakers[name] = {u.speaker for u in utts}
        tag = name.replace("_", "")
        nums.update({f"data-{tag}-utts": len(utts), f"data-{tag}-words": words,
                     f"data-{tag}-speakers": len(speakers[name])})
        share = " & ".join(f"{100.0 * counts[k] / words:.1f}" for k in "012")
        rows.append(f"\\texttt{{{name.replace('_', '\\_')}}} & {subset} & {len(utts):,} & {words:,} & "
                    f"{len(speakers[name])} & {share} & {use} \\\\")
    names = list(speakers)
    nums["data-shared-speakers"] = sum(len(speakers[a] & speakers[b]) for i, a in enumerate(names) for b in names[i + 1:])
    tex = "\n".join(["% Generated by python -m prosody_llms.analysis; do not edit by hand.",
                     "\\begin{tabular}{@{}llrrrrrrl@{}}", "\\toprule",
                     " & & & & & \\multicolumn{3}{c}{Labels (\\%)} & \\\\", "\\cmidrule(lr){6-8}",
                     "Split & LibriTTS subset & Utterances & Words & Speakers & 0 & 1 & 2 & Used for \\\\",
                     "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]) + "\n"
    return tex, nums


def write_all(root: Path, data_dir: Path, paper_dir: Path) -> dict:
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (paper_dir / "generated").mkdir(parents=True, exist_ok=True)
    nums: dict = {}
    agr = agreement(data_dir)
    if "agreement-3" in agr:
        nums.update({"agreement-2": agr["agreement-2"], "agreement-3": agr["agreement-3"],
                     "agreement-groups": agr["groups"], "agreement-pairs": agr["agreement-pairs"]})
        # Bounds on the best text-only accuracy if readers are independent draws from P(label | text):
        # agreement = E[sum p^2] <= E[max p] <= E[sqrt(sum p^2)] <= sqrt(E[sum p^2]) (Jensen).
        for t in (2, 3):
            a = agr[f"agreement-{t}"] / 100.0
            nums[f"bound-low-{t}"] = 100.0 * a
            nums[f"bound-high-{t}"] = 100.0 * a ** 0.5
    cons, cons_tags = consensus(root, data_dir)
    nums.update(cons)
    ceil = reader_ceiling(root, data_dir)
    nums.update({k: v for k, v in ceil.items() if not k.endswith("-system-2") and not k.endswith("-system-3")})
    ov_rows, ov_info = overlap(root, data_dir)
    nums.update(ov_info)
    for t in (2, 3):
        diffs = [r["diff"] for r in ov_rows if r["task"] == t]
        if diffs:
            nums[f"overlap-diff-min-{t}"] = -max(diffs)  # smallest drop
            nums[f"overlap-diff-max-{t}"] = -min(diffs)  # largest drop
    nums.update(corpus_values(data_dir))
    data_tex, data_nums = corpus_table(data_dir)
    nums.update(data_nums)
    (paper_dir / "generated" / "data_table.tex").write_text(data_tex, encoding="utf-8")
    sample = root / "audio" / "reconstruction" / "sample" / "result.json"
    if sample.exists():
        import json

        d = json.loads(sample.read_text(encoding="utf-8"))
        for k, tag in (("toolkit_textgrid", "own"), ("public_textgrid", "public")):
            if k in d:
                nums[f"sample-{tag}-maxdiff"] = float(d[k]["max_abs_diff"])
                nums[f"sample-{tag}-same"] = int(d[k]["labels_equal"])
        nums["sample-words"] = len(d["words"])
        pairs = [(w["toolkit_textgrid"], w["public_textgrid"]) for w in d["words"]
                 if w.get("toolkit_textgrid") is not None and w.get("public_textgrid") is not None]
        if pairs:
            nums["sample-between-maxdiff"] = max(abs(a - b) for a, b in pairs)
    bd_rows, examples, bd_nums = breakdown(root, data_dir, cons_tags)
    nums.update(bd_nums)
    for name, rows_ in (("overlap.csv", ov_rows), ("breakdown.csv", bd_rows)):
        if rows_:
            with open(out / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_[0]))
                w.writeheader()
                w.writerows(rows_)
    (paper_dir / "generated" / "breakdown_table.tex").write_text(breakdown_tex(bd_rows), encoding="utf-8")
    agr_line = (f"**Agreement between readers of the same text** ({agr['groups']} texts, "
                f"{agr['agreement-pairs']:,} word pairs): {agr['agreement-2']:.1f} (2-way), "
                f"{agr['agreement-3']:.1f} (3-way)." if "agreement-3" in agr else
                "**Agreement between readers of the same text**: no text read by two speakers.")
    lines = ["# Error analysis", "", agr_line, "",
             "## Do the text models agree with each other?", "",
             ("" if not cons else
              f"{cons['consensus-systems']} text probes on the whole test set. Mean accuracy {cons['consensus-acc-3']:.1f} "
              f"(3-way) / {cons['consensus-acc-2']:.1f} (2-way); mean agreement between two models "
              f"{cons['consensus-mutual-3']:.1f} / {cons['consensus-mutual-2']:.1f}. All models give the same label to "
              f"{cons['consensus-unanimous-3']:.1f}% / {cons['consensus-unanimous-2']:.1f}% of the words, and the same "
              f"wrong label to {cons['consensus-unanimous-wrong-3']:.1f}% / {cons['consensus-unanimous-wrong-2']:.1f}%."),
             "",
             "## Text-only ceiling from other readers (exploratory: too few texts)", "",
             (f"Texts read by at least three speakers: {ceil['ceiling-groups']} texts, "
              f"{ceil['ceiling-readings']} readings, {ceil['ceiling-words']:,} words. Each reading predicted "
              f"from the other readers' mean prominence score: {ceil['ceiling-2']:.1f} (2-way), "
              f"{ceil['ceiling-3']:.1f} (3-way)." if "ceiling-3" in ceil else "No text read by three or more speakers."),
             "" if "ceiling-test-3" not in ceil else
             (f"On the {ceil['ceiling-test-utterances']} test readings among them ({ceil['ceiling-test-words']:,} "
              f"words): ceiling {ceil['ceiling-test-2']:.1f} / {ceil['ceiling-test-3']:.1f}; best text system "
              f"{ceil.get('ceiling-test-besttext-2', float('nan')):.1f} "
              f"({ceil.get('ceiling-test-besttext-system-2', '')}) / "
              f"{ceil.get('ceiling-test-besttext-3', float('nan')):.1f} "
              f"({ceil.get('ceiling-test-besttext-system-3', '')})."), "",
             f"## Test utterances whose text also occurs in train_360 or dev ", "",
             f"{ov_info['overlap-utterances']} utterances, {ov_info['overlap-words']:,} scored words.", "",
             "| system | task | overlap | rest | difference |", "|---|---|---|---|---|"]
    for r in sorted(ov_rows, key=lambda r: (r["task"], -r["rest_acc"])):
        lines.append(f"| {r['system']} | {r['task']} | {r['overlap_acc']:.2f} | {r['rest_acc']:.2f} | {r['diff']:+.2f} |")
    lines += ["", "## Best text against best audio system on the common words", ""]
    if bd_rows:
        for task in (3, 2):
            sub = [r for r in bd_rows if r["task"] == task]
            if not sub:
                continue
            lines += [f"### {task}-way: {sub[0]['text_system']} against {sub[0]['audio_system']}", "",
                      "| dimension | value | words | text | audio | audio - text | audio right, text wrong |",
                      "|---|---|---|---|---|---|---|"]
            for r in sub:
                lines.append(f"| {r['dimension']} | {r['value']} | {r['words']:,} | {r['text_acc']:.2f} | "
                             f"{r['audio_acc']:.2f} | {r['diff']:+.2f} | {r['audio_right_text_wrong']:.2f} |")
            lines.append("")
        lines += ["## Examples (3-way; word[gold|text|audio]; first five by a fixed hash)", ""]
        lines += [f"- `{e['utt_id']}`: {e['text']}" for e in examples]
    else:
        lines.append("Not yet available: needs the audio comparison (results/audio/comparison/).")
    (out / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {k: (round(v, 2) if isinstance(v, float) and not math.isnan(v) else v) for k, v in nums.items()}
