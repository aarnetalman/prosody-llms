"""Reconstruct the corpus labels with the Wavelet Prosody Toolkit on our alignments.

    python -m prosody_llms.audio.reconstruct --check-sample      # the toolkit's own LibriTTS sample
    python -m prosody_llms.audio.reconstruct --splits dev test   # all of dev and test

The corpus labels were made with ``asuni/wavelet_prosody_toolkit`` (``prosody_labeller``
with ``configs/libritts.yaml``) on alignments that are no longer available. This runs the
same labeller on the LibriTTS audio with the public alignments and compares, word by word:

* the real-valued prominence with the corpus's 4th column (Pearson and Spearman);
* the labels after the corpus's cut-offs (prominence < 0.4 -> 0, < 1.2 -> 1, else 2)
  with the corpus labels, as 3-way and 2-way accuracy.

The corpus stores ``max(0, score)``: no corpus value is negative and 9.3% are exactly
0.000, while the toolkit gives small negative scores to unstressed words. Reconstructed
scores are clipped the same way (labels are unaffected: everything below 0.4 is 0).

Agreement measures how faithful our word boundaries (and toolkit version) are, and the
reconstructed labels are the acoustic upper reference for the audio models: the
labelling procedure itself, on our boundaries.

The toolkit runs from a source checkout pinned to ``TOOLKIT_COMMIT``, with its
dependencies in a separate folder (both on ``$WPT_PATH``, set up by ``scripts/install_toolkit.sh``),
as a separate process; it is not part of this package. (Its packaging installs only the
top-level package, so ``pip install`` alone does not give a working toolkit.) Scoring is in the same
format as every other experiment: ``full`` counts words without a reconstructed score
as non-prominent ("0", as unparsed words in prompting), ``aligned`` scores only words
with a score.

``--check-sample`` fetches the toolkit's sample ``7127_75947_000010_000000`` (a test
utterance) with its own TextGrid, which may be one of the original alignments, and
labels it with that TextGrid and with the public one: the scores against the corpus show
whether the toolkit version reproduces the labels and how much the alignment alone moves
them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

from prosody_llms.audio.base import LIBRITTS_SUBSETS, libritts_path
from prosody_llms.audio.download import default_audio_dir, textgrid_root
from prosody_llms.audio.timings import UtteranceTimings, load_timings
from prosody_llms.data.corpus import LABEL_NA, Token, Utterance, read_split
from prosody_llms.data.download import default_data_dir
from prosody_llms.eval.io import evaluate, make_record
from prosody_llms.utils.runinfo import environment_info, write_json

log = logging.getLogger("prosody_llms.audio.reconstruct")

TOOLKIT_REPO = "asuni/wavelet_prosody_toolkit"
TOOLKIT_COMMIT = "a19fa7c87ff6cd2bb7072b673c8f20bf91aef6d7"  # v1.1.0, 2026-09-22
TOOLKIT_LIBRITTS_COMMIT = "768f5cc20a3d69f6383d7bcdc64fcde333603ce7"  # 2020-01-24, LibriTTS tools added
SAMPLE = "7127_75947_000010_000000"
PROMINENCE_CUTS = (0.4, 1.2)
FRAME_RATE = 200  # the toolkit's label times are frames / 200


def discretise(value: float, cuts=PROMINENCE_CUTS) -> str:
    return "0" if value < cuts[0] else "1" if value < cuts[1] else "2"


# ----------------------------------------------------------------------------- toolkit


def toolkit_python() -> str:
    """The interpreter for the toolkit: ``$WPT_PYTHON`` if set, else this one."""
    return os.environ.get("WPT_PYTHON") or sys.executable


def toolkit_env() -> dict:
    """Environment for the toolkit process: ``$WPT_PATH`` (the toolkit's source checkout
    and the folder its dependencies were installed into with ``pip install --target``,
    separated by ``os.pathsep``) goes first on the path, so the toolkit's packages never
    mix with this environment's."""
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")  # the labeller imports matplotlib; no display here
    tools = Path(__file__).resolve().parents[3] / "tools"
    if not env.get("WPT_PATH") and (tools / "wpt-src").is_dir():  # scripts/install_toolkit.sh
        env["WPT_PATH"] = os.pathsep.join(str(tools / d) for d in ("wpt-src", "wpt-lib"))
    if env.get("WPT_PATH"):
        env["PYTHONPATH"] = os.pathsep.join(p for p in (env["WPT_PATH"], env.get("PYTHONPATH")) if p)
    return env


def toolkit_config(python: str) -> Path:
    out = subprocess.run([python, "-c", "import os, wavelet_prosody_toolkit as w; "
                          "print(os.path.join(os.path.dirname(w.__file__), 'configs', 'libritts.yaml'))"],
                         check=True, capture_output=True, text=True, env=toolkit_env())
    return Path(out.stdout.strip())


def run_toolkit(input_dir: Path, out_dir: Path, jobs: int, python: str | None = None) -> None:
    python = python or toolkit_python()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-m", "wavelet_prosody_toolkit.prosody_labeller", str(input_dir),
           "-c", str(toolkit_config(python)), "-o", str(out_dir), "-j", str(jobs)]
    log.info("running %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env=toolkit_env())


def read_prom(path: Path) -> dict[tuple[int, int], float]:
    """{(start frame, end frame): prominence} from a toolkit ``.prom`` file."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 6:
            out[(round(float(parts[1]) * FRAME_RATE), round(float(parts[2]) * FRAME_RATE))] = float(parts[4])
    return out


def word_scores(utt: Utterance, t: UtteranceTimings | None, prom: dict) -> list[float | None]:
    """Reconstructed prominence per corpus token (None: punctuation, no timing, a shared
    interval, or no toolkit row for the word's interval)."""
    out = []
    for i, tok in enumerate(utt.tokens):
        if t is None or tok.is_punct or not t.timings[i].aligned or t.shared[i]:
            out.append(None)
            continue
        key = (round(t.timings[i].start * FRAME_RATE), round(t.timings[i].end * FRAME_RATE))
        v = prom.get(key)
        out.append(None if v is None else max(0.0, v))  # the corpus stores max(0, score)
    return out


# ----------------------------------------------------------------------------- scoring


def mask_utterances(utts: list[Utterance], masks: list[list[bool]]) -> list[Utterance]:
    """Copies in which unmasked words count as punctuation, i.e. are not scored."""
    out = []
    for u, m in zip(utts, masks):
        toks = [t if keep else Token(t.word, LABEL_NA, LABEL_NA, None, None) for t, keep in zip(u.tokens, m)]
        out.append(Utterance(u.utt_id, toks))
    return out


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def correlations(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return {"pearson": None, "spearman": None, "n": len(a)}
    return {"pearson": round(float(np.corrcoef(a, b)[0, 1]), 4),
            "spearman": round(float(np.corrcoef(ranks(a), ranks(b))[0, 1]), 4), "n": len(a)}


def score(utts, scores: list[list[float | None]]) -> dict:
    """Metrics for 3-way and 2-way, on all words (missing -> "0") and on scored words."""
    preds = [[LABEL_NA if tok.is_punct else ("0" if s is None else discretise(s))
              for tok, s in zip(u.tokens, ss)] for u, ss in zip(utts, scores)]
    masks = [[s is not None for s in ss] for ss in scores]
    masked = mask_utterances(utts, masks)
    out = {}
    for task in (3, 2):
        p = preds if task == 3 else [["1" if x == "2" else x for x in ps] for ps in preds]
        full = evaluate(utts, p, task)
        out[task] = {**full, "aligned": evaluate(masked, p, task)["full"]}
    pairs = [(tok.prominence_value, s) for u, ss in zip(utts, scores) for tok, s in zip(u.tokens, ss)
             if s is not None and tok.prominence_value is not None]
    words = sum(not t.is_punct for u in utts for t in u.tokens)
    out["values"] = {**correlations([x for x, _ in pairs], [y for _, y in pairs]),
                     "scored_words": len(pairs), "words": words,
                     "coverage": round(len(pairs) / words, 5) if words else None}
    return out


# ----------------------------------------------------------------------------- runs


def prepare_inputs(utts, split: str, audio_dir: Path, work: Path, done: set[str]) -> int:
    """Flat directory of wav and TextGrid symlinks for the utterances still to label."""
    work.mkdir(parents=True, exist_ok=True)
    for f in work.iterdir():
        f.unlink()
    subset = LIBRITTS_SUBSETS[split]
    n = 0
    for u in utts:
        if u.utt_id in done:
            continue
        sp, ch = u.utt_id.split("_")[:2]
        tg = textgrid_root(audio_dir) / subset / sp / ch / f"{u.utt_id}.TextGrid"
        wav = libritts_path(audio_dir, split, u.utt_id)
        if tg.exists() and wav.exists():
            (work / f"{u.utt_id}.wav").symlink_to(wav)
            (work / f"{u.utt_id}.TextGrid").symlink_to(tg)
            n += 1
    return n


def reconstruct_split(split: str, args) -> dict:
    utts = read_split(args.data_dir / f"{split}.txt")
    if args.max_utterances:
        utts = utts[: args.max_utterances]
    out = args.out_root / split
    prom_dir = out / "prom"
    prom_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in prom_dir.glob("*.prom")}
    todo = prepare_inputs(utts, split, args.audio_dir, out / "input", done)
    log.info("%s: %d utterances, %d already labelled, %d to label", split, len(utts), len(done), todo)
    if todo:
        run_toolkit(out / "input", prom_dir, args.jobs)
    subset = LIBRITTS_SUBSETS[split]
    timings = [load_timings(textgrid_root(args.audio_dir), subset, u) for u in utts]
    scores = []
    for u, t in zip(utts, timings):
        p = prom_dir / f"{u.utt_id}.prom"
        scores.append(word_scores(u, t, read_prom(p) if p.exists() else {}))
    metrics = score(utts, scores)
    write_outputs(out, split, utts, scores, metrics)
    return metrics


def write_outputs(out: Path, split: str, utts, scores, metrics) -> None:
    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for task in (3, 2):
            rec = make_record(experiment="reconstruction", model="wavelet-toolkit", task=task, split=split,
                              condition={"alignments": "LibriTTSLabel", "toolkit": TOOLKIT_COMMIT[:7],
                                         "config": "libritts.yaml"},
                              metrics=metrics[task])
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    from prosody_llms.probing.run import write_predictions_gz

    preds3 = [[LABEL_NA if tok.is_punct else ("0" if v is None else discretise(v))
               for tok, v in zip(u.tokens, ss)] for u, ss in zip(utts, scores)]
    write_predictions_gz(out / f"predictions_{split}_3way.tsv.gz", utts, preds3, 3)
    write_predictions_gz(out / f"predictions_{split}_2way.tsv.gz", utts,
                         [["1" if x == "2" else x for x in ps] for ps in preds3], 2)
    with gzip.open(out / "scores.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("utt_id\tword\tlabel\tcorpus_value\treconstructed\n")
        for u, ss in zip(utts, scores):
            for tok, s in zip(u.tokens, ss):
                if not tok.is_punct:
                    f.write(f"{u.utt_id}\t{tok.word}\t{tok.prominence}\t{tok.prominence_value}\t"
                            f"{'' if s is None else s}\n")
    per_utt = []
    for u, ss in zip(utts, scores):
        pairs = [(t.prominence, discretise(s)) for t, s in zip(u.tokens, ss) if s is not None]
        if pairs:
            per_utt.append((sum(a == b for a, b in pairs) / len(pairs), len(pairs), u.utt_id))
    per_utt.sort()
    result = {"experiment": "reconstruction", "split": split, "toolkit_commit": TOOLKIT_COMMIT,
              "accuracy": {t: metrics[t]["full"]["accuracy"] for t in (3, 2)},
              "accuracy_aligned": {t: metrics[t]["aligned"]["accuracy"] for t in (3, 2)},
              "values": metrics["values"],
              "worst_utterances": [{"utt_id": i, "agreement_3way": round(a * 100, 1), "words": n}
                                   for a, n, i in per_utt[:50]],
              "environment": environment_info()}
    write_json(out / "result.json", result)


def check_sample(args) -> dict:
    """The toolkit's LibriTTS sample, labelled with its own TextGrid and with ours."""
    out = args.out_root / "sample"
    base = f"https://raw.githubusercontent.com/{TOOLKIT_REPO}/{TOOLKIT_COMMIT}/samples/libritts/{SAMPLE}"
    test = {u.utt_id: u for u in read_split(args.data_dir / "test.txt")}
    utt = test[SAMPLE]
    rows = {}
    for name in ("toolkit_textgrid", "public_textgrid"):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        wav = d / f"{SAMPLE}.wav"
        if not wav.exists():
            urllib.request.urlretrieve(f"{base}.wav", wav)
        tg = d / f"{SAMPLE}.TextGrid"
        if not tg.exists():
            if name == "toolkit_textgrid":
                urllib.request.urlretrieve(f"{base}.TextGrid", tg)
            else:
                sp, ch = SAMPLE.split("_")[:2]
                tg.write_bytes((textgrid_root(args.audio_dir) / "test-clean" / sp / ch /
                                f"{SAMPLE}.TextGrid").read_bytes())
        run_toolkit(wav, d / "prom", 1)
        from prosody_llms.audio.timings import map_tokens, read_textgrid

        tiers = read_textgrid(tg)
        t = map_tokens(utt, tiers["words"], tiers.get("phones"))
        rows[name] = word_scores(utt, t, read_prom(d / "prom" / f"{SAMPLE}.prom"))
    table = []
    for i, tok in enumerate(utt.tokens):
        if not tok.is_punct:
            table.append({"word": tok.word, "corpus": tok.prominence_value, "label": tok.prominence,
                          **{k: v[i] for k, v in rows.items()}})
    result = {"utterance": SAMPLE, "toolkit_commit": TOOLKIT_COMMIT, "words": table}
    for k in rows:
        pairs = [(r["corpus"], r[k]) for r in table if r[k] is not None]
        result[k] = {"max_abs_diff": round(max(abs(a - b) for a, b in pairs), 3) if pairs else None,
                     "labels_equal": sum(r["label"] == discretise(r[k]) for r in table if r[k] is not None),
                     "words": len(table), **(correlations(*zip(*pairs)) if len(pairs) >= 3 else {})}
    write_json(out / "result.json", result)
    lines = [f"Sample {SAMPLE}: corpus prominence vs toolkit on two alignments", "",
             f"{'word':14} {'corpus':>7} {'toolkit TG':>11} {'public TG':>10}"]
    for r in table:
        f = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
        lines.append(f"{r['word']:14} {f(r['corpus']):>7} {f(r['toolkit_textgrid']):>11} "
                     f"{f(r['public_textgrid']):>10}")
    for k in rows:
        lines.append(f"{k}: max |diff| {result[k]['max_abs_diff']}, same label for "
                     f"{result[k]['labels_equal']}/{result[k]['words']} words")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="*", choices=["dev", "test", "train_100"], default=["dev", "test"])
    p.add_argument("--check-sample", action="store_true")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    p.add_argument("--max-utterances", type=int, default=None, help="first N per split (checks)")
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--audio-dir", type=Path, default=default_audio_dir())
    p.add_argument("--out-root", type=Path,
                   default=Path(os.environ.get("PROSODY_RUNS", "runs")) / "audio" / "reconstruction")
    args = p.parse_args(argv)
    if args.check_sample:
        check_sample(args)
    rows = []
    for split in args.splits:
        m = reconstruct_split(split, args)
        rows.append((split, m))
    if rows:
        lines = ["# Label reconstruction (wavelet toolkit on public alignments)", "",
                 "| split | 3-way | 2-way | 3-way (scored words) | 2-way (scored words) | Pearson | Spearman | coverage |",
                 "|---|---|---|---|---|---|---|---|"]
        for split, m in rows:
            v = m["values"]
            lines.append(f"| {split} | {m[3]['full']['accuracy']:.2f} | {m[2]['full']['accuracy']:.2f} | "
                         f"{m[3]['aligned']['accuracy']:.2f} | {m[2]['aligned']['accuracy']:.2f} | "
                         f"{v['pearson']} | {v['spearman']} | {v['coverage']} |")
        (args.out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(args.out_root / "result.json", {
            "experiment": "reconstruction", "toolkit_commit": TOOLKIT_COMMIT,
            "splits": {split: {"accuracy": {t: m[t]["full"]["accuracy"] for t in (3, 2)},
                               "accuracy_aligned": {t: m[t]["aligned"]["accuracy"] for t in (3, 2)},
                               "values": m["values"]} for split, m in rows}})
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
