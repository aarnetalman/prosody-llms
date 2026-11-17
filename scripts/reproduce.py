#!/usr/bin/env python3
"""Every command behind the paper's results, grouped by experiment.

    python scripts/reproduce.py groups                 # the groups, in order, with their needs
    python scripts/reproduce.py list probe-linear      # print the commands of a group
    python scripts/reproduce.py run baselines          # run a group's commands one by one
    python scripts/reproduce.py run probe-linear --dry-run

The commands only use the environment variables PROSODY_DATA (corpus, default
``data``), PROSODY_AUDIO (audio, default ``$PROSODY_DATA/../audio``) and
PROSODY_RUNS (outputs, default ``runs``). Every command can be rerun: finished runs are
skipped, and interrupted probing runs resume from their last checkpoint.

Groups have to run in the order listed by ``groups`` (stronger probes read the best layer
of the linear probes; the comparison reads everything). Within a group the commands are
independent, so on a cluster they can run as one array job, one command per task:

    python scripts/reproduce.py list probe-linear > jobs.txt
    sbatch --array=1-$(wc -l < jobs.txt) slurm/array.sbatch jobs.txt
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PY = ["python", "-m"]
ENCODERS = ["bert-base-uncased", "bert-base-cased", "bert-large-uncased", "modernbert-base",
            "modernbert-large", "xlm-roberta-base", "xlm-roberta-large"]
DECODERS = ["qwen3.5-0.8b-base", "qwen3.5-2b-base", "qwen3.5-4b-base", "qwen3.5-9b-base",
            "qwen3.8-27b", "olmo-3-7b", "olmo-3-32b", "muse-glimmer-30b"]
SPEECH = ["wavlm-large", "hubert-large", "whisper-large-v3"]
FINETUNED = ["bert-large-uncased", "modernbert-large", "xlm-roberta-large"]
REPRO_SEEDS = [1234, 1, 2, 3, 4]
FINETUNE_SEEDS = [1234, 1, 2]
EXEMPLARS = [10, 25, 100, 250, 500]
KINDS = ["encoder", "decoder", "speech", "speech_text"]


@dataclass
class Group:
    name: str
    needs: str
    about: str
    commands: list[list[str]]


def runs() -> str:
    return os.environ.get("PROSODY_RUNS", "runs")


def linear(model: str, *args: str) -> list[str]:
    return PY + ["prosody_llms.probing.run", "--model", model, "--resume", *args]


def nonlinear(model: str, *args: str) -> list[str]:
    return PY + ["prosody_llms.probing.nonlinear", "--model", model, *args]


# Probing conditions: (model, arguments) for the linear probes and, after them, the
# stronger probes on the best linear layer.
def text_conditions() -> list[tuple[str, list[str]]]:
    conds = [(m, ["--context", "plain"]) for m in ENCODERS]
    conds += [(m, ["--context", c]) for m in DECODERS for c in ("plain", "echo")]
    return conds


def strong_conditions() -> list[tuple[str, list[str]]]:
    """Stronger probes: encoders with plain input, decoders with echo (their best linear
    condition), the speech models, and the text controls trained on train_100."""
    conds = [(m, ["--context", "plain"]) for m in ENCODERS]
    conds += [(m, ["--context", "echo"]) for m in DECODERS]
    return conds


def audio_conditions() -> list[tuple[str, list[str]]]:
    conds = [(m, []) for m in SPEECH]
    conds += [("qwen2-audio-7b", ["--part", "llm"]), ("qwen2-audio-7b", ["--part", "encoder"])]
    conds += [("bert-base-uncased", ["--context", "plain", "--train-set", "train_100"]),
              ("muse-glimmer-30b", ["--context", "echo", "--train-set", "train_100"])]
    return conds


def finetune_commands() -> list[list[str]]:
    from prosody_llms.config import get_model

    out = []
    for name in FINETUNED:
        m = get_model(name)
        for task in (2, 3):
            for seed in FINETUNE_SEEDS:
                out.append(PY + ["prosody_llms.legacy.train", "--name", name, "--model", m.id,
                                 "--revision", m.revision, "--n-classes", str(task), "--seed", str(seed),
                                 "--attention-mask", "--learning-rate", "2e-5", "--no-checkpoint",
                                 "--out-dir", f"{runs()}/finetune/{name}/{task}way/seed{seed}"])
    return out


def groups() -> list[Group]:
    r = runs()
    repro = [PY + ["prosody_llms.legacy.train", "--n-classes", str(t), "--seed", str(s), "--resume",
                   "--out-dir", f"{r}/repro_2019/train_360-1.0/{t}way/seed{s}"]
             for t in (2, 3) for s in REPRO_SEEDS]
    repro += [PY + ["prosody_llms.legacy.train", "--n-classes", str(t), "--seed", "1234", "--resume",
                    "--fraction-of-train-data", "0.1",
                    "--out-dir", f"{r}/repro_2019/train_360-0.1/{t}way/seed1234"] for t in (2, 3)]

    probe_linear = [linear(m, *a) for m, a in text_conditions()]
    probe_linear += [linear("bert-base-uncased", "--context", "plain", "--train-utterances", str(n))
                     for n in (500, 5000)]
    probe_linear += [linear(m, "--context", "plain", "--train-utterances", "5000")
                     for m in ("qwen3.8-27b", "muse-glimmer-30b")]
    probe_linear += [linear("bert-base-uncased", "--context", c) for c in ("prev1", "prev3")]
    probe_linear += [linear("modernbert-large", "--context", "prev3")]

    probe_strong = [nonlinear(m, *a) for m, a in strong_conditions()]
    probe_strong += [nonlinear("bert-base-uncased", "--context", c, "--probes", "bilstm")
                     for c in ("prev1", "prev3")]
    probe_strong += [nonlinear("modernbert-large", "--context", "prev3", "--probes", "bilstm")]

    controls = [
        nonlinear("bert-base-uncased", "--context", "plain", "--layers", "0,0", "--probes", "bilstm",
                  "--out-root", f"{r}/probing_layer0"),
        nonlinear("qwen3.8-27b", "--context", "echo", "--layers", "0,0", "--probes", "bilstm",
                  "--out-root", f"{r}/probing_layer0"),
        nonlinear("random-words-1024", "--random-features", "1024", "--layers", "0,0",
                  "--probes", "bilstm", "--out-root", f"{r}/probing_control"),
    ]

    matched = [linear("qwen3.8-27b", "--context", "echo", "--train-exemplars", str(k)) for k in EXEMPLARS]
    matched += [nonlinear("qwen3.8-27b", "--context", "echo", "--train-exemplars", str(k), "--probes", "bilstm")
                for k in EXEMPLARS]

    audio_prep = [
        ["bash", "scripts/install_toolkit.sh"],
        PY + ["prosody_llms.audio.download"],
        PY + ["prosody_llms.audio.reconstruct", "--check-sample", "--splits"],
    ]
    audio_ref = [
        PY + ["prosody_llms.audio.baselines"],
        PY + ["prosody_llms.audio.reconstruct", "--splits", "dev", "test"],
    ]
    audio_linear = [linear(m, *a) for m, a in audio_conditions()]
    audio_strong = [nonlinear(m, *a) for m, a in audio_conditions()]
    audio_prompt = [
        PY + ["prosody_llms.audio.prompt", "--model", "qwen2-audio-7b-instruct", "--split", "test", "--both"],
        PY + ["prosody_llms.audio.freegen", "--model", "qwen2-audio-7b-instruct"],
    ]

    icl = [PY + ["prosody_llms.prompting.local", "--model", "qwen3.8-27b", "--split", "dev500"],
           PY + ["prosody_llms.prompting.local", "--model", "qwen3.8-27b", "--split", "test",
                 "--shots", "0", "500"]]

    prompting = [PY + ["prosody_llms.prompting.sweep"],
                 PY + ["prosody_llms.prompting.sweep", "--confirm-spend", "--max-usd", "100"],
                 PY + ["prosody_llms.prompting.costs"]]

    return [
        Group("data", "internet, CPU", "download and verify the corpus",
              [PY + ["prosody_llms.data.download"]]),
        Group("models", "internet, about 300 GB of disk",
              "download every model at its pinned revision into the Hugging Face cache",
              [PY + ["prosody_llms.prefetch", "--kinds", *KINDS], PY + ["prosody_llms.prefetch", "--verify", "--kinds", *KINDS]]),
        Group("baselines", "CPU, minutes", "majority, per-word and random baselines",
              [PY + ["prosody_llms.legacy.baselines", "--out", f"{r}/repro_2019/baselines.json"]]),
        Group("repro-2019", "GPU", "the 2019 BERT-base recipe: 2 tasks x 5 seeds, and the 10% point",
              repro),
        Group("finetune", "GPU", "BERT-large, ModernBERT-large and XLM-R large fine-tuned: 2 tasks x 3 seeds",
              finetune_commands()),
        Group("probe-linear", "GPU (up to about 95 GB for the 30B models)",
              "linear probes on every layer of every text model; learning-curve and discourse-context runs",
              probe_linear),
        Group("probe-strong", "GPU; after probe-linear",
              "logistic, MLP and BiLSTM classifiers on each model's best linear layer", probe_strong),
        Group("controls", "GPU; after probe-linear",
              "BiLSTM on the embedding layer (BERT-base, Qwen3.8 27B) and on random word vectors", controls),
        Group("icl", "GPU", "many-shot prompting of Qwen3.8 27B with constrained decoding", icl),
        Group("prompting", "internet, API key; paid",
              "prompting sweep of the hosted models: plan and cost estimate, then the paid run "
              "(see configs/prompting.toml)", prompting),
        Group("audio-prep", "internet, CPU, about 10 GB of disk",
              "the labelling toolkit, LibriTTS audio and alignments, and the toolkit's own sample", audio_prep),
        Group("audio-reference", "CPU", "duration-only baseline and the labels reconstructed from audio",
              audio_ref),
        Group("audio-linear", "GPU", "linear probes on the speech models and the train_100 text controls",
              audio_linear),
        Group("audio-strong", "GPU; after audio-linear", "stronger probes on the same", audio_strong),
        Group("audio-prompting", "GPU", "Qwen2-Audio-Instruct prompted with and without the recording",
              audio_prompt),
        Group("compare", "CPU; after everything else", "all systems on identical words",
              [PY + ["prosody_llms.audio.compare"]]),
        Group("collect", "CPU", "copy the small result files from the runs folder into results/",
              [["python", "scripts/collect_results.py"]]),
        Group("analysis", "CPU, seconds", "tables, numbers and figure data of the paper from results/",
              [PY + ["prosody_llms.analysis"]]),
        Group("matched-supervision", "GPU; optional, not in the paper",
              "Qwen3.8 27B probed after training on only the k prompt examples", matched),
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("groups", help="list the groups in order")
    for name in ("list", "run"):
        s = sub.add_parser(name)
        s.add_argument("group", nargs="+")
        if name == "run":
            s.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    gs = {g.name: g for g in groups()}
    if args.cmd == "groups":
        for i, g in enumerate(gs.values(), 1):
            print(f"{i:2d}. {g.name:<20} {len(g.commands):3d} command(s)  [{g.needs}]\n    {g.about}")
        return 0
    unknown = [n for n in args.group if n not in gs]
    if unknown:
        p.error(f"unknown group(s): {', '.join(unknown)}; see `groups`")
    for name in args.group:
        for cmd in gs[name].commands:
            line = shlex.join(cmd)
            if args.cmd == "list":
                print(line)
                continue
            print(f"+ {line}", flush=True)
            if args.dry_run:
                continue
            if name == "prompting" and "--confirm-spend" in cmd:
                if input("This sends paid requests. Type 'yes' to continue: ").strip() != "yes":
                    print("skipped")
                    continue
            code = subprocess.call(cmd, cwd=ROOT)
            if code:
                print(f"failed with exit code {code}: {line}", file=sys.stderr)
                return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
