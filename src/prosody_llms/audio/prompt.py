"""Prompting a speech-text model with the recording: Qwen2-Audio-Instruct.

    python -m prosody_llms.audio.prompt --model qwen2-audio-7b-instruct --split test
    python -m prosody_llms.audio.prompt --model qwen2-audio-7b-instruct --split test --no-audio

The prompt is the ``lines`` prompt of the text prompting experiment (instructions, the sentence, its numbered
words), preceded by the utterance's recording and the line "The recording above is the
sentence below, read aloud." With ``--no-audio`` the same model gets the same prompt
without the recording and that line: the difference is what the audio adds inside one
model.

As in the open-model many-shot runs (``prosody_llms.prompting.local``), the model does not
generate freely: the answer is written word by word, and for every word the model's
probabilities for the labels are read and the most probable label is written. Nothing
is parsed, and every word gets a full distribution (for AUC). Zero-shot only.

Outputs in ``$PROSODY_RUNS/audio_prompting/<model>/<split>/<variant>-<audio|no_audio>/``,
in the format of the other experiments (``records.jsonl``, ``predictions.tsv.gz``,
``probabilities.json.gz``, ``result.json``). Finished settings are skipped.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from prosody_llms.audio.base import libritts_path
from prosody_llms.audio.download import default_audio_dir
from prosody_llms.audio.extractors import SAMPLE_RATE, read_wav, resample
from prosody_llms.data.download import default_data_dir
from prosody_llms.eval.io import evaluate, make_record
from prosody_llms.prompting.local import LabelScorer
from prosody_llms.prompting.prompts import VARIANTS, PromptSpec, user_message, word_items
from prosody_llms.prompting.run import load_split, roc_auc, write_predictions_gz
from prosody_llms.utils.runinfo import environment_info, write_json

log = logging.getLogger("prosody_llms.audio.prompt")
AUDIO_LINE = "The recording above is the sentence below, read aloud.\n"


def conversation(utt, spec: PromptSpec, with_audio: bool) -> list[dict]:
    text = user_message(utt, spec)
    if not with_audio:
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return [{"role": "user", "content": [{"type": "audio", "audio_url": "recording.wav"},
                                         {"type": "text", "text": AUDIO_LINE + text}]}]


class AudioPrompter:
    def __init__(self, model_id: str, revision: str | None, dtype: str = "bfloat16", device: str | None = None):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        from prosody_llms.audio.extractors import _load

        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = _load(Qwen2AudioForConditionalGeneration, model_id, revision, dtype, device)
        self.device = next(self.model.parameters()).device
        self.scorer = LabelScorer(self.model, self.processor.tokenizer, self.device)

    def prompt_state(self, conv: list[dict], audio: np.ndarray | None):
        """Model state after the prompt (up to the start of the assistant's answer)."""
        import torch

        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        kw = {"text": text, "return_tensors": "pt"}
        if audio is not None:
            kw.update(audio=audio, sampling_rate=SAMPLE_RATE)
        inp = {k: v.to(self.device) for k, v in self.processor(**kw).items()}
        if "input_features" in inp:
            inp["input_features"] = inp["input_features"].to(next(self.model.parameters()).dtype)
        with torch.no_grad():
            out = self.model(**inp, use_cache=True)
        return out.past_key_values, int(inp["input_ids"].shape[1])


def run_setting(prompter: AudioPrompter, name: str, split: str, utts, spec: PromptSpec, with_audio: bool,
                audio_dir: Path, out_root: Path) -> dict:
    cond = f"{spec.variant}-{'audio' if with_audio else 'no_audio'}"
    out = out_root / name / split / cond
    if (out / "result.json").exists():
        log.info("%s already finished", out)
        return json.loads((out / "result.json").read_text(encoding="utf-8"))
    preds, probs, prompt_tokens = [], {}, []
    t0 = time.time()
    for j, u in enumerate(utts):
        audio = None
        if with_audio:
            x, sr = read_wav(libritts_path(audio_dir, "dev" if split.startswith("dev") else "test", u.utt_id))
            audio = resample(x, sr)
        cache, n = prompter.prompt_state(conversation(u, spec, with_audio), audio)
        prompt_tokens.append(n)
        labels, p = prompter.scorer.score_utterance(cache, "", u, task=3)
        preds.append(labels)
        probs[u.utt_id] = p
        if j % 200 == 0:
            log.info("%s: %d/%d utterances (%.0fs)", cond, j + 1, len(utts), time.time() - t0)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {3: evaluate(utts, preds, 3),
               2: evaluate(utts, [["1" if x == "2" else x for x in ps] for ps in preds], 2)}
    scores, gold = [], []
    for u in utts:
        for num, (i, _) in enumerate(word_items(u), start=1):
            scores.append(sum(probs[u.utt_id][num][1:]))
            gold.append(u.labels(2)[i] == "1")
    auc = roc_auc(np.array(scores), np.array(gold))
    metrics[2]["full"]["auc_prominent"] = auc
    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for task, m in metrics.items():
            f.write(json.dumps(make_record(experiment="audio_prompting", model=name, task=task, split=split,
                                           condition={"variant": spec.variant, "format": "lines", "shots": 0,
                                                      "audio": with_audio, "derived": task == 2},
                                           metrics=m), sort_keys=True) + "\n")
    write_predictions_gz(out / "predictions.tsv.gz", utts, preds, 3)
    with gzip.open(out / "probabilities.json.gz", "wt", encoding="utf-8") as f:
        json.dump(probs, f)
    result = {"experiment": "audio_prompting", "model": name, "split": split, "audio": with_audio,
              "prompt": {"variant": spec.variant, "format": "lines", "shots": 0},
              "utterances": len(utts), "mean_prompt_tokens": round(float(np.mean(prompt_tokens)), 1),
              "accuracy": {t: m["full"]["accuracy"] for t, m in metrics.items()},
              "macro_f1_3way": metrics[3]["full"]["macro_f1"], "auc_prominent": auc,
              "time_s": round(time.time() - t0, 1),
              "example_prompt": conversation(utts[0], spec, with_audio), "environment": environment_info()}
    write_json(out / "result.json", result)
    log.info("%s: 3-way %.2f, 2-way %.2f, AUC %s", cond, result["accuracy"][3], result["accuracy"][2], auc)
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="registry name or a local path")
    p.add_argument("--revision", default=None)
    p.add_argument("--split", default="test", help="test, dev, dev500 or test<N>")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=["linguistic"])
    p.add_argument("--no-audio", action="store_true", help="only the text-only control")
    p.add_argument("--both", action="store_true", help="with and without audio")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--audio-dir", type=Path, default=default_audio_dir())
    p.add_argument("--out-root", type=Path,
                   default=Path(os.environ.get("PROSODY_RUNS", "runs")) / "audio_prompting")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    from prosody_llms.config import get_model

    try:
        spec = get_model(args.model)
        model_id, revision, name = spec.id, spec.revision, spec.name
    except KeyError:
        model_id, revision, name = args.model, args.revision, Path(args.model).name
    prompter = AudioPrompter(model_id, revision, args.dtype, args.device)
    utts = load_split(args.data_dir, args.split)
    if args.max_examples:
        utts = utts[: args.max_examples]
    modes = [True, False] if args.both else [not args.no_audio]
    for variant in args.variants:
        for with_audio in modes:
            run_setting(prompter, name, args.split, utts, PromptSpec(variant, "lines", 3, 0), with_audio,
                        args.audio_dir, args.out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
