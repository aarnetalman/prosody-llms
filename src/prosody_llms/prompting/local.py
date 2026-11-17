"""Many-shot in-context learning with open models on a local GPU (no API, no cost).

    python -m prosody_llms.prompting.local --model qwen3.8-27b --shots 0 10 25 100 250 500

The prompts are the ``lines`` prompts of the API experiment (same instructions, same
fixed examples from ``train_360``, given as earlier chat turns). The model does not
generate freely. It writes the answer word by word: for every word the line
``<number> <word> `` is fed, the model's probabilities for the labels are read at the
point where the candidate lines first differ, the most probable label is written, and the
next line follows. There is nothing to parse, and every word gets its full label
distribution (also used for AUC).

The examples are the same for every item, so the model reads them once: the prompt is
split into a shared prefix (instructions and examples) and the item, and the model state
after the prefix is reused for every item. Example sets are nested (the first 10 of the
25, the first 25 of the 100, ...), so a curve over the number of shots compares like with
like. Settings whose prompt would not fit the model's context window are skipped.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from prosody_llms.config import get_model
from prosody_llms.data.corpus import LABEL_NA, Utterance, read_split
from prosody_llms.data.download import default_data_dir
from prosody_llms.eval.io import evaluate, make_record
from prosody_llms.prompting.prompts import VARIANTS, PromptSpec, chat_messages, select_exemplars, word_items
from prosody_llms.prompting.run import load_split, roc_auc, write_predictions_gz
from prosody_llms.utils.runinfo import environment_info, write_json

log = logging.getLogger("prosody_llms.prompting.local")
MARKER = "\u0000ITEM\u0000"
LABELS = ("0", "1", "2")


def render(tokenizer, messages: list[dict]) -> str:
    """The chat prompt as text, ending with the assistant's turn. Thinking is switched off
    for templates that support it (Qwen); models without a chat template get a plain
    User/Assistant layout."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    parts = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n" for m in messages]
    return "".join(parts) + "Assistant: "


def split_prompt(tokenizer, exemplar_messages: list[dict]) -> tuple[str, str]:
    """(prefix shared by all items, text that follows each item's user message)."""
    text = render(tokenizer, exemplar_messages + [{"role": "user", "content": MARKER}])
    i = text.index(MARKER)
    return text[:i], text[i + len(MARKER):]


class LabelScorer:
    """Writes the ``lines`` answer for one utterance word by word and reads the label
    distribution of every word."""

    def __init__(self, model, tokenizer, device):
        self.model, self.tok, self.device = model, tokenizer, device

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def forward(self, ids: list[int], cache):
        import torch

        x = torch.tensor([ids], device=self.device)
        out = self.model(input_ids=x, past_key_values=cache, use_cache=True)
        return out.logits[0, -1].float(), out.past_key_values

    def encode_prefix(self, text: str, chunk: int = 8192):
        """Model state after the shared prefix (processed in chunks to bound memory)."""
        import torch

        ids = self.ids(text)
        cache, logits = None, None
        with torch.no_grad():
            for s in range(0, len(ids), chunk):
                logits, cache = self.forward(ids[s:s + chunk], cache)
        return cache, logits, len(ids)

    def label_lines(self, n: int, word: str, labels: tuple[str, ...]):
        """Token ids of ``"<n> <word> <label>\\n"`` for every label, the length of their
        common prefix, and each label's first distinguishing token."""
        lines = [self.ids(f"{n} {word} {lab}\n") for lab in labels]
        p = 0
        while all(len(line) > p for line in lines) and len({line[p] for line in lines}) == 1:
            p += 1
        firsts = [line[p] if len(line) > p else None for line in lines]
        if None in firsts or len(set(firsts)) != len(labels):
            raise ValueError(f"labels are not distinguishable by one token after {lines[0][:p]}")
        return lines, p, firsts

    def score_utterance(self, prefix_cache, item_text: str, utt: Utterance, task: int = 3):
        """Labels (one per token, NA for punctuation) and {word number: [P(label)]}."""
        import torch

        labels = LABELS[:task]
        cache = copy.deepcopy(prefix_cache) if prefix_cache is not None else None
        pending = self.ids(item_text)
        out = [LABEL_NA if t.is_punct else "0" for t in utt.tokens]
        probs = {}
        with torch.no_grad():
            for n, (i, word) in enumerate(word_items(utt), start=1):
                lines, p, firsts = self.label_lines(n, word, labels)
                logits, cache = self.forward(pending + lines[0][:p], cache)
                dist = torch.softmax(logits[firsts], dim=0).tolist()
                best = int(np.argmax(dist))
                out[i] = labels[best]
                probs[n] = dist
                pending = lines[best][p:]
        return out, probs


def run_setting(scorer, model_name: str, split: str, utts, spec: PromptSpec, train, out_root: Path,
                max_positions: int | None, min_words: int = 6) -> dict | None:
    out = out_root / model_name / split / f"{spec.variant}-k{spec.shots}"
    if (out / "result.json").exists():
        log.info("%s already finished", out)
        return json.loads((out / "result.json").read_text(encoding="utf-8"))
    exemplars = select_exemplars(train, spec.shots, min_words=min_words)
    ex_messages = chat_messages(exemplars[0], spec, exemplars)[:-1] if exemplars else []
    prefix_text, suffix = split_prompt(scorer.tok, ex_messages)
    item_texts = [chat_messages(u, spec, [])[-1]["content"] + suffix for u in utts]
    longest_item = max(len(scorer.ids(t)) for t in item_texts) + 12 * max(len(word_items(u)) for u in utts)
    prefix_tokens = len(scorer.ids(prefix_text))
    if max_positions and prefix_tokens + longest_item > max_positions:
        log.warning("k=%d needs about %d tokens, more than the context of %d; skipped",
                    spec.shots, prefix_tokens + longest_item, max_positions)
        return None
    t0 = time.time()
    cache, _, _ = scorer.encode_prefix(prefix_text)
    t1 = time.time()
    preds, probs = [], {}
    for j, (u, text) in enumerate(zip(utts, item_texts)):
        labels, p = scorer.score_utterance(cache, text, u, task=3)
        preds.append(labels)
        probs[u.utt_id] = p
        if j % 100 == 0:
            log.info("k=%d: %d/%d utterances (%.0fs)", spec.shots, j + 1, len(utts), time.time() - t1)
    t2 = time.time()

    out.mkdir(parents=True, exist_ok=True)
    metrics = {3: evaluate(utts, preds, 3),
               2: evaluate(utts, [["1" if lab == "2" else lab for lab in ps] for ps in preds], 2)}
    scores, gold = [], []
    for u in utts:
        for num, (i, _) in enumerate(word_items(u), start=1):
            scores.append(sum(probs[u.utt_id][num][1:]))
            gold.append(u.labels(2)[i] == "1")
    auc = roc_auc(np.array(scores), np.array(gold))
    metrics[2]["full"]["auc_prominent"] = auc
    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for task, m in metrics.items():
            rec = make_record(experiment="icl_local", model=model_name, task=task, split=split,
                              condition={"variant": spec.variant, "format": "lines",
                                         "shots": spec.shots, "derived": task == 2},
                              metrics=m)
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    write_predictions_gz(out / "predictions.tsv.gz", utts, preds, 3)
    with gzip.open(out / "probabilities.json.gz", "wt", encoding="utf-8") as f:
        json.dump(probs, f)
    result = {"experiment": "icl_local", "model": model_name, "split": split,
              "prompt": {"variant": spec.variant, "format": "lines", "shots": spec.shots},
              "prefix_tokens": prefix_tokens, "utterances": len(utts),
              "accuracy": {t: m["full"]["accuracy"] for t, m in metrics.items()},
              "macro_f1_3way": metrics[3]["full"]["macro_f1"], "auc_prominent": auc,
              "time_s": {"prefix": round(t1 - t0, 1), "items": round(t2 - t1, 1)},
              "example_prompt_start": prefix_text[:2000], "example_item": item_texts[0],
              "environment": environment_info()}
    try:
        import torch

        if torch.cuda.is_available():
            result["environment"]["max_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    except ImportError:
        pass
    write_json(out / "result.json", result)
    log.info("k=%d: 3-way %.2f, 2-way %.2f, AUC %.3f", spec.shots, result["accuracy"][3],
             result["accuracy"][2], auc or float("nan"))
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="registry name or a local path")
    p.add_argument("--revision", default=None)
    p.add_argument("--split", default="dev500")
    # About 300 tokens per example: 500 examples (about 150K tokens) fit a 262K context.
    p.add_argument("--shots", type=int, nargs="+", default=[0, 10, 25, 100, 250, 500])
    p.add_argument("--variant", choices=VARIANTS, default="linguistic")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--min-exemplar-words", type=int, default=6)
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--out-root", type=Path,
                   default=Path(os.environ.get("PROSODY_RUNS", "runs")) / "icl")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from prosody_llms.features.loading import load_text_model

    try:
        spec = get_model(args.model)
        model_id, revision, name = spec.id, spec.revision, spec.name
    except KeyError:
        model_id, revision, name = args.model, args.revision, Path(args.model).name
    loaded = load_text_model(model_id, revision, dtype=args.dtype, device=args.device, head="causal")
    model = loaded.model
    device = next(model.parameters()).device
    cfg = getattr(model.config, "text_config", model.config)
    max_positions = getattr(cfg, "max_position_embeddings", None)
    scorer = LabelScorer(model, loaded.tokenizer, device)
    utts = load_split(args.data_dir, args.split)
    if args.max_examples:
        utts = utts[: args.max_examples]
    train = read_split(args.data_dir / "train_360.txt")
    for k in sorted(args.shots):
        run_setting(scorer, name, args.split, utts, PromptSpec(args.variant, "lines", 3, k), train,
                    args.out_root, max_positions, args.min_exemplar_words)
    return 0


if __name__ == "__main__":
    sys.exit(main())
