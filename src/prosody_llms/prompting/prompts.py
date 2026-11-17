"""Prompts for marking prominent words.

Each item shows the utterance as natural text and a numbered list of its words.
Punctuation is not numbered: it is never scored. The model answers in one of three
formats:

* ``lines``: one line per word, ``<number> <word> <label>``. The label is a single token,
  so its top log-probabilities give a distribution over labels for every word;
* ``json``: the numbers of the prominent words, per level;
* ``marked``: the sentence rewritten with prominent words marked (3-way: strongly
  prominent in CAPITALS, somewhat prominent between *asterisks*; 2-way: CAPITALS).

Three instruction variants are used: ``minimal``, ``linguistic`` (what prominence is and
how it is realised) and ``narrator`` (an audiobook narrator's perspective). Few-shot
examples come from ``train_360``, chosen once by a fixed hash so every model and item
sees the same ones, and are given as earlier user/assistant turns in the same format.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from prosody_llms.data.corpus import Utterance
from prosody_llms.data.subsets import fixed_subset, scored_words
from prosody_llms.features.alignment import render_text

FORMATS = ("lines", "json", "marked")
VARIANTS = ("minimal", "linguistic", "narrator")

LEVELS = {
    3: "0 = not prominent, 1 = somewhat prominent, 2 = strongly prominent",
    2: "0 = not prominent, 1 = prominent",
}

INSTRUCTIONS = {
    "minimal": "Mark which words in the sentence are prosodically prominent when it is read aloud.",
    "linguistic": (
        "Prosodic prominence is the emphasis a speaker gives to some words when reading aloud: "
        "prominent words stand out from their neighbours through higher or more varied pitch, "
        "greater loudness and longer duration. Prominence typically falls on content words that "
        "carry new, important or contrastive information, and towards the end of a phrase; "
        "function words and words repeating given information are usually not prominent. "
        "Mark which words in the sentence would be prominent when it is read aloud in a "
        "natural way."),
    "narrator": (
        "You are an experienced audiobook narrator reading a classic novel aloud. Decide which "
        "words of the following sentence you would emphasise, as you would naturally read it "
        "to listeners."),
}

FORMAT_INSTRUCTIONS = {
    ("lines", 3): "Answer with one line per numbered word, in order: the number, the word and its "
                  "label (0, 1 or 2), separated by spaces. Write nothing else.",
    ("lines", 2): "Answer with one line per numbered word, in order: the number, the word and its "
                  "label (0 or 1), separated by spaces. Write nothing else.",
    ("json", 3): 'Answer with a JSON object only: {"2": [numbers of strongly prominent words], '
                 '"1": [numbers of somewhat prominent words]}. All other words are not prominent.',
    ("json", 2): 'Answer with a JSON object only: {"1": [numbers of prominent words]}. All other '
                 "words are not prominent.",
    ("marked", 3): "Answer by rewriting the sentence exactly, with strongly prominent words in "
                   "CAPITALS and somewhat prominent words between *asterisks*. Write nothing else.",
    ("marked", 2): "Answer by rewriting the sentence exactly, with prominent words in CAPITALS. "
                   "Write nothing else.",
}


@dataclass(frozen=True)
class PromptSpec:
    variant: str = "minimal"
    fmt: str = "lines"
    task: int = 3  # 3: labels 0/1/2; 2: native 2-way prompt (labels 0/1)
    shots: int = 0

    def __post_init__(self):
        if self.variant not in VARIANTS or self.fmt not in FORMATS or self.task not in (2, 3):
            raise ValueError(f"invalid prompt spec {self}")

    @property
    def name(self) -> str:
        return f"{self.variant}-{self.fmt}-{self.task}way-k{self.shots}"


def word_items(utt: Utterance) -> list[tuple[int, str]]:
    """(token index in the utterance, word) for every scored (non-punctuation) word."""
    return [(i, t.word) for i, t in enumerate(utt.tokens) if not t.is_punct]


def user_message(utt: Utterance, spec: PromptSpec) -> str:
    text, _ = render_text(utt.words, "natural")
    lines = "\n".join(f"{n} {w}" for n, (_, w) in enumerate(word_items(utt), start=1))
    return (f"{INSTRUCTIONS[spec.variant]}\nLabels: {LEVELS[spec.task]}.\n"
            f"{FORMAT_INSTRUCTIONS[(spec.fmt, spec.task)]}\n\nSentence: {text}\nWords:\n{lines}")


def gold_answer(utt: Utterance, spec: PromptSpec) -> str:
    """The answer for an example utterance, in the requested format (few-shot turns)."""
    items = word_items(utt)
    labels = [utt.labels(spec.task)[i] for i, _ in items]
    if spec.fmt == "lines":
        return "\n".join(f"{n} {w} {lab}" for n, ((_, w), lab) in enumerate(zip(items, labels), 1))
    if spec.fmt == "json":
        levels = ("2", "1") if spec.task == 3 else ("1",)
        return json.dumps({lev: [n for n, lab in enumerate(labels, 1) if lab == lev] for lev in levels})
    marked = list(utt.words)
    for (i, w), lab in zip(items, labels):
        if (spec.task == 3 and lab == "2") or (spec.task == 2 and lab == "1"):
            marked[i] = w.upper()
        elif spec.task == 3 and lab == "1":
            marked[i] = f"*{w}*"
    return render_text(marked, "natural")[0]


def select_exemplars(train: Sequence[Utterance], k: int, seed: int = 0,
                     min_words: int = 6, max_words: int = 20) -> list[Utterance]:
    """``k`` training utterances of moderate length, chosen by a fixed hash."""
    if k == 0:
        return []
    pool = [u for u in train if min_words <= scored_words(u) <= max_words]
    if len(pool) < k:
        raise ValueError(f"only {len(pool)} training utterances have {min_words}-{max_words} "
                         f"words; cannot choose {k} examples")
    return fixed_subset(pool, k, seed=seed)


def chat_messages(utt: Utterance, spec: PromptSpec, exemplars: Sequence[Utterance]) -> list[dict]:
    messages = []
    for ex in exemplars:
        messages.append({"role": "user", "content": user_message(ex, spec)})
        messages.append({"role": "assistant", "content": gold_answer(ex, spec)})
    messages.append({"role": "user", "content": user_message(utt, spec)})
    return messages
