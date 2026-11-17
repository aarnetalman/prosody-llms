"""Parse model answers into one label per corpus word.

Every format is parsed tolerantly (code fences, reasoning tags, extra text, minor format
deviations). Words the parser cannot label get ``DEFAULT_LABEL`` (non-prominent) and are
counted, so every run reports its coverage. Punctuation always gets ``NA``.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field

from prosody_llms.data.corpus import LABEL_NA, Utterance
from prosody_llms.prompting.prompts import PromptSpec, word_items

DEFAULT_LABEL = "0"
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```[a-zA-Z]*\n?|```")
_LINE = re.compile(r"^\s*(\d+)\s*[.):\-]?\s*(.*?)\s*[:=\-]?\s*\b([0-2])\s*$")
_WORD_CHARS = re.compile(r"[^\w']+", re.U)


@dataclass
class Parsed:
    labels: list[str]  # one per token; NA for punctuation
    n_words: int
    parsed: int  # words that received a label from the answer
    stats: dict = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return self.parsed / self.n_words if self.n_words else 1.0


def clean(text: str | None) -> str:
    text = _THINK.sub("", text or "")
    return _FENCE.sub("", text).strip()


def _assemble(utt: Utterance, by_number: dict[int, str], stats: dict) -> Parsed:
    items = word_items(utt)
    labels = [LABEL_NA if t.is_punct else DEFAULT_LABEL for t in utt.tokens]
    parsed = 0
    for n, (i, _) in enumerate(items, start=1):
        if n in by_number:
            labels[i] = by_number[n]
            parsed += 1
    return Parsed(labels, len(items), parsed, stats)


def parse_lines(text: str, utt: Utterance, task: int) -> Parsed:
    valid = {"0", "1", "2"} if task == 3 else {"0", "1"}
    n = len(word_items(utt))
    by_number, bad = {}, 0
    rows = [ln for ln in clean(text).splitlines() if ln.strip()]
    for ln in rows:
        m = _LINE.match(ln)
        if not m:
            bad += 1
            continue
        num, label = int(m.group(1)), m.group(3)
        if 1 <= num <= n and label in valid and num not in by_number:
            by_number[num] = label
        else:
            bad += 1
    if not by_number and len(rows) == n:  # no numbers: fall back to the order of the lines
        for num, ln in enumerate(rows, start=1):
            tail = ln.strip().split()[-1] if ln.strip() else ""
            if tail in valid:
                by_number[num] = tail
    return _assemble(utt, by_number, {"bad_lines": bad})


def parse_pairs(text: str, utt: Utterance, task: int) -> Parsed:
    """Labels given as "word label" pairs in running text, in the order of the words, e.g.
    ``1 Stephen's 0 heart 0 began 0 ...`` (all on one line, numbers optional). Each corpus word
    is matched to the next token that equals it (ignoring case and punctuation) within a short
    window, and takes the label directly after it."""
    valid = {"0", "1", "2"} if task == 3 else {"0", "1"}
    toks = clean(text).split()
    items = word_items(utt)
    by_number, pos, skipped = {}, 0, 0
    for num, (_, word) in enumerate(items, start=1):
        target = _norm(word)
        for j in range(pos, min(pos + 6, len(toks) - 1)):
            if _norm(toks[j]) == target and toks[j + 1].strip(".,;:") in valid:
                by_number[num] = toks[j + 1].strip(".,;:")
                pos = j + 2
                break
        else:
            skipped += 1
    return _assemble(utt, by_number, {"unmatched_words": skipped})


def parse_json(text: str, utt: Utterance, task: int) -> Parsed:
    n = len(word_items(utt))
    cleaned = clean(text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    obj = None
    if start != -1 and end > start:
        blob = cleaned[start:end + 1]
        for candidate in (blob, blob.replace("'", '"')):
            try:
                obj = json.loads(candidate)
                break
            except ValueError:
                continue
    if not isinstance(obj, dict):
        return _assemble(utt, {}, {"json_error": True})
    levels = ("2", "1") if task == 3 else ("1",)
    aliases = {"prominent": "1" if task == 2 else "2", "strongly prominent": "2",
               "somewhat prominent": "1"}
    by_number: dict[int, str] = {}
    for key, values in obj.items():
        level = str(key).strip().lower()
        level = aliases.get(level, level)
        if level not in levels or not isinstance(values, list):
            continue
        for v in values:
            try:
                num = int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= num <= n:
                by_number.setdefault(num, level)  # a word listed twice keeps its first level
    # Every word not listed is non-prominent: the whole utterance counts as parsed.
    for num in range(1, n + 1):
        by_number.setdefault(num, "0")
    return _assemble(utt, by_number, {"json_error": False})


def _norm(token: str) -> str:
    return _WORD_CHARS.sub("", token.strip("*")).lower()


def parse_marked(text: str, utt: Utterance, task: int) -> Parsed:
    items = word_items(utt)
    out_tokens = [t for t in re.findall(r"\*[^*\s][^*]*?\*|\S+", clean(text).splitlines()[0]
                                        if clean(text) else "")]
    # split multi-word *...* spans into single marked words
    expanded = []
    for t in out_tokens:
        if t.startswith("*") and t.endswith("*") and " " in t:
            expanded += [f"*{w}*" for w in t.strip("*").split()]
        else:
            expanded.append(t)
    out_norm = [_norm(t) for t in expanded]
    ref_norm = [_norm(w) for _, w in items]
    by_number: dict[int, str] = {}
    caps_ambiguous = 0
    matcher = difflib.SequenceMatcher(a=ref_norm, b=out_norm, autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            n_idx, o_idx = block.a + k, block.b + k
            orig = items[n_idx][1]
            tok = expanded[o_idx]
            core = re.sub(r"[^\w]", "", tok)
            starred = "*" in tok
            has_lower = any(c.islower() for c in orig)
            caps = has_lower and core.isupper() and any(c.isalpha() for c in core)
            if not has_lower and any(c.isalpha() for c in orig):
                caps_ambiguous += 1
            if task == 3:
                label = "2" if caps else "1" if starred else "0"
            else:
                label = "1" if caps or starred else "0"
            by_number[n_idx + 1] = label
    return _assemble(utt, by_number, {"caps_ambiguous": caps_ambiguous,
                                      "unmatched_output": len(out_norm) - len(by_number)})


def parse(text: str | None, utt: Utterance, spec: PromptSpec) -> Parsed:
    if spec.fmt == "lines":
        return parse_lines(text or "", utt, spec.task)
    if spec.fmt == "json":
        return parse_json(text or "", utt, spec.task)
    return parse_marked(text or "", utt, spec.task)


def label_positions(text: str, utt: Utterance, task: int) -> dict[int, int]:
    """For the ``lines`` format: {word number: character offset of its label in ``text``}
    (offsets in the raw text, for matching log-probability tokens)."""
    valid = {"0", "1", "2"} if task == 3 else {"0", "1"}
    n = len(word_items(utt))
    out, pos = {}, 0
    for raw in text.splitlines(keepends=True):
        m = _LINE.match(raw.rstrip("\n"))
        if m and m.group(3) in valid:
            num = int(m.group(1))
            if 1 <= num <= n and num not in out:
                out[num] = pos + m.start(3)
        pos += len(raw)
    return out
