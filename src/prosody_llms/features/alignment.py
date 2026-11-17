"""Render corpus utterances as text and align corpus words to model tokens.

The corpus stores one token per line with punctuation split off (``mr Quilter is
entirely free from affectation of any kind .``). Encoders in 2019 saw each word
tokenised separately; LLMs should see ordinary running text instead. ``render_text``
produces either form together with the character span of every corpus word, and
``word_token_map`` maps those spans to token indices using the tokenizer's offsets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

STYLES = ("natural", "spaced")
NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}", "...", "…", "%",
                   "”", "’"}
NO_SPACE_AFTER = {"(", "[", "{", "“", "‘", "$"}
SYMMETRIC_QUOTES = {'"'}


def render_text(words: Sequence[str], style: str = "natural") -> tuple[str, list[tuple[int, int]]]:
    """Return ``(text, spans)`` where ``text[s:e]`` is word *i* for ``spans[i] == (s, e)``.

    ``spaced`` joins the words with single spaces. ``natural`` removes the space before
    closing punctuation and after opening brackets, and treats a standalone ``"`` as an
    opening quote the first time and a closing quote the next.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    quote_open = False
    glue_next = False
    for i, w in enumerate(words):
        space = i > 0
        if style == "natural" and i > 0:
            if w in NO_SPACE_BEFORE or glue_next:
                space = False
            if w in SYMMETRIC_QUOTES and quote_open:
                space = False
        glue_next = False
        if style == "natural":
            if w in NO_SPACE_AFTER:
                glue_next = True
            elif w in SYMMETRIC_QUOTES:
                if not quote_open:
                    glue_next = True
                quote_open = not quote_open
        if space:
            parts.append(" ")
            pos += 1
        spans.append((pos, pos + len(w)))
        parts.append(w)
        pos += len(w)
    return "".join(parts), spans


def word_token_map(
    offsets: Sequence[tuple[int, int]], spans: Sequence[tuple[int, int]], text: str
) -> list[list[int]]:
    """Token indices for every word, from the tokenizer's character ``offsets``.

    A token belongs to every word whose span it overlaps, ignoring whitespace inside the
    token (byte-level BPE tokens usually start with the preceding space). Special tokens
    (empty offsets) belong to no word. A token can be shared by two words when the
    tokenizer merges them (for example ``word.`` as one token); ``AlignedText.shared``
    counts such cases.
    """
    out: list[list[int]] = [[] for _ in spans]
    for t, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e <= s:
            continue
        for w, (ws, we) in enumerate(spans):
            if s < we and ws < e:
                out[w].append(t)
    return out


@dataclass
class AlignedText:
    text: str
    input_ids: list[int]
    word_tokens: list[list[int]]  # token indices (into input_ids) for every corpus word
    shared: int  # tokens assigned to more than one word

    @property
    def unaligned(self) -> list[int]:
        """Indices of words that received no token (should be empty)."""
        return [i for i, t in enumerate(self.word_tokens) if not t]


CONTEXTS = ("plain", "echo", "prev1", "prev3")
ECHO_SEPARATOR = "\n"


def context_size(context: str) -> int:
    """Number of preceding utterances for a ``prevK`` context (0 otherwise)."""
    return int(context[4:]) if context.startswith("prev") else 0


def discourse_prefixes(utts, k: int) -> dict[str, list[str]]:
    """For every utterance, the words of the ``k`` utterances before it in the same chapter
    (same speaker and chapter in the LibriTTS id, ordered by paragraph and sentence), as
    they occur in the split. The first utterances of a chapter get fewer (or none)."""
    import collections

    chapters = collections.defaultdict(list)
    for u in utts:
        spk, ch, para, sent = u.utt_id.split("_")
        chapters[(spk, ch)].append(((int(para), int(sent)), u))
    out = {}
    for items in chapters.values():
        items.sort(key=lambda x: x[0])
        for i, (_, u) in enumerate(items):
            out[u.utt_id] = [w for _, v in items[max(0, i - k):i] for w in v.words]
    return out


def align(tokenizer, words: Sequence[str], style: str = "natural",
          add_special_tokens: bool = True, context: str = "plain",
          prefix: Sequence[str] = (), max_length: int | None = None) -> AlignedText:
    """Tokenise an utterance and map every corpus word to its tokens.

    ``context="echo"`` feeds the utterance twice (separated by a newline) and maps the
    words to the **second** copy, so a left-to-right model has seen the whole utterance
    when it reaches each word (the "echo embeddings" idea). ``plain`` feeds it once.
    ``prevK`` puts the words of the preceding utterances (``prefix``) before it; words are
    dropped from the start of the prefix until the input fits ``max_length`` tokens.
    Requires a fast tokenizer (``return_offsets_mapping``).
    """
    if context not in CONTEXTS:
        raise ValueError(f"context must be one of {CONTEXTS}")
    text, spans = render_text(words, style)
    if context.startswith("prev") and prefix:
        prefix = list(prefix)
        while prefix:
            ptext, _ = render_text(prefix, style)
            shift = len(ptext) + 1
            full = ptext + " " + text
            n = len(tokenizer(full, add_special_tokens=add_special_tokens)["input_ids"])
            if max_length is None or n <= max_length:
                text, spans = full, [(s + shift, e + shift) for s, e in spans]
                break
            prefix = prefix[max(1, (n - max_length) // 2):]  # drop at least one word
    if context == "echo":
        shift = len(text) + len(ECHO_SEPARATOR)
        text = text + ECHO_SEPARATOR + text
        spans = [(s + shift, e + shift) for s, e in spans]
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    word_tokens = word_token_map(offsets, spans, text)
    counts: dict[int, int] = {}
    for toks in word_tokens:
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
    shared = sum(1 for c in counts.values() if c > 1)
    return AlignedText(text, list(enc["input_ids"]), word_tokens, shared)
