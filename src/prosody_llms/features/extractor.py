"""Feature extractor interface and the Hugging Face implementation.

Every extractor returns, per utterance, an array of shape ``(layers, words, dim)`` with
one vector per corpus word (punctuation included, so indices line up with the corpus)
and per layer, where layer 0 is the embedding output.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

import numpy as np

from prosody_llms.data.corpus import Utterance
from prosody_llms.features.alignment import AlignedText, align
from prosody_llms.features.pooling import pool_words


class FeatureExtractor(Protocol):
    name: str

    @property
    def n_layers(self) -> int:
        """Number of returned layers, including the embedding layer."""

    @property
    def dim(self) -> int: ...

    def extract(self, utts: Sequence[Utterance]) -> Iterator[np.ndarray]:
        """Yield one ``(layers, words, dim)`` array per utterance, in order."""


def length_batches(lengths: Sequence[int], max_tokens: int) -> list[list[int]]:
    """Group indices into batches of at most ``max_tokens`` padded tokens.

    Indices are sorted by length (ties by index), so the batches are deterministic and
    padding is small. A single sequence longer than ``max_tokens`` gets its own batch.
    """
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if cur and (len(cur) + 1) * lengths[i] > max_tokens:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


class HFTextExtractor:
    """All hidden layers of a Hugging Face model, pooled to corpus words.

    Text is rendered with ``style`` (``natural`` running text, or ``spaced``) and fed
    once (``context="plain"``) or twice (``"echo"``, words read from the second copy).
    Multimodal checkpoints are loaded as their text-only model.
    """

    def __init__(self, model: str, revision: str | None = None, *, style: str = "natural",
                 pooling: str = "first", context: str = "plain", device: str | None = None,
                 dtype: str = "float32", batch_size: int = 16, add_special_tokens: bool = True,
                 out_dtype=np.float32, name: str | None = None):
        from prosody_llms.features.loading import load_text_model

        loaded = load_text_model(model, revision, dtype=dtype, device=device)
        self.name = name or model
        self.model, self.tokenizer = loaded.model, loaded.tokenizer
        self.model_class = loaded.model_class
        self.key_mapping = loaded.key_mapping
        self._n_layers, self._dim = loaded.n_layers, loaded.dim
        self.device = next(self.model.parameters()).device
        self.style, self.pooling, self.context = style, pooling, context
        self.batch_size = batch_size
        self.add_special_tokens = add_special_tokens
        self.out_dtype = out_dtype
        self.alignment_stats = {"utterances": 0, "shared_tokens": 0}

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def dim(self) -> int:
        return self._dim

    def register_discourse(self, splits) -> None:
        """For a ``prevK`` context: the preceding utterances of every utterance, from the
        full splits (before any subsampling)."""
        from prosody_llms.features.alignment import context_size, discourse_prefixes

        k = context_size(self.context)
        self.prefixes = {}
        for utts in splits:
            self.prefixes.update(discourse_prefixes(utts, k))

    def align(self, u: Utterance) -> AlignedText:
        prefix = getattr(self, "prefixes", {}).get(u.utt_id, ())
        limit = getattr(self.tokenizer, "model_max_length", None)
        limit = limit if limit and limit < 100000 else None
        a = align(self.tokenizer, u.words, self.style, self.add_special_tokens, self.context,
                  prefix=prefix, max_length=limit)
        if a.unaligned:
            raise ValueError(f"{u.utt_id}: words without tokens: {a.unaligned}")
        return a

    def forward(self, aligned: Sequence[AlignedText]):
        """Pooled hidden states for a batch: a list of ``(layers, words, dim)`` tensors on
        the model's device and in its dtype.

        If the GPU runs out of memory, the batch is split in half and retried. Features do
        not depend on how utterances are batched, so the results are the same.
        """
        import torch

        try:
            return self._forward(aligned)
        except torch.OutOfMemoryError:
            if len(aligned) == 1:
                raise
        # Retry outside the except block, so the failed attempt's tensors (still referenced
        # by the traceback inside it) can be freed first.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.oom_splits = getattr(self, "oom_splits", 0) + 1
        half = len(aligned) // 2
        return self.forward(aligned[:half]) + self.forward(aligned[half:])

    def _forward(self, aligned: Sequence[AlignedText]):
        import torch

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        maxlen = max(len(a.input_ids) for a in aligned)
        ids = torch.full((len(aligned), maxlen), pad_id, dtype=torch.long)
        mask = torch.zeros((len(aligned), maxlen), dtype=torch.long)
        for i, a in enumerate(aligned):  # right padding keeps token positions unchanged
            ids[i, : len(a.input_ids)] = torch.tensor(a.input_ids)
            mask[i, : len(a.input_ids)] = 1
        with torch.no_grad():
            out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                             output_hidden_states=True)
        hidden = out.hidden_states  # tuple of (batch, tokens, dim), one per layer
        del out
        for a in aligned:
            self.alignment_stats["utterances"] += 1
            self.alignment_stats["shared_tokens"] += a.shared
        # Pool layer by layer: stacking all layers first would copy every hidden state.
        return [torch.stack([pool_words(h[i : i + 1, : len(a.input_ids)], a.word_tokens,
                                        self.pooling)[0] for h in hidden])
                for i, a in enumerate(aligned)]

    def extract(self, utts: Sequence[Utterance]) -> Iterator[np.ndarray]:
        for start in range(0, len(utts), self.batch_size):
            batch = [self.align(u) for u in utts[start : start + self.batch_size]]
            for h in self.forward(batch):
                yield h.float().cpu().numpy().astype(self.out_dtype)


class RandomWordExtractor:
    """A control representation with no learned knowledge: every word type (lower-cased) gets a
    fixed random vector, derived from a hash of the word, so the same word always has the same
    vector in every split and run. One "layer" (0). Same interface as ``HFTextExtractor`` for
    ``prosody_llms.probing.nonlinear.extract_layers``."""

    def __init__(self, dim: int = 1024, device: str | None = None, name: str | None = None):
        import torch

        self._dim = dim
        self.name = name or f"random-words-{dim}"
        self.model_class = "random word vectors"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._cache: dict[str, np.ndarray] = {}

    @property
    def n_layers(self) -> int:
        return 1

    @property
    def dim(self) -> int:
        return self._dim

    def vector(self, word: str) -> np.ndarray:
        import hashlib

        w = word.lower()
        v = self._cache.get(w)
        if v is None:
            seed = int.from_bytes(hashlib.sha256(w.encode("utf-8")).digest()[:8], "little")
            v = (np.random.default_rng(seed).standard_normal(self._dim) / np.sqrt(self._dim)).astype(np.float32)
            self._cache[w] = v
        return v

    def align(self, u: Utterance):
        from types import SimpleNamespace

        return SimpleNamespace(input_ids=[0] * len(u.tokens), words=u.words)

    def forward(self, aligned):
        import torch

        return [torch.from_numpy(np.stack([self.vector(w) for w in a.words]))[None] for a in aligned]
