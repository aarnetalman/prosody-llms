"""Word-level representations from pre-trained models.

* ``alignment``: render an utterance as text and map every corpus word to the sub-word
  tokens of any Hugging Face fast tokenizer (needed for LLMs, whose tokenisers do not
  follow the corpus words);
* ``pooling``: turn token-level hidden states into one vector per word and layer;
* ``extractor``: the extractor interface and a basic local implementation.
"""

from prosody_llms.features.alignment import AlignedText, align, render_text, word_token_map
from prosody_llms.features.pooling import pool_words

__all__ = ["AlignedText", "align", "pool_words", "render_text", "word_token_map"]
