import numpy as np
import pytest

from prosody_llms.data.corpus import read_split
from prosody_llms.features.alignment import align, render_text, word_token_map
from prosody_llms.features.pooling import pool_words


def test_render_text_natural_and_spaced():
    words = ["He", "said", '"', "yes", '"', ",", "(", "twice", ")", "."]
    text, spans = render_text(words, "natural")
    assert text == 'He said "yes", (twice).'
    assert [text[s:e] for s, e in spans] == words
    text, spans = render_text(words, "spaced")
    assert text == " ".join(words)
    assert [text[s:e] for s, e in spans] == words
    with pytest.raises(ValueError):
        render_text(words, "other")


def test_word_token_map_handles_leading_spaces_and_shared_tokens():
    text = "Hi there."
    spans = [(0, 2), (3, 8), (8, 9)]
    # tokens: [BOS], "Hi", " there", "." ; and a variant where "there." is one token
    assert word_token_map([(0, 0), (0, 2), (2, 8), (8, 9)], spans, text) == [[1], [2], [3]]
    assert word_token_map([(0, 2), (2, 9)], spans, text) == [[0], [1], [1]]


def test_align_with_wordpiece(tiny_bert, corpus_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_bert)
    u = read_split(corpus_dir / "test.txt")[2]  # "dogs" -> dog ##s
    a = align(tok, u.words)
    assert tok.convert_ids_to_tokens(a.input_ids) == ["[CLS]", "dog", "##s", "[SEP]"]
    assert a.word_tokens == [[1, 2]] and a.shared == 0 and a.unaligned == []


def test_pool_words_numpy_and_torch():
    h = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    words = [[1], [2, 3]]
    assert np.array_equal(pool_words(h, words, "first"), h[:, [1, 2]])
    assert np.array_equal(pool_words(h, words, "last"), h[:, [1, 3]])
    assert np.allclose(pool_words(h, words, "mean")[:, 1], h[:, 2:4].mean(axis=1))
    torch = pytest.importorskip("torch")
    t = torch.tensor(h)
    assert torch.allclose(pool_words(t, words, "mean"), torch.tensor(pool_words(h, words, "mean")))
    with pytest.raises(ValueError):
        pool_words(h, [[]], "first")


def test_hf_extractor_shapes_and_batch_invariance(tiny_bert, corpus_dir):
    from prosody_llms.features.extractor import HFTextExtractor

    utts = read_split(corpus_dir / "test.txt")
    one = list(HFTextExtractor(str(tiny_bert), device="cpu", batch_size=1).extract(utts))
    ext = HFTextExtractor(str(tiny_bert), device="cpu", batch_size=3)
    many = list(ext.extract(utts))
    assert ext.n_layers == 3 and ext.dim == 16
    assert [f.shape for f in many] == [(3, len(u.tokens), 16) for u in utts]
    for a, b in zip(one, many):  # padding must not change the features
        assert np.allclose(a, b, atol=1e-5)
    assert ext.alignment_stats == {"utterances": 3, "shared_tokens": 0}


def test_discourse_prefixes_and_alignment(tiny_bert, corpus_dir):
    from transformers import AutoTokenizer

    from prosody_llms.features.alignment import context_size, discourse_prefixes

    train = read_split(corpus_dir / "train_360.txt")
    pre = discourse_prefixes(train, 1)
    assert pre["100_1_000000_000000"] == [] and pre["100_1_000000_000001"] == ["The", "cat", "sat", "."]
    assert pre["200_2_000001_000000"] == []  # another chapter
    assert context_size("prev3") == 3 and context_size("plain") == 0
    tok = AutoTokenizer.from_pretrained(tiny_bert)
    u = read_split(corpus_dir / "test.txt")[2]  # "dogs"
    plain = align(tok, u.words)
    a = align(tok, u.words, context="prev1", prefix=["the", "dog"])
    words = tok.convert_ids_to_tokens(a.input_ids)
    assert words[:3] == ["[CLS]", "the", "dog"] and [words[i] for i in a.word_tokens[0]] == ["dog", "##s"]
    assert len(a.input_ids) == len(plain.input_ids) + 2
    short = align(tok, u.words, context="prev1", prefix=["the", "dog", "sat"], max_length=5)
    assert len(short.input_ids) <= 5 and [tok.convert_ids_to_tokens(short.input_ids)[i] for i in short.word_tokens[0]] == ["dog", "##s"]
