import gzip
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from prosody_llms.data.corpus import read_split  # noqa: E402
from prosody_llms.prompting import local  # noqa: E402
from prosody_llms.prompting.prompts import PromptSpec, chat_messages, select_exemplars  # noqa: E402

TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")


@pytest.fixture
def tiny_causal(tmp_path, corpus_dir):
    """A 2-layer causal LM with a byte-level BPE tokenizer and a chat template."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    train = read_split(corpus_dir / "train_360.txt")
    texts = [m["content"] for u in train for m in chat_messages(u, PromptSpec("linguistic"), train)]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts * 3, trainers.BpeTrainer(
        vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|user|>", "<|assistant|>", "<|end|>"]))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|end|>")
    fast.chat_template = TEMPLATE
    d = tmp_path / "tiny-causal"
    fast.save_pretrained(d)
    cfg = transformers.LlamaConfig(vocab_size=len(fast), hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                   max_position_embeddings=4096)
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(cfg).save_pretrained(d)
    return d


def _scorer(path):
    from prosody_llms.features.loading import load_text_model

    loaded = load_text_model(str(path), dtype="float32", device="cpu", head="causal")
    return local.LabelScorer(loaded.model, loaded.tokenizer, torch.device("cpu"))


def test_split_prompt_and_render(tiny_causal, corpus_dir):
    scorer = _scorer(tiny_causal)
    train = read_split(corpus_dir / "train_360.txt")
    spec = PromptSpec("minimal", "lines", 3, 2)
    ex = select_exemplars(train, 2, min_words=2)
    prefix, suffix = local.split_prompt(scorer.tok, chat_messages(ex[0], spec, ex)[:-1])
    assert prefix.count("<|assistant|>") == 2 and prefix.endswith("<|user|>\n")
    assert suffix == "<|end|>\n<|assistant|>\n"
    full = local.render(scorer.tok, chat_messages(train[0], spec, ex))
    item = chat_messages(train[0], spec, [])[-1]["content"]
    assert full == prefix + item + suffix


def test_reused_prefix_state_matches_a_full_pass(tiny_causal, corpus_dir):
    """Label probabilities with the cached prefix equal those from processing everything
    from scratch."""
    scorer = _scorer(tiny_causal)
    train = read_split(corpus_dir / "train_360.txt")
    utt = read_split(corpus_dir / "test.txt")[1]
    spec = PromptSpec("minimal", "lines", 3, 2)
    ex = select_exemplars(train, 2, min_words=2)
    prefix, suffix = local.split_prompt(scorer.tok, chat_messages(ex[0], spec, ex)[:-1])
    item = chat_messages(utt, spec, [])[-1]["content"] + suffix
    cache, _, _ = scorer.encode_prefix(prefix, chunk=7)  # small chunks: exercises chunking
    labels_a, probs_a = scorer.score_utterance(cache, item, utt)
    # From scratch: feed the prefix tokens as part of the item.
    orig = scorer.ids
    scorer.ids = lambda text: (orig(prefix) + orig(item)) if text == item else orig(text)
    labels_b, probs_b = scorer.score_utterance(None, item, utt)
    scorer.ids = orig
    assert labels_a == labels_b
    for n in probs_a:
        assert probs_a[n] == pytest.approx(probs_b[n], abs=1e-5)
        assert sum(probs_a[n]) == pytest.approx(1.0)
    # The cached prefix is not modified by scoring an item.
    labels_c, probs_c = scorer.score_utterance(cache, item, utt)
    assert probs_c == probs_a and labels_c[3] == "NA"  # "," is punctuation


def test_local_run_end_to_end(tiny_causal, corpus_dir, tmp_path):
    out = tmp_path / "icl"
    assert local.main(["--model", str(tiny_causal), "--split", "test", "--shots", "0", "2",
                       "--data-dir", str(corpus_dir), "--out-root", str(out), "--dtype", "float32",
                       "--device", "cpu", "--min-exemplar-words", "2"]) == 0
    for k in (0, 2):
        d = out / "tiny-causal" / "test" / f"linguistic-k{k}"
        res = json.loads((d / "result.json").read_text())
        assert res["prompt"]["shots"] == k and set(res["accuracy"]) == {"3", "2"}
        recs = [json.loads(line) for line in open(d / "records.jsonl")]
        assert {r["task"] for r in recs} == {2, 3}
        assert recs[0]["metrics"]["full"]["scored_words"] in (7,)
        with gzip.open(d / "probabilities.json.gz", "rt") as f:
            probs = json.load(f)
        assert len(probs) == 3 and all(len(v) == 3 for p in probs.values() for v in p.values())
    k2 = json.loads((out / "tiny-causal" / "test" / "linguistic-k2" / "result.json").read_text())
    k0 = json.loads((out / "tiny-causal" / "test" / "linguistic-k0" / "result.json").read_text())
    assert k2["prefix_tokens"] > k0["prefix_tokens"]
    # Finished settings are not repeated; settings beyond the context are skipped.
    with pytest.raises(ValueError, match="cannot choose"):
        local.main(["--model", str(tiny_causal), "--split", "test", "--shots", "4",
                    "--data-dir", str(corpus_dir), "--out-root", str(out), "--dtype", "float32",
                    "--device", "cpu", "--min-exemplar-words", "2"])
