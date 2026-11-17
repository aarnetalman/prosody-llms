import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from prosody_llms.data.corpus import read_split  # noqa: E402
from prosody_llms.features.alignment import align, render_text  # noqa: E402
from prosody_llms.features.extractor import length_batches  # noqa: E402
from prosody_llms.probes.linear import LinearStats, solve  # noqa: E402
from prosody_llms.probes.ridge import RidgeAccumulator  # noqa: E402
from prosody_llms.probing import run as probing  # noqa: E402


def test_echo_maps_words_to_the_second_copy(tiny_bert, corpus_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_bert)
    u = read_split(corpus_dir / "test.txt")[0]  # The dog sat
    plain = align(tok, u.words)
    echo = align(tok, u.words, context="echo")
    assert echo.text == render_text(u.words)[0] + "\n" + render_text(u.words)[0]
    assert len(echo.input_ids) == 2 * len(plain.input_ids) - 2  # one [CLS] and one [SEP]
    first_copy = len(plain.input_ids) - 1
    assert all(t >= first_copy for toks in echo.word_tokens for t in toks)
    assert [echo.input_ids[t[0]] for t in echo.word_tokens] == \
        [plain.input_ids[t[0]] for t in plain.word_tokens]
    with pytest.raises(ValueError):
        align(tok, u.words, context="twice")


def test_length_batches():
    lengths = [5, 1, 3, 3, 9]
    assert length_batches(lengths, 6) == [[1, 2], [3], [0], [4]]
    assert sorted(i for b in length_batches(lengths, 100) for i in b) == list(range(5))
    assert length_batches(lengths, 6) == length_batches(lengths, 6)


def _data(n=300, layers=2, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    x = rng.normal(size=(layers, n, dim))
    x[1, :, 0] += 2.0 * y
    x[1, np.arange(n), 1 + y] += 1.0
    y[:7] = -1  # punctuation
    return x, y


def test_linear_stats_match_numpy_reference():
    x, y = _data()
    st = LinearStats(2, 4, 3)
    st.update(torch.tensor(x[:, :150]), torch.tensor(y[:150]))
    st.update(torch.tensor(x[:, 150:]), torch.tensor(y[150:]))
    ref = RidgeAccumulator(2, 4, 3)
    ref.update(x, y)
    assert st.n == ref.n == 293
    assert np.allclose(st.xx.numpy(), ref.xx) and np.allclose(st.xy.numpy(), ref.xy)
    ridge = solve(st, "ridge", (0.01,))
    lda = solve(st, "lda", (0.1,))
    for layer in range(2):
        xs = torch.tensor(x[layer])
        r = (xs.double() @ ridge.weights[layer, 0] + ridge.bias[layer, 0]).numpy()
        assert np.allclose(r, ref.solve(alpha=0.01 * ref.n).scores(x)[layer], atol=1e-8)
        ld = (xs.double() @ lda.weights[layer, 0] + lda.bias[layer, 0]).numpy()
        assert np.allclose(ld, ref.solve_lda(0.1).scores(x)[layer], atol=1e-8)
        assert torch.equal(lda.predict_layer(layer, xs)[0], torch.tensor(ld.argmax(-1)))


def test_two_way_statistics_are_collapsed_exactly():
    x, y = _data()
    st = LinearStats(2, 4, 3)
    st.update(torch.tensor(x), torch.tensor(y))
    y2 = np.where(y == 2, 1, y)
    direct = LinearStats(2, 4, 2)
    direct.update(torch.tensor(x), torch.tensor(y2))
    c = st.collapse(2)
    assert torch.allclose(c.xy, direct.xy) and torch.allclose(c.sy, direct.sy)
    assert st.collapse(3) is st


def test_state_roundtrip():
    x, y = _data()
    st = LinearStats(2, 4, 3)
    st.update(torch.tensor(x), torch.tensor(y))
    other = LinearStats(2, 4, 3)
    other.load_state_dict(st.state_dict())
    assert other.n == st.n and torch.equal(other.xx, st.xx)


def test_text_model_from_multimodal_checkpoint(tmp_path):
    """A (tiny) Qwen3.5 multimodal checkpoint loads as its text model, with the same
    hidden states as the full model's language model."""
    transformers = pytest.importorskip("transformers")
    from prosody_llms.features.loading import load_text_model

    tc = dict(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
              num_attention_heads=2, num_key_value_heads=1, head_dim=16,
              linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
              linear_num_value_heads=2,
              layer_types=["linear_attention"] * 3 + ["full_attention"])
    vc = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32)
    cfg = transformers.Qwen3_5Config(text_config=tc, vision_config=vc)
    torch.manual_seed(0)
    transformers.Qwen3_5ForConditionalGeneration(cfg).save_pretrained(tmp_path)
    full = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        tmp_path, dtype=torch.float32).eval()
    from tests.conftest import VOCAB

    transformers.BertTokenizer(vocab={w: i for i, w in enumerate(VOCAB)}).save_pretrained(tmp_path)
    loaded = load_text_model(str(tmp_path), dtype="float32", device="cpu")
    assert loaded.model_class == "Qwen3_5TextModel" and (loaded.n_layers, loaded.dim) == (5, 32)
    text = loaded.model
    ids = torch.tensor([[1, 5, 7, 9]])
    with torch.no_grad():
        a = full.model.language_model(input_ids=ids, output_hidden_states=True).hidden_states
        b = text(input_ids=ids, output_hidden_states=True).hidden_states
    assert len(b) == 5 and all(torch.equal(x, y) for x, y in zip(a, b))


def _args(corpus_dir, tiny_bert, out_root, *extra):
    return ["--model", str(tiny_bert), "--data-dir", str(corpus_dir), "--out-root", str(out_root),
            "--dtype", "float32", "--device", "cpu", *extra]


def test_probing_end_to_end(corpus_dir, tiny_bert, tmp_path):
    assert probing.main(_args(corpus_dir, tiny_bert, tmp_path)) == 0
    out = tmp_path / "tiny-bert" / "plain-first-natural"
    res = json.loads((out / "result.json").read_text())
    assert res["n_layers"] == 3 and res["train_words"] == 9
    from prosody_llms.eval.io import read_records

    recs = read_records(out / "records.jsonl")
    assert len(recs) == 3 * 2 * 2 * 2  # layers x families x tasks x splits
    test = [r for r in recs if r["split"] == "test"]
    assert all(r["metrics"]["full"]["scored_words"] == 7 for r in test)
    assert all(r["metrics"]["legacy_2019"]["scored_words"] == 6 for r in test)
    assert {r["condition"]["probe"] for r in recs} == {"ridge", "lda"}
    assert (out / "summary.md").read_text().startswith("# Probing tiny-bert")
    assert (out / "predictions_test_3way_lda_best.tsv.gz").exists()
    npz = np.load(out / "predictions_layers.npz")
    assert npz["3way_ridge"].shape == (3, 8)
    assert not (out / "pass1_state.pt").exists()
    # A finished run is not repeated.
    assert probing.main(_args(corpus_dir, tiny_bert, tmp_path)) == 0


def test_probing_resume_matches_uninterrupted_run(corpus_dir, tiny_bert, tmp_path):
    from prosody_llms.eval.io import read_records

    full_root, part_root = tmp_path / "a", tmp_path / "b"
    probing.main(_args(corpus_dir, tiny_bert, full_root, "--max-tokens", "8"))
    # Stop pass 1 after one batch (the training file gives three batches here).
    assert probing.main(_args(corpus_dir, tiny_bert, part_root, "--max-tokens", "8",
                              "--max-batches", "1")) == probing.EXIT_INTERRUPTED
    state = part_root / "tiny-bert" / "plain-first-natural" / "pass1_state.pt"
    assert state.exists()
    with pytest.raises(SystemExit, match="--resume"):
        probing.main(_args(corpus_dir, tiny_bert, part_root, "--max-tokens", "8"))
    assert probing.main(_args(corpus_dir, tiny_bert, part_root, "--max-tokens", "8",
                              "--resume")) == 0
    a = read_records(full_root / "tiny-bert" / "plain-first-natural" / "records.jsonl")
    b = read_records(part_root / "tiny-bert" / "plain-first-natural" / "records.jsonl")
    assert a == b


def test_echo_condition_and_pooling_default(corpus_dir, tiny_bert, tmp_path):
    probing.main(_args(corpus_dir, tiny_bert, tmp_path, "--context", "echo", "--pooling", "mean",
                       "--families", "lda"))
    out = tmp_path / "tiny-bert" / "echo-mean-natural"
    res = json.loads((out / "result.json").read_text())
    assert res["condition"]["context"] == "echo" and res["families"] == ["lda"]


def test_forward_splits_batches_on_out_of_memory(tiny_bert, corpus_dir):
    from prosody_llms.features.extractor import HFTextExtractor

    utts = read_split(corpus_dir / "test.txt")
    ext = HFTextExtractor(str(tiny_bert), device="cpu")
    aligned = [ext.align(u) for u in utts]
    expected = ext.forward(aligned)
    real = ext._forward
    calls = []

    def flaky(batch):  # fail for any batch with more than one utterance
        calls.append(len(batch))
        if len(batch) > 1:
            raise torch.OutOfMemoryError("simulated")
        return real(batch)

    ext._forward = flaky
    got = ext.forward(aligned)
    assert ext.oom_splits == 2 and calls == [3, 1, 2, 1, 1]
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(expected, got))


def _tiny_glimmer(path):
    transformers = pytest.importorskip("transformers")
    from tests.conftest import VOCAB

    text = dict(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                layer_types=["sliding_attention", "full_attention"])
    vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                  layer_types=["full_attention"])
    try:
        cfg = transformers.MuseGlimmerConfig(text_config=text, vision_config=vision)
    except (AttributeError, TypeError, ValueError) as e:
        pytest.skip(f"cannot build a tiny Muse Glimmer config: {e}")
    torch.manual_seed(0)
    transformers.MuseGlimmerForConditionalGeneration(cfg).save_pretrained(path)
    transformers.BertTokenizer(vocab={w: i for i, w in enumerate(VOCAB)}).save_pretrained(path)
    return transformers.MuseGlimmerForConditionalGeneration.from_pretrained(
        path, dtype=torch.float32).eval()


def test_glimmer_text_model_loads_real_weights(tmp_path):
    """Regression: the text-only class used to load Muse Glimmer with *random* weights
    (names differ and transformers only warns). The loader must map the names."""
    from prosody_llms.features.loading import load_text_model

    full = _tiny_glimmer(tmp_path)
    loaded = load_text_model(str(tmp_path), dtype="float32", device="cpu")
    assert loaded.key_mapping is not None
    expected = full.model.language_model.state_dict()
    got = loaded.model.state_dict()
    assert set(got) == set(expected)
    assert all(torch.equal(got[k], expected[k]) for k in got)


def test_loader_refuses_random_weights(tmp_path, monkeypatch):
    from prosody_llms.features import loading

    _tiny_glimmer(tmp_path)
    monkeypatch.setattr(loading, "KEY_MAPPINGS", ())  # no name mapping available
    with pytest.raises(loading.MissingWeightsError, match="refusing"):
        loading.load_text_model(str(tmp_path), dtype="float32", device="cpu")


def _nl_args(corpus_dir, tiny_bert, tmp_path, *extra):
    return ["--model", str(tiny_bert), "--data-dir", str(corpus_dir),
            "--linear-root", str(tmp_path / "linear"), "--out-root", str(tmp_path / "nl"),
            "--dtype", "float32", "--device", "cpu", "--epochs", "3", "--seeds", "0", "1", *extra]


def test_nonlinear_probes_end_to_end(corpus_dir, tiny_bert, tmp_path):
    from prosody_llms.eval.io import read_records
    from prosody_llms.probing import nonlinear

    probing.main(_args(corpus_dir, tiny_bert, tmp_path / "linear"))
    linear = json.loads((tmp_path / "linear" / "tiny-bert" / "plain-first-natural" /
                         "result.json").read_text())
    expected_layers = {s["task"]: s["best_layer"] for s in linear["summary"] if s["probe"] == "lda"}
    assert nonlinear.main(_nl_args(corpus_dir, tiny_bert, tmp_path)) == 0
    out = tmp_path / "nl" / "tiny-bert" / "plain-first-natural"
    res = json.loads((out / "result.json").read_text())
    assert {int(k): v for k, v in res["layers"].items()} == expected_layers
    recs = read_records(out / "records.jsonl")
    assert len(recs) == 3 * 2 * 2 * 2  # probes x seeds x tasks x splits
    assert {r["condition"]["probe"] for r in recs} == {"logistic", "mlp", "bilstm"}
    test = [r for r in recs if r["split"] == "test"]
    assert all(r["metrics"]["full"]["scored_words"] == 7 for r in test)
    assert (out / "predictions_test_3way_bilstm.tsv.gz").exists()
    assert "| 3-way | bilstm |" in (out / "summary.md").read_text()
    # Same seeds, same results.
    other = tmp_path / "again"
    nonlinear.main(_nl_args(corpus_dir, tiny_bert, tmp_path, "--out-root", str(other)))
    again = read_records(other / "tiny-bert" / "plain-first-natural" / "records.jsonl")
    assert [r["metrics"]["full"]["accuracy"] for r in again] == \
        [r["metrics"]["full"]["accuracy"] for r in recs]


def test_nonlinear_explicit_layers_and_bfloat16_storage(corpus_dir, tiny_bert, tmp_path):
    from prosody_llms.features.extractor import HFTextExtractor
    from prosody_llms.probing import nonlinear

    nonlinear.main(_nl_args(corpus_dir, tiny_bert, tmp_path, "--layers", "1,2",
                            "--probes", "logistic", "--seeds", "0"))
    res = json.loads((tmp_path / "nl" / "tiny-bert" / "plain-first-natural" /
                      "result.json").read_text())
    assert res["layers"] == {"2": 1, "3": 2}
    ext = HFTextExtractor(str(tiny_bert), device="cpu")
    utts = read_split(corpus_dir / "test.txt")
    feats, offsets = nonlinear.extract_layers(ext, utts, [2], 4096)
    assert feats[2].dtype == torch.bfloat16 and feats[2].shape == (8, 16)
    assert list(offsets) == [0, 3, 7, 8]
    # bfloat16 keeps the range of float32 (float16 would overflow above 65504).
    assert torch.isfinite(torch.tensor([1e6]).to(torch.bfloat16)).all()

def test_probing_report(corpus_dir, tiny_bert, tmp_path):
    from prosody_llms.probing import report

    root = tmp_path / "probing"
    probing.main(_args(corpus_dir, tiny_bert, root))
    probing.main(_args(corpus_dir, tiny_bert, root, "--context", "echo"))
    probing.main(_args(corpus_dir, tiny_bert, root, "--train-utterances", "2"))  # not a full run
    from prosody_llms.probing import nonlinear

    nonlinear.main(["--model", str(tiny_bert), "--data-dir", str(corpus_dir), "--linear-root",
                    str(root), "--out-root", str(tmp_path / "probing_nonlinear"), "--dtype", "float32",
                    "--device", "cpu", "--epochs", "2", "--seeds", "0", "1"])
    assert report.main([str(root), "--data-dir", str(corpus_dir), "--resamples", "50"]) == 0
    md = (root / "overview.md").read_text()
    assert "## Size series" in md and "## Stronger probes" in md
    assert "| tiny-bert | plain |" in md
    assert "## 2-way" in md and "## 3-way" in md
    assert "tiny-bert (echo) | tiny-bert (plain)" in md  # echo vs plain comparison
    rows = list(csv.DictReader(open(root / "overview.csv")))
    assert len(rows) == 2 * 2  # two full runs x two tasks; the subset run is left out
    curves = list(csv.DictReader(open(root / "layer_curves.csv")))
    assert len(curves) == 2 * 2 * 2 * 3  # runs x tasks x probes x layers
    assert all(c["dev"] for c in curves)


def test_probing_with_preceding_utterances(corpus_dir, tiny_bert, tmp_path):
    from prosody_llms.probing import nonlinear

    root = tmp_path / "probing"
    probing.main(_args(corpus_dir, tiny_bert, root, "--context", "prev1"))
    out = root / "tiny-bert" / "prev1-first-natural"
    assert (out / "result.json").exists() and (out / "records.jsonl").exists()
    nonlinear.main(["--model", str(tiny_bert), "--data-dir", str(corpus_dir), "--linear-root", str(root),
                    "--out-root", str(tmp_path / "nl"), "--dtype", "float32", "--device", "cpu",
                    "--context", "prev1", "--epochs", "1", "--seeds", "0", "--probes", "bilstm"])
    assert (tmp_path / "nl" / "tiny-bert" / "prev1-first-natural" / "result.json").exists()


def test_probing_on_prompt_exemplars(corpus_dir, tiny_bert, tmp_path, monkeypatch):
    from prosody_llms.probing import nonlinear
    from prosody_llms.prompting import prompts

    monkeypatch.setattr(prompts, "select_exemplars", lambda train, k, **kw: list(train)[:k])
    root = tmp_path / "probing"
    probing.main(_args(corpus_dir, tiny_bert, root, "--train-exemplars", "2"))
    assert (root / "tiny-bert" / "plain-first-natural-ex2" / "result.json").exists()
    nonlinear.main(["--model", str(tiny_bert), "--data-dir", str(corpus_dir), "--linear-root", str(root),
                    "--out-root", str(tmp_path / "nl"), "--dtype", "float32", "--device", "cpu",
                    "--train-exemplars", "2", "--epochs", "1", "--seeds", "0", "--probes", "bilstm"])
    res = json.loads((tmp_path / "nl" / "tiny-bert" / "plain-first-natural-ex2" / "result.json").read_text())
    assert res["train_utterances"] == 2
    nonlinear.main(["--model", str(tiny_bert), "--data-dir", str(corpus_dir), "--linear-root", str(root),
                    "--out-root", str(tmp_path / "layer0"), "--dtype", "float32", "--device", "cpu",
                    "--layers", "0,0", "--epochs", "1", "--seeds", "0", "--probes", "bilstm"])
    res = json.loads((tmp_path / "layer0" / "tiny-bert" / "plain-first-natural" / "result.json").read_text())
    assert res["layers"] == {"2": 0, "3": 0}


def test_random_word_vector_control(corpus_dir, tmp_path):
    from prosody_llms.features.extractor import RandomWordExtractor
    from prosody_llms.probing import nonlinear

    ext = RandomWordExtractor(8, device="cpu")
    a, b = ext.vector("Dog"), ext.vector("dog")
    assert np.array_equal(a, b) and not np.array_equal(a, ext.vector("cat"))
    assert np.array_equal(RandomWordExtractor(8, device="cpu").vector("dog"), a)  # same in every run
    nonlinear.main(["--model", "random-words-8", "--random-features", "8", "--layers", "0,0",
                    "--data-dir", str(corpus_dir), "--out-root", str(tmp_path / "rand"), "--device", "cpu",
                    "--epochs", "1", "--seeds", "0", "--probes", "bilstm"])
    res = json.loads((tmp_path / "rand" / "random-words-8" / "plain-first-natural" / "result.json").read_text())
    assert res["layers"] == {"2": 0, "3": 0} and res["dim"] == 8
