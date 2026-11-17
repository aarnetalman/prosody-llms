import json
import wave

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from prosody_llms.audio import extractors as ex  # noqa: E402
from prosody_llms.audio.base import libritts_path  # noqa: E402
from prosody_llms.data.corpus import read_split  # noqa: E402


def write_wav(path, seconds, sr=24000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def textgrid(words, xmax):
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "xmin = 0", f"xmax = {xmax}",
             "tiers? <exists>", "size = 1", "item []:", "    item [1]:", '        class = "IntervalTier"',
             '        name = "words"', "        xmin = 0", f"        xmax = {xmax}",
             f"        intervals: size = {len(words)}"]
    for k, (s, e, w) in enumerate(words, 1):
        lines += [f"        intervals [{k}]:", f"            xmin = {s}", f"            xmax = {e}",
                  f'            text = "{w}"']
    return "\n".join(lines) + "\n"


@pytest.fixture
def audio_corpus(tmp_path, corpus_dir):
    """Audio and alignments for every fixture utterance; the last scored word of each dev
    utterance is left out of the alignment."""
    audio = tmp_path / "audio"
    subsets = {"train_100": "train-clean-100", "dev": "dev-clean", "test": "test-clean"}
    for split, subset in subsets.items():
        for n, u in enumerate(read_split(corpus_dir / f"{split}.txt")):
            words = [t.word.lower() for t in u.tokens if not t.is_punct]
            if split == "dev":
                words = words[:-1]
            spans = [(0.1 + 0.3 * k, 0.4 + 0.3 * k, w) for k, w in enumerate(words)]
            dur = 0.6 + 0.3 * len(words)
            write_wav(libritts_path(audio, split, u.utt_id), dur, seed=n)
            sp, ch = u.utt_id.split("_")[:2]
            tg = audio / "alignments" / "textgrid" / subset / sp / ch / f"{u.utt_id}.TextGrid"
            tg.parent.mkdir(parents=True, exist_ok=True)
            tg.write_text(textgrid(spans, dur))
    return audio


@pytest.fixture
def tiny_wavlm(tmp_path):
    cfg = transformers.WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                   intermediate_size=64, conv_dim=(16,) * 7, num_conv_pos_embeddings=16,
                                   num_conv_pos_embedding_groups=2, num_buckets=16, max_bucket_distance=32)
    torch.manual_seed(0)
    d = tmp_path / "tiny-wavlm"
    transformers.WavLMModel(cfg).save_pretrained(d)
    transformers.Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, do_normalize=True,
                                          return_attention_mask=True).save_pretrained(d)
    return d


def test_resample_keeps_a_tone():
    sr = 24000
    t = np.arange(sr) / sr
    y = ex.resample(np.sin(2 * np.pi * 440 * t).astype(np.float32), sr, 16000)
    assert len(y) == 16000
    spec = np.abs(np.fft.rfft(y))
    assert abs(np.argmax(spec) - 440) <= 1


def test_pool_frames_means_frames_inside_spans():
    h = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)  # frame k has value k
    out, beyond = ex.pool_frames(h, [(0.0, 0.06), None, (0.105, 0.11), (5.0, 6.0)], 0.02, 0.01)
    assert out[0, :, 0].tolist() == [1.0, 0.0, 5.0, 0.0] and beyond == 1


def test_speech_probing_end_to_end(tiny_wavlm, audio_corpus, corpus_dir, tmp_path):
    from prosody_llms.probing import run as prun

    out = tmp_path / "probing"
    args = prun.build_parser().parse_args(
        ["--model", str(tiny_wavlm), "--kind", "speech", "--audio-dir", str(audio_corpus),
         "--data-dir", str(corpus_dir), "--out-root", str(out), "--dtype", "float32", "--device", "cpu"])
    res = prun.run(args)
    run_dir = out / "tiny-wavlm" / "audio-mean"
    assert res["n_layers"] == 3 and res["condition"]["train_set"] == "train_100"
    assert (run_dir / "result.json").exists()
    recs = [json.loads(line) for line in open(run_dir / "records.jsonl")]
    dev = [r for r in recs if r["split"] == "dev"]
    # The unaligned last word of each dev utterance is scored as "0" in "full" and left out
    # of "aligned".
    assert dev[0]["metrics"]["aligned"]["scored_words"] == dev[0]["metrics"]["full"]["scored_words"] - 2
    st = res["alignment"]
    assert st["aligned_tokens"] < st["tokens"]


def test_word_mask_respects_the_input_window(audio_corpus, corpus_dir):
    e = ex.AudioExtractor(audio_corpus, "x")
    utts = read_split(corpus_dir / "test.txt")
    e.register({"test": utts})
    assert e.word_mask(utts[1]) == [True, True, True, False]  # A cat runs ,
    e.max_seconds = 0.5
    assert e.word_mask(utts[1]) == [True, True, False, False]
    a = e.align(utts[1])
    assert len(a.samples()) == int(0.6 * 16000 + 0.3 * 3 * 16000)


def test_whisper_encoder_extractor(tmp_path, audio_corpus, corpus_dir):
    cfg = transformers.WhisperConfig(num_mel_bins=80, d_model=32, encoder_layers=2, decoder_layers=1,
                                     encoder_attention_heads=2, decoder_attention_heads=2,
                                     encoder_ffn_dim=64, decoder_ffn_dim=64, vocab_size=100,
                                     max_source_positions=1500, max_target_positions=32, pad_token_id=0,
                                     bos_token_id=1, eos_token_id=2, decoder_start_token_id=1)
    d = tmp_path / "tiny-whisper"
    torch.manual_seed(0)
    transformers.WhisperModel(cfg).save_pretrained(d)
    transformers.WhisperFeatureExtractor(feature_size=80).save_pretrained(d)
    e = ex.WhisperEncoderExtractor(str(d), None, audio_corpus, "tiny-whisper", device="cpu")
    utts = read_split(corpus_dir / "test.txt")
    e.register({"test": utts})
    (h,) = e.forward([e.align(utts[0])])
    assert h.shape == (3, len(utts[0].tokens), 32) and e.n_layers == 3
    assert bool(h[:, 0].abs().sum() > 0)


@pytest.fixture
def tiny_qwen2_audio(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz 0123456789\n:=,.|<>_")}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="a"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="a")
    fast.add_special_tokens({"additional_special_tokens": ["<|AUDIO|>", "<|audio_bos|>", "<|audio_eos|>"]})
    proc = transformers.Qwen2AudioProcessor(feature_extractor=transformers.WhisperFeatureExtractor(feature_size=80),
                                            tokenizer=fast)
    cfg = transformers.Qwen2AudioConfig(
        audio_config={"model_type": "qwen2_audio_encoder", "num_mel_bins": 80, "d_model": 32,
                      "encoder_layers": 2, "encoder_attention_heads": 2, "encoder_ffn_dim": 64,
                      "max_source_positions": 1500},
        text_config={"model_type": "qwen2", "vocab_size": len(fast), "hidden_size": 32, "num_hidden_layers": 3,
                     "num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": 64},
        audio_token_index=fast.convert_tokens_to_ids("<|AUDIO|>"))
    torch.manual_seed(0)
    d = tmp_path / "tiny-qwen2-audio"
    transformers.Qwen2AudioForConditionalGeneration(cfg).save_pretrained(d)
    proc.save_pretrained(d)
    return d


@pytest.mark.parametrize("part,layers,dim", [("llm", 4, 32), ("encoder", 3, 32)])
def test_qwen2_audio_extractor(tiny_qwen2_audio, audio_corpus, corpus_dir, part, layers, dim):
    e = ex.Qwen2AudioExtractor(str(tiny_qwen2_audio), None, audio_corpus, "q", dtype="float32",
                               device="cpu", part=part)
    utts = read_split(corpus_dir / "test.txt")
    e.register({"test": utts})
    (h,) = e.forward([e.align(utts[1])])
    assert e.n_layers == layers and h.shape == (layers, len(utts[1].tokens), dim)
    assert bool(h[:, :3].abs().sum(dim=-1).gt(0).all()) and bool(h[:, 3].eq(0).all())  # "," is zero


def test_stronger_probes_on_audio(tiny_wavlm, audio_corpus, corpus_dir, tmp_path):
    from prosody_llms.probing import nonlinear
    from prosody_llms.probing import run as prun

    common = ["--model", str(tiny_wavlm), "--kind", "speech", "--audio-dir", str(audio_corpus),
              "--data-dir", str(corpus_dir), "--dtype", "float32", "--device", "cpu"]
    prun.run(prun.build_parser().parse_args(common + ["--out-root", str(tmp_path / "probing")]))
    assert nonlinear.main(common + ["--linear-root", str(tmp_path / "probing"),
                                    "--out-root", str(tmp_path / "nl"), "--probes", "logistic", "bilstm",
                                    "--seeds", "0", "--epochs", "2"]) == 0
    recs = [json.loads(line) for line in open(tmp_path / "nl" / "tiny-wavlm" / "audio-mean" / "records.jsonl")]
    assert {r["condition"]["probe"] for r in recs} == {"logistic", "bilstm"}
    assert all("aligned" in r["metrics"] for r in recs)


def test_compare_text_and_audio_on_common_words(tmp_path, corpus_dir, audio_corpus):
    from prosody_llms.audio import compare
    from prosody_llms.probing.run import write_predictions_gz

    utts = read_split(corpus_dir / "test.txt")
    runs = tmp_path / "runs"
    gold3 = [u.labels(3) for u in utts]
    zeros = [["NA" if g == "NA" else "0" for g in gs] for gs in gold3]
    text_dir = runs / "probing" / "bert" / "plain-first-natural-train_100"
    audio_dir = runs / "probing" / "wavlm" / "audio-mean"
    for d, preds in ((text_dir, zeros), (audio_dir, gold3)):
        d.mkdir(parents=True)
        write_predictions_gz(d / "predictions_test_3way_lda_best.tsv.gz", utts, preds, 3)
    p = runs / "prompting" / "gpt" / "test" / "minimal-lines-3way-k0"
    p.mkdir(parents=True)
    write_predictions_gz(p / "predictions.tsv.gz", utts, gold3, 3)
    rows, info = compare.compare(runs, corpus_dir, audio_corpus)
    assert info["common_words"] == info["test_words"] == 7
    three = {r["system"]: r for r in rows if r["task"] == 3}
    assert three["wavlm (audio-mean) LDA"]["accuracy_common"] == 100.0
    assert three["wavlm (audio-mean) LDA"]["modality"] == "audio"
    assert {r["system"] for r in rows if r["task"] == 2} == {"gpt minimal-lines-3way-k0"}  # derived only
    text = compare.write(rows, info, tmp_path / "out")
    assert "Common words: 7 of 7" in text and (tmp_path / "out" / "comparison.csv").exists()
    import gzip
    rows = gzip.open(tmp_path / "out" / "common_words.tsv.gz", "rt").read().split()
    assert rows[:2] == ["400_4_000000_000000", "111"] and "mask" not in json.loads(
        (tmp_path / "out" / "common_words.json").read_text())


def test_audio_prompting_with_and_without_audio(tiny_qwen2_audio, audio_corpus, corpus_dir, tmp_path):
    from prosody_llms.audio import prompt

    out = tmp_path / "ap"
    assert prompt.main(["--model", str(tiny_qwen2_audio), "--split", "test", "--both", "--data-dir",
                        str(corpus_dir), "--audio-dir", str(audio_corpus), "--out-root", str(out),
                        "--dtype", "float32", "--device", "cpu"]) == 0
    a = json.loads((out / "tiny-qwen2-audio" / "test" / "linguistic-audio" / "result.json").read_text())
    b = json.loads((out / "tiny-qwen2-audio" / "test" / "linguistic-no_audio" / "result.json").read_text())
    assert a["audio"] and not b["audio"] and a["mean_prompt_tokens"] > b["mean_prompt_tokens"]
    recs = [json.loads(line) for line in open(out / "tiny-qwen2-audio" / "test" / "linguistic-audio" / "records.jsonl")]
    assert {r["task"] for r in recs} == {2, 3} and recs[0]["metrics"]["full"]["scored_words"] == 7
