import json

import pytest

from prosody_llms.legacy import report, train


def _args(corpus_dir, tiny_bert, out_dir, *extra):
    return ["--data-dir", str(corpus_dir), "--out-dir", str(out_dir), "--n-classes", "3",
            "--model", str(tiny_bert), "--revision", "main", "--batch-size", "2",
            "--epochs", "2", "--device", "cpu", "--log-every", "1", *extra]


def test_encode_matches_2019_layout(tiny_bert, corpus_dir):
    from transformers import AutoTokenizer

    from prosody_llms.data.corpus import read_split

    tok = AutoTokenizer.from_pretrained(tiny_bert)
    utt = read_split(corpus_dir / "test.txt")[2]  # "dogs" -> dog ##s
    tags = ["<pad>", "0", "1", "2", "NA"]
    ids, ys, heads = train.encode(utt, tok, {t: i for i, t in enumerate(tags)}, 3)
    assert tok.convert_ids_to_tokens(ids) == ["[CLS]", "dog", "##s", "[SEP]"]
    assert ys == [0, 3, 0, 0]
    assert heads == [1, 1, 0, 1]


def test_training_end_to_end(corpus_dir, tiny_bert, tmp_path):
    out = tmp_path / "run"
    assert train.main(_args(corpus_dir, tiny_bert, out)) == 0
    res = json.loads((out / "result.json").read_text())
    assert res["labels"] == ["<pad>", "0", "1", "2", "NA"]
    assert [h["epoch"] for h in res["history"]] == [1, 2]
    assert res["test"]["full"]["scored_words"] == 7
    assert res["test"]["legacy_2019"]["scored_words"] == 6
    assert res["data"]["train_utterances"] == 2  # legacy protocol drops the last one
    assert not (out / "checkpoint.pt").exists()
    lines = (out / "predictions_test.tsv").read_text().splitlines()
    assert lines[0] == "<file>\t400_4_000000_000000" and lines[1].startswith("The\t0\t")
    # A finished run is not repeated.
    assert train.main(_args(corpus_dir, tiny_bert, out)) == 0


def test_training_is_deterministic(corpus_dir, tiny_bert, tmp_path):
    for name in ("a", "b"):
        train.main(_args(corpus_dir, tiny_bert, tmp_path / name, "--protocol", "full"))
    a, b = (json.loads((tmp_path / n / "result.json").read_text()) for n in ("a", "b"))
    assert a["test"] == b["test"] and a["history"][0]["dev_accuracy_full"] == \
        b["history"][0]["dev_accuracy_full"]
    assert (tmp_path / "a" / "predictions_test.tsv").read_text() == \
        (tmp_path / "b" / "predictions_test.tsv").read_text()


def test_resume_matches_uninterrupted_run(corpus_dir, tiny_bert, tmp_path):
    full = tmp_path / "full"
    train.main(_args(corpus_dir, tiny_bert, full))
    part = tmp_path / "part"
    # Stop after one epoch, keeping the checkpoint, then resume to two.
    args = _args(corpus_dir, tiny_bert, part, "--keep-checkpoint")
    args[args.index("--epochs") + 1] = "1"
    train.main(args)
    (part / "result.json").unlink()
    train.main(_args(corpus_dir, tiny_bert, part, "--resume"))
    a, b = (json.loads((d / "result.json").read_text()) for d in (full, part))
    assert a["test"] == b["test"]


def test_report(corpus_dir, tiny_bert, tmp_path, capsys):
    for seed in ("1", "2"):
        train.main(_args(corpus_dir, tiny_bert, tmp_path / "runs" / seed, "--seed", seed))
    csv_path = tmp_path / "summary.csv"
    assert report.main([str(tmp_path / "runs"), "--csv", str(csv_path)]) == 0
    md = capsys.readouterr().out
    assert "| 3-way | train_360 | 1 | legacy, eager | 2 |" in md
    assert csv_path.read_text().count("\n") == 2


@pytest.mark.parametrize("flag", ["--attention-mask"])
def test_attention_mask_option(corpus_dir, tiny_bert, tmp_path, flag):
    out = tmp_path / "m"
    train.main(_args(corpus_dir, tiny_bert, out, flag))
    assert json.loads((out / "result.json").read_text())["config"]["attention_mask"] is True


def test_word_start_space_only_for_byte_level_tokenizers(tiny_bert, corpus_dir):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    from prosody_llms.data.corpus import read_split

    assert not train.needs_word_space(AutoTokenizer.from_pretrained(tiny_bert))  # 2019 unchanged
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.train_from_iterator(["the dog sat", "a cat runs", "dogs"] * 5, trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[CLS]", "[SEP]", "[PAD]", "[UNK]"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, cls_token="[CLS]", sep_token="[SEP]",
                                  pad_token="[PAD]", unk_token="[UNK]")
    assert train.needs_word_space(tok)
    utt = read_split(corpus_dir / "test.txt")[0]  # The dog sat
    tags = ["<pad>", "0", "1", "2", "NA"]
    ids, _, heads = train.encode(utt, tok, {t: i for i, t in enumerate(tags)}, 3, word_space=True)
    pieces = tok.convert_ids_to_tokens(ids)
    assert pieces[0] == "[CLS]" and pieces[-1] == "[SEP]"
    assert all(p.startswith("Ġ") for p, h in zip(pieces[1:-1], heads[1:-1]) if h)  # "Ġdog"


def test_finetune_runs_are_labelled_and_pad_with_the_model_token(corpus_dir, tiny_bert, tmp_path):
    out = tmp_path / "ft"
    assert train.main(_args(corpus_dir, tiny_bert, out, "--attention-mask", "--name", "tiny")) == 0
    res = json.loads((out / "result.json").read_text())
    assert res["experiment"] == "finetune" and res["model"] == "tiny"
    assert res["tokenisation"] == {"word_start_space": False, "pad_id": 0}  # BERT's [PAD] is 0
    legacy = tmp_path / "legacy"
    train.main(_args(corpus_dir, tiny_bert, legacy))
    res = json.loads((legacy / "result.json").read_text())
    assert res["experiment"] == "repro_2019" and res["model"] == "bert-base-uncased"


def test_no_checkpoint_option(corpus_dir, tiny_bert, tmp_path, monkeypatch):
    import torch

    saved = []
    monkeypatch.setattr(torch, "save", lambda *a, **k: saved.append(a[1]))
    out = tmp_path / "nockpt"
    assert train.main(_args(corpus_dir, tiny_bert, out, "--no-checkpoint")) == 0
    assert not saved and (out / "result.json").exists()
