import pytest

from prosody_llms.data.corpus import read_split
from prosody_llms.eval.metrics import accuracy, confusion
from prosody_llms.legacy.common import evaluate_predictions, tag_vocab


def test_accuracy_ignores_punctuation():
    gold = ["0", "1", "NA", "2"]
    pred = ["0", "2", "0", "2"]  # wrong prediction on punctuation is not scored
    acc, n = accuracy(gold, pred)
    assert n == 3 and acc == pytest.approx(200 / 3)


def test_predicted_na_or_pad_counts_as_error():
    assert accuracy(["0", "1"], ["NA", "<pad>"]) == (0.0, 2)


def test_accuracy_errors():
    with pytest.raises(ValueError, match="length"):
        accuracy(["0"], ["0", "1"])
    with pytest.raises(ValueError, match="no scored"):
        accuracy(["NA"], ["0"])


def test_confusion():
    c = confusion(["0", "0", "1", "NA"], ["0", "1", "1", "1"])
    assert c == {"0": {"0": 1, "1": 1}, "1": {"1": 1}}


def test_evaluate_predictions_full_and_2019_subset(corpus_dir):
    utts = read_split(corpus_dir / "test.txt")
    # Correct everywhere except the last utterance (which 2019 never scored).
    preds = [u.labels(3) for u in utts[:-1]] + [["0"]]
    res = evaluate_predictions(utts, preds, 3)
    assert res["legacy_2019"]["accuracy"] == 100.0
    assert res["legacy_2019"]["scored_words"] == 6
    assert res["full"]["scored_words"] == 7
    assert res["full"]["accuracy"] == pytest.approx(600 / 7)


def test_tag_vocab_is_sorted(corpus_dir):
    splits = {"test": read_split(corpus_dir / "test.txt")}
    assert tag_vocab(splits, 3) == ["<pad>", "0", "1", "2", "NA"]
    assert tag_vocab(splits, 2) == ["<pad>", "0", "1", "NA"]
