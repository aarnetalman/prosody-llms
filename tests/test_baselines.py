import json

import pytest

from prosody_llms.data.corpus import read_split
from prosody_llms.legacy import baselines
from prosody_llms.legacy.common import tag_vocab


@pytest.fixture
def splits(corpus_dir):
    return {s: read_split(corpus_dir / f"{s}.txt") for s in ("train_360", "test")}


def test_majority(splits):
    train, test = splits["train_360"], splits["test"]
    # 3-way train counts: 0 x4, 1 x2, 2 x3 -> "0"; 2-way: 0 x4, 1 x5 -> "1"
    assert baselines.majority(train, test, 3)[0] == ["0", "0", "0"]
    assert baselines.majority(train, test, 2)[0] == ["1", "1", "1"]


def test_word_majority(splits):
    train, test = splits["train_360"], splits["test"]
    preds = baselines.word_majority(train, test, 3)
    # dog: 2,1 -> tie -> "1"; sat: 1,2 -> tie -> "1"; the: 0,0; a: 0; cat: 2; runs: 0
    assert preds[0] == ["0", "1", "1"]
    assert preds[1] == ["0", "2", "0", "0"]  # "," never seen with a label -> majority "0"
    assert preds[2] == ["0"]  # unseen word -> global majority


def test_random_baselines_are_seeded(splits):
    train, test = splits["train_360"], splits["test"]
    a = baselines.random_labels(train, test, 3, seed=1, prior=False)
    b = baselines.random_labels(train, test, 3, seed=1, prior=False)
    assert a == b and {p for ps in a for p in ps} <= {"0", "1", "2"}


def test_word_majority_2019_counts_all_pieces(splits, tiny_bert):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_bert)
    train, test = splits["train_360"], splits["test"]
    tags = tag_vocab({"train": train, "test": test}, 3)
    preds = baselines.word_majority_2019(train, test, 3, tok, tags)
    assert preds[0] == ["0", "1", "1"]
    # "dogs" -> dog ##s: keyed by the first piece, so it gets the statistics of "dog"
    assert preds[2] == ["1"]
    # "," never occurs in training: unseen pieces get tags[1], i.e. "0"
    assert preds[1][3] == "0"


def test_baselines_cli(corpus_dir, tmp_path):
    out = tmp_path / "b.json"
    baselines.main(["--data-dir", str(corpus_dir), "--out", str(out),
                    "--baselines", "majority", "word_majority", "random_uniform"])
    res = json.loads(out.read_text())
    assert set(res["results"]) == {f"{b}/{n}way" for b in ("majority", "word_majority",
                                                           "random_uniform") for n in (2, 3)}
    maj = res["results"]["majority/3way"]["test"]
    assert maj["full"]["scored_words"] == 7 and maj["legacy_2019"]["scored_words"] == 6
