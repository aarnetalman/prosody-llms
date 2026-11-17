import pytest

from prosody_llms.data.corpus import read_split
from prosody_llms.eval.io import (
    append_records,
    evaluate,
    make_record,
    read_predictions,
    read_records,
    records_to_csv,
    write_predictions,
)
from prosody_llms.eval.metrics import macro_f1, per_class, summary
from prosody_llms.eval.stats import bootstrap_accuracy, mcnemar, mean_sd


def test_per_class_and_macro_f1():
    gold = ["0", "0", "1", "1", "NA"]
    pred = ["0", "1", "1", "1", "0"]
    pc = per_class(gold, pred)
    assert pc["0"] == {"precision": 100.0, "recall": 50.0, "f1": pytest.approx(200 / 3),
                       "support": 2}
    assert pc["1"]["precision"] == pytest.approx(200 / 3) and pc["1"]["recall"] == 100.0
    assert macro_f1(gold, pred) == pytest.approx((200 / 3 + 80.0) / 2)
    s = summary(gold, pred)
    assert s["accuracy"] == 75.0 and s["scored_words"] == 4


def test_bootstrap():
    gold = [["0", "1", "NA"], ["1", "1"], ["0", "0", "1"]]
    perfect = [g[:] for g in gold]
    wrong = [["1", "0", "NA"], ["0", "0"], ["1", "1", "0"]]
    res = bootstrap_accuracy(gold, perfect, wrong, n_resamples=200, seed=1)
    assert res["accuracy"] == [100.0, 0.0]
    assert res["ci"][0] == (100.0, 100.0)
    assert res["difference"]["value"] == 100.0 and res["difference"]["p_value"] == 0.0
    same = bootstrap_accuracy(gold, perfect, perfect, n_resamples=50)
    assert same["difference"]["p_value"] == 1.0
    with pytest.raises(ValueError):
        bootstrap_accuracy(gold, [["0"]])


def test_mcnemar_and_mean_sd():
    gold = ["0"] * 10 + ["NA"]
    a = ["0"] * 10 + ["1"]
    b = ["1"] * 10 + ["1"]
    r = mcnemar(gold, a, b)
    assert (r["b"], r["c"], r["method"]) == (10, 0, "exact")
    assert r["p_value"] == pytest.approx(2 / 2 ** 10)
    assert mcnemar(gold, a, a)["p_value"] == 1.0
    m = mean_sd([1.0, 2.0, 3.0])
    assert m["mean"] == 2.0 and m["sd"] == 1.0 and m["n"] == 3


def test_predictions_roundtrip_and_evaluate(corpus_dir, tmp_path):
    utts = read_split(corpus_dir / "test.txt")
    preds = [u.labels(3) for u in utts]
    write_predictions(tmp_path / "p.tsv", utts, preds, 3)
    back = read_predictions(tmp_path / "p.tsv")
    assert list(back) == [u.utt_id for u in utts]
    assert [p for _, _, p in back[utts[0].utt_id]] == preds[0]
    res = evaluate(utts, preds, 3)
    assert res["full"]["accuracy"] == 100.0 and res["full"]["macro_f1"] == 100.0
    with pytest.raises(ValueError):
        write_predictions(tmp_path / "q.tsv", utts, [["0"]] * len(utts), 3)


def test_records(tmp_path):
    recs = [make_record(experiment="probe", model="m", task=2, split="test", layer=layer,
                        condition={"pooling": "first"}, metrics={"full": {"accuracy": 80.0 + layer,
                                                                          "macro_f1": 70.0,
                                                                          "scored_words": 9}})
            for layer in (0, 1)]
    append_records(tmp_path / "r.jsonl", recs)
    back = read_records(tmp_path / "r.jsonl")
    assert back == recs
    records_to_csv(back, tmp_path / "r.csv")
    lines = (tmp_path / "r.csv").read_text().splitlines()
    assert lines[0] == "experiment,model,layer,task,split,cond.pooling,accuracy,macro_f1,scored_words"
    assert lines[2] == "probe,m,1,2,test,first,81.0,70.0,9"
    with pytest.raises(ValueError, match="missing"):
        append_records(tmp_path / "r.jsonl", [{"experiment": "x"}])
