import gzip
import json

from prosody_llms.analysis import collect, errors, tables
from prosody_llms.data.corpus import read_split
from prosody_llms.probing.run import write_predictions_gz


def _results(tmp_path, corpus_dir):
    """A small results/ tree: 2019 runs, a text probe, a prompting run, an audio probe and
    the comparison files."""
    root = tmp_path / "results"
    test = read_split(corpus_dir / "test.txt")
    for task in (2, 3):
        for seed, acc in (("1", 83.0), ("2", 83.2)):
            d = root / "repro_2019" / "train_360-1.0" / f"{task}way" / f"seed{seed}"
            d.mkdir(parents=True)
            m = {"accuracy": acc - (15 if task == 3 else 0), "macro_f1": 80.0}
            (d / "result.json").write_text(json.dumps({"test": {"full": m, "legacy_2019": m},
                                                       "dev": {"full": m}}))
    gold = [u.labels(3) for u in test]
    zeros = [["NA" if g == "NA" else "0" for g in gs] for gs in gold]
    for model, cond, preds, dev in (("bert-base-uncased", "plain-first-natural", zeros, 80.0),
                                    ("wavlm-large", "audio-mean", gold, 90.0)):
        d = root / "probing" / model / cond
        d.mkdir(parents=True)
        summary = [{"task": t, "probe": "lda", "best_layer": 1, "layers": 3, "dev": dev,
                    "test_2019": 70.0 + t, "test_full": 70.0 + t, "test_macro_f1": 60.0} for t in (2, 3)]
        (d / "result.json").write_text(json.dumps({"model": model, "summary": summary}))
        recs = [{"split": "test", "condition": {"probe": "lda"}, "layer": layer, "task": t,
                 "metrics": {"legacy_2019": {"accuracy": 60.0 + layer + t}}} for layer in range(3) for t in (2, 3)]
        (d / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        write_predictions_gz(d / "predictions_test_3way_lda_best.tsv.gz", test, preds, 3)
    p = root / "prompting" / "gpt-4o" / "test" / "linguistic-lines-3way-k5"
    p.mkdir(parents=True)
    recs = [{"model": "gpt-4o", "task": t, "condition": {"derived": t == 2},
             "metrics": {"full": {"accuracy": 50.0, "macro_f1": 40.0}, "legacy_2019": {"accuracy": 50.0}}}
            for t in (3, 2)]
    (p / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    write_predictions_gz(p / "predictions.tsv.gz", test, zeros, 3)
    c = root / "audio" / "comparison"
    c.mkdir(parents=True)
    with gzip.open(c / "common_words.tsv.gz", "wt") as f:
        for u in test:
            f.write(u.utt_id + "\t" + "".join("0" if t.is_punct else "1" for t in u.tokens) + "\n")
    (c / "common_words.json").write_text(json.dumps({"common_words": 7, "test_words": 7}))
    rows = [("bert-base-uncased (plain-first-natural) LDA", "text", "probe_linear",
             "probing/bert-base-uncased/plain-first-natural/predictions_test_3way_lda_best.tsv.gz", 40.0),
            ("wavlm-large (audio-mean) LDA", "audio", "probe_linear",
             "probing/wavlm-large/audio-mean/predictions_test_3way_lda_best.tsv.gz", 100.0)]
    with open(c / "comparison.csv", "w") as f:
        f.write("task,system,family,modality,accuracy_common,mcnemar_p,path\n")
        for name, mod, fam, path, acc in rows:
            for t in (2, 3):
                f.write(f"{t},{name},{fam},{mod},{acc},,{path}\n")
    return root


def test_collect_and_tables(tmp_path, corpus_dir):
    root = _results(tmp_path, corpus_dir)
    rows = collect.collect(root)
    ft = [r for r in rows if r["family"] == "finetune" and r["task"] == 2][0]
    assert ft["test_2019"] == 83.1 and ft["n"] == 2 and round(ft["sd"], 3) == 0.141
    assert {r["modality"] for r in rows if r["family"] == "probe_linear"} == {"text", "audio"}
    info = tables.write_all(root, tmp_path / "paper", {"agreement-3": "61.1"})
    gen = tmp_path / "paper" / "results"
    nums = (gen / "numbers.tex").read_text()
    assert "res@repro-2\\endcsname{83.10}" in nums and "res@agreement-3\\endcsname{61.1}" in nums
    assert "res@gap-3\\endcsname{60.0}" in nums  # audio 100 - text 40 on the common words
    assert "\\resval" in nums and "??" in nums
    main = (gen / "main_table.tex").read_text()
    assert "BERT-base (2019 recipe)" in main and "best audio: WavLM-large" in main
    assert info["figure_files"] and (gen / "layers_wavlm_large.csv").exists()


def test_best_is_chosen_on_dev_not_test(tmp_path, corpus_dir):
    rows = [collect.row("probe_linear", "a", "a", "c", "text", 3, 70.0, 70.0, dev=60.0),
            collect.row("probe_linear", "b", "b", "c", "text", 3, 60.0, 60.0, dev=65.0)]
    assert tables.best(rows, "probe_linear", "text", 3)["model"] == "b"


def test_errors_breakdown_overlap_and_agreement(tmp_path, corpus_dir):
    root = _results(tmp_path, corpus_dir)
    nums = errors.write_all(root, corpus_dir, tmp_path / "paper")
    md = (root / "analysis" / "error_analysis.md").read_text()
    assert "Agreement between readers" in md and "word[gold|text|audio]" in md
    bd = (root / "analysis" / "breakdown.csv").read_text()
    assert "content" in bd and "function" in bd
    assert nums["bd-3-content-audio"] == 100.0
    assert nums["corpus-negative"] == 0 and "train-utterances" in nums
    assert "\\midrule" in (tmp_path / "paper" / "results" / "breakdown_table.tex").read_text()


def test_word_features_givenness_and_position(corpus_dir):
    test = read_split(corpus_dir / "test.txt")
    f = errors.word_features(test, {"the": 2000, "dog": 5})
    first = f[test[0].utt_id]  # The dog sat
    assert first[0]["class"] == "function" and first[0]["position"] == "initial"
    assert first[1]["frequency"] == "1-10" and first[1]["givenness"] == "new"
    assert first[2]["position"] == "final" and first[2]["frequency"] == "unseen"


def test_consensus_counts_shared_errors(tmp_path, corpus_dir):
    root = _results(tmp_path, corpus_dir)
    # A second text probe with the same (all "0") predictions: the two always agree.
    src = root / "probing" / "bert-base-uncased" / "plain-first-natural"
    dst = root / "probing" / "xlm-roberta-base" / "plain-first-natural"
    dst.mkdir(parents=True)
    (dst / "predictions_test_3way_lda_best.tsv.gz").write_bytes((src / "predictions_test_3way_lda_best.tsv.gz").read_bytes())
    nums, tags = errors.consensus(root, corpus_dir)
    assert nums["consensus-systems"] == 2 and nums["consensus-mutual-3"] == 100.0
    # Every error is shared, since the two models predict the same labels.
    assert nums["consensus-shared-errors-3"] == 100.0
    assert {t for ts in tags.values() for t in ts if t} <= {"all-right", "all-wrong"}


def test_finetuning_runs_that_did_not_train_are_left_out(tmp_path):
    root = tmp_path / "results"
    (root / "repro_2019").mkdir(parents=True)
    maj = {"dev": {"full": {"accuracy": 52.0}}, "test": {"full": {"accuracy": 52.0}, "legacy_2019": {"accuracy": 52.0}}}
    (root / "repro_2019" / "baselines.json").write_text(json.dumps({"results": {"majority/2way": maj}}))
    for seed, dev, test in (("1", 52.1, 51.8), ("2", 83.9, 83.2), ("3", 84.0, 83.4)):
        d = root / "finetune" / "xlm-roberta-large" / "2way" / f"seed{seed}"
        d.mkdir(parents=True)
        m = {"accuracy": test, "macro_f1": 80.0}
        (d / "result.json").write_text(json.dumps({"test": {"full": m, "legacy_2019": m},
                                                   "dev": {"full": {"accuracy": dev}}}))
    r = [x for x in collect.finetuned(root) if x["task"] == 2][0]
    assert r["n"] == 2 and round(r["test_2019"], 2) == 83.3 and "1 of 3 seeds did not train" in r["note"]


def test_visual_paper_figures():
    from prosody_llms.analysis import visuals

    agr = {"labels": ["Gold labels", "Whisper large-v3", "BERT-base", "BERT_x"], "groups": ["gold", "speech", "text", "text"],
           "matrix": [[100, 80, 60, 61], [80, 100, 70, 71], [60, 70, 100, 90], [61, 71, 90, 100]], "words": 10}
    tex = visuals.heatmap_tex(agr)
    assert tex.count("rectangle") == 6 and "BERT\\_x" in tex and "{Gold}" in tex
    nums = visuals.agreement_numbers(agr)
    assert nums["agr-text-text"] == 90 and nums["agr-gold-text"] == 60.5 and nums["agr-text-speech"] == 70.5
    words = [{"word": w, "punct": p, "gold": g, "text": t, "speech": s} for w, p, g, t, s in
             [("Oh", False, "2", "1", "1"), (",", True, "NA", "NA", "NA"), ("Paris", False, "0", "1", "0")]]
    ex = visuals.example_tex([{"utt_id": "a_b", "words": words}])
    assert "\\textbf{\\large Oh}," in ex and "\\textbf{Paris}" in ex
    assert "pending" in visuals.example_tex([])


def test_readings_tables_and_profile():
    from prosody_llms.analysis import readings
    from prosody_llms.data.corpus import Token, Utterance

    toks = [Token(w, s, "0", None, None) for w, s in (("the", "0"), ("cat", "2"), ("sat", "1"), (".", "NA"))]
    u = Utterance("1_1_000000_000000", toks)
    prof = readings._profile([u], None)
    assert prof["content-top"] == 50.0 and prof["function-top"] == 0.0 and prof["last-top"] == 0.0
    wrong = {u.utt_id: ["2", "2", "2", "NA"]}
    prof = readings._profile([u], wrong)
    assert prof["last-top"] == 100.0 and prof["function-acc"] == 0.0
    rows = [{"system": "S", "kind": "speech", "own": 80.0, "other": 60.0, "diff": 20.0}]
    tex = readings.reader_tex(rows, {"rt-readers": 64.0})
    assert "+20.0" in tex and "64.0" in tex
    assert "pending" in readings.reader_tex([], {}) and "pending" in readings.profile_tex([])
    tex = readings.profile_tex([{"system": "Gold labels", "tag": "gold", "content-top": 42.0, "function-top": 9.0,
                                 "last-top": 32.0}])
    assert "--" in tex and "42.0" in tex


def test_bold_best_cells():
    from prosody_llms.analysis.tables import best_cells, bold

    marks = best_cells([(0, "text", 68.79), (1, "text", 68.81), (2, "audio", 84.7), (3, "text", 60.0), (4, "audio", None)])
    assert marks == {0, 1, 2}  # ties at one decimal are all bold
    assert bold("68.8 $\\pm$ 0.1", True) == "\\textbf{68.8} $\\pm$ 0.1"
    assert bold("83.2$^{\\circ}$", True) == "\\textbf{83.2}$^{\\circ}$"
    assert bold("--", True) == "--" and bold("60.0", False) == "60.0"


def test_params_label_and_class_tables():
    from prosody_llms.analysis import readings
    from prosody_llms.analysis.tables import params_label

    assert params_label(110106428) == "110M" and params_label(27781427952) == "27.8B" and params_label(None) == "--"
    conf = {g: {p: (100.0 if g == p else 0.0) for p in "012"} for g in "012"}
    rows = [{"system": "S", "tag": "qwen", "kind": "t", "macro_f1": 60.0, "recall-0": 80.0, "recall-1": 40.0,
             "recall-2": 60.0, "confusion": conf}]
    assert "60.0" in readings.class_tex(rows) and "pending" in readings.class_tex([])
    tex = readings.confusion_tex(rows)
    assert tex.count("rectangle") == 9 and "predicted" in tex


def test_lexical_default_helpers(corpus_dir):
    from prosody_llms.analysis import readings

    lookup, share = readings.word_majority(corpus_dir)
    assert lookup("zzz-unseen") == "0" and 0.0 <= share("zzz-unseen") <= 1.0
    rows = [{"system": "S", "tag": "qwen", "kind": "t", "agree": 70.0, "departs": 30.0, "dep_right": 51.0,
             "dep_wm_right": 31.0}]
    tex = readings.lexical_tex(rows)
    assert "51.0" in tex and "31.0" in tex and "pending" in readings.lexical_tex([])
