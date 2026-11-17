import pytest

from prosody_llms.data.corpus import collapse_label, corpus_stats, load_splits, read_split


def test_read_split_parses_tokens(corpus_dir):
    utts = read_split(corpus_dir / "train_360.txt")
    assert [u.utt_id for u in utts] == ["100_1_000000_000000", "100_1_000000_000001",
                                        "200_2_000001_000000"]
    assert utts[0].speaker == "100"
    assert utts[0].words == ["The", "cat", "sat", "."]
    t = utts[0].tokens[1]
    assert (t.prominence, t.boundary, t.prominence_value, t.boundary_value) == ("2", "2", 2.1, 1.5)
    punct = utts[0].tokens[3]
    assert punct.is_punct and punct.prominence_value is None


def test_full_protocol_keeps_last_utterance(corpus_dir):
    assert len(read_split(corpus_dir / "test.txt")) == 3


def test_legacy_protocol_drops_last_utterance(corpus_dir):
    # 2019 loader only flushed an utterance on the next <file> line.
    utts = read_split(corpus_dir / "test.txt", legacy=True)
    assert [u.utt_id for u in utts] == ["400_4_000000_000000", "400_4_000000_000001"]


def test_legacy_fraction_slices_raw_lines(corpus_dir):
    # train has 14 lines; 0.4 -> round(5.6) = 6 lines: <file>, 4 tokens, <file>.
    # The slice ends on a <file> line, so the first utterance is kept.
    utts = read_split(corpus_dir / "train_360.txt", fraction=0.4, legacy=True)
    assert [u.utt_id for u in utts] == ["100_1_000000_000000"]
    # 0.6 -> round(8.4) = 8 lines: ends inside the second utterance, which is dropped.
    utts = read_split(corpus_dir / "train_360.txt", fraction=0.6, legacy=True)
    assert [u.utt_id for u in utts] == ["100_1_000000_000000"]
    # Without legacy the partial utterance is kept.
    utts = read_split(corpus_dir / "train_360.txt", fraction=0.6)
    assert len(utts) == 2 and utts[1].words == ["A", "dog"]


def test_collapse_label():
    assert [collapse_label(x, 2) for x in ["0", "1", "2", "NA"]] == ["0", "1", "1", "NA"]
    assert [collapse_label(x, 3) for x in ["0", "1", "2", "NA"]] == ["0", "1", "2", "NA"]
    with pytest.raises(ValueError):
        collapse_label("1", 4)


def test_labels_per_task(corpus_dir):
    u = read_split(corpus_dir / "train_360.txt")[0]
    assert u.labels(3) == ["0", "2", "1", "NA"]
    assert u.labels(2) == ["0", "1", "1", "NA"]


def test_load_splits_and_stats(corpus_dir):
    s = load_splits(corpus_dir, "train_360")
    assert {k: len(v) for k, v in s.items()} == {"train": 3, "dev": 2, "test": 3}
    st = corpus_stats(s["test"])
    assert st == {"utterances": 3, "tokens": 8, "scored_words": 7,
                  "labels": {"0": 3, "1": 1, "2": 3, "NA": 1}, "speakers": 2}
    with pytest.raises(ValueError):
        load_splits(corpus_dir, "train_500")


def test_malformed_input(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("word\t0\t0\t0.1\t0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="before the first"):
        read_split(p)
    p.write_text("<file>\tx.txt\nword\t0\t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="5 tab-separated"):
        read_split(p)


def test_real_corpus_counts(real_data_dir):
    """Counts of the pinned corpus and the 2019 loader's counts."""
    expected = {"train_100": (33042, 570592), "train_360": (116263, 2076289),
                "dev": (5727, 99200), "test": (4822, 90063)}
    for name, (n_utts, n_words) in expected.items():
        utts = read_split(real_data_dir / f"{name}.txt")
        st = corpus_stats(utts)
        assert (st["utterances"], st["scored_words"]) == (n_utts, n_words)
        # The 2019 loader dropped exactly one utterance per file (README counts).
        assert len(read_split(real_data_dir / f"{name}.txt", legacy=True)) == n_utts - 1


def test_real_2019_test_subset_size(real_data_dir):
    """The 2019 test numbers are over 4,821 utterances and 90,004 scored words."""
    utts = read_split(real_data_dir / "test.txt", legacy=True)
    st = corpus_stats(utts)
    assert (st["utterances"], st["scored_words"]) == (4821, 90004)
