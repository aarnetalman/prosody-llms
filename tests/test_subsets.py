import pytest

from prosody_llms.data.corpus import Token, Utterance, read_split
from prosody_llms.data.subsets import (
    chapter_position,
    cross_speaker_agreement,
    fixed_subset,
    is_boilerplate,
    multi_reader_groups,
    overlapping_texts,
    preceding_context,
)


def utt(utt_id, words, labels):
    return Utterance(utt_id, [Token(w, l, "0", None, None) for w, l in zip(words, labels)])


def test_fixed_subset_is_stable(corpus_dir):
    utts = read_split(corpus_dir / "train_360.txt") + read_split(corpus_dir / "test.txt")
    a = fixed_subset(utts, 3, seed=1)
    assert [u.utt_id for u in a] == [u.utt_id for u in fixed_subset(utts, 3, seed=1)]
    assert [u.utt_id for u in a] == [u.utt_id for u in utts if u in a]  # corpus order
    # Removing an unchosen utterance does not change the choice.
    rest = [u for u in utts if u.utt_id != next(x.utt_id for x in utts if x not in a)]
    assert [u.utt_id for u in fixed_subset(rest, 3, seed=1)] == [u.utt_id for u in a]
    assert fixed_subset(utts, 100) == utts


def test_multi_reader_groups_and_agreement():
    words = ["the", "cat", "sat", "on", "mats", "."]
    a = utt("1_10_000001_000000", words, ["0", "2", "1", "0", "2", "NA"])
    b = utt("2_20_000001_000000", words, ["0", "1", "1", "0", "0", "NA"])
    c = utt("1_11_000001_000000", words, ["0", "2", "1", "0", "2", "NA"])  # same speaker as a
    groups = multi_reader_groups({"train": [a, c], "test": [b]})
    assert len(groups) == 1 and len(groups[0]) == 3
    # Pairs a-b and c-b (a-c is the same speaker): 3 of 5 agree in each -> 60%.
    assert cross_speaker_agreement(groups, 3) == (60.0, 10)
    # 2-way: 1 and 2 merge, so only "mats" differs.
    assert cross_speaker_agreement(groups, 2) == (80.0, 10)
    assert multi_reader_groups({"train": [a], "test": [c]}) == []


def test_overlap_and_boilerplate():
    words = ["this", "is", "a", "librivox", "recording", "."]
    lv1 = utt("1_1_000000_000000", words, ["0"] * 5 + ["NA"])
    lv2 = utt("2_2_000000_000000", words, ["0"] * 5 + ["NA"])
    s1 = utt("3_3_000001_000000", ["one", "two", "three", "four", "five"], ["0"] * 5)
    s2 = utt("4_4_000001_000000", ["one", "two", "three", "four", "five"], ["1"] * 5)
    assert is_boilerplate(lv1) and not is_boilerplate(s1)
    assert overlapping_texts([lv1, s1], [[lv2, s2]]) == {s1.utt_id}
    assert overlapping_texts([lv1, s1], [[lv2, s2]], include_boilerplate=True) == {
        lv1.utt_id, s1.utt_id}


def test_preceding_context():
    u = [utt(i, ["w"], ["0"]) for i in ("5_9_000002_000001", "5_9_000001_000000",
                                         "5_9_000002_000000", "6_9_000001_000000")]
    ctx = preceding_context(u, 2)
    assert [x.utt_id for x in ctx["5_9_000002_000001"]] == ["5_9_000001_000000",
                                                             "5_9_000002_000000"]
    assert ctx["5_9_000001_000000"] == [] and ctx["6_9_000001_000000"] == []
    assert chapter_position("5_9_000002_000001") == ("5_9", (2, 1))
    with pytest.raises(ValueError):
        chapter_position("bad")


def test_real_cross_speaker_agreement(real_data_dir):
    """Counts of the multi-reader groups in the pinned corpus."""
    splits = {s: read_split(real_data_dir / f"{s}.txt")
              for s in ("train_360", "train_100", "dev", "test")}
    groups = multi_reader_groups(splits)
    plain = [g for g in groups if not is_boilerplate(g[0][1])]
    assert len(plain) == 449
    acc2, pairs = cross_speaker_agreement(plain, 2)
    acc3, _ = cross_speaker_agreement(plain, 3)
    assert pairs == 7343
    assert round(acc2, 1) == 77.0 and round(acc3, 1) == 63.7
    leak = overlapping_texts(splits["test"], [splits["train_360"], splits["train_100"],
                                              splits["dev"]])
    assert len(leak) > 0
