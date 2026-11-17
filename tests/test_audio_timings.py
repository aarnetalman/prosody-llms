from prosody_llms.audio.timings import Interval, coverage, load_timings, map_tokens, read_textgrid
from prosody_llms.data.corpus import Token, Utterance

TG = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 2.0
tiers? <exists>
size = 2
item []:
\titem [1]:
\t\tclass = "IntervalTier"
\t\tname = "words"
\t\txmin = 0.0
\t\txmax = 2.0
\t\tintervals: size = 6
\t\t\tintervals [1]:
\t\t\t\txmin = 0.000
\t\t\t\txmax = 0.300
\t\t\t\ttext = "mister"
\t\t\tintervals [2]:
\t\t\t\txmin = 0.300
\t\t\t\txmax = 0.400
\t\t\t\ttext = ""
\t\t\tintervals [3]:
\t\t\t\txmin = 0.400
\t\t\t\txmax = 0.900
\t\t\t\ttext = "<unk>"
\t\t\tintervals [4]:
\t\t\t\txmin = 0.900
\t\t\t\txmax = 1.200
\t\t\t\ttext = "don't"
\t\t\tintervals [5]:
\t\t\t\txmin = 1.200
\t\t\t\txmax = 1.500
\t\t\t\ttext = "pshaw"
\t\t\tintervals [6]:
\t\t\t\txmin = 1.500
\t\t\t\txmax = 2.000
\t\t\t\ttext = "sil"
\titem [2]:
\t\tclass = "IntervalTier"
\t\tname = "phones"
\t\txmin = 0.0
\t\txmax = 2.0
\t\tintervals: size = 1
\t\t\tintervals [1]:
\t\t\t\txmin = 0.000
\t\t\t\txmax = 0.100
\t\t\t\ttext = "M"
'''


def utt(words, uid="1_2_000000_000000"):
    return Utterance(uid, [Token(w, "NA" if w in ",." else "0", "0", None, None) for w in words])


def test_read_textgrid(tmp_path):
    p = tmp_path / "x.TextGrid"
    p.write_text(TG)
    tiers = read_textgrid(p)
    assert [i.text for i in tiers["words"]] == ["mister", "", "<unk>", "don't", "pshaw", "sil"]
    assert tiers["phones"] == [Interval(0.0, 0.1, "M")]


def test_mapping_handles_abbreviations_unknowns_and_splits(tmp_path):
    words = read_textgrid(_write(tmp_path))["words"]
    u = utt(["Mr", "Bozzle", ",", "do", "n't", "p", "shaw", "."])
    t = map_tokens(u, words)
    spans = [(x.start, x.end) for x in t.timings]
    assert spans == [(0.0, 0.3), (0.4, 0.9), (None, None), (0.9, 1.2), (0.9, 1.2),
                     (1.2, 1.5), (1.2, 1.5), (None, None)]
    assert t.shared == [False, False, False, True, True, True, True, False]
    assert not t.exact and t.duration == 2.0


def test_unmatched_words_stay_without_timing(tmp_path):
    words = read_textgrid(_write(tmp_path))["words"]
    t = map_tokens(utt(["Mr", "X", "twenty", "one", "don't", "pshaw"]), words)
    assert t.mask() == [True, True, False, False, True, True]  # X is the <unk>


def test_load_and_coverage(tmp_path):
    root = tmp_path / "tg"
    (root / "dev-clean" / "1" / "2").mkdir(parents=True)
    (root / "dev-clean" / "1" / "2" / "1_2_000000_000000.TextGrid").write_text(TG)
    a = utt(["Mr", "Bozzle", "don't", "pshaw", "."])
    b = utt(["Hello"], uid="1_2_000000_000001")
    ts = [load_timings(root, "dev-clean", u) for u in (a, b)]
    assert ts[1] is None
    cov = coverage([a, b], ts)
    assert cov["words"] == 5 and cov["aligned_words"] == 4 and cov["without_textgrid"] == 1


def _write(tmp_path):
    p = tmp_path / "x.TextGrid"
    p.write_text(TG)
    return p
