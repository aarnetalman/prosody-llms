import os
import json
import sys

import pytest

from prosody_llms.audio import reconstruct as rc
from prosody_llms.audio.base import libritts_path
from prosody_llms.audio.timings import Interval, map_tokens
from prosody_llms.data.corpus import read_split

FAKE = '''#!{python}
"""Stands in for the toolkit: every word interval gets prominence = its corpus value + offset."""
import os, sys, glob, json
if sys.argv[1] == "-c":
    print("/dev/null"); sys.exit(0)
args = sys.argv[3:]
inp, out = args[0], args[args.index("-o") + 1]
values = json.load(open(os.environ["FAKE_VALUES"]))
files = [inp] if inp.endswith(".wav") else glob.glob(inp + "/*.wav")
for wav in files:
    base = os.path.splitext(os.path.basename(wav))[0]
    rows = values.get(base, [])
    with open(os.path.join(out, base + ".prom"), "w") as f:
        for s, e, w, v in rows:
            f.write(f"{{base}}\\t{{s:.3f}}\\t{{e:.3f}}\\t{{w}}\\t{{v:.3f}}\\t0.000\\n")
'''


def _textgrid(words):
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "xmin = 0", "xmax = 9",
             "tiers? <exists>", "size = 1", "item []:", "    item [1]:", '        class = "IntervalTier"',
             '        name = "words"', "        xmin = 0", "        xmax = 9",
             f"        intervals: size = {len(words)}"]
    for k, (s, e, w) in enumerate(words, 1):
        lines += [f"        intervals [{k}]:", f"            xmin = {s}", f"            xmax = {e}",
                  f'            text = "{w}"']
    return "\n".join(lines) + "\n"


@pytest.fixture
def setup(tmp_path, corpus_dir, monkeypatch):
    audio = tmp_path / "audio"
    values = {}
    for u in read_split(corpus_dir / "dev.txt"):
        words = [(0.1 * k, 0.1 * k + 0.1, t.word.lower()) for k, t in enumerate(u.tokens) if not t.is_punct]
        sp, ch = u.utt_id.split("_")[:2]
        tg = audio / "alignments" / "textgrid" / "dev-clean" / sp / ch / f"{u.utt_id}.TextGrid"
        tg.parent.mkdir(parents=True, exist_ok=True)
        tg.write_text(_textgrid(words))
        wav = libritts_path(audio, "dev", u.utt_id)
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"RIFF")
        vals = [t.prominence_value for t in u.tokens if not t.is_punct]
        values[u.utt_id] = [(s, e, w, v + 0.05) for (s, e, w), v in zip(words, vals)]
    vfile = tmp_path / "values.json"
    vfile.write_text(json.dumps(values))
    fake = tmp_path / "fake_python"
    fake.write_text(FAKE.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv("WPT_PYTHON", str(fake))
    monkeypatch.setenv("FAKE_VALUES", str(vfile))
    return audio


def test_reconstruction_scores_against_corpus(setup, corpus_dir, tmp_path):
    out = tmp_path / "rec"
    assert rc.main(["--splits", "dev", "--audio-dir", str(setup), "--data-dir", str(corpus_dir),
                    "--out-root", str(out), "--jobs", "1"]) == 0
    res = json.loads((out / "dev" / "result.json").read_text())
    assert res["values"]["coverage"] == 1.0 and res["values"]["pearson"] == pytest.approx(1.0)
    # +0.05 moves no corpus value across a cut-off in the fixture, so labels agree fully.
    assert res["accuracy"] == {"3": 100.0, "2": 100.0}
    recs = [json.loads(line) for line in open(out / "dev" / "records.jsonl")]
    assert {r["task"] for r in recs} == {2, 3} and "aligned" in recs[0]["metrics"]
    assert (out / "summary.md").exists()
    # Finished utterances are not labelled again.
    (out / "dev" / "prom").joinpath("marker").write_text("")
    rc.main(["--splits", "dev", "--audio-dir", str(setup), "--data-dir", str(corpus_dir),
             "--out-root", str(out), "--jobs", "1"])
    assert not list((out / "dev" / "input").iterdir())


def test_discretise_matches_corpus_cutoffs(corpus_dir):
    assert [rc.discretise(v) for v in (0.0, 0.399, 0.4, 1.199, 1.2, 6.0)] == ["0", "0", "1", "1", "2", "2"]


def test_word_scores_skip_shared_and_missing(corpus_dir):
    u = read_split(corpus_dir / "dev.txt")[1]  # A dog .
    t = map_tokens(u, [Interval(0.0, 0.1, "a"), Interval(0.1, 0.4, "dog")])
    assert rc.word_scores(u, t, {(0, 20): 0.7, (20, 80): 1.5}) == [0.7, 1.5, None]
    assert rc.word_scores(u, t, {(0, 20): 0.7}) == [0.7, None, None]
    assert rc.word_scores(u, None, {}) == [None, None, None]
    assert rc.word_scores(u, t, {(0, 20): -0.116, (20, 80): 1.5}) == [0.0, 1.5, None]  # clipped at 0


def test_toolkit_from_a_target_folder(tmp_path, monkeypatch):
    """The toolkit installed with ``pip install --target`` is found through WPT_PATH and
    run with this interpreter in a separate process."""
    (tmp_path / "wpt-lib").mkdir()  # dependencies folder; the package is in the source folder
    pkg = tmp_path / "wpt-src" / "wavelet_prosody_toolkit"
    (pkg / "configs").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "configs" / "libritts.yaml").write_text("x: 1\n")
    (pkg / "prosody_labeller.py").write_text(
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "assert os.environ['MPLBACKEND'] == 'Agg'\n"
        "out = args[args.index('-o') + 1]\n"
        "open(os.path.join(out, 'x.prom'), 'w').write('x\\t0.000\\t0.100\\ta\\t1.500\\t0.000\\n')\n")
    monkeypatch.delenv("WPT_PYTHON", raising=False)
    monkeypatch.setenv("WPT_PATH", os.pathsep.join([str(tmp_path / "wpt-src"), str(tmp_path / "wpt-lib")]))
    assert rc.toolkit_python() == sys.executable
    assert rc.toolkit_config(rc.toolkit_python()).name == "libritts.yaml"
    rc.run_toolkit(tmp_path / "in", tmp_path / "out", 1)
    assert rc.read_prom(tmp_path / "out" / "x.prom") == {(0, 20): 1.5}
    import importlib.util
    assert importlib.util.find_spec("wavelet_prosody_toolkit") is None  # not in this process
