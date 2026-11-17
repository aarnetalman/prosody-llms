from prosody_llms.data.corpus import read_split
from prosody_llms.prompting.parse import parse_lines
from prosody_llms.prompting.prompts import PromptSpec, gold_answer


def test_freegen_compare_and_summary(corpus_dir):
    from prosody_llms.audio.freegen import compare, summary_md

    u = read_split(corpus_dir / "test.txt")[0]
    spec = PromptSpec("linguistic", "lines", 3, 0)
    expected = gold_answer(u, spec)
    perfect = parse_lines(expected, u, 3)
    assert perfect.coverage == 1.0
    c = compare(u, perfect.labels, ["0" if x != "NA" else x for x in u.labels(3)])
    assert c["free_correct"] == c["words"] and sum(c["gold_dist"].values()) == c["words"]
    prose = parse_lines("The prominent words are the nouns.", u, 3)
    assert prose.coverage == 0.0
    row = {"utt_id": u.utt_id, "answer": "", "expected": expected, "coverage": 0.0,
           "compare": compare(u, prose.labels, prose.labels)}
    md = summary_md("m", "linguistic-freegen-audio", [row, dict(row, coverage=1.0, answer=expected)])
    assert "1 of 2" in md and "(empty)" in md


def test_parse_pairs_and_rescore(corpus_dir, tmp_path):
    import json

    from prosody_llms.audio.freegen import parse_free, rescore

    u = read_split(corpus_dir / "test.txt")[0]  # The dog sat
    parsed, fmt = parse_free("1 The 0 dog 2 sat 1", u)
    assert fmt == "pairs" and parsed.coverage == 1.0 and parsed.labels[:3] == ["0", "2", "1"]
    parsed, fmt = parse_free("1 The 0\n2 dog 2\n3 sat 1", u)
    assert fmt == "lines" and parsed.coverage == 1.0
    d = tmp_path / "m" / "test1" / "linguistic-freegen-audio"
    d.mkdir(parents=True)
    row = {"utt_id": u.utt_id, "answer": "The 0 dog 2 sat 1", "expected": "", "scored_labels": ["0"] * len(u.tokens)}
    (d / "answers.partial.jsonl").write_text(json.dumps(row) + "\n")
    res = rescore(tmp_path, "m", "test1", [u])
    assert res["linguistic-freegen-audio"]["kinds"] == {"labels": 1}
    assert res["linguistic-freegen-audio"]["labels_given"] == {"0": 1, "1": 1, "2": 1}
    from prosody_llms.audio.freegen import classify_answer

    assert classify_answer("1 The 2 dog 3 sat", u)[0] == "enumeration"
    assert (d / "rescored.json").exists() and (d / "summary.md").exists()
