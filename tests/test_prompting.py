import gzip
import json
import math

import numpy as np
import pytest

from prosody_llms.data.corpus import read_split
from prosody_llms.prompting import run as prun
from prosody_llms.prompting.cache import CostLog, ResponseCache
from prosody_llms.prompting.client import APIError, filtered_response, label_distribution
from prosody_llms.prompting.parse import label_positions, parse
from prosody_llms.prompting.prompts import (
    FORMATS,
    PromptSpec,
    chat_messages,
    gold_answer,
    select_exemplars,
    user_message,
    word_items,
)


@pytest.fixture
def utts(corpus_dir):
    return read_split(corpus_dir / "test.txt")


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("task", [2, 3])
def test_gold_answers_parse_back_exactly(utts, fmt, task):
    spec = PromptSpec("minimal", fmt, task, 0)
    for u in utts:
        p = parse(gold_answer(u, spec), u, spec)
        assert p.labels == u.labels(task) and p.coverage == 1.0


def test_user_message_numbers_only_words(utts):
    u = utts[1]  # A cat runs ,
    msg = user_message(u, PromptSpec("linguistic", "json", 3, 0))
    assert "Sentence: A cat runs," in msg
    assert msg.endswith("Words:\n1 A\n2 cat\n3 runs")
    assert "0 = not prominent, 1 = somewhat prominent, 2 = strongly prominent" in msg
    assert word_items(u) == [(0, "A"), (1, "cat"), (2, "runs")]


def test_parser_tolerates_noise(utts):
    u = utts[0]  # The dog sat -> 0 2 1
    spec = PromptSpec("minimal", "lines", 3, 0)
    noisy = "<think>hmm</think>```\nHere you go:\n1. The: 0\n2 dog - 2\n3) sat 1\n```"
    assert parse(noisy, u, spec).labels == ["0", "2", "1"]
    partial = parse("1 The 0\n3 sat 1\n7 extra 2", u, spec)
    assert partial.labels == ["0", "0", "1"] and partial.parsed == 2
    no_numbers = parse("The 0\ndog 2\nsat 1", u, spec)
    assert no_numbers.labels == ["0", "2", "1"]
    js = PromptSpec("minimal", "json", 3, 0)
    assert parse("Sure: {'2': [2], '1': ['3']}", u, js).labels == ["0", "2", "1"]
    assert parse("not json", u, js).coverage == 0.0
    mk = PromptSpec("minimal", "marked", 3, 0)
    assert parse("The DOG *sat*.", u, mk).labels == ["0", "2", "1"]


def test_marked_capitals_are_ambiguous_for_capitalised_words(corpus_dir):
    from prosody_llms.data.corpus import Token, Utterance

    u = Utterance("1_1_000000_000000", [Token(w, l, "0", None, None)
                                        for w, l in [("I", "2"), ("saw", "0"), ("NASA", "1")]])
    p = parse("I saw NASA", u, PromptSpec("minimal", "marked", 2, 0))
    assert p.labels == ["0", "0", "0"] and p.stats["caps_ambiguous"] == 2


def test_label_distribution_and_positions(utts):
    entry = {"top_logprobs": [{"token": "2", "logprob": math.log(0.6)},
                              {"token": " 1", "logprob": math.log(0.3)},
                              {"token": "x", "logprob": math.log(0.05)}]}
    d = label_distribution(entry, ("0", "1", "2"))
    assert d[2] > d[1] > d[0] > 0 and abs(sum(d) - 1) < 1e-9
    assert label_distribution({"top_logprobs": [{"token": "a", "logprob": 0.0}]}, ("0", "1")) is None
    text = "1 The 0\n2 dog 2\n3 sat 1"
    pos = label_positions(text, utts[0], 3)
    assert [text[pos[n]] for n in (1, 2, 3)] == ["0", "2", "1"]


def test_exemplars_are_fixed_and_moderate(corpus_dir):
    train = read_split(corpus_dir / "train_360.txt")
    assert select_exemplars(train, 0) == []
    ex = select_exemplars(train, 2, min_words=2)
    assert ex == select_exemplars(train, 2, min_words=2) and len(ex) == 2
    msgs = chat_messages(train[0], PromptSpec("minimal", "lines", 3, 2), ex)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]


def test_roc_auc():
    assert prun.roc_auc(np.array([0.9, 0.8, 0.1]), np.array([True, True, False])) == 1.0
    assert prun.roc_auc(np.array([0.5, 0.5]), np.array([True, False])) == 0.5
    assert prun.roc_auc(np.array([0.5]), np.array([True])) is None


# ----------------------------------------------------------------------------- runner


class FakeClient:
    """Answers with the gold labels of the item, with log-probabilities in the lines
    format; can simulate failures and content-filter refusals."""

    def __init__(self, utts, fail=0, filter_ids=(), fatal=False):
        self.by_text = {}
        self.utts = utts
        self.calls = 0
        self.fail, self.filter_ids, self.fatal = fail, set(filter_ids), fatal

    def chat(self, body):
        self.calls += 1
        if self.fatal:
            raise APIError("HTTP 401: unauthorised", transient=False)
        if self.fail:
            self.fail -= 1
            raise APIError("HTTP 429: slow down", transient=True)
        last = body["messages"][-1]["content"]
        u = next(u for u in self.utts if last.endswith(
            "Words:\n" + "\n".join(f"{n} {w}" for n, (_, w) in enumerate(word_items(u), 1))))
        if u.utt_id in self.filter_ids:
            return filtered_response('{"error": {"code": "content_filter"}}')
        task = 3 if "2 = strongly" in last else 2
        fmt = "lines" if "one line per" in last else "json" if "JSON" in last else "marked"
        text = gold_answer(u, PromptSpec("minimal", fmt, task, 0))
        choice = {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        if body.get("logprobs"):
            toks, content = [], []
            for ch in text:  # one token per character keeps offsets simple
                top = [{"token": ch, "logprob": math.log(0.7)}]
                if ch in "012":
                    top += [{"token": o, "logprob": math.log(0.15)} for o in "012" if o != ch][:2]
                content.append({"token": ch, "top_logprobs": top})
            choice["logprobs"] = {"content": content}
        return {"choices": [choice], "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                                               "total_tokens": 120}}


def _args(corpus_dir, tmp_path, *extra):
    return prun.build_parser().parse_args(
        ["--model", "mock-model", "--split", "test", "--data-dir", str(corpus_dir),
         "--out-root", str(tmp_path / "prompting"), "--concurrency", "1", *extra])


def test_dry_run_sends_nothing(corpus_dir, tmp_path, utts):
    client = FakeClient(utts)
    res = prun.run(_args(corpus_dir, tmp_path, "--grid", "stage1"), client=client)
    assert res["dry_run"] and client.calls == 0
    assert res["plan"]["to_send"] == 9 * 3 and res["plan"]["est_usd_upper_bound"] > 0


def test_confirmed_run_scores_and_caches(corpus_dir, tmp_path, utts):
    client = FakeClient(utts)
    res = prun.run(_args(corpus_dir, tmp_path, "--confirm-spend", "--max-usd", "1"), client=client)
    s = res["summaries"][0]
    assert s[3] == 100.0 and s[2] == 100.0 and s["coverage"] == 1.0
    assert s["auc"] == 1.0  # gold labels with 0.7 probability on the right label
    out = tmp_path / "prompting" / "mock-model" / "test" / "minimal-lines-3way-k0"
    recs = [json.loads(line) for line in open(out / "records.jsonl")]
    assert {(r["task"], r["condition"]["derived"]) for r in recs} == {(3, False), (2, True)}
    assert recs[0]["metrics"]["legacy_2019"]["scored_words"] == 6
    with gzip.open(out / "responses.jsonl.gz", "rt") as f:
        assert len(f.readlines()) == 3
    assert (out / "probabilities.json.gz").exists() and (out / "predictions.tsv.gz").exists()
    assert CostLog(tmp_path / "prompting" / "costs.jsonl").total() > 0
    # Rerunning sends nothing: everything is cached.
    again = FakeClient(utts)
    prun.run(_args(corpus_dir, tmp_path, "--confirm-spend", "--max-usd", "1"), client=again)
    assert again.calls == 0


def test_spend_needs_budget_and_key(corpus_dir, tmp_path, utts, monkeypatch):
    with pytest.raises(prun.SpendNotConfirmed, match="max-usd"):
        prun.run(_args(corpus_dir, tmp_path, "--confirm-spend"), client=FakeClient(utts))
    monkeypatch.delenv("MOCK_API_KEY", raising=False)
    with pytest.raises(prun.SpendNotConfirmed, match="MOCK_API_KEY"):
        prun.run(_args(corpus_dir, tmp_path, "--confirm-spend", "--max-usd", "1"))


def test_budgets_stop_the_run(corpus_dir, tmp_path, utts, monkeypatch):
    client = FakeClient(utts)
    prun.run(_args(corpus_dir, tmp_path, "--confirm-spend", "--max-usd", "0.0001"), client=client)
    assert client.calls == 0  # the worst case of a single request is over the run budget
    cfg = prun.load_config()
    cfg["budget"]["total_usd"] = 0.0
    monkeypatch.setattr(prun, "load_config", lambda path=None: cfg)
    client = FakeClient(utts)
    prun.run(_args(corpus_dir, tmp_path, "--confirm-spend", "--max-usd", "5"), client=client)
    assert client.calls == 0  # the project cap is reached


def test_failures_filters_and_fatal_errors(corpus_dir, tmp_path, utts):
    client = FakeClient(utts, fail=1, filter_ids={utts[1].utt_id})
    res = prun.run(_args(corpus_dir, tmp_path, "--confirm-spend", "--max-usd", "1"), client=client)
    s = res["summaries"][0]
    assert s["missing"] == 1 and s["filtered"] == 1  # one skipped (not cached), one refused
    assert res["sent"]["failed"] == 1
    retry = FakeClient(utts)
    prun.run(_args(corpus_dir, tmp_path, "--confirm-spend", "--max-usd", "1"), client=retry)
    assert retry.calls == 1  # only the skipped item is sent again
    with pytest.raises(APIError):
        prun.run(_args(corpus_dir, tmp_path, "--split", "dev", "--confirm-spend", "--max-usd", "1"),
                 client=FakeClient(read_split(corpus_dir / "dev.txt"), fatal=True))


def test_reasoning_model_request_body(corpus_dir, utts):
    cfg = prun.load_config()
    body = prun.build_body(cfg["models"]["gpt-5.5"], utts[0], PromptSpec(), [])
    assert "temperature" not in body and "max_tokens" not in body
    assert body["max_completion_tokens"] == 4096 and body["reasoning_effort"] == "low"
    gpt4o = prun.build_body(cfg["models"]["gpt-4o"], utts[0], PromptSpec(), [])
    assert gpt4o["temperature"] == 0 and gpt4o["top_logprobs"] == 10
    mistral = prun.build_body(cfg["models"]["mistral-large-3"], utts[0], PromptSpec(), [])
    assert "logprobs" not in mistral
    assert prun.billed_tokens({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 40},
                              0, 0) == (10, 30)


def test_cache_roundtrip(tmp_path):
    c = ResponseCache(tmp_path / "c.sqlite")
    c.put("k", "p", {"model": "m"}, {"choices": [], "usage": {"prompt_tokens": 1}}, 0.5)
    assert c.has("k") and c.get("k")["usage"] == {"prompt_tokens": 1} and c.total_cost() == 0.5


# ----------------------------------------------------------------------------- sweep


class PickyClient(FakeClient):
    """Answers correctly only when ``good(messages)`` holds; otherwise answers nothing."""

    def __init__(self, utts, good=lambda msgs: False):
        super().__init__(utts)
        self.good = good

    def chat(self, body):
        resp = super().chat(body)
        if not self.good(body["messages"]):
            resp["choices"][0]["message"]["content"] = ""
            resp["choices"][0].pop("logprobs", None)
        return resp


@pytest.fixture
def sweep_env(corpus_dir, monkeypatch):
    from prosody_llms.prompting import sweep as sw

    monkeypatch.setattr(prun, "SHOTS_GRID", (1, 2))
    monkeypatch.setattr(prun, "select_exemplars", lambda train, k, **kw: list(train[:k]))
    return sw


def _sweep_args(sw, corpus_dir, tmp_path, *extra):
    return sw.build_parser().parse_args(
        ["--models", "mock-model", "--reasoning-model", "mock-model", "--reasoning-follows",
         "mock-model", "--dev-split", "dev", "--test-split", "test", "--reasoning-split", "test2",
         "--data-dir", str(corpus_dir), "--out-root", str(tmp_path / "prompting"),
         "--concurrency", "1", *extra])


def test_sweep_dry_run_sends_nothing(sweep_env, corpus_dir, tmp_path):
    sw = sweep_env
    calls = []
    res = sw.sweep(_sweep_args(sw, corpus_dir, tmp_path),
                   client_for=lambda m: calls.append(m) or FakeClient([]))
    assert res["dry_run"] and not calls and res["upper"] >= res["expected"] >= 0
    assert all(r["to_send"] > 0 for r in res["rows"])
    assert [r["stage"] for r in res["rows"]] == list(sw.STAGES)
    assert all("not chosen" in r["setting"] for r in res["rows"][2:])


def test_sweep_selects_stage_by_stage_and_resumes(sweep_env, corpus_dir, tmp_path):
    sw = sweep_env
    utts = read_split(corpus_dir / "dev.txt") + read_split(corpus_dir / "test.txt")
    clients = []

    def client_for(model):
        clients.append(PickyClient(utts))
        return clients[-1]

    # Every answer is empty, so every setting ties and the first one is chosen each time.
    sw.sweep(_sweep_args(sw, corpus_dir, tmp_path, "--confirm-spend", "--max-usd", "5",
                         "--until", "stage1"), client_for=client_for)
    root = tmp_path / "prompting"
    assert (root / "sweep_summary.md").exists()
    sent_before = sum(c.calls for c in clients)
    assert sent_before == 1 + 9 * 2 - 1  # the connection check's item is reused by stage 1
    res = sw.sweep(_sweep_args(sw, corpus_dir, tmp_path, "--confirm-spend", "--max-usd", "5"),
                   client_for=client_for)
    assert not res["dry_run"]
    sel = json.loads((root / "sweep_selection.json").read_text())
    assert sel["mock-model"] == {"variant": "minimal", "fmt": "lines", "shots": 0}
    assert (root / "mock-model" / "test" / "minimal-lines-3way-k0" / "result.json").exists()
    assert (root / "mock-model" / "dev" / "minimal-lines-2way-k0" / "result.json").exists()
    # Everything is cached now: a rerun sends nothing.
    n = sum(c.calls for c in clients)
    sw.sweep(_sweep_args(sw, corpus_dir, tmp_path, "--confirm-spend", "--max-usd", "5"),
             client_for=client_for)
    assert sum(c.calls for c in clients) == n


def test_sweep_selection_follows_dev_accuracy(sweep_env, corpus_dir, tmp_path):
    sw = sweep_env
    utts = read_split(corpus_dir / "dev.txt") + read_split(corpus_dir / "test.txt")
    root = tmp_path / "prompting"
    args = _sweep_args(sw, corpus_dir, tmp_path, "--confirm-spend", "--max-usd", "5")
    # Right only for json prompts with 0 or 1 example: stage 1 picks minimal/json (the
    # first json setting), and 0 shots ties with 1 shot, so 0 is kept.
    good = lambda msgs: "JSON" in msgs[-1]["content"] and len(msgs) in (1, 3)  # noqa: E731
    sw.sweep(args, client_for=lambda m: PickyClient(utts, good))
    sel = json.loads((root / "sweep_selection.json").read_text())
    assert sel["mock-model"] == {"variant": "minimal", "fmt": "json", "shots": 0}
    assert (root / "mock-model" / "test" / "minimal-json-3way-k0" / "result.json").exists()
    assert (root / "mock-model" / "test2" / "minimal-json-3way-k0" / "result.json").exists()


def test_sweep_stops_on_incomplete_stage(sweep_env, corpus_dir, tmp_path):
    sw = sweep_env
    utts = read_split(corpus_dir / "dev.txt")
    args = _sweep_args(sw, corpus_dir, tmp_path, "--confirm-spend", "--max-usd", "5")
    with pytest.raises(sw.SweepIncomplete, match="missing"):
        sw.sweep(args, client_for=lambda m: FakeClient(utts, filter_ids=(), fail=10**6))
    with pytest.raises(prun.SpendNotConfirmed):
        sw.sweep(_sweep_args(sw, corpus_dir, tmp_path, "--confirm-spend"))
