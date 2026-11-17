"""Plan, cost-check and run prompting experiments on hosted models.

    python -m prosody_llms.prompting.run --model gpt-4o --split dev500 --grid stage1
    python -m prosody_llms.prompting.run --model gpt-4o --split dev500 --grid stage1 \\
        --confirm-spend --max-usd 10

Without ``--confirm-spend`` nothing is sent: the runner builds every request, checks the
cache, prints the number of requests and a cost estimate, and stops. With it, it also
needs ``--max-usd`` (budget for this invocation) and known prices; it reserves the
worst-case cost of each request before sending it, stops before the run's budget or the
project's total budget (``[budget] total_usd`` in configs/prompting.toml, checked against
the cost log) would be exceeded, and stops after 20 transient failures in a row. Failed
items are not cached, so rerunning the same command sends only what is missing. Outputs
are always written from the cache.

Splits: ``dev500`` (the fixed 500-utterance dev subset for the grid), ``dev``, ``test``
(scored on all utterances and on the 2019 subset), ``test<N>`` (a fixed subset of N test
utterances, e.g. for the reasoning model).

Grids (one invocation, several prompt settings):

* ``stage1``: 3 instruction variants x 3 formats, 3-way, 0 shots (9 settings);
* ``shots``: the given --variant/--format with 1, 5 and 10 shots;
* ``twoway``: the given --variant/--format/--shots as a native 2-way prompt.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from prosody_llms.data.corpus import LABEL_NA, Utterance, read_split
from prosody_llms.data.download import default_data_dir
from prosody_llms.data.subsets import fixed_subset
from prosody_llms.eval.io import evaluate, make_record
from prosody_llms.prompting.cache import CostLog, ResponseCache, default_root, request_key
from prosody_llms.prompting.client import (
    APIError,
    ChatClient,
    is_filtered,
    label_distribution,
    response_text,
    token_logprobs,
)
from prosody_llms.prompting.parse import label_positions, parse
from prosody_llms.prompting.prompts import (
    FORMATS,
    VARIANTS,
    PromptSpec,
    chat_messages,
    select_exemplars,
    word_items,
)
from prosody_llms.utils.runinfo import environment_info, write_json

log = logging.getLogger("prosody_llms.prompting")


def write_predictions_gz(path: Path, utts, preds, task: int) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for u, ps in zip(utts, preds):
            f.write(f"<file>\t{u.utt_id}\n")
            for w, g, p in zip(u.words, u.labels(task), ps):
                f.write(f"{w}\t{g}\t{p}\n")
            f.write("\n")
MAX_FAILURES_IN_A_ROW = 20
SHOTS_GRID = (1, 5, 10)  # larger counts: the open-model many-shot run (prompting.local)


class SpendNotConfirmed(RuntimeError):
    pass


def config_path() -> Path:
    return Path(os.environ.get("PROSODY_PROMPTING_CONFIG",
                               Path(__file__).resolve().parents[3] / "configs" / "prompting.toml"))


def load_config(path: Path | None = None) -> dict:
    with open(path or config_path(), "rb") as f:
        return tomllib.load(f)


def model_entry(alias: str, cfg: dict | None = None) -> tuple[dict, dict]:
    cfg = cfg or load_config()
    if alias not in cfg["models"]:
        raise KeyError(f"model {alias!r} not in configs/prompting.toml; known: {', '.join(cfg['models'])}")
    m = cfg["models"][alias]
    return m, cfg["providers"][m["provider"]]


# ----------------------------------------------------------------------------- requests


@dataclass
class Request:
    utt: Utterance
    spec: PromptSpec
    body: dict
    key: str


def max_output_tokens(utt: Utterance, spec: PromptSpec) -> int:
    n = len(word_items(utt))
    return {"lines": 10 * n + 40, "json": 6 * n + 40, "marked": 8 * n + 60}[spec.fmt]


def build_body(m: dict, utt: Utterance, spec: PromptSpec, exemplars) -> dict:
    body = {"model": m["name"], "messages": chat_messages(utt, spec, exemplars)}
    if m.get("reasoning") == "always":
        body["max_completion_tokens"] = int(m.get("max_completion_tokens", 4096))
        if m.get("reasoning_effort"):
            body["reasoning_effort"] = m["reasoning_effort"]
    else:
        body.update(temperature=0, max_tokens=max_output_tokens(utt, spec), seed=0)
        if m.get("logprobs") and spec.fmt == "lines":
            body.update(logprobs=True, top_logprobs=10)
    return body


def build_requests(model: str, utts, specs, train, cfg=None) -> list[Request]:
    m, _ = model_entry(model, cfg)
    reqs = []
    for spec in specs:
        exemplars = select_exemplars(train, spec.shots)
        for u in utts:
            body = build_body(m, u, spec, exemplars)
            reqs.append(Request(u, spec, body, request_key(m["provider"], body)))
    return reqs


def estimate_input_tokens(body: dict) -> int:
    return sum(len(msg["content"]) for msg in body["messages"]) // 4 + 8 * len(body["messages"]) + 8


def output_limit(body: dict) -> int:
    return int(body.get("max_completion_tokens") or body.get("max_tokens") or 512)


def expected_output_tokens(req: Request) -> int:
    n = len(word_items(req.utt))
    return {"lines": 6 * n + 4, "json": 2 * n + 12, "marked": 3 * n + 10}[req.spec.fmt]


def cost_usd(m: dict, in_tok: int, out_tok: int) -> float:
    return (in_tok * m["usd_per_m_input"] + out_tok * m["usd_per_m_output"]) / 1e6


def billed_tokens(usage: dict, est_in: int, est_out: int) -> tuple[int, int]:
    """Hidden reasoning tokens are billed as output: OpenAI counts them in
    completion_tokens, other providers only in total_tokens. Output = the larger of
    completion_tokens and total_tokens - prompt_tokens."""
    in_tok = int(usage.get("prompt_tokens", est_in))
    out_tok = int(usage.get("completion_tokens", est_out))
    if usage.get("total_tokens") is not None:
        out_tok = max(out_tok, int(usage["total_tokens"]) - in_tok)
    return in_tok, out_tok


def plan(model: str, reqs: list[Request], cache: ResponseCache, cfg=None) -> dict:
    m, _ = model_entry(model, cfg)
    todo = [r for r in reqs if not cache.has(r.key)]
    in_tok = sum(estimate_input_tokens(r.body) for r in todo)
    exp_out = sum(expected_output_tokens(r) for r in todo)
    max_out = sum(output_limit(r.body) for r in todo)
    reasoning = m.get("reasoning") == "always"
    return {
        "model": model, "deployment": m["name"], "requests": len(reqs),
        "cached": len(reqs) - len(todo), "to_send": len(todo),
        "est_input_tokens": in_tok, "est_output_tokens": exp_out, "max_output_tokens": max_out,
        "est_usd_expected": None if reasoning else round(cost_usd(m, in_tok, exp_out), 2),
        "est_usd_upper_bound": round(cost_usd(m, in_tok, max_out), 2),
        "todo": todo,
    }


# ----------------------------------------------------------------------------- sending


def base_url(prov: dict) -> str:
    """The provider's endpoint: from the environment variable named in ``base_url_env``
    (e.g. your own Azure resource), or else ``base_url``."""
    env = prov.get("base_url_env")
    if env:
        url = os.environ.get(env)
        if url:
            return url
        if not prov.get("base_url"):
            raise SpendNotConfirmed(f"environment variable {env} is not set (the endpoint URL)")
    return prov["base_url"]


def make_client(model: str, cfg=None) -> ChatClient:
    m, prov = model_entry(model, cfg)
    key = os.environ.get(prov["api_key_env"])
    if not key:
        raise SpendNotConfirmed(f"environment variable {prov['api_key_env']} is not set")
    return ChatClient(base_url(prov), key, auth=prov.get("auth", "bearer"),
                      timeout=float(m.get("timeout", 120)), api_version=prov.get("api_version"),
                      deployment_in_path=bool(prov.get("deployment_in_path")))


def send(model: str, run_label: str, todo: list[Request], client, cache: ResponseCache,
         costs: CostLog, max_usd: float, total_cap: float, concurrency: int = 4,
         cfg=None) -> dict:
    m, _ = model_entry(model, cfg)
    lock = threading.Lock()
    already = costs.total()
    state = {"committed": 0.0, "reserved": 0.0, "stopped": None, "failed": 0, "in_a_row": 0,
             "sent": 0}

    def one(req: Request):
        worst = cost_usd(m, estimate_input_tokens(req.body) * 2, output_limit(req.body))
        with lock:
            if state["stopped"]:
                return
            spent = state["committed"] + state["reserved"] + worst
            if spent > max_usd:
                state["stopped"] = f"run budget of ${max_usd:.2f} reached"
                return
            if already + spent > total_cap:
                state["stopped"] = f"project budget of ${total_cap:.2f} reached"
                return
            state["reserved"] += worst
        try:
            resp = client.chat(req.body)
        except APIError as e:
            with lock:
                state["reserved"] -= worst
                if not e.transient:
                    state["stopped"] = f"error: {e}"
                    raise
                state["failed"] += 1
                state["in_a_row"] += 1
                log.warning("request failed and was skipped (%s): %s", e, req.utt.utt_id)
                if state["in_a_row"] >= MAX_FAILURES_IN_A_ROW:
                    state["stopped"] = f"{MAX_FAILURES_IN_A_ROW} failures in a row; last: {e}"
            return
        in_tok, out_tok = billed_tokens(resp.get("usage") or {}, estimate_input_tokens(req.body),
                                        output_limit(req.body))
        c = cost_usd(m, in_tok, out_tok)
        cache.put(req.key, m["provider"], req.body, resp, c)
        costs.add(run=run_label, model=model, deployment=m["name"], prompt=req.spec.name,
                  utt_id=req.utt.utt_id, input_tokens=in_tok, output_tokens=out_tok, cost_usd=c,
                  filtered=is_filtered(resp))
        with lock:
            state["reserved"] -= worst
            state["committed"] += c
            state["in_a_row"] = 0
            state["sent"] += 1

    if concurrency <= 1:
        for r in todo:
            one(r)
    else:
        with ThreadPoolExecutor(concurrency) as ex:
            list(ex.map(one, todo))
    if state["failed"]:
        log.warning("%d request(s) failed and were skipped; rerun to send them", state["failed"])
    if state["stopped"]:
        log.warning("stopped early: %s", state["stopped"])
    log.info("sent %d request(s), spent about $%.4f (project total about $%.2f)", state["sent"],
             state["committed"], already + state["committed"])
    return {k: v for k, v in state.items() if k != "reserved"}


# ----------------------------------------------------------------------------- outputs


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Area under the ROC curve (ties get half credit); None if one class is missing."""
    pos, neg = scores[positive], scores[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def word_probabilities(resp: dict, text: str, utt: Utterance, task: int):
    """{word number: [P(label)]} from the label tokens' top log-probabilities (lines format)."""
    entries = token_logprobs(resp)
    if not entries or not text:
        return None
    labels = ("0", "1", "2") if task == 3 else ("0", "1")
    positions = label_positions(text, utt, task)
    starts, pos = [], 0
    for e in entries:
        starts.append(pos)
        pos += len(e.get("token") or "")
    out = {}
    for num, off in positions.items():
        idx = next((k for k, s in enumerate(starts)
                    if s <= off < s + len(entries[k].get("token") or "")), None)
        if idx is not None:
            dist = label_distribution(entries[idx], labels)
            if dist is not None:
                out[num] = dist
    return out or None


def run_dir(root: Path, model: str, split: str, spec: PromptSpec) -> Path:
    return root / model / split / spec.name


def write_outputs(model: str, split: str, utts, reqs: list[Request], cache: ResponseCache,
                  root: Path, plan_summary: dict, sent: dict | None) -> list[dict]:
    by_spec: dict[PromptSpec, list[Request]] = {}
    for r in reqs:
        by_spec.setdefault(r.spec, []).append(r)
    summaries = []
    for spec, rs in by_spec.items():
        out = run_dir(root, model, split, spec)
        out.mkdir(parents=True, exist_ok=True)
        preds, missing, filtered, parsed, n_words, stats, probs = [], 0, 0, 0, 0, {}, {}
        raw_rows = []
        for r in rs:
            resp = cache.get(r.key)
            if resp is None:
                missing += 1
                labels = [LABEL_NA if t.is_punct else "0" for t in r.utt.tokens]
                n_words += len(word_items(r.utt))
                preds.append(labels)
                raw_rows.append({"utt_id": r.utt.utt_id, "missing": True})
                continue
            text = response_text(resp)
            blocked = is_filtered(resp)
            filtered += blocked
            p = parse(text, r.utt, spec)
            preds.append(p.labels)
            parsed += p.parsed
            n_words += p.n_words
            for k, v in p.stats.items():
                stats[k] = stats.get(k, 0) + (int(v) if isinstance(v, bool) else v)
            wp = word_probabilities(resp, text or "", r.utt, spec.task) if spec.fmt == "lines" else None
            if wp:
                probs[r.utt.utt_id] = wp
            raw_rows.append({"utt_id": r.utt.utt_id, "response": text, "filtered": blocked,
                             "usage": resp.get("usage"), "coverage": round(p.coverage, 4)})
        task_metrics = {spec.task: evaluate(utts, preds, spec.task)}
        if spec.task == 3:  # 2-way derived from the 3-way answers
            collapsed = [["1" if lab == "2" else lab for lab in ps] for ps in preds]
            task_metrics[2] = evaluate(utts, collapsed, 2)
        auc = None
        if probs:
            scores, gold = [], []
            for u in utts:
                wp = probs.get(u.utt_id, {})
                for num, (i, _) in enumerate(word_items(u), start=1):
                    if num in wp:
                        scores.append(sum(wp[num][1:]))  # P(prominent)
                        gold.append(u.labels(2)[i] == "1")
            if scores:
                auc = roc_auc(np.array(scores), np.array(gold))
        coverage = {"utterances": len(rs), "missing_responses": missing, "content_filtered": filtered,
                    "words": n_words, "words_parsed": parsed,
                    "word_coverage": round(parsed / n_words, 4) if n_words else None,
                    "words_with_probabilities": sum(len(v) for v in probs.values()), **stats}
        records = []
        for task, metrics in task_metrics.items():
            for k in ("full", "legacy_2019"):
                if k in metrics:
                    metrics[k]["coverage"] = coverage
            if task == 2 and auc is not None:
                metrics["full"]["auc_prominent"] = auc
            records.append(make_record(
                experiment="prompting", model=model, task=task, split=split,
                condition={"variant": spec.variant, "format": spec.fmt, "shots": spec.shots,
                           "prompt_task": spec.task, "derived": task != spec.task},
                metrics=metrics))
        with open(out / "records.jsonl", "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        write_predictions_gz(out / "predictions.tsv.gz", utts, preds, spec.task)
        with gzip.open(out / "responses.jsonl.gz", "wt", encoding="utf-8") as f:
            for row in raw_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if probs:
            with gzip.open(out / "probabilities.json.gz", "wt", encoding="utf-8") as f:
                json.dump(probs, f)
        result = {"experiment": "prompting", "model": model, "split": split, "prompt": asdict(spec),
                  "example_request": rs[0].body, "coverage": coverage,
                  "accuracy": {t: m["full"]["accuracy"] for t, m in task_metrics.items()},
                  "auc_prominent": auc,
                  "plan": {k: v for k, v in plan_summary.items() if k != "todo"},
                  "sent": sent, "environment": environment_info()}
        write_json(out / "result.json", result)
        summaries.append({"prompt": spec.name, **result["accuracy"], "coverage": coverage["word_coverage"],
                          "missing": missing, "filtered": filtered, "auc": auc})
    return summaries


# ----------------------------------------------------------------------------- CLI


def load_split(data_dir: Path, split: str) -> list[Utterance]:
    if split == "dev500":
        return fixed_subset(read_split(data_dir / "dev.txt"), 500, seed=0)
    if split in ("dev", "test"):
        return read_split(data_dir / f"{split}.txt")
    if split.startswith("test") and split[4:].isdigit():
        return fixed_subset(read_split(data_dir / "test.txt"), int(split[4:]), seed=0)
    raise ValueError(f"unknown split {split!r}")


def specs_for(args) -> list[PromptSpec]:
    if args.grid == "stage1":
        return [PromptSpec(v, f, 3, 0) for v in VARIANTS for f in FORMATS]
    if args.grid == "shots":
        return [PromptSpec(args.variant, args.format, 3, k) for k in SHOTS_GRID]
    if args.grid == "twoway":
        return [PromptSpec(args.variant, args.format, 2, args.shots)]
    return [PromptSpec(args.variant, args.format, args.task, args.shots)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="dev500")
    p.add_argument("--grid", choices=["stage1", "shots", "twoway"], default=None)
    p.add_argument("--variant", choices=VARIANTS, default="minimal")
    p.add_argument("--format", choices=FORMATS, default="lines")
    p.add_argument("--task", type=int, choices=[2, 3], default=3)
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--max-examples", type=int, default=None, help="first N utterances only (checks)")
    p.add_argument("--confirm-spend", action="store_true")
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--out-root", type=Path, default=default_root())
    return p


def run(args, client=None, cache=None, costs=None) -> dict:
    cfg = load_config()
    m, _ = model_entry(args.model, cfg)
    utts = load_split(args.data_dir, args.split)
    if args.max_examples:
        utts = utts[: args.max_examples]
    train = read_split(args.data_dir / "train_360.txt")
    specs = specs_for(args)
    cache = cache or ResponseCache(args.out_root / "cache.sqlite")
    costs = costs or CostLog(args.out_root / "costs.jsonl")
    reqs = build_requests(args.model, utts, specs, train, cfg)
    p = plan(args.model, reqs, cache, cfg)
    label = f"{args.model}/{args.split}/{args.grid or specs[0].name}"
    log.info("plan for %s: %s", label, json.dumps({k: v for k, v in p.items() if k != "todo"}))
    sent = None
    if p["todo"]:
        if not args.confirm_spend:
            print(json.dumps({k: v for k, v in p.items() if k != "todo"}, indent=2))
            print(f"DRY RUN: {p['to_send']} request(s) would be sent "
                  f"(expected ${p['est_usd_expected']}, at most ${p['est_usd_upper_bound']}). "
                  f"Project spending so far: ${costs.total():.2f} of ${cfg['budget']['total_usd']:.2f}. "
                  "Nothing was sent; add --confirm-spend --max-usd N after approval.")
            return {"dry_run": True, "plan": {k: v for k, v in p.items() if k != "todo"}}
        if args.max_usd is None:
            raise SpendNotConfirmed("--confirm-spend needs --max-usd")
        client = client or make_client(args.model, cfg)
        sent = send(args.model, label, p["todo"], client, cache, costs, args.max_usd,
                    float(cfg["budget"]["total_usd"]), args.concurrency, cfg)
    summaries = write_outputs(args.model, args.split, utts, reqs, cache, args.out_root, p, sent)
    for s in summaries:
        print(json.dumps(s))
    return {"dry_run": False, "summaries": summaries, "sent": sent}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(build_parser().parse_args(argv))
    except SpendNotConfirmed as e:
        print(f"NOT SENT: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
