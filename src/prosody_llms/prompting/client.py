"""Minimal OpenAI-compatible chat client (standard library only).

Plain OpenAI-compatible endpoints (bearer token) and both Azure endpoint styles are
supported: Azure OpenAI (deployment in the URL, ``api-key`` header) and Azure AI model inference
(OpenAI-compatible, bearer token).
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


class APIError(RuntimeError):
    """A failed request. ``transient=True``: it failed only after retries on errors that
    usually pass (timeouts, rate limits, server errors), so a run can skip the item and a
    rerun will send it again. Other errors (bad request, authentication) stop the run."""

    def __init__(self, msg: str, transient: bool = False):
        super().__init__(msg)
        self.transient = transient


@dataclass
class ChatClient:
    base_url: str
    api_key: str
    timeout: float = 120.0
    max_retries: int = 6
    auth: str = "bearer"  # "bearer" or "api-key" (Azure OpenAI)
    api_version: str | None = None  # Azure: ?api-version=...
    deployment_in_path: bool = False  # Azure OpenAI: /openai/deployments/<model>/chat/completions

    def url_for(self, body: dict) -> str:
        base = self.base_url.rstrip("/")
        if self.deployment_in_path:
            url = f"{base}/openai/deployments/{urllib.parse.quote(body['model'], safe='')}/chat/completions"
        else:
            url = base + "/chat/completions"
        if self.api_version:
            url += "?" + urllib.parse.urlencode({"api-version": self.api_version})
        return url

    def chat(self, body: dict) -> dict:
        url = self.url_for(body)
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.auth == "api-key":
            headers["api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", "replace")
                if e.code == 400 and "content_filter" in raw:
                    return filtered_response(raw)
                retryable = e.code in (408, 409, 429) or e.code >= 500
                if not retryable or attempt == self.max_retries:
                    raise APIError(f"HTTP {e.code}: {raw[:500]}", transient=retryable) from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise APIError(str(e), transient=True) from e
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise APIError("unreachable")


def filtered_response(raw: str) -> dict:
    """A content-filter refusal (Azure answers HTTP 400) as an ordinary response with no
    answer, so the item is recorded, cached and scored as wrong instead of stopping the run."""
    try:
        body = json.loads(raw)
    except ValueError:
        body = {}
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": None},
                         "finish_reason": "content_filter"}],
            "usage": body.get("usage") or {}, "content_filter": True, "filter_detail": raw[:1000]}


def is_filtered(resp: dict) -> bool:
    if resp.get("content_filter"):
        return True
    return any(c.get("finish_reason") == "content_filter" for c in resp.get("choices") or [])


def response_text(resp: dict) -> str | None:
    """The final answer text; reasoning traces returned in separate fields are ignored."""
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    content = msg.get("content")
    if isinstance(content, list):  # some providers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content


def token_logprobs(resp: dict) -> list[dict] | None:
    """``choices[0].logprobs.content``: one entry per generated token with ``token`` and
    ``top_logprobs``; None if the response has no log-probabilities."""
    try:
        content = resp["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return content or None


def label_distribution(entry: dict, labels: tuple[str, ...]) -> list[float] | None:
    """Probabilities of ``labels`` at one token position, from its top log-probabilities.

    Candidate tokens are stripped; a token equal to a label counts for it, and several such
    tokens are summed. Labels missing from the top list share the remaining probability
    mass evenly (at most the smallest listed probability each). The result is normalised
    over the labels. None if no label is among the candidates.
    """
    tops = entry.get("top_logprobs") or []
    probs = [0.0] * len(labels)
    found = [False] * len(labels)
    for t in tops:
        tok = (t.get("token") or "").strip()
        if tok in labels:
            i = labels.index(tok)
            probs[i] += math.exp(t["logprob"])
            found[i] = True
    if not any(found):
        return None
    floor = min(math.exp(t["logprob"]) for t in tops)
    missing = [i for i, f in enumerate(found) if not f]
    if missing:
        rest = max(0.0, 1.0 - sum(math.exp(t["logprob"]) for t in tops))
        share = min(floor, rest / len(missing)) if rest > 0 else 0.0
        for i in missing:
            probs[i] = share
    total = sum(probs)
    return [p / total for p in probs]
