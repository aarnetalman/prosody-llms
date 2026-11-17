"""Response cache (SQLite) and cost log (JSONL).

Every request is keyed by a
hash of the provider and the full request body, and stored with its raw response, so a
rerun never pays twice and every number can be traced back to the exact prompt. Request
headers (which carry the API key) are never stored.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path


def default_root() -> Path:
    return Path(os.environ.get("PROSODY_RUNS", "runs")) / "prompting"


def request_key(provider: str, body: dict) -> str:
    blob = json.dumps({"provider": provider, "body": body}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or default_root() / "cache.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, provider TEXT, model TEXT, "
            "request TEXT, response TEXT, usage TEXT, cost_usd REAL, created REAL)")
        self.db.commit()

    def get(self, key: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def has(self, key: str) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM responses WHERE key = ?", (key,)).fetchone() is not None

    def put(self, key: str, provider: str, body: dict, response: dict, cost: float) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (key, provider, body.get("model"), json.dumps(body, ensure_ascii=False),
                 json.dumps(response, ensure_ascii=False), json.dumps(response.get("usage")),
                 cost, time.time()))
            self.db.commit()

    def total_cost(self) -> float:
        with self.lock:
            row = self.db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM responses").fetchone()
        return float(row[0])


class CostLog:
    """Append-only log of every paid call: time, run, model, tokens, estimated cost."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or default_root() / "costs.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def add(self, **entry) -> None:
        entry.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def total(self, run: str | None = None, model: str | None = None) -> float:
        return sum(float(e.get("cost_usd", 0.0)) for e in self.entries()
                   if (run is None or e.get("run") == run) and (model is None or e.get("model") == model))
