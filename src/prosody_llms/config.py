"""Model registry (``configs/models.toml``): every model pinned to an exact revision."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("encoder", "decoder", "speech", "speech_text")
CHECKPOINTS = ("base", "post_trained")
STATUSES = ("planned", "proposed", "dropped")


def default_registry_path() -> Path:
    env = os.environ.get("PROSODY_MODELS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "configs" / "models.toml"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    id: str
    revision: str
    kind: str
    checkpoint: str
    status: str
    layers: int | None = None
    hidden: int | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind must be one of {KINDS}")
        if self.checkpoint not in CHECKPOINTS:
            raise ValueError(f"{self.name}: checkpoint must be one of {CHECKPOINTS}")
        if self.status not in STATUSES:
            raise ValueError(f"{self.name}: status must be one of {STATUSES}")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError(f"{self.name}: revision must be a full 40-character commit hash")


def load_registry(path: str | Path | None = None) -> dict[str, ModelSpec]:
    path = Path(path) if path else default_registry_path()
    with open(path, "rb") as f:
        raw = tomllib.load(f)["models"]
    known = {"id", "revision", "kind", "checkpoint", "status", "layers", "hidden", "notes"}
    out = {}
    for name, entry in raw.items():
        missing = {"id", "revision", "kind", "checkpoint", "status"} - set(entry)
        if missing:
            raise ValueError(f"{name}: missing {sorted(missing)}")
        out[name] = ModelSpec(
            name=name, id=entry["id"], revision=entry["revision"], kind=entry["kind"],
            checkpoint=entry["checkpoint"], status=entry["status"], layers=entry.get("layers"),
            hidden=entry.get("hidden"), notes=entry.get("notes", ""),
            extra={k: v for k, v in entry.items() if k not in known},
        )
    return out


def get_model(name: str, path: str | Path | None = None) -> ModelSpec:
    registry = load_registry(path)
    if name not in registry:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(sorted(registry))}")
    return registry[name]
