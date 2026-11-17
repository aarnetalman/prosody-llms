"""Run metadata recorded with every result: library versions, device, git commit."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path


def git_commit() -> str | None:
    """Commit of this repository, with ``-dirty`` if ``src/`` has uncommitted changes."""
    try:
        root = Path(__file__).resolve().parents[3]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "src"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "git_commit": git_commit()}
    try:
        import numpy
        info["numpy"] = numpy.__version__
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


def write_json(path: Path, obj) -> None:
    """Write atomically, so an interrupted job never leaves a half-written result."""
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)
