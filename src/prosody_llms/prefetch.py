"""Download and verify registry models (pinned revisions) in the Hugging Face cache.

Run it on a machine with internet before running jobs offline (``HF_HUB_OFFLINE=1``):

    python -m prosody_llms.prefetch --check             # what is cached / missing (needs network)
    python -m prosody_llms.prefetch --verify            # integrity of cached models (offline)
    python -m prosody_llms.prefetch qwen3.5-0.8b-base   # download selected models
    python -m prosody_llms.prefetch --max-gb 120        # download planned models up to 120 GB
    python -m prosody_llms.prefetch --dry-run           # sizes only

Only inference files are fetched (configs, tokenizer files, safetensors); duplicate
weight formats (.bin, .pth, .gguf, ``original/``) are excluded. Downloads go into
``$HF_HOME/hub`` and resume where an earlier download stopped. Every downloaded model is
verified straight away. Nothing is ever deleted from the cache.

``--verify`` checks, per model: the snapshot for the pinned revision exists; every
weight file listed in ``model.safetensors.index.json`` (or the single
``model.safetensors``) is a readable file, not a broken link, and its size matches its
own safetensors header (truncated downloads fail); ``config.json`` is present; and
there are no ``*.incomplete`` files under the model's ``blobs/``.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

from prosody_llms.config import load_registry

PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "tokenizer*", "*.jinja"]
IGNORE = ["*.bin", "*.pth", "*.pt", "*.h5", "*.msgpack", "*.onnx", "*.gguf", "original/*", "*.fp32*"]
# Checkpoints published only as PyTorch pickles (WavLM, HuBERT at their pinned revisions).
BIN_WEIGHTS = "pytorch_model.bin"


def patterns_for(filenames: list[str]) -> tuple[list[str], list[str]]:
    """(allow, ignore) patterns: safetensors when the revision has them, else the single
    ``pytorch_model.bin``."""
    if any(f.endswith(".safetensors") for f in filenames) or BIN_WEIGHTS not in filenames:
        return PATTERNS, IGNORE
    return PATTERNS + [BIN_WEIGHTS], [p for p in IGNORE if p != "*.bin"]


def selected(names: list[str], kinds: tuple[str, ...]) -> list:
    reg = load_registry()
    if names:
        missing = [n for n in names if n not in reg]
        if missing:
            raise SystemExit(f"unknown model(s): {', '.join(missing)}")
        return [reg[n] for n in names]
    return [m for m in reg.values() if m.status == "planned" and m.kind in kinds]


def hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return Path(home) / "hub"


def repo_dir(model_id: str, cache: Path | None = None) -> Path:
    return (cache or hub_cache()) / ("models--" + model_id.replace("/", "--"))


def safetensors_ok(path: Path) -> str | None:
    """None if the file is a complete safetensors file, else the problem."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or 8 + n > size:
                return "header length larger than the file"
            header = json.loads(f.read(n))
    except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError) as e:
        return f"unreadable ({e.__class__.__name__})"
    ends = [v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"]
    expected = 8 + n + (max(ends) if ends else 0)
    if size != expected:
        return f"size {size} bytes, header says {expected} (truncated?)"
    return None


def bin_ok(path: Path) -> str | None:
    """None if a PyTorch zip checkpoint is complete (its central directory reads)."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            z.infolist()
    except (OSError, zipfile.BadZipFile) as e:
        return f"unreadable ({e.__class__.__name__}; truncated?)"
    return None


def verify(model_id: str, revision: str, cache: Path | None = None) -> dict:
    """Integrity of one cached model. Works offline."""
    rdir = repo_dir(model_id, cache)
    snap = rdir / "snapshots" / revision
    problems: list[str] = []
    out = {"model": model_id, "revision": revision, "snapshot": str(snap), "weight_files": 0,
           "bytes": 0, "problems": problems}
    if not snap.is_dir():
        problems.append("no snapshot for the pinned revision")
        return out
    incomplete = list((rdir / "blobs").glob("*.incomplete")) if (rdir / "blobs").is_dir() else []
    # Leftovers of an interrupted download only matter if a weight file is also bad; they
    # are reported as a warning when every weight file checks out (see below).
    out["stale_incomplete"] = len(incomplete)
    if not (snap / "config.json").is_file():
        problems.append("config.json missing")
    index = snap / "model.safetensors.index.json"
    if index.exists():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        except (OSError, KeyError, json.JSONDecodeError) as e:
            problems.append(f"unreadable index ({e.__class__.__name__})")
            return out
        files = sorted(set(weight_map.values()))
    elif (snap / "model.safetensors").exists() or (snap / "model.safetensors").is_symlink():
        files = ["model.safetensors"]
    elif (snap / BIN_WEIGHTS).exists() or (snap / BIN_WEIGHTS).is_symlink():
        files = [BIN_WEIGHTS]
    else:
        problems.append("no weights (no safetensors index, model.safetensors or pytorch_model.bin)")
        return out
    for name in files:
        p = snap / name
        if p.is_symlink() and not p.exists():
            problems.append(f"{name}: broken link")
        elif not p.is_file():
            problems.append(f"{name}: missing")
        else:
            bad = bin_ok(p) if name == BIN_WEIGHTS else safetensors_ok(p)
            if bad:
                problems.append(f"{name}: {bad}")
            else:
                out["weight_files"] += 1
                out["bytes"] += p.stat().st_size
    out["expected_weight_files"] = len(files)
    if incomplete and problems:
        problems.append(f"{len(incomplete)} *.incomplete file(s) in blobs/")
    return out


def format_verify(name: str, r: dict) -> str:
    size = f"{r['bytes'] / 1e9:6.1f} GB ({r['bytes'] / 2**30:5.1f} GiB)"
    if not r["problems"]:
        stale = r.get("stale_incomplete", 0)
        note = f"  (warning: {stale} stale *.incomplete file(s) in blobs/)" if stale else ""
        return f"{name:26} complete   {r['weight_files']:3d} weight file(s), {size}{note}"
    shown = "; ".join(r["problems"][:4]) + (" ..." if len(r["problems"]) > 4 else "")
    return f"{name:26} INCOMPLETE {shown}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*")
    p.add_argument("--kinds", nargs="+", default=["encoder", "decoder"])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="sizes only")
    mode.add_argument("--check", action="store_true",
                      help="report which pinned files are cached; download nothing")
    mode.add_argument("--verify", action="store_true",
                      help="check the integrity of cached models; offline, download nothing")
    p.add_argument("--max-gb", type=float, default=None,
                   help="download only models whose missing files fit within this many GB in total")
    args = p.parse_args(argv)
    models = selected(args.models, tuple(args.kinds))
    print(f"cache: {hub_cache()}", flush=True)

    if args.verify:
        ok = True
        for m in models:
            r = verify(m.id, m.revision)
            ok &= not r["problems"]
            print(format_verify(m.name, r), flush=True)
        return 0 if ok else 1

    from fnmatch import fnmatch

    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache

    api = HfApi()
    total = missing_total = downloaded = 0.0
    failures: list[str] = []

    def download(m, allow, ignore):
        """Download, retrying once without the Xet backend, which can fail mid-download
        ("Background writer channel closed"); partial files resume."""
        kwargs = dict(revision=m.revision, allow_patterns=allow, ignore_patterns=ignore)
        try:
            snapshot_download(m.id, **kwargs)
        except Exception as e:  # noqa: BLE001
            if os.environ.get("HF_HUB_DISABLE_XET") == "1":
                raise
            print(f"    retrying without Xet after {type(e).__name__}: {e}", flush=True)
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            try:
                snapshot_download(m.id, **kwargs)
            finally:
                os.environ.pop("HF_HUB_DISABLE_XET", None)
    for m in models:
        info = api.model_info(m.id, revision=m.revision, files_metadata=True)
        allow, ignore = patterns_for([s.rfilename for s in info.siblings])
        files = [s for s in info.siblings
                 if any(fnmatch(s.rfilename, pat) for pat in allow)
                 and not any(fnmatch(s.rfilename, pat) for pat in ignore)]
        size = sum(s.size or 0 for s in files) / 1e9
        missing = [s for s in files if not isinstance(
            try_to_load_from_cache(m.id, s.rfilename, revision=m.revision), str)]
        gb = sum(s.size or 0 for s in missing) / 1e9
        total += size
        missing_total += gb
        line = f"{m.name:26} {m.id:38} {m.revision[:10]} {size:7.1f} GB"
        line += "   cached" if not missing else f"   missing {len(missing)}/{len(files)} files, {gb:.1f} GB"
        if args.dry_run or args.check or not missing:
            print(line, flush=True)
            continue
        if args.max_gb is not None and downloaded + gb > args.max_gb:
            print(line + f"   SKIPPED (over --max-gb {args.max_gb:g})", flush=True)
            continue
        print(line + "   downloading ...", flush=True)
        try:
            download(m, allow, ignore)
        except Exception as e:  # noqa: BLE001 - report and continue with the next model
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            failures.append(m.name)
            continue
        downloaded += gb
        r = verify(m.id, m.revision)
        if r["problems"]:
            failures.append(m.name)
        print("    " + format_verify(m.name, r), flush=True)
    print(f"total {total:.1f} GB; missing before this run {missing_total:.1f} GB; "
          f"downloaded {downloaded:.1f} GB" + (" (dry run)" if args.dry_run else ""))
    if failures:
        print(f"FAILED: {', '.join(failures)} (re-run to resume)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
