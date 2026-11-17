"""Download the prominence corpus of Talman et al. (2019) from the upstream repository at a pinned commit.

The files are verified against SHA-256 checksums so that every environment trains and
evaluates on exactly the same data.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

UPSTREAM_REPO = "Helsinki-NLP/prosody"
UPSTREAM_COMMIT = "75a56c5790ca487e2bb52380bd98722ff0c147b7"
BASE_URL = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/data"

# name -> (sha256, size in bytes)
FILES: dict[str, tuple[str, int]] = {
    "train_100.txt": ("43cbf73852b55183dd12b14d50b7956104e6963db81d8fcd944bc44f95351104", 14462197),
    "train_360.txt": ("5b3aff9d465c65b6f8dfcdd28c8054993b403eb20d9c9663de37906d40b4c548", 52504349),
    "dev.txt": ("ce5c73f9bccd59bed0c2a1054854357f157c74cc95723472bb09f7f0a6f15567", 2513731),
    "test.txt": ("70f5f21e79a40f2d7163089f2682e0d3c464b50514467a9e389c3f6d139d1f93", 2272072),
}


def default_data_dir() -> Path:
    return Path(os.environ.get("PROSODY_DATA", "data"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(data_dir: Path, names: list[str] | None = None) -> dict[str, bool]:
    """Return {file name: checksum ok} for the requested files (missing files are False)."""
    out = {}
    for name in names or list(FILES):
        path = data_dir / name
        out[name] = path.is_file() and sha256(path) == FILES[name][0]
    return out


def download(data_dir: Path, names: list[str] | None = None, force: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in names or list(FILES):
        expected, _ = FILES[name]
        path = data_dir / name
        if path.is_file() and not force and sha256(path) == expected:
            print(f"ok       {name}")
            continue
        tmp = path.with_suffix(path.suffix + ".part")
        print(f"download {name}")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", tmp)
        got = sha256(tmp)
        if got != expected:
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch for {name}: expected {expected}, got {got}")
        tmp.replace(path)
        print(f"ok       {name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--files", nargs="*", choices=list(FILES), default=None)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.verify_only:
        res = verify(args.data_dir, args.files)
        for name, ok in res.items():
            print(f"{'ok' if ok else 'FAILED':8} {name}")
        return 0 if all(res.values()) else 1
    download(args.data_dir, args.files, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
