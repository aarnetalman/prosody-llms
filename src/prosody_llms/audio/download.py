"""Download the LibriTTS audio and the public word alignments.

    python -m prosody_llms.audio.download                 # dev, test and train_100 (about 10 GB)
    python -m prosody_llms.audio.download --check         # what is there; downloads nothing
    python -m prosody_llms.audio.download --splits dev test

Needs internet. Layout under ``$PROSODY_AUDIO`` (default
``$PROSODY_DATA/../audio``):

    LibriTTS/<subset>/<speaker>/<chapter>/<utt_id>.wav       (from OpenSLR resource 60)
    alignments/textgrid/<subset>/<speaker>/<chapter>/<utt_id>.TextGrid
    MANIFEST.json                                            (checksums, counts)

* **Audio.** The ``*.tar.gz`` of each LibriTTS subset is fetched from OpenSLR (trying
  its mirrors in turn, resuming partial downloads) and only the ``.wav`` files of the
  corpus's utterances are kept. OpenSLR publishes no checksums we could pin in advance,
  so the archive's SHA-256 is recorded in MANIFEST.json on the first download and
  checked against it afterwards; every utterance of the corpus split must be present.
* **Alignments.** ``kan-bayashi/LibriTTSLabel`` (Montreal Forced Aligner output) at a
  pinned commit, each archive part verified by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from prosody_llms.audio.base import LIBRITTS_SUBSETS, libritts_path
from prosody_llms.data.corpus import read_split
from prosody_llms.data.download import default_data_dir, sha256

MIRRORS = ("https://www.openslr.org/resources/60", "https://us.openslr.org/resources/60",
           "https://openslr.elda.org/resources/60", "https://openslr.magicdatatech.com/resources/60")

LABEL_REPO = "kan-bayashi/LibriTTSLabel"
LABEL_COMMIT = "19e3e7ca86ed1dc156ac2ab2337d51b967c1e735"  # 2020-03-16
LABEL_URL = f"https://raw.githubusercontent.com/{LABEL_REPO}/{LABEL_COMMIT}"
TEXTGRID_PARTS = {  # part -> sha256
    "textgrid.tar.gz-0": "6ae4b3794637f6aa941a7ee5232ea2d22a11b889262c62f6ee24c89f44bd36d3",
    "textgrid.tar.gz-1": "240e6b46d7bf424db76ff58541f01ed5bbc8623801cfcd8fb173c5b59f02c622",
    "textgrid.tar.gz-2": "896b314d31deb76c3d6fb564b53de8d7a34b2a1e43ccf62a3444b00ebd7184fe",
    "textgrid.tar.gz-3": "58b7af1d8c7e48a9c92381328a283c57f8b1d37a2ee5e24c05f05fa9c1ee705a",
    "textgrid.tar.gz-4": "05911157f88bc9b26edc4b337631037721aba90dd0a401e23e1332931deee529",
    "textgrid.tar.gz-5": "fb681ad40fcb9bc57aef0949155f48d97b50f043b909889d66648d581d821770",
    "textgrid.tar.gz-6": "2014bb85525409cea1864bb18da63472e6eb2a81585f5b216b4b59b01ab0d0cb",
    "textgrid.tar.gz-7": "d13be45452f492139b9e633b4c9a6bdef87ea8726a15b0a0cdd817643d79c4f1",
}
TEXTGRID_SHA256 = "1cd6a48882b662fbdf674d53b57da0d3f946d460d052f3b6ae3ce6a70791db31"  # parts joined

DEFAULT_SPLITS = ("dev", "test", "train_100")


def default_audio_dir() -> Path:
    if os.environ.get("PROSODY_AUDIO"):
        return Path(os.environ["PROSODY_AUDIO"])
    return default_data_dir().resolve().parent / "audio"


def textgrid_root(audio_dir: Path) -> Path:
    return audio_dir / "alignments" / "textgrid"


def fetch(urls: list[str], dest: Path, retries: int = 3) -> str:
    """Download to ``dest`` (resuming a ``.part`` file) from the first URL that works."""
    part = dest.with_name(dest.name + ".part")
    last = None
    for url in urls:
        for attempt in range(retries):
            try:
                have = part.stat().st_size if part.exists() else 0
                req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
                with urllib.request.urlopen(req, timeout=60) as r:
                    mode = "ab" if have and r.status == 206 else "wb"
                    total = r.headers.get("Content-Length")
                    print(f"  {url} ({'resuming' if mode == 'ab' else 'starting'}"
                          f"{', ' + str(round(int(total) / 1e9, 2)) + ' GB to go' if total else ''})",
                          flush=True)
                    with open(part, mode) as f:
                        while chunk := r.read(1 << 22):
                            f.write(chunk)
                part.replace(dest)
                return url
            except Exception as e:  # noqa: BLE001 - network errors of all kinds: try again
                last = e
                print(f"  failed ({e}); retrying", flush=True)
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not download {dest.name}: {last}")


def load_manifest(audio_dir: Path) -> dict:
    path = audio_dir / "MANIFEST.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_manifest(audio_dir: Path, manifest: dict) -> None:
    (audio_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")


def missing_wavs(audio_dir: Path, data_dir: Path, split: str) -> list[str]:
    return [u.utt_id for u in read_split(data_dir / f"{split}.txt")
            if not libritts_path(audio_dir, split, u.utt_id).exists()]


def get_audio(audio_dir: Path, data_dir: Path, split: str, keep_archive: bool = False) -> dict:
    subset = LIBRITTS_SUBSETS[split]
    if not missing_wavs(audio_dir, data_dir, split):
        print(f"ok       audio {split} ({subset})")
        return load_manifest(audio_dir).get(subset, {})
    archive = audio_dir / "archives" / f"{subset}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(audio_dir)
    if not archive.exists():
        print(f"download audio {split} ({subset})")
        fetch([f"{m}/{subset}.tar.gz" for m in MIRRORS], archive)
    digest = sha256(archive)
    known = manifest.get(subset, {}).get("archive_sha256")
    if known and known != digest:
        raise RuntimeError(f"{archive} has SHA-256 {digest}, the manifest has {known}; "
                           "delete the archive and download again")
    wanted = {u.utt_id for u in read_split(data_dir / f"{split}.txt")}
    n = 0
    print(f"extract  {subset} ({len(wanted)} utterances)", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            name = Path(m.name)
            if m.isfile() and name.suffix == ".wav" and name.stem in wanted:
                dest = libritts_path(audio_dir, split, name.stem)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, open(dest, "wb") as out:
                    out.write(src.read())
                n += 1
    missing = missing_wavs(audio_dir, data_dir, split)
    if missing:
        raise RuntimeError(f"{len(missing)} utterances of {split} not in {subset}, e.g. {missing[:3]}")
    recorded = manifest[subset].get("recorded") if known else time.strftime("%Y-%m-%d")
    entry = {"archive_sha256": digest, "archive_bytes": archive.stat().st_size, "wav_files": n,
             "recorded": recorded}
    manifest[subset] = entry
    save_manifest(audio_dir, manifest)
    if not keep_archive:
        archive.unlink()
    return entry


def get_alignments(audio_dir: Path, subsets: list[str]) -> None:
    root = textgrid_root(audio_dir)
    todo = [s for s in subsets if not (root / s).is_dir()]
    if not todo:
        print("ok       alignments")
        return
    parts_dir = audio_dir / "archives" / "textgrid"
    parts_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    for part, digest in TEXTGRID_PARTS.items():
        dest = parts_dir / part
        if not (dest.exists() and sha256(dest) == digest):
            print(f"download alignments {part}")
            fetch([f"{LABEL_URL}/{part}"], dest)
            if sha256(dest) != digest:
                raise RuntimeError(f"{part}: checksum mismatch")
        with open(dest, "rb") as f:
            h.update(f.read())
    if h.hexdigest() != TEXTGRID_SHA256:
        raise RuntimeError("joined TextGrid archive: checksum mismatch")

    class Joined:  # the parts read as one stream
        def __init__(self, paths):
            self.files = [open(p, "rb") for p in paths]

        def read(self, n=-1):
            out = b""
            while self.files and (n < 0 or len(out) < n):
                chunk = self.files[0].read(-1 if n < 0 else n - len(out))
                if not chunk:
                    self.files.pop(0).close()
                    continue
                out += chunk
            return out

    print(f"extract  alignments for {', '.join(todo)}", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=Joined([parts_dir / p for p in TEXTGRID_PARTS]), mode="r|gz") as tar:
        for m in tar:
            parts = Path(m.name).parts  # textgrid/<subset>/<speaker>/<chapter>/<file>
            if m.isfile() and len(parts) == 5 and parts[1] in todo and m.name.endswith(".TextGrid"):
                dest = root.joinpath(*parts[1:])
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, open(dest, "wb") as out:
                    out.write(src.read())
    manifest = load_manifest(audio_dir)
    manifest["alignments"] = {"repo": LABEL_REPO, "commit": LABEL_COMMIT, "sha256": TEXTGRID_SHA256}
    save_manifest(audio_dir, manifest)


def check(audio_dir: Path, data_dir: Path, splits: list[str]) -> bool:
    ok = True
    for split in splits:
        miss = missing_wavs(audio_dir, data_dir, split)
        tg = textgrid_root(audio_dir) / LIBRITTS_SUBSETS[split]
        print(f"{split:10} audio: {'ok' if not miss else f'{len(miss)} missing'}; "
              f"alignments: {'ok' if tg.is_dir() else 'missing'}")
        ok = ok and not miss and tg.is_dir()
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", choices=list(LIBRITTS_SUBSETS), default=list(DEFAULT_SPLITS))
    p.add_argument("--audio-dir", type=Path, default=default_audio_dir())
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--check", action="store_true", help="report what is there; download nothing")
    p.add_argument("--keep-archive", action="store_true")
    args = p.parse_args(argv)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    if args.check:
        return 0 if check(args.audio_dir, args.data_dir, args.splits) else 1
    get_alignments(args.audio_dir, [LIBRITTS_SUBSETS[s] for s in args.splits])
    for split in args.splits:
        get_audio(args.audio_dir, args.data_dir, split, args.keep_archive)
    return 0 if check(args.audio_dir, args.data_dir, args.splits) else 1


if __name__ == "__main__":
    sys.exit(main())
