import io
import tarfile

import pytest

from prosody_llms.audio import download as dl
from prosody_llms.audio.base import libritts_path
from prosody_llms.data.corpus import read_split


def _tar(path, utt_ids, subset):
    with tarfile.open(path, "w:gz") as tar:
        for uid in utt_ids + ["9_9_000000_000000"]:  # plus one not in the corpus
            sp, ch = uid.split("_")[:2]
            for suffix, data in ((".wav", b"RIFF" + uid.encode()), (".normalized.txt", b"x")):
                info = tarfile.TarInfo(f"LibriTTS/{subset}/{sp}/{ch}/{uid}{suffix}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


def test_audio_download_keeps_only_corpus_wavs(tmp_path, corpus_dir, monkeypatch):
    ids = [u.utt_id for u in read_split(corpus_dir / "dev.txt")]
    src = tmp_path / "src.tar.gz"
    _tar(src, ids, "dev-clean")
    calls = []

    def fake_fetch(urls, dest, retries=3):
        calls.append(urls[0])
        dest.write_bytes(src.read_bytes())
        return urls[0]

    monkeypatch.setattr(dl, "fetch", fake_fetch)
    audio = tmp_path / "audio"
    entry = dl.get_audio(audio, corpus_dir, "dev")
    assert entry["wav_files"] == len(ids) and calls[0].endswith("/dev-clean.tar.gz")
    assert libritts_path(audio, "dev", ids[0]).read_bytes() == b"RIFF" + ids[0].encode()
    assert not list(audio.rglob("*.txt")) and not (audio / "archives" / "dev-clean.tar.gz").exists()
    assert dl.load_manifest(audio)["dev-clean"]["archive_sha256"] == dl.sha256(src)
    dl.get_audio(audio, corpus_dir, "dev")  # complete: nothing fetched again
    assert len(calls) == 1
    # A different archive later is refused.
    libritts_path(audio, "dev", ids[0]).unlink()
    other = tmp_path / "other.tar.gz"
    _tar(other, ids[:1], "dev-clean")
    monkeypatch.setattr(dl, "fetch", lambda urls, dest, retries=3: dest.write_bytes(other.read_bytes()))
    with pytest.raises(RuntimeError, match="manifest"):
        dl.get_audio(audio, corpus_dir, "dev")


def test_missing_utterances_are_an_error(tmp_path, corpus_dir, monkeypatch):
    src = tmp_path / "src.tar.gz"
    _tar(src, [read_split(corpus_dir / "test.txt")[0].utt_id], "test-clean")
    monkeypatch.setattr(dl, "fetch", lambda urls, dest, retries=3: dest.write_bytes(src.read_bytes()))
    with pytest.raises(RuntimeError, match="not in test-clean"):
        dl.get_audio(tmp_path / "audio", corpus_dir, "test")
    assert not dl.check(tmp_path / "audio", corpus_dir, ["test"])
