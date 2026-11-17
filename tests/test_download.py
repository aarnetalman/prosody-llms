import hashlib

import pytest

from prosody_llms.data import download


@pytest.fixture
def fake_files(monkeypatch):
    content = {"dev.txt": b"dev data\n", "test.txt": b"test data\n"}
    files = {k: (hashlib.sha256(v).hexdigest(), len(v)) for k, v in content.items()}
    monkeypatch.setattr(download, "FILES", files)
    return content


def test_download_and_verify(tmp_path, monkeypatch, fake_files):
    calls = []

    def fake_urlretrieve(url, dest):
        calls.append(url)
        dest.write_bytes(fake_files[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(download.urllib.request, "urlretrieve", fake_urlretrieve)
    download.download(tmp_path)
    assert all(download.UPSTREAM_COMMIT in u for u in calls)
    assert download.verify(tmp_path) == {"dev.txt": True, "test.txt": True}
    download.download(tmp_path)  # second call is a no-op
    assert len(calls) == 2
    assert download.main(["--data-dir", str(tmp_path), "--verify-only"]) == 0


def test_checksum_mismatch_is_rejected(tmp_path, monkeypatch, fake_files):
    monkeypatch.setattr(download.urllib.request, "urlretrieve",
                        lambda url, dest: dest.write_bytes(b"tampered"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download.download(tmp_path, ["dev.txt"])
    assert not any(tmp_path.iterdir())
    assert download.main(["--data-dir", str(tmp_path), "--verify-only"]) == 1


def test_pinned_checksums_cover_all_splits():
    assert set(download.FILES) == {"train_100.txt", "train_360.txt", "dev.txt", "test.txt"}
    assert all(len(h) == 64 for h, _ in download.FILES.values())
