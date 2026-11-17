import json
import os

import pytest

from prosody_llms import prefetch

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

REV = "a" * 40


def make_model(cache, model_id="org/model", shards=2, rev=REV):
    """A cached model in the Hugging Face layout: blobs + snapshot symlinks."""
    rdir = prefetch.repo_dir(model_id, cache)
    blobs, snap = rdir / "blobs", rdir / "snapshots" / rev
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)

    def add(name, data: bytes):
        blob = blobs / f"blob-{name}"
        blob.write_bytes(data)
        os.symlink(os.path.relpath(blob, snap), snap / name)

    add("config.json", b"{}")
    weight_map = {}
    for i in range(shards):
        name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors" if shards > 1 else "model.safetensors"
        tmp = cache / f"tmp{i}.safetensors"
        safetensors_torch.save_file({f"w{i}": torch.randn(4, 3)}, str(tmp))
        add(name, tmp.read_bytes())
        weight_map[f"w{i}"] = name
    if shards > 1:
        add("model.safetensors.index.json", json.dumps({"weight_map": weight_map}).encode())
    return rdir, snap


def test_complete_sharded_and_single_file(tmp_path):
    make_model(tmp_path, "org/a", shards=3)
    make_model(tmp_path, "org/b", shards=1)
    for mid, n in (("org/a", 3), ("org/b", 1)):
        r = prefetch.verify(mid, REV, tmp_path)
        assert r["problems"] == [] and r["weight_files"] == n and r["bytes"] > 0
        assert "complete" in prefetch.format_verify(mid, r)


def test_detects_truncated_missing_broken_and_incomplete(tmp_path):
    rdir, snap = make_model(tmp_path, shards=2)
    target = (snap / "model-00001-of-00002.safetensors").resolve()
    target.write_bytes(target.read_bytes()[:-5])  # truncated download
    os.remove((snap / "model-00002-of-00002.safetensors").resolve())  # broken link
    (rdir / "blobs" / "partial.incomplete").write_bytes(b"x")
    problems = prefetch.verify("org/model", REV, tmp_path)["problems"]
    assert any("truncated" in p for p in problems)
    assert any("incomplete" in p for p in problems)  # reported because weights are bad too
    assert any("broken link" in p for p in problems)
    assert "INCOMPLETE" in prefetch.format_verify("m", prefetch.verify("org/model", REV, tmp_path))


def test_missing_snapshot_and_missing_weights(tmp_path):
    assert prefetch.verify("org/none", REV, tmp_path)["problems"] == [
        "no snapshot for the pinned revision"]
    rdir, snap = make_model(tmp_path, "org/c", shards=1)
    (snap / "model.safetensors").unlink()
    assert "no weights" in prefetch.verify("org/c", REV, tmp_path)["problems"][0]
    make_model(tmp_path, "org/d", shards=1, rev="b" * 40)
    assert prefetch.verify("org/d", REV, tmp_path)["problems"] == [
        "no snapshot for the pinned revision"]  # a different revision does not count


def test_verify_cli_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    assert prefetch.hub_cache() == tmp_path / "hub"
    # No registry model is cached here, so every model is reported incomplete.
    assert prefetch.main(["--verify", "bert-base-uncased"]) == 1
    assert "INCOMPLETE" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        prefetch.main(["--verify", "no-such-model"])


def test_stale_incomplete_blobs_are_a_warning_when_weights_are_intact(tmp_path):
    rdir, _ = make_model(tmp_path, shards=2)
    (rdir / "blobs" / "leftover.incomplete").write_bytes(b"x")
    r = prefetch.verify("org/model", REV, tmp_path)
    assert r["problems"] == [] and r["stale_incomplete"] == 1
    line = prefetch.format_verify("m", r)
    assert "complete" in line and "warning: 1 stale" in line


def test_bin_only_checkpoints(tmp_path):
    allow, ignore = prefetch.patterns_for(["config.json", "pytorch_model.bin", "tf_model.h5"])
    assert "pytorch_model.bin" in allow and "*.bin" not in ignore and "*.h5" in ignore
    allow, ignore = prefetch.patterns_for(["model.safetensors", "pytorch_model.bin", "model.fp32-00001-of-00002.safetensors"])
    assert "*.bin" in ignore and "*.fp32*" in ignore
    rdir = prefetch.repo_dir("org/bin", tmp_path)
    snap = rdir / "snapshots" / REV
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    torch.save({"w": torch.zeros(3)}, snap / "pytorch_model.bin")
    r = prefetch.verify("org/bin", REV, tmp_path)
    assert not r["problems"] and r["weight_files"] == 1
    data = (snap / "pytorch_model.bin").read_bytes()
    (snap / "pytorch_model.bin").write_bytes(data[: len(data) // 2])
    assert "truncated" in prefetch.verify("org/bin", REV, tmp_path)["problems"][0]
