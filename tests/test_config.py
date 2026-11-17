import pytest

from prosody_llms.config import ModelSpec, get_model, load_registry


def test_registry_loads_and_is_pinned():
    reg = load_registry()
    assert "bert-base-uncased" in reg and "qwen3.8-27b" in reg
    assert reg["bert-base-uncased"].revision == "86b5e0934494bd15c9632b12f734a8a67f723594"
    ids = [m.id for m in reg.values()]
    assert len(ids) == len(set(ids)), "each model listed once"
    for m in reg.values():
        assert len(m.revision) == 40
    # Dotted names must be single keys, not nested tables.
    assert reg["qwen3.5-0.8b-base"].id == "Qwen/Qwen3.5-0.8B-Base"


def test_unknown_model():
    with pytest.raises(KeyError, match="unknown model"):
        get_model("no-such-model")


def test_validation(tmp_path):
    with pytest.raises(ValueError, match="revision"):
        ModelSpec("x", "a/b", "main", "encoder", "base", "planned")
    with pytest.raises(ValueError, match="kind"):
        ModelSpec("x", "a/b", "0" * 40, "vision", "base", "planned")
    p = tmp_path / "m.toml"
    p.write_text('[models.x]\nid = "a/b"\nkind = "encoder"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_registry(p)
