"""Load the text part of any registry model.

Several LLMs ship as multimodal "conditional generation" models (Qwen3.5/3.8, Muse
Glimmer). For probing only the language model is needed, so its text-only class is
loaded from the multimodal checkpoint and the vision tower is skipped.

Weight names differ between checkpoints and text-only classes. transformers translates
them for some models (Qwen3.5) but not for others (Muse Glimmer), and when it does not,
it only *warns* and leaves the weights randomly initialised. This loader therefore
checks for missing weights: if any are missing it retries with an explicit name mapping,
and if weights are still missing it raises an error instead of returning a random model.
"""

from __future__ import annotations

from dataclasses import dataclass


# Missing weights that are acceptable: encoders' pooler layer is never used for probing,
# and some checkpoints do not include it.
IGNORABLE_MISSING = ("pooler",)
# Name mappings tried when the plain load leaves weights missing.
KEY_MAPPINGS = ({r"^model\.language_model\.": ""}, {r"^language_model\.model\.": ""})


class MissingWeightsError(RuntimeError):
    pass


def _missing(info: dict) -> list[str]:
    return [k for k in info.get("missing_keys", []) if not any(s in k for s in IGNORABLE_MISSING)]


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    n_layers: int  # hidden states returned, including the embedding layer
    dim: int
    model_class: str
    key_mapping: dict | None = None


# Name mappings for the causal-LM class (it keeps the "model." prefix and the LM head).
CAUSAL_KEY_MAPPINGS = ({r"^model\.language_model\.": "model."},
                       {r"^language_model\.model\.": "model.", r"^language_model\.lm_head\.": "lm_head."})


def load_text_model(model_id: str, revision: str | None = None, *, dtype: str = "bfloat16",
                    device: str | None = None, head: str = "none") -> LoadedModel:
    """``head="none"``: the base text model (hidden states, for probing).
    ``head="causal"``: the text model with its language-modelling head (for scoring)."""
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING, MODEL_MAPPING

    if head not in ("none", "causal"):
        raise ValueError("head must be 'none' or 'causal'")
    mapping_table = MODEL_MAPPING if head == "none" else MODEL_FOR_CAUSAL_LM_MAPPING
    mappings = KEY_MAPPINGS if head == "none" else CAUSAL_KEY_MAPPINGS
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and type(text_config) in mapping_table:
        cls = mapping_table[type(text_config)]  # e.g. Qwen3_5TextModel / Qwen3_5ForCausalLM
    elif type(config) in mapping_table:
        text_config = config
        cls = mapping_table[type(config)]
    else:
        raise ValueError(f"{model_id}: transformers has no {head} text class for "
                         f"{type(config).__name__}")
    kwargs = dict(revision=revision, dtype=getattr(torch, dtype), output_loading_info=True)
    model, info = cls.from_pretrained(model_id, **kwargs)
    used_mapping = None
    for mapping in mappings:
        if not _missing(info):
            break
        model, info = cls.from_pretrained(model_id, key_mapping=mapping, **kwargs)
        used_mapping = mapping
    missing = _missing(info)
    if missing:
        raise MissingWeightsError(
            f"{model_id}: {len(missing)} weights of {cls.__name__} are missing from the "
            f"checkpoint (e.g. {missing[:3]}); refusing to continue with random weights")
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(f"{model_id}: a fast tokenizer is needed for word alignment")
    return LoadedModel(model, tokenizer, text_config.num_hidden_layers + 1,
                       text_config.hidden_size, cls.__name__, used_mapping)
