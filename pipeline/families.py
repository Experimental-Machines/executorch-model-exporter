"""Model families: which HF architectures each backend can export, and how.

For XNNPACK, ExecuTorch 1.4.0's ``export_llm`` builds every model from one transformer
definition (examples/models/llama/llama_transformer.py). ``base.model_class`` only picks
the example directory and a few class-specific behaviours; the params file defines the
architecture. So a new size or finetune of a known architecture is exported by
generating the params from its HF config.json, not by waiting for ExecuTorch to list it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.sizing import Architecture

# HF config keys that mean the checkpoint is a mixture of experts.
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_intermediate_size")

# Llama 3.1+ RoPE scaling that ExecuTorch's non-HF RoPE implements: rope.apply_scaling
# hard-codes low_freq_factor 1 and an 8192-token original context, and model.py forces
# rope_scale_factor 32 for every scaled-RoPE model class except llama3/llama3_1.
_LLAMA32_ROPE = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


class UnsupportedModel(Exception):
    """The checkpoint's architecture is outside what the recipe can export correctly."""


@dataclass(frozen=True)
class Family:
    key: str
    architectures: tuple[str, ...]
    # Backend → None when supported, or the reason it is not (yet).
    unsupported: dict[str, str] = field(default_factory=dict)

    def supports(self, backend: str) -> bool:
        return backend not in self.unsupported


_NOT_IN_EXPORT_LLM = (
    "not in ExecuTorch 1.4.0 export_llm's model list; needs a validated export path "
    "(optimum-executorch or params support) before it is published"
)
_NPU_PENDING = "backend pipeline not built yet (docs/PLAN.md phases 3-4)"

FAMILIES: tuple[Family, ...] = (
    Family("qwen3", ("Qwen3ForCausalLM",), {"qnn": _NPU_PENDING, "mtk": _NPU_PENDING}),
    Family("qwen2_5", ("Qwen2ForCausalLM",), {"qnn": _NPU_PENDING, "mtk": _NPU_PENDING}),
    Family("llama", ("LlamaForCausalLM",), {"qnn": _NPU_PENDING, "mtk": _NPU_PENDING}),
    Family(
        "gemma3",
        ("Gemma3ForCausalLM",),
        {"xnnpack": _NOT_IN_EXPORT_LLM, "qnn": _NPU_PENDING, "mtk": _NPU_PENDING},
    ),
    Family(
        "smollm3",
        ("SmolLM3ForCausalLM",),
        {"xnnpack": _NOT_IN_EXPORT_LLM, "qnn": _NPU_PENDING, "mtk": _NPU_PENDING},
    ),
)
BACKENDS = ("xnnpack", "qnn", "mtk")


def family_for(config: dict) -> Family | None:
    architectures = config.get("architectures") or []
    for family in FAMILIES:
        if any(a in family.architectures for a in architectures):
            return family
    return None


def is_moe(config: dict) -> bool:
    if any("moe" in a.lower() for a in config.get("architectures") or []):
        return True
    for key in _MOE_KEYS:
        value = config.get(key)
        if value not in (None, 0, False):
            return True
    return False


def text_config(config: dict) -> dict:
    return config.get("text_config") or config


def rope_theta(config: dict) -> float:
    if "rope_theta" in config:
        return float(config["rope_theta"])
    params = config.get("rope_parameters") or {}
    if "rope_theta" in params:
        return float(params["rope_theta"])
    raise UnsupportedModel("config has no rope_theta")


def rope_scaling(config: dict) -> dict | None:
    """The rope scaling block, or None for plain RoPE (transformers v4 and v5 spellings)."""
    scaling = config.get("rope_scaling") or config.get("rope_parameters")
    if not scaling:
        return None
    kind = scaling.get("rope_type") or scaling.get("type")
    if kind in (None, "default"):
        return None
    return scaling


def architecture(config: dict, total_params: int) -> Architecture:
    c = text_config(config)
    n_heads = int(c["num_attention_heads"])
    dim = int(c["hidden_size"])
    return Architecture(
        n_layers=int(c["num_hidden_layers"]),
        n_heads=n_heads,
        n_kv_heads=int(c.get("num_key_value_heads") or n_heads),
        head_dim=int(c.get("head_dim") or dim // n_heads),
        vocab_size=int(c["vocab_size"]),
        dim=dim,
        intermediate=int(c["intermediate_size"]),
        total_params=int(total_params),
        tied_embeddings=bool(c.get("tie_word_embeddings", False)),
    )


def _common_params(c: dict) -> dict:
    n_heads = int(c["num_attention_heads"])
    dim = int(c["hidden_size"])
    return {
        "dim": dim,
        "ffn_dim_multiplier": 1,
        "hidden_dim": int(c["intermediate_size"]),
        "n_heads": n_heads,
        "head_dim": int(c.get("head_dim") or dim // n_heads),
        "n_kv_heads": int(c.get("num_key_value_heads") or n_heads),
        "n_layers": int(c["num_hidden_layers"]),
        "norm_eps": float(c["rms_norm_eps"]),
        "rope_theta": rope_theta(c),
        "vocab_size": int(c["vocab_size"]),
    }


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise UnsupportedModel(reason)


def _check_common(c: dict) -> None:
    _require(c.get("hidden_act", "silu") == "silu", f"activation {c.get('hidden_act')!r} is not silu")
    _require(not c.get("use_sliding_window"), "sliding-window attention is enabled")
    _require(not c.get("mlp_bias"), "MLP biases are not supported by the recipe")


@dataclass(frozen=True)
class XnnpackPlan:
    """What export_llm needs besides the shared recipe."""

    model_class: str
    params: dict
    converter: str  # key into pipeline.convert.CONVERTERS


def xnnpack_plan(family: Family, config: dict) -> XnnpackPlan:
    """ExecuTorch params and model class for this checkpoint, or UnsupportedModel."""
    if not family.supports("xnnpack"):
        raise UnsupportedModel(family.unsupported["xnnpack"])
    c = text_config(config)
    _check_common(c)
    params = _common_params(c)

    if family.key == "qwen3":
        _require(rope_scaling(c) is None, "RoPE scaling (e.g. YaRN) is not supported")
        _require(not c.get("attention_bias"), "Qwen3 with attention biases is not supported")
        params.update(
            use_scaled_rope=False,
            use_hf_rope=True,
            attention_qkv_bias=False,
            use_qk_norm=True,
            qk_norm_before_rope=True,
        )
        return XnnpackPlan("qwen3_1_7b", params, "qwen3")

    if family.key == "qwen2_5":
        _require(rope_scaling(c) is None, "RoPE scaling (e.g. YaRN) is not supported")
        params.update(use_scaled_rope=False, use_hf_rope=True, attention_qkv_bias=True)
        return XnnpackPlan("qwen2_5_1_5b", params, "qwen2")

    if family.key == "llama":
        _require(not c.get("attention_bias"), "Llama with attention biases is not supported")
        scaling = rope_scaling(c)
        if scaling is None:
            # Plain RoPE (SmolLM2). Weights are un-permuted to Meta's interleaved layout
            # and run through ExecuTorch's Meta RoPE, as torchtune's converter does.
            params.update(use_scaled_rope=False, use_hf_rope=False, attention_qkv_bias=False)
            return XnnpackPlan("smollm2", params, "llama")
        for key, expected in _LLAMA32_ROPE.items():
            actual = scaling.get(key, scaling.get("type") if key == "rope_type" else None)
            if isinstance(expected, float):
                ok = actual is not None and float(actual) == expected
            else:
                ok = actual == expected
            _require(ok, f"rope_scaling {key}={actual!r}, recipe only implements {expected!r}")
        params.update(use_scaled_rope=True, use_hf_rope=False, attention_qkv_bias=False)
        return XnnpackPlan("llama3_2", params, "llama")

    raise UnsupportedModel(f"no XNNPACK recipe for family {family.key}")
