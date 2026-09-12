"""HF safetensors → the checkpoint layout ExecuTorch's llama_transformer loads.

Key names match ExecuTorch 1.4.0's own converters (examples/models/{qwen3,qwen2_5,
smollm2}/convert_weights.py). Those load everything through torchtune or into one dict
and only handle single-file or specific checkpoints; this one streams shards, handles
tied embeddings from the config, and fails on any tensor it does not know rather than
silently dropping it.

The checkpoint's file name must not contain "int8" or "8da4w": examples/models/llama/
model.py switches to pre-quantized loading when the path contains either string.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

# Per-layer HF suffix → ExecuTorch suffix, shared by every supported family.
_LAYER_KEYS = {
    "self_attn.q_proj.weight": "attention.wq.weight",
    "self_attn.k_proj.weight": "attention.wk.weight",
    "self_attn.v_proj.weight": "attention.wv.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    # ExecuTorch applies the activation to w1, as HF does to gate_proj.
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
}
_QKV_BIAS_KEYS = {
    "self_attn.q_proj.bias": "attention.wq.bias",
    "self_attn.k_proj.bias": "attention.wk.bias",
    "self_attn.v_proj.bias": "attention.wv.bias",
}
_QK_NORM_KEYS = {
    "self_attn.q_norm.weight": "attention.q_norm_fn.weight",
    "self_attn.k_norm.weight": "attention.k_norm_fn.weight",
}
_TOP_KEYS = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}
# Buffers some checkpoints carry that the model recomputes.
_IGNORED = re.compile(r"(rotary_emb\.inv_freq|\.masked_bias|\.attn\.bias)$")

CONVERTERS = {
    # family converter key → (extra per-layer keys, un-permute q/k for Meta RoPE)
    "qwen3": ({**_QK_NORM_KEYS}, False),
    "qwen2": ({**_QKV_BIAS_KEYS}, False),
    "llama": ({}, True),
}


def unpermute(weight, n_heads: int):
    """Inverse of the q/k permutation HF's Llama conversion applies for rotate_half RoPE.

    HF: w.view(n_heads, head_dim // 2, 2, dim).transpose(1, 2).reshape(out, dim)
    """
    out_features, in_features = weight.shape
    return (
        weight.view(n_heads, 2, out_features // n_heads // 2, in_features)
        .transpose(1, 2)
        .reshape(out_features, in_features)
    )


def _shards(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [model_dir / name for name in sorted(set(weight_map.values()))]
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no safetensors checkpoint in {model_dir}")


def map_key(hf_key: str, converter: str) -> str | None:
    """ExecuTorch name for an HF tensor, None for ignorable buffers, KeyError if unknown."""
    extra, _ = CONVERTERS[converter]
    if hf_key in _TOP_KEYS:
        return _TOP_KEYS[hf_key]
    if _IGNORED.search(hf_key):
        return None
    match = _LAYER.match(hf_key)
    if match:
        index, suffix = match.groups()
        target = _LAYER_KEYS.get(suffix) or extra.get(suffix)
        if target:
            return f"layers.{index}.{target}"
    raise KeyError(f"tensor {hf_key!r} has no mapping for converter {converter!r}")


def convert(model_dir: Path, output: Path, converter: str, config: dict) -> dict:
    """Write the converted checkpoint to ``output``; returns a summary for the report."""
    import torch
    from safetensors import safe_open

    if "int8" in output.name or "8da4w" in output.name:
        raise ValueError(f"checkpoint name {output.name!r} would trigger pre-quantized loading")
    _, permute = CONVERTERS[converter]
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config.get("num_key_value_heads") or n_heads)

    state: dict[str, torch.Tensor] = {}
    for shard in _shards(model_dir):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                target = map_key(key, converter)
                if target is None:
                    continue
                tensor = f.get_tensor(key)
                if permute and target.endswith("attention.wq.weight"):
                    tensor = unpermute(tensor, n_heads)
                elif permute and target.endswith("attention.wk.weight"):
                    tensor = unpermute(tensor, n_kv_heads)
                state[target] = tensor

    tied = "output.weight" not in state
    if tied:
        if not config.get("tie_word_embeddings", True):
            raise KeyError("checkpoint has no lm_head.weight but config says embeddings are untied")
        state["output.weight"] = state["tok_embeddings.weight"]

    n_layers = int(config["num_hidden_layers"])
    expected_per_layer = len(_LAYER_KEYS) + len(CONVERTERS[converter][0])
    per_layer = sum(1 for k in state if k.startswith("layers."))
    if per_layer != n_layers * expected_per_layer:
        raise ValueError(f"expected {n_layers} layers x {expected_per_layer} tensors, found {per_layer}")

    dtypes = sorted({str(t.dtype) for t in state.values()})
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    return {"tensors": len(state), "tied_embeddings": tied, "dtypes": dtypes}
