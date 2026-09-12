"""Memory arithmetic for choosing an export's context window.

Units are bytes throughout. Every estimate here is for a full-attention decoder with an
fp32 KV cache, which is what the XNNPACK recipe exports (docs/PLAN.md, "Context auto-fit").
"""

from __future__ import annotations

from dataclasses import dataclass

FP32_BYTES = 4
# 8da4w with group size 32: a 4-bit weight is half a byte, plus one fp32 scale per group
# of 32 weights (4 / 32 = 0.125 bytes per weight).
LINEAR_BYTES_PER_PARAM_8DA4W_G32 = 0.5 + FP32_BYTES / 32
# int8 per-channel embedding table: one byte per weight (one scale per row is negligible).
EMBEDDING_BYTES_PER_PARAM_INT8 = 1.0
# Export holds fp32 weights plus transient copies while quantizing, on top of a fixed cost
# for torch.export and lowering. To be calibrated by the probe workflow. Measured so far:
# LFM2.5-1.2B 6.7 GB and LFM2.5-2.6B 12.2 GB peak (openweights export notes); SmolLM2-135M
# 2,748,440,576 B peak for 538 MB of fp32 weights (local Docker run, 2026-09-12), so the
# fixed part is at least ~2.2 GB.
EXPORT_PEAK_WEIGHT_MULTIPLE = 1.5
EXPORT_FIXED_OVERHEAD_BYTES = 2_500_000_000


@dataclass(frozen=True)
class Architecture:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    dim: int
    total_params: int
    tied_embeddings: bool


def kv_cache_bytes(arch: Architecture, context: int) -> int:
    """K and V for every layer, every KV head, every position, fp32."""
    return arch.n_layers * 2 * arch.n_kv_heads * arch.head_dim * context * FP32_BYTES


def pte_bytes_estimate(arch: Architecture) -> int:
    embedding = arch.vocab_size * arch.dim
    linear = arch.total_params - embedding
    if arch.tied_embeddings:
        # The export gives the output projection its own quantized copy of the table.
        linear += embedding
    return int(embedding * EMBEDDING_BYTES_PER_PARAM_INT8 + linear * LINEAR_BYTES_PER_PARAM_8DA4W_G32)


def device_resident_bytes(arch: Architecture, context: int, overhead: int) -> int:
    return pte_bytes_estimate(arch) + kv_cache_bytes(arch, context) + overhead


def export_peak_bytes(arch: Architecture, context: int) -> int:
    weights = arch.total_params * FP32_BYTES
    return int(weights * EXPORT_PEAK_WEIGHT_MULTIPLE + kv_cache_bytes(arch, context) + EXPORT_FIXED_OVERHEAD_BYTES)


@dataclass(frozen=True)
class WindowChoice:
    context: int | None
    reason: str
    table: tuple[dict, ...]


def choose_context(
    arch: Architecture,
    tiers: tuple[int, ...],
    device_budget: int,
    runtime_overhead: int,
    host_budget: int | None,
) -> WindowChoice:
    """Largest tier whose phone residency and host export peak both fit."""
    rows = []
    chosen = None
    for context in sorted(tiers, reverse=True):
        resident = device_resident_bytes(arch, context, runtime_overhead)
        peak = export_peak_bytes(arch, context)
        fits_device = resident <= device_budget
        fits_host = host_budget is None or peak <= host_budget
        rows.append(
            {
                "context": context,
                "kv_cache_bytes": kv_cache_bytes(arch, context),
                "device_resident_bytes": resident,
                "export_peak_bytes": peak,
                "fits_device": fits_device,
                "fits_host": fits_host,
            }
        )
        if chosen is None and fits_device and fits_host:
            chosen = context
    if chosen is None:
        smallest = rows[-1]
        reason = (
            f"no window fits: at {smallest['context']} tokens the phone would hold "
            f"{smallest['device_resident_bytes']:,} B (budget {device_budget:,} B) and the "
            f"export would peak near {smallest['export_peak_bytes']:,} B"
        )
        return WindowChoice(None, reason, tuple(rows))
    return WindowChoice(chosen, f"largest window within budgets: {chosen}", tuple(rows))
