from conftest import TOTAL_PARAMS, hf_config

from pipeline import families, sizing

TIERS = (32768, 16384, 8192, 4096, 2048)
BUDGET = 5_000_000_000
OVERHEAD = 500_000_000


def arch(model_id):
    return families.architecture(hf_config(model_id), TOTAL_PARAMS[model_id])


def test_kv_cache_matches_the_plan_literal():
    # docs/PLAN.md: 28 x 2 x 8 x 128 x 32,768 x 4 = 7,516,192,768 bytes.
    qwen = arch("Qwen/Qwen3-1.7B")
    assert sizing.kv_cache_bytes(qwen, 1) == 229_376
    assert sizing.kv_cache_bytes(qwen, 32768) == 7_516_192_768


def test_kv_cache_qwen3_4b():
    # 36 layers x 2 x 8 KV heads x 128 x 4 bytes = 294,912 per token.
    assert sizing.kv_cache_bytes(arch("Qwen/Qwen3-4B"), 1) == 294_912
    assert sizing.kv_cache_bytes(arch("Qwen/Qwen3-4B"), 32768) == 9_663_676_416


def test_pte_estimate_counts_tied_output_separately():
    # embeddings 151,936 x 2,048 = 311,164,928 at 1 B; the tied output projection adds a
    # 4-bit copy, so all 2,031,739,904 parameters count as linear at 0.625 B.
    assert sizing.pte_bytes_estimate(arch("Qwen/Qwen3-1.7B")) == 311_164_928 + 1_269_837_440
    assert sizing.pte_bytes_estimate(arch("Qwen/Qwen3-1.7B")) == 1_581_002_368


def test_pte_estimate_untied():
    a = sizing.Architecture(
        n_layers=1, n_kv_heads=1, head_dim=1, vocab_size=1000, dim=100, total_params=1_100_000, tied_embeddings=False
    )
    # 100,000 embedding params at 1 B + 1,000,000 linear at 0.625 B.
    assert sizing.pte_bytes_estimate(a) == 725_000


def test_resident_bytes_around_the_qwen3_1_7b_boundary():
    qwen = arch("Qwen/Qwen3-1.7B")
    assert sizing.device_resident_bytes(qwen, 8192, OVERHEAD) == 3_960_050_560
    assert sizing.device_resident_bytes(qwen, 16384, OVERHEAD) == 5_839_098_752


def test_window_choice_per_model():
    expected = {
        "Qwen/Qwen3-0.6B": 16384,  # 625,352,704 + 3,758,096,384 + 500,000,000 = 4,883,449,088
        "Qwen/Qwen3-1.7B": 8192,
        "Qwen/Qwen3-4B": 4096,  # 2,902,998,720 + 1,207,959,552 + 500,000,000 = 4,610,958,272
        "meta-llama/Llama-3.2-1B-Instruct": 32768,  # 65,536 B/token: 1,035,052,288 + 2,147,483,648 + 500,000,000
    }
    for model_id, context in expected.items():
        choice = sizing.choose_context(arch(model_id), TIERS, BUDGET, OVERHEAD, None)
        assert choice.context == context, model_id


def test_causal_masks_dominate_export_memory_at_32k():
    # The probe's Qwen3-0.6B 32k export killed a 16.8 GB + 24 GB swap runner: 28 layers of
    # 32,768 x 32,768 one-byte masks is 30,064,771,072 bytes before weights or KV cache.
    qwen = arch("Qwen/Qwen3-0.6B")
    assert sizing.causal_mask_bytes(qwen, 32768) == 30_064_771_072
    host_budget = 16_766_414_848 + 25_769_799_680 - 1_000_000_000
    assert sizing.export_peak_bytes(qwen, 32768) > host_budget
    assert sizing.export_peak_bytes(qwen, 16384) < host_budget


def test_window_choice_respects_host_budget():
    qwen = arch("Qwen/Qwen3-0.6B")
    unconstrained = sizing.choose_context(qwen, TIERS, 10**12, OVERHEAD, None)
    assert unconstrained.context == 32768
    peak_at_16k = sizing.export_peak_bytes(qwen, 16384)
    limited = sizing.choose_context(qwen, TIERS, 10**12, OVERHEAD, peak_at_16k)
    assert limited.context == 16384


def test_no_window_fits():
    choice = sizing.choose_context(arch("Qwen/Qwen3-4B"), TIERS, 1_000_000_000, OVERHEAD, None)
    assert choice.context is None
    assert "no window fits" in choice.reason
    assert [row["context"] for row in choice.table] == list(TIERS)
