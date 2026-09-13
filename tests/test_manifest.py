from pipeline import manifest, smoke


def report(source_id="Qwen/Qwen3-1.7B", backend="xnnpack", license=None):
    return {
        "backend": backend,
        "target": None,
        "tokenizer": "tokenizer.json",
        "output_repo": "experimentalmachines/Qwen3-1.7B-ExecuTorch",
        "source": {
            "id": source_id,
            "sha": "abcdef0123456789",
            "license": license or {"license": "apache-2.0"},
            "license_files": ["LICENSE"],
        },
        "toolchain": {"executorch": "1.4.0"},
        "recipe": {"label": "8da4w-g32, int8 embeddings", "description": "recipe text."},
        "window": {"context": 8192, "kv_cache_bytes_per_token": 229_376},
        "files": [{"path": "xnnpack/Qwen3-1.7B-8da4w-8k.pte", "bytes": 1_500_000_000, "sha256": "ff"}],
        "metadata": {
            "get_max_context_len": 8192,
            "get_max_seq_len": 2048,
            "get_eos_ids": [151645, 151643],
            "methods": ["forward", "get_max_context_len"],
        },
        "smoke": {"passed": True, "answered": True},
        "run": {"url": "https://github.com/o/r/actions/runs/1"},
    }


def test_backend_config_is_in_the_form_the_app_reads():
    config = manifest.backend_config(report())
    assert config["runtime_version"] == "1.4.0"
    [variant] = config["variants"]
    assert variant["file"] == "Qwen3-1.7B-8da4w-8k.pte"  # bare file name
    assert variant["methods"]["get_max_context_len"] == 8192
    assert "methods" not in variant["methods"]


def test_readme_has_card_metadata_and_kv_arithmetic():
    text = manifest.readme("experimentalmachines/Qwen3-1.7B-ExecuTorch", [report()], ["executorch"], ["LICENSE"])
    assert text.startswith("---\nlicense: apache-2.0\n")
    assert "base_model_relation: quantized" in text
    assert "- xnnpack" in text and "- executorch" in text
    # 229,376 x 8,192 = 1,879,048,192
    assert "at 8,192 tokens: the KV cache costs 229,376 bytes per token (fp32), 1,879,048,192 bytes" in text
    assert "[`LICENSE`](LICENSE)" in text
    assert "NOTICE" not in text


def test_llama_derivatives_carry_the_attribution_the_license_requires():
    r = report("meta-llama/Llama-3.2-1B-Instruct", license={"license": "llama3.2"})
    text = manifest.readme("experimentalmachines/Llama-3.2-1B-Instruct-ExecuTorch", [r], [], ["LICENSE.txt"])
    assert "**Built with Llama**" in text
    assert "[`NOTICE`](NOTICE)" in text
    assert "Llama 3.2 Community License" in manifest.NOTICES["llama32"]


def test_other_license_names_are_used():
    r = report(
        "Qwen/Qwen2.5-3B-Instruct",
        license={"license": "other", "license_name": "qwen-research", "license_link": "https://x/LICENSE"},
    )
    text = manifest.readme("experimentalmachines/Qwen2.5-3B-Instruct-ExecuTorch", [r], [], ["LICENSE"])
    assert "license_name: qwen-research" in text
    assert "(qwen-research)" in text


def test_degenerate_output():
    assert smoke.degenerate(["a"] * 8)
    assert not smoke.degenerate(["a"] * 7)
    assert not smoke.degenerate(["a"] * 7 + ["b"])
