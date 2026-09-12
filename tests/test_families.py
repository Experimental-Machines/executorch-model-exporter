import copy

import pytest
from conftest import hf_config, load_json

from pipeline import families


def plan_for(config):
    return families.xnnpack_plan(families.family_for(config), config)


def expected_params(name):
    """ExecuTorch's shipped params, with head_dim made explicit where it is implied."""
    params = load_json(name)
    params.setdefault("head_dim", params["dim"] // params["n_heads"])
    return params


@pytest.mark.parametrize(
    "config, reference",
    [
        ("qwen3-0.6b.config.json", "et-qwen3-0_6b.params.json"),
        ("qwen3-1.7b.config.json", "et-qwen3-1_7b.params.json"),
        ("qwen3-4b.config.json", "et-qwen3-4b.params.json"),
        ("qwen2.5-1.5b.config.json", "et-qwen2_5-1_5b.params.json"),
        ("smollm2-135m.config.json", "et-smollm2-135m.params.json"),
    ],
)
def test_generated_params_equal_executorchs_own(config, reference):
    assert plan_for(load_json(config)).params == expected_params(reference)


def test_model_classes_and_converters():
    assert (plan_for(hf_config("Qwen/Qwen3-1.7B")).model_class, plan_for(hf_config("Qwen/Qwen3-1.7B")).converter) == (
        "qwen3_1_7b",
        "qwen3",
    )
    qwen25 = plan_for(hf_config("Qwen/Qwen2.5-1.5B-Instruct"))
    assert (qwen25.model_class, qwen25.converter) == ("qwen2_5_1_5b", "qwen2")
    smol = plan_for(hf_config("HuggingFaceTB/SmolLM2-360M-Instruct"))
    assert (smol.model_class, smol.converter, smol.params["use_hf_rope"]) == ("smollm2", "llama", False)
    assert smol.params["head_dim"] == 64  # 960 / 15


def test_llama32_uses_scaled_meta_rope():
    plan = plan_for(hf_config("meta-llama/Llama-3.2-1B-Instruct"))
    assert plan.model_class == "llama3_2"  # model.py then sets rope_scale_factor = 32
    assert plan.params["use_scaled_rope"] is True
    assert plan.params["use_hf_rope"] is False
    assert plan.params["head_dim"] == 64
    assert plan.params["rope_theta"] == 500000.0


def test_other_llama_rope_scaling_is_refused():
    config = copy.deepcopy(hf_config("meta-llama/Llama-3.2-1B-Instruct"))
    config["rope_scaling"]["factor"] = 8.0  # Llama 3.1's factor
    with pytest.raises(families.UnsupportedModel, match="factor"):
        plan_for(config)


def test_yarn_is_refused():
    config = copy.deepcopy(hf_config("Qwen/Qwen3-1.7B"))
    config["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
    with pytest.raises(families.UnsupportedModel, match="RoPE scaling"):
        plan_for(config)


def test_sliding_window_is_refused():
    config = copy.deepcopy(hf_config("Qwen/Qwen2.5-1.5B-Instruct"))
    config["use_sliding_window"] = True
    with pytest.raises(families.UnsupportedModel, match="sliding"):
        plan_for(config)


def test_transformers_v5_rope_parameters():
    config = copy.deepcopy(hf_config("Qwen/Qwen3-1.7B"))
    del config["rope_theta"]
    config.pop("rope_scaling", None)
    config["rope_parameters"] = {"rope_type": "default", "rope_theta": 1000000.0}
    assert plan_for(config).params["rope_theta"] == 1000000.0


def test_moe_detection():
    assert families.is_moe(hf_config("Qwen/Qwen3-30B-A3B"))
    for model_id in ("Qwen/Qwen3-1.7B", "meta-llama/Llama-3.2-1B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"):
        assert not families.is_moe(hf_config(model_id)), model_id


def test_gemma3_has_a_family_but_no_xnnpack_recipe_yet():
    family = families.family_for(hf_config("google/gemma-3-1b-it"))
    assert family.key == "gemma3"
    assert not family.supports("xnnpack")
    with pytest.raises(families.UnsupportedModel):
        families.xnnpack_plan(family, hf_config("google/gemma-3-1b-it"))
