from conftest import make_source

from pipeline import hub


def test_special_tokens_fall_back_to_config_when_generation_config_has_nulls():
    source = make_source(
        "Qwen/Qwen3-0.6B",
        config={"bos_token_id": 151643, "eos_token_id": 151645},
        generation_config={"bos_token_id": None, "eos_token_id": [151645, 151643, 151645]},
    )
    assert hub.special_token_ids(source) == (151643, [151645, 151643])


def test_special_tokens_prefer_generation_config():
    source = make_source(
        "meta-llama/Llama-3.2-1B-Instruct",
        config={"bos_token_id": 128000, "eos_token_id": 128001},
        generation_config={"bos_token_id": 128000, "eos_token_id": [128001, 128008, 128009]},
    )
    assert hub.special_token_ids(source) == (128000, [128001, 128008, 128009])
    assert hub.special_token_ids(make_source("Qwen/Qwen3-0.6B", config={})) == (None, [])
