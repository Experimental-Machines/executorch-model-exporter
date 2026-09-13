import dataclasses
import json

import pytest
from conftest import hf_config, load_json

from pipeline import export_mtk, families, manifest, naming, publish, settings

CFG = settings.load()


def flag(command, name):
    return command[command.index(name) + 1]


def plan(model_id="Qwen/Qwen3-0.6B", config=None):
    config = config or hf_config(model_id)
    return families.mtk_plan(families.family_for(config), config, CFG.mtk.max_chunks)


@pytest.mark.parametrize(("layers", "chunks"), [(28, 4), (24, 4), (36, 4), (30, 3), (26, 2), (1, 1)])
def test_chunks_split_the_layers_evenly(layers, chunks):
    assert families.mtk_chunks(layers, 4) == chunks


def test_qwen_families_use_mediateks_qwen_script_and_their_chat_templates():
    assert plan() == families.MtkPlan("qwen.py", "qwen3.json", 4)
    qwen25 = load_json("qwen2.5-1.5b-instruct.config.json")
    assert families.mtk_plan(families.family_for(qwen25), qwen25, 4) == families.MtkPlan("qwen.py", "qwen.json", 4)


def test_rope_scaling_and_other_model_types_are_refused():
    config = hf_config("Qwen/Qwen3-0.6B") | {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}
    with pytest.raises(families.UnsupportedModel, match="RoPE scaling"):
        plan(config=config)
    with pytest.raises(families.UnsupportedModel, match="model_type"):
        plan(config=hf_config("Qwen/Qwen3-0.6B") | {"model_type": "qwen2"})


def test_export_command_matches_mediateks_shell_scripts(tmp_path):
    command = export_mtk.export_command("/venv/bin/python", plan(), CFG.mtk, "MT6989", tmp_path / "config.json")
    assert command[:3] == ["/venv/bin/python", "model_export_scripts/qwen.py", str(tmp_path / "config.json")]
    assert flag(command, "--precision") == "A16W4"
    assert flag(command, "--num_chunks") == "4"
    assert flag(command, "--dataset") == "aot_utils/llm_utils/prompts/alpaca.txt"
    assert flag(command, "--preformatter") == "aot_utils/llm_utils/preformatter_templates/qwen3.json"
    shapes = command.index("-shapes")
    assert command[shapes + 1 : shapes + 3] == ["128t512c", "1t512c"]
    assert flag(command, "--response_cap") == "9"
    assert flag(command, "--platform") == "DX3"
    assert flag(export_mtk.export_command("p", plan(), CFG.mtk, "MT6991", tmp_path), "--platform") == "DX4"


def test_output_names_follow_the_scripts(tmp_path):
    exp = export_mtk.exp_name(tmp_path / "Qwen3-0.6B", "A16W4", 4)
    assert exp == "Qwen3-0.6B_A16W4_4_chunks"
    assert export_mtk.method_names(exp, CFG.mtk, 3) == [
        "Qwen3-0.6B_A16W4_4_chunks_128t512c_3",
        "Qwen3-0.6B_A16W4_4_chunks_1t512c_3",
    ]


def test_every_chunk_name_reads_as_neuropilot_in_the_app():
    for model_id in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "Qwen/Qwen2.5-1.5B-Instruct"):
        repo = naming.output_repo(model_id, CFG.hub_org, CFG.repo_suffix)
        for soc in CFG.mtk.socs:
            for i in range(4):
                path = f"{naming.mtk_folder(soc)}/{naming.mtk_chunk_file(model_id, 'A16W4', 2048, i, 4)}"
                assert naming.check_app_rules(repo, path, "mtk") == [], path
    assert (
        naming.mtk_chunk_file("Qwen/Qwen3-0.6B", "A16W4", 2048, 0, 4) == "Qwen3-0.6B-neuropilot-a16w4-2k-chunk1of4.pte"
    )


def test_runner_settings_for_qwen3_0_6b():
    runner = export_mtk.runner_settings(hf_config("Qwen/Qwen3-0.6B"), CFG.mtk, 151643, [151645, 151643])
    assert runner["hidden_size"] == 1024
    assert runner["num_head"] == 16
    assert runner["num_layer"] == 28
    assert runner["head_dim"] == 128
    assert runner["rot_emb_base"] == 1000000.0
    assert runner["cache_size"] == 512 and runner["prompt_token_batch_size"] == 128
    assert runner["eos_token"] == 151645 and runner["eos_tokens"] == [151645, 151643]
    assert runner["vocab_size"] == 151936
    assert runner["tokenizer_type"] == "hf" and runner["cache_type"] == "fp32"


def test_a_forced_window_changes_the_shapes(tmp_path):
    recipe = dataclasses.replace(CFG.mtk, cache_size=4096)
    command = export_mtk.export_command("p", plan(), recipe, "MT6991", tmp_path)
    shapes = command.index("-shapes")
    assert command[shapes + 1 : shapes + 3] == ["128t4096c", "1t4096c"]


def test_unknown_chip_and_short_window_are_refused_before_any_download(tmp_path):
    with pytest.raises(export_mtk.ExportError, match="unknown MediaTek chip"):
        export_mtk.run("Qwen/Qwen3-0.6B", "main", "MT6878", tmp_path, tmp_path, "p", tmp_path)
    with pytest.raises(export_mtk.ExportError, match="below the prompt length"):
        export_mtk.run("Qwen/Qwen3-0.6B", "main", "MT6991", tmp_path, tmp_path, "p", tmp_path, context=64)


def mtk_report(soc="mt6991"):
    chunks = [f"mtk/{soc}/Qwen3-0.6B-neuropilot-a16w4-2k-chunk{i}of4.pte" for i in range(1, 5)]
    embedding = f"mtk/{soc}/Qwen3-0.6B-neuropilot-embedding-fp32.bin"
    return {
        "backend": "mtk",
        "target": soc,
        "target_name": export_mtk.SOC_NAMES[soc.upper()],
        "tokenizer": "tokenizer.json",
        "output_repo": "experimentalmachines/Qwen3-0.6B-ExecuTorch",
        "source": {"id": "Qwen/Qwen3-0.6B", "sha": "c1899de289a0", "license": {"license": "apache-2.0"}},
        "toolchain": {"executorch": "1.4.0"},
        "neuropilot": {
            "name": "NeuroPilot Express SDK",
            "build": "20250327",
            "mtk_converter": "8.13.0+public",
            "mtk_neuron": "8.2.19",
        },
        "recipe": {"label": "NeuroPilot A16W4, 4 chunks", "description": "recipe."},
        "window": {"context": 2048, "kv_cache_bytes_per_token": None},
        "runner": {"token_embedding_path": embedding.rsplit("/", 1)[-1], "cache_size": 2048},
        "files": [{"path": p, "bytes": 150_000_000, "sha256": "ab"} for p in chunks]
        + [{"path": embedding, "bytes": 622_329_856, "sha256": "cd"}],
        "metadata": {},
        "smoke": {"kind": "structural", "passed": True, "problems": []},
        "run": {},
    }


def test_config_describes_one_model_in_several_files():
    config = manifest.backend_config(mtk_report())
    assert len(config["variants"]) == 1
    variant = config["variants"][0]
    assert variant["files"] == [f"Qwen3-0.6B-neuropilot-a16w4-2k-chunk{i}of4.pte" for i in range(1, 5)]
    assert variant["embedding"] == "Qwen3-0.6B-neuropilot-embedding-fp32.bin"
    assert variant["size_bytes"] == 4 * 150_000_000 + 622_329_856
    assert config["runner"]["cache_size"] == 2048
    assert config["neuropilot_sdk"]["build"] == "20250327"


def test_readme_credits_mediatek_without_claiming_the_sdk_is_included():
    text = manifest.readme("experimentalmachines/Qwen3-0.6B-ExecuTorch", [mtk_report(), mtk_report("mt6989")], [], [])
    assert "MT6991 (Dimensity 9400)" in text and "MT6989 (Dimensity 9300)" in text
    assert "MediaTek NeuroPilot Express SDK (build 20250327" in text and "MediaTek Inc." in text
    assert "No MediaTek SDK or runtime library is included" in text
    assert "neuropilot-embedding-fp32.bin" in text
    assert "- mtk" in text


def test_publish_refuses_a_tokenizer_inside_a_mediatek_folder(tmp_path):
    folder = tmp_path / "mtk" / "mt6991"
    folder.mkdir(parents=True)
    (folder / "export-report.json").write_text(json.dumps(mtk_report()))
    (folder / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="repo root"):
        publish.publish_hf(tmp_path, "mtk", "MT6991")


def test_calibration_memory_estimate_matches_the_run_that_took_the_runner_down():
    # Qwen3-0.6B: 28 layers x K,V x 8 KV heads x 128 x 4 B = 229,376 B of fp32 cache per token.
    config = hf_config("Qwen/Qwen3-0.6B")
    at_2k = dataclasses.replace(CFG.mtk, cache_size=2048, response_cap=9)
    assert export_mtk.calibration_bytes(config, at_2k, 8) == 8 * 10 * 229_376 * 2048 * 2 == 75_161_927_680
    assert export_mtk.calibration_bytes(config, CFG.mtk, 8) == 18_790_481_920  # the 512 default
    # 16.8 GB RAM + 24 GB swap minus the reserve: 2048 is refused, 512 fits.
    budget = 16_766_414_848 + 25_769_799_680 - 1_000_000_000
    assert export_mtk.calibration_bytes(config, at_2k, 8) > budget > export_mtk.calibration_bytes(config, CFG.mtk, 8)
