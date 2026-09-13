import dataclasses
import json
import os
import subprocess
from pathlib import Path

import pytest
from conftest import load_json

from pipeline import export_qnn, families, manifest, naming, publish, settings

CFG = settings.load()


def flag(command, name):
    return command[command.index(name) + 1]


def test_compile_only_command_for_a_registered_checkpoint(tmp_path):
    command = export_qnn.llama_command("qwen3-0_6b", "SM8650", CFG.qnn, tmp_path / "art", None)
    assert command[1:3] == ["-m", "executorch.examples.qualcomm.oss_scripts.llama.llama"]
    assert "--compile_only" in command
    assert flag(command, "--decoder_model") == "qwen3-0_6b"
    assert flag(command, "--soc_model") == "SM8650"
    assert flag(command, "--model_mode") == "hybrid"
    assert flag(command, "--max_context_len") == flag(command, "--max_seq_len") == "2048"
    assert flag(command, "--prefill_ar_len") == "128"
    assert flag(command, "--calib_tasks") == "wikitext"
    assert flag(command, "--build_folder")  # llama.py realpath()s it even when compiling only
    assert "--checkpoint" not in command
    # The wheel lacks the registry's params .json, so it is always handed over.
    params = Path(flag(command, "--params"))
    assert params == settings.ROOT / "third_party/executorch/examples/models/qwen3/config/0_6b_config.json"


def test_a_forced_window_reaches_the_command_and_the_file_name(tmp_path):
    recipe = dataclasses.replace(CFG.qnn, max_context_len=4096)
    command = export_qnn.llama_command("qwen3-0_6b", "SM8750", recipe, tmp_path / "art", None)
    assert flag(command, "--max_context_len") == flag(command, "--max_seq_len") == "4096"
    assert naming.qnn_file("Qwen/Qwen3-0.6B", recipe.model_mode, 4096) == "Qwen3-0.6B-qnn-hybrid-4k.pte"


def test_a_window_below_the_prefill_length_is_refused_before_any_download(tmp_path):
    with pytest.raises(export_qnn.ExportError, match="below the prefill length"):
        export_qnn.run("Qwen/Qwen3-0.6B", "main", "SM8750", tmp_path / "out", tmp_path / "work", context=64)


def test_llama_gets_metas_original_checkpoint(tmp_path):
    meta = tmp_path / "original"
    command = export_qnn.llama_command("llama3_2-1b_instruct", "SM8750", CFG.qnn, tmp_path / "art", meta)
    assert flag(command, "--checkpoint") == str(meta / "consolidated.00.pth")
    assert flag(command, "--params") == str(meta / "params.json")
    assert flag(command, "--tokenizer_model") == str(meta / "tokenizer.model")
    assert command.count("--params") == 1


def test_every_registered_checkpoint_has_params_or_metas_checkpoint():
    for decoder in families.QNN_DECODERS.values():
        if decoder in families.QNN_META_CHECKPOINT:
            assert decoder not in families.QNN_PARAMS
        else:
            json.loads(export_qnn.params_file(decoder).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "decoder, hf_fixture",
    [
        ("qwen3-0_6b", "qwen3-0.6b"),
        ("qwen3-1_7b", "qwen3-1.7b"),
        ("qwen2_5-1_5b", "qwen2.5-1.5b"),
        ("gemma3-1b", "gemma-3-1b-it"),
        ("smollm2_135m", "smollm2-135m"),
    ],
)
def test_copied_params_describe_the_hf_checkpoint(decoder, hf_fixture):
    params = json.loads(export_qnn.params_file(decoder).read_text(encoding="utf-8"))
    arch = families.architecture(load_json(f"{hf_fixture}.config.json"), 1)
    assert params["n_layers"] == arch.n_layers
    assert params["n_heads"] == arch.n_heads
    assert params["n_kv_heads"] == arch.n_kv_heads
    assert params["dim"] == arch.dim
    assert params["hidden_dim"] == arch.intermediate
    assert params["vocab_size"] == arch.vocab_size
    assert params.get("head_dim", arch.dim // arch.n_heads) == arch.head_dim


SDK = {
    "version": "2.37.0.250724",
    "root": "/home/runner/.cache/executorch/qnn/sdk-2.37.0.250724",
    "libcxx_dir": "/home/runner/.cache/executorch/qnn/libcxx-14.0.0",
    "libcxx_files": ["libc++.so.1.0", "libc++abi.so.1.0", "libunwind.so.1"],
}


def test_the_sdk_is_probed_in_a_child_so_this_process_keeps_its_environment(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(SDK) + "\n", stderr="Loaded libc++.so.1.0\n")

    monkeypatch.setattr(export_qnn.subprocess, "run", fake_run)
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    assert export_qnn.qnn_sdk() == SDK
    assert calls and calls[0][1] == "-c"
    assert "QNN_SDK_ROOT" not in os.environ


def test_sonames():
    assert export_qnn.soname("libc++.so.1.0") == "libc++.so.1"
    assert export_qnn.soname("libc++abi.so.1.0") == "libc++abi.so.1"
    assert export_qnn.soname("libunwind.so.1") == "libunwind.so.1"


def test_the_script_starts_with_the_sdk_and_libcxx_on_the_loader_path(tmp_path):
    if os.name == "nt":
        pytest.skip("symlinks need privileges on Windows; the export runs on Linux")
    links = tmp_path / "links"
    env = export_qnn.qnn_env(SDK, links, {"LD_LIBRARY_PATH": "/opt/x", "HOME": "/h"})
    assert env["QNN_SDK_ROOT"] == SDK["root"]
    assert env["LD_LIBRARY_PATH"] == f"{SDK['root']}/lib/x86_64-linux-clang:{links}:/opt/x"
    assert env["HOME"] == "/h" and env["PYTHONUNBUFFERED"] == "1"
    assert os.readlink(links / "libc++.so.1") == f"{SDK['libcxx_dir']}/libc++.so.1.0"
    assert sorted(p.name for p in links.iterdir()) == ["libc++.so.1", "libc++abi.so.1", "libunwind.so.1"]
    export_qnn.qnn_env(SDK, links, {})  # re-running over existing links is fine


def test_decoder_pte_is_found_by_mode(tmp_path):
    (tmp_path / "qwen3-0_6b_hybrid_llama_qnn.pte").write_bytes(b"x")
    assert export_qnn.decoder_pte(tmp_path, "hybrid").name == "qwen3-0_6b_hybrid_llama_qnn.pte"
    (tmp_path / "hybrid_llama_qnn.pte").write_bytes(b"x")
    with pytest.raises(export_qnn.ExportError, match="expected one hybrid"):
        export_qnn.decoder_pte(tmp_path, "hybrid")


def test_contains_finds_a_needle_across_block_boundaries(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"a" * 30 + b"QnnBackend" + b"b" * 30)
    for block in (4, 7, 16, 35, 1 << 20):
        assert export_qnn.contains(path, b"QnnBackend", block=block), block
    assert not export_qnn.contains(path, b"XnnpackBackend", block=8)


def test_every_registered_qnn_checkpoint_gets_names_the_app_reads_as_qnn():
    for model_id in families.QNN_DECODERS:
        repo = naming.output_repo(model_id, CFG.hub_org, CFG.repo_suffix)
        for soc in CFG.qnn.socs:
            path = f"{naming.qnn_folder(soc)}/{naming.qnn_file(model_id, CFG.qnn.model_mode, 2048)}"
            assert naming.check_app_rules(repo, path, "qnn") == [], (repo, path)


def qnn_report(soc="sm8650"):
    return {
        "backend": "qnn",
        "target": soc,
        "target_name": export_qnn.SOC_NAMES[soc.upper()],
        "tokenizer": "tokenizer.json",
        "output_repo": "experimentalmachines/Qwen3-0.6B-ExecuTorch",
        "source": {"id": "Qwen/Qwen3-0.6B", "sha": "c1899de289a0", "license": {"license": "apache-2.0"}},
        "toolchain": {"executorch": "1.4.0", "qairt": "2.37.0.250724"},
        "recipe": {"label": "QNN HTP", "description": "recipe."},
        "window": {"context": 2048, "kv_cache_bytes_per_token": None},
        "files": [{"path": f"qnn/{soc}/Qwen3-0.6B-qnn-hybrid-2k.pte", "bytes": 700_000_000, "sha256": "ab"}],
        "metadata": {},
        "smoke": {"kind": "structural", "passed": True, "problems": []},
        "run": {},
    }


def test_readme_and_config_for_qnn():
    config = manifest.backend_config(qnn_report())
    assert config["qnn_sdk_version"] == "2.37.0.250724"
    assert config["target"] == "sm8650"
    text = manifest.readme("experimentalmachines/Qwen3-0.6B-ExecuTorch", [qnn_report(), qnn_report("sm8750")], [], [])
    assert "SM8650 (Snapdragon 8 Gen 3)" in text and "SM8750 (Snapdragon 8 Elite)" in text
    assert "structure checked (no host HTP runtime)" in text
    assert "QAIRT) 2.37.0.250724" in text and "AI Stack License" in text
    assert "- qnn" in text


def test_publish_refuses_a_tokenizer_inside_a_backend_folder(tmp_path):
    folder = tmp_path / "qnn" / "sm8650"
    folder.mkdir(parents=True)
    (folder / "export-report.json").write_text(json.dumps(qnn_report()))
    (folder / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="repo root"):
        publish.publish_hf(tmp_path, "qnn", "SM8650")
