"""Export one HF model to a Qualcomm QNN (HTP) .pte for one Snapdragon chip.

Drives ExecuTorch 1.4.0's own Qualcomm LLM script (examples/qualcomm/oss_scripts/llama)
in compile-only mode: it downloads the checkpoint, calibrates the quantization recipe it
registers for that model, and compiles hybrid prefill + decode graphs into QNN context
binaries for the chip. Produces, under ``out_dir``:

    tokenizer.json | tokenizer.model, LICENSE…, NOTICE   (repo root, shared with other backends)
    qnn/<soc>/<name>-qnn-hybrid-<window>.pte
    qnn/<soc>/config.json
    qnn/<soc>/export-report.json

There is no host runtime for HTP binaries here, so the check after export is structural:
the program loads, carries the decoder graphs, and delegates to QnnBackend.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pipeline import eligibility, families, hub, manifest, naming, settings, smoke
from pipeline.exporting import (
    ExportError,
    children_peak_rss,
    copy_side_files,
    host_info,
    sha256,
    source_report,
    toolchain,
)

BACKEND = "qnn"
# examples/qualcomm/oss_scripts/llama/decoder_constants.py: DECODER_GRAPH_NAMES
HYBRID_METHODS = ("kv_forward", "prefill_forward")
META_FILES = ("original/consolidated.00.pth", "original/params.json", "original/tokenizer.model")
SOC_NAMES = {"SM8650": "Snapdragon 8 Gen 3", "SM8750": "Snapdragon 8 Elite"}
EXECUTORCH_SOURCE_FILES = settings.ROOT / "third_party" / "executorch"


def params_file(decoder: str) -> Path:
    """The registry's params file for ``decoder`` (copied from ExecuTorch; the wheel lacks it)."""
    path = EXECUTORCH_SOURCE_FILES / families.QNN_PARAMS[decoder]
    if not path.is_file():
        raise ExportError(f"missing {path}: copy it from the ExecuTorch source at the pinned version")
    return path


def llama_command(
    decoder: str,
    soc: str,
    recipe: settings.QnnRecipe,
    artifact: Path,
    meta_dir: Path | None,
) -> list[str]:
    window = recipe.max_context_len
    command = [
        sys.executable,
        "-m",
        "executorch.examples.qualcomm.oss_scripts.llama.llama",
        "--decoder_model",
        decoder,
        "--soc_model",
        soc,
        "--compile_only",
        "--model_mode",
        recipe.model_mode,
        "--max_seq_len",
        str(window),
        "--max_context_len",
        str(window),
        "--prefill_ar_len",
        str(recipe.prefill_ar_len),
        "--calib_tasks",
        *recipe.calib_tasks,
        "--calib_limit",
        str(recipe.calib_limit),
        "--artifact",
        str(artifact),
        # main() calls os.path.realpath(args.build_folder) even in compile-only mode, where the
        # on-device runner it points at is never used.
        "--build_folder",
        str(artifact.parent / "build-android"),
        # Required by the script; used only when it runs on a device, which compile-only skips.
        "--prompt",
        "What is the capital of France?",
        "--temperature",
        "0",
    ]
    if meta_dir is not None:
        command += [
            "--checkpoint",
            str(meta_dir / "consolidated.00.pth"),
            "--params",
            str(meta_dir / "params.json"),
            "--tokenizer_model",
            str(meta_dir / "tokenizer.model"),
        ]
    else:
        # Every reader of the params prefers --params over the registry's params_path, which
        # points into the source tree the wheel does not carry.
        command += ["--params", str(params_file(decoder))]
    return command


def decoder_pte(artifact: Path, model_mode: str) -> Path:
    """The text decoder the script wrote: ``[<decoder>_]<mode>_llama_qnn.pte``."""
    matches = sorted(artifact.glob(f"*{model_mode}_llama_qnn.pte"))
    if len(matches) != 1:
        found = sorted(p.name for p in artifact.glob("*.pte"))
        raise ExportError(f"expected one {model_mode} decoder .pte in {artifact}, found {found}")
    return matches[0]


def contains(path: Path, needle: bytes, block: int = 1 << 24) -> bool:
    """Whether ``needle`` occurs in the file, scanning in blocks (a .pte can be gigabytes)."""
    tail = b""
    with path.open("rb") as f:
        while chunk := f.read(block):
            window = tail + chunk
            if needle in window:
                return True
            tail = window[-(len(needle) - 1) :]
    return False


def structural_check(pte: Path, model_mode: str) -> dict:
    problems = []
    try:
        metadata = smoke.read_metadata(pte)
    except Exception as error:  # a program that does not even parse
        return {"kind": "structural", "passed": False, "problems": [f"program did not load: {error}"]}
    methods = metadata.get("methods", [])
    wanted = HYBRID_METHODS if model_mode == "hybrid" else ("kv_forward",)
    missing = [m for m in wanted if m not in methods]
    if missing:
        problems.append(f"missing decoder methods {missing} (has {methods})")
    if not contains(pte, b"QnnBackend"):
        problems.append("no QnnBackend delegate in the program")
    return {
        "kind": "structural",
        "passed": not problems,
        "problems": problems,
        "methods": methods,
        "metadata": metadata,
    }


def qairt_version() -> str | None:
    """The QAIRT SDK the executorch wheel pins (and downloads on first import).

    Asked of a separate interpreter: importing executorch.backends.qualcomm sets QNN_SDK_ROOT
    and preloads libc++ in the importing process, and a child that inherits QNN_SDK_ROOT
    skips that preload, so libQnnHtp.so then fails to find libc++.so.1.
    """
    probe = "from executorch.backends.qualcomm.scripts.download_qnn_sdk import QNN_VERSION; print(QNN_VERSION)"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip().splitlines()[-1]


def run(
    model_id: str,
    revision: str,
    soc: str,
    out_dir: Path,
    work_dir: Path,
    keep_work: bool = False,
) -> dict:
    cfg = settings.load()
    recipe = cfg.qnn
    source = hub.fetch(model_id, revision)
    verdict = eligibility.evaluate(source, cfg)
    if verdict.reasons:
        raise ExportError(f"{model_id} is not eligible: {'; '.join(verdict.reasons)}")
    if verdict.backends[BACKEND] is not None:
        raise ExportError(f"{model_id} cannot be exported to {BACKEND}: {verdict.backends[BACKEND]}")
    decoder = families.qnn_decoder(source.id)
    window = recipe.max_context_len

    output_repo = naming.output_repo(model_id, cfg.hub_org, cfg.repo_suffix)
    folder = naming.qnn_folder(soc)
    pte_name = naming.qnn_file(model_id, recipe.model_mode, window)
    path_in_repo = f"{folder}/{pte_name}"
    problems = naming.check_app_rules(output_repo, path_in_repo, BACKEND)
    if problems:
        raise ExportError("; ".join(problems))

    backend_dir = out_dir / folder
    backend_dir.mkdir(parents=True, exist_ok=True)
    src_dir = work_dir / "source"
    artifact = work_dir / "artifact"
    meta = decoder in families.QNN_META_CHECKPOINT
    print(f"==> {model_id}@{source.sha[:12]}: --decoder_model {decoder}, {soc}, window {window}")

    qairt = qairt_version()
    expected = settings.read_env_file(settings.CONFIG_DIR / "versions.env").get("QAIRT_VERSION")
    if qairt != expected:
        raise ExportError(f"executorch's QAIRT is {qairt}, config/versions.env pins {expected}")

    started = time.time()
    # The script fetches the weights itself (repo_id in its registry); only Llama 3.2 needs
    # Meta's original checkpoint handed over.
    hub.download(source, src_dir, weights=False, extra=META_FILES if meta else ())
    tokenizer, licenses = copy_side_files(source, src_dir, out_dir)

    command = llama_command(decoder, soc, recipe, artifact, src_dir / "original" if meta else None)
    print("==> " + " ".join(command))
    export_started = time.time()
    subprocess.run(command, check=True, cwd=work_dir, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    export_seconds = time.time() - export_started

    built = decoder_pte(artifact, recipe.model_mode)
    pte = backend_dir / pte_name
    shutil.move(str(built), pte)
    check = structural_check(pte, recipe.model_mode)
    metadata = check.pop("metadata", {})

    report = {
        "backend": BACKEND,
        "target": soc.lower(),
        "target_name": SOC_NAMES.get(soc, soc),
        "tokenizer": tokenizer,
        "output_repo": output_repo,
        "source": source_report(source, verdict.family, verdict.variant, licenses),
        "toolchain": {**toolchain("transformers", "lm_eval"), "qairt": qairt},
        "recipe": {
            "decoder_model": decoder,
            "params": f"{model_id}: original/params.json" if meta else f"executorch: {families.QNN_PARAMS[decoder]}",
            "model_mode": recipe.model_mode,
            "prefill_ar_len": recipe.prefill_ar_len,
            "max_context_len": window,
            "calibration": {"tasks": list(recipe.calib_tasks), "limit": recipe.calib_limit},
            "label": f"QNN HTP, ExecuTorch {decoder} recipe",
            "description": (
                f"ExecuTorch {toolchain()['executorch']} Qualcomm static LLM "
                f"(`examples/qualcomm/oss_scripts/llama`, `--decoder_model {decoder}`): the "
                f"quantization recipe ExecuTorch registers for this model, calibrated on "
                f"{'/'.join(recipe.calib_tasks)} ({recipe.calib_limit} sample), {recipe.model_mode} "
                f"prefill ({recipe.prefill_ar_len} tokens per step) and decode graphs compiled "
                f"with QAIRT {qairt} for {soc} ({SOC_NAMES.get(soc, soc)})."
            ),
        },
        "window": {
            "context": window,
            "reason": "fixed by qnn.max_context_len (static NPU graphs)",
            "kv_cache_bytes_per_token": None,
        },
        "files": [{"path": path_in_repo, "bytes": pte.stat().st_size, "sha256": sha256(pte)}],
        "host": {
            **host_info(),
            "peak_rss_export_bytes": children_peak_rss(),
            "export_seconds": round(export_seconds, 1),
            "total_seconds": round(time.time() - started, 1),
        },
        "metadata": metadata,
        "smoke": check,
        "run": manifest.run_info(),
    }
    (backend_dir / "config.json").write_text(
        json.dumps(manifest.backend_config(report), indent=2) + "\n", encoding="utf-8"
    )
    (backend_dir / "export-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    if not check["passed"]:
        raise ExportError("structural check failed: " + "; ".join(check["problems"]))
    return report
