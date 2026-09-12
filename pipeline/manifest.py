"""The files published beside each export: config.json, README.md, NOTICE."""

from __future__ import annotations

import os

from pipeline import naming

# Methods the app reads from config.json before download (openweights ExportConfig.kt),
# plus the rest of the runtime's LLM metadata, as the .pte reports them.
CONFIG_METHODS = (
    "get_max_context_len",
    "get_max_seq_len",
    "get_bos_id",
    "get_eos_ids",
    "get_vocab_size",
    "get_n_layers",
    "use_kv_cache",
    "use_sdpa_with_kv_cache",
    "enable_dynamic_shape",
)

# Attribution the upstream licenses require of redistributed derivatives, keyed by the
# app family token of the source name.
NOTICES = {
    # Llama 3.2 Community License, section 1.b.iii.
    "llama32": (
        "Llama 3.2 is licensed under the Llama 3.2 Community License, "
        "Copyright © Meta Platforms, Inc. All Rights Reserved.\n"
    ),
    # Gemma Terms of Use, section 3.1.
    "gemma3": ("Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms\n"),
}
BUILT_WITH = {"llama32": "Built with Llama"}

BACKEND_TITLES = {"xnnpack": "XNNPACK (CPU)", "qnn": "Qualcomm QNN (HTP)", "mtk": "MediaTek NeuroPilot"}


def backend_config(report: dict) -> dict:
    """config.json for one backend folder, in the variants form the app reads."""
    variants = []
    for f in report["files"]:
        if not f["path"].endswith(".pte"):
            continue
        methods = {k: report["metadata"][k] for k in CONFIG_METHODS if k in report["metadata"]}
        variants.append(
            {
                "file": f["path"].rsplit("/", 1)[-1],
                "size_bytes": f["bytes"],
                "sha256": f["sha256"],
                "quantization": report["recipe"]["label"],
                "methods": methods,
            }
        )
    return {
        "runtime": "executorch",
        "runtime_version": report["toolchain"]["executorch"],
        "backend": report["backend"],
        "target": report.get("target"),
        "tokenizer": report["tokenizer"],
        "source_model": report["source"]["id"],
        "source_revision": report["source"]["sha"],
        "variants": variants,
    }


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def _card_metadata(source: dict, tags: list[str]) -> str:
    lic = source.get("license") or {}
    lines = ["---"]
    if lic.get("license"):
        lines.append(f"license: {lic['license']}")
    for key in ("license_name", "license_link"):
        if lic.get(key):
            lines.append(f"{key}: {lic[key]}")
    lines += [
        "base_model:",
        f"- {source['id']}",
        "base_model_relation: quantized",
        "library_name: executorch",
        "pipeline_tag: text-generation",
        "tags:",
        *[f"- {t}" for t in tags],
        "---",
    ]
    return "\n".join(lines)


def readme(repo_id: str, reports: list[dict], hub_tags: list[str], license_files: list[str]) -> str:
    """Model card covering every backend present in the repo."""
    reports = sorted(reports, key=lambda r: (list(BACKEND_TITLES).index(r["backend"]), r.get("target") or ""))
    source = reports[0]["source"]
    token = naming.app_family(naming.source_name(source["id"]))
    tags = sorted(set(hub_tags) | {r["backend"] for r in reports})
    upstream = f"https://huggingface.co/{source['id']}"
    out = [_card_metadata(source, tags), ""]
    if token in BUILT_WITH:
        out += [f"**{BUILT_WITH[token]}**", ""]
    out += [
        f"# {naming.source_name(source['id'])} for ExecuTorch",
        "",
        f"ExecuTorch exports of [{source['id']}]({upstream}) (revision `{source['sha'][:12]}`) "
        "for on-device inference with the "
        "[openweights](https://github.com/alpharomercoma/openweights) Android app or any "
        f"ExecuTorch {reports[0]['toolchain']['executorch']} runtime.",
        "",
        "## Files",
        "",
        "| Backend | Target | File | Window | Size | Smoke test |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        smoke = r.get("smoke") or {}
        verdict = "passed" if smoke.get("passed") else ("not run" if not smoke else "failed")
        if smoke.get("answered"):
            verdict += ' ("Paris")'
        for f in r["files"]:
            if f["path"].endswith(".pte"):
                out.append(
                    f"| {BACKEND_TITLES[r['backend']]} | {r.get('target') or 'any arm64'} | "
                    f"[`{f['path']}`]({f['path']}) | {r['window']['context']:,} tokens | "
                    f"{_gb(f['bytes'])} | {verdict} |"
                )
    out += [
        "",
        f"Tokenizer: [`{reports[0]['tokenizer']}`]({reports[0]['tokenizer']}), copied unchanged "
        "from the source repo. Each backend folder has a `config.json` with the metadata the "
        "`.pte` reports and an `export-report.json` with the full export record.",
        "",
        "## Memory",
        "",
    ]
    for r in reports:
        kv = r["window"].get("kv_cache_bytes_per_token")
        if kv:
            ctx = r["window"]["context"]
            out.append(
                f"- {BACKEND_TITLES[r['backend']]}: the KV cache costs {kv:,} bytes per token "
                f"(fp32), {kv * ctx:,} bytes at the exported window of {ctx:,} tokens, allocated "
                "in full when the model loads."
            )
    out += ["", "## How it was made", ""]
    for r in reports:
        recipe = r["recipe"]
        line = f"- {BACKEND_TITLES[r['backend']]}: {recipe['description']}"
        if r.get("run", {}).get("url"):
            line += f" Built by [this workflow run]({r['run']['url']})."
        out.append(line)
    lic = source.get("license") or {}
    name = lic.get("license_name") or lic.get("license") or "the upstream license"
    out += [
        "",
        "## License",
        "",
        f"A quantized derivative of [{source['id']}]({upstream}), distributed under the same terms ({name}).",
    ]
    if license_files:
        out.append(
            "The upstream license files are included unchanged: "
            + ", ".join(f"[`{f}`]({f})" for f in license_files)
            + "."
        )
    if token in NOTICES:
        out.append("See [`NOTICE`](NOTICE) for the attribution the license requires.")
    return "\n".join(out) + "\n"


def run_info() -> dict:
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repo and run_id:
        return {"url": f"{server}/{repo}/actions/runs/{run_id}", "id": run_id}
    return {}
