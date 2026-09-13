"""Publishing an export: Hugging Face Hub, GitHub Releases, and the job summary."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from pipeline import exporting, hub, manifest, settings

# GitHub rejects release assets of 2 GiB or more.
RELEASE_ASSET_LIMIT = 2 * 1024**3


def _folder(backend: str, target: str | None) -> str:
    return backend if target is None else f"{backend}/{target.lower()}"


def load_report(out_dir: Path, backend: str, target: str | None = None) -> dict:
    return json.loads((out_dir / _folder(backend, target) / "export-report.json").read_text(encoding="utf-8"))


def publish_hf(out_dir: Path, backend: str, target: str | None = None, attempts: int = 6) -> str:
    """Commit this backend's folder plus the shared root files; regenerate README.md.

    Each backend's run publishes on its own, possibly at the same moment as another, so the
    commit is pinned to the revision the README was rendered from and retried on conflict.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    cfg = settings.load()
    report = load_report(out_dir, backend, target)
    repo_id = report["output_repo"]
    folder = _folder(backend, target)
    local_folder = out_dir / folder
    new_paths = {f"{folder}/{p.name}": p for p in local_folder.iterdir() if p.is_file()}
    root_files = {p.name: p for p in out_dir.iterdir() if p.is_file()}
    scoped = [path for path in new_paths if path.rsplit("/", 1)[-1] in exporting.TOKENIZER_FILES]
    if scoped:
        # A tokenizer inside any backend folder makes the app stop lending the root one to
        # every other folder (HuggingFaceClient.tokenizerFor), breaking the other backends.
        raise ValueError(f"tokenizer files must stay at the repo root, found {scoped}")

    hf = hub.api()
    hf.create_repo(repo_id, repo_type="model", exist_ok=True)

    for attempt in range(attempts):
        info = hf.model_info(repo_id, expand=["sha", "siblings"])
        existing = [s.rfilename for s in info.siblings or []]
        reports = [report]
        for path in existing:
            if path.endswith("/export-report.json") and not path.startswith(f"{folder}/"):
                local = hf_hub_download(repo_id, path, revision=info.sha, token=hf.token)
                reports.append(json.loads(Path(local).read_text(encoding="utf-8")))
        license_files = sorted({f for r in reports for f in r["source"].get("license_files", [])})
        readme = manifest.readme(repo_id, reports, list(cfg.hub_tags), license_files)

        operations = [
            CommitOperationDelete(path_in_repo=path)
            for path in existing
            if path.startswith(f"{folder}/") and path not in new_paths
        ]
        operations += [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in new_paths.items()]
        operations += [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in root_files.items()]
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")))
        try:
            commit = hf.create_commit(
                repo_id,
                operations,
                commit_message=f"{folder}: {report['source']['id']}@{report['source']['sha'][:12]}",
                parent_commit=info.sha,
            )
            return commit.commit_url
        except HfHubHTTPError as error:
            status = getattr(error.response, "status_code", None)
            if status != 412 or attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def release_tag(report: dict) -> str:
    name = report["source"]["id"].rsplit("/", 1)[-1]
    target = f"-{report['target']}" if report.get("target") else ""
    return f"{name}-{report['backend']}{target}-{report['source']['sha'][:7]}"


def release_notes(report: dict, hf_url: str | None) -> str:
    lines = [
        f"{manifest.BACKEND_TITLES[report['backend']]} export of "
        f"[{report['source']['id']}](https://huggingface.co/{report['source']['id']}) "
        f"at `{report['source']['sha'][:12]}`, window {report['window']['context']:,} tokens, "
        f"ExecuTorch {report['toolchain']['executorch']}.",
        "",
    ]
    if hf_url:
        lines += [f"Hugging Face: {hf_url}", ""]
    too_big = [f for f in report["files"] if f["bytes"] >= RELEASE_ASSET_LIMIT]
    for f in too_big:
        lines.append(
            f"- `{f['path']}` is {f['bytes']:,} bytes, over GitHub's 2 GiB asset limit: download it from Hugging Face."
        )
    return "\n".join(lines) + "\n"


def publish_release(out_dir: Path, backend: str, target: str | None = None) -> str:
    """Create (or update) a GitHub release with the export's files under the 2 GiB limit."""
    report = load_report(out_dir, backend, target)
    folder = out_dir / _folder(backend, target)
    tag = release_tag(report)
    hf_url = f"https://huggingface.co/{report['output_repo']}"
    candidates = [*sorted(folder.iterdir()), out_dir / report["tokenizer"]]
    assets = [str(p) for p in candidates if p.is_file() and p.stat().st_size < RELEASE_ASSET_LIMIT]
    notes = release_notes(report, hf_url)
    exists = subprocess.run(["gh", "release", "view", tag], capture_output=True).returncode == 0
    if exists:
        subprocess.run(["gh", "release", "upload", tag, "--clobber", *assets], check=True)
        subprocess.run(["gh", "release", "edit", tag, "--notes", notes], check=True)
    else:
        subprocess.run(
            ["gh", "release", "create", tag, "--title", tag, "--notes", notes, *assets],
            check=True,
        )
    return tag


def summary(out_dir: Path, backend: str, target: str | None = None) -> str:
    """Markdown for $GITHUB_STEP_SUMMARY."""
    report = load_report(out_dir, backend, target)
    smoke = report.get("smoke") or {}
    host = report["host"]
    estimates = report.get("estimates") or {}
    rows = [
        ("Source", f"{report['source']['id']} @ `{report['source']['sha'][:12]}`"),
        ("Family / variant", f"{report['source']['family']} / {report['source']['variant']}"),
        ("Parameters", f"{report['source']['total_params']:,}"),
        ("Window", f"{report['window']['context']:,} ({report['window']['reason']})"),
        ("Files", ", ".join(f"`{f['path']}` {f['bytes']:,} B" for f in report["files"])),
        ("Peak RSS export", f"{host.get('peak_rss_export_bytes') or 0:,} B"),
        ("Peak swap in use", f"{host.get('peak_swap_used_bytes') or 0:,} B"),
        ("CPU", host.get("cpu_model") or "?"),
        ("Export time", f"{host.get('export_seconds')} s of {host.get('total_seconds')} s"),
    ]
    if estimates:
        rows += [
            (".pte estimate vs actual", f"{estimates['pte_bytes_estimate']:,} / {estimates['pte_bytes_actual']:,} B"),
            ("Export peak estimate", f"{estimates['export_peak_bytes_estimate']:,} B"),
        ]
    if not smoke:
        rows.append(("Check", "not run"))
    elif smoke.get("kind") == "structural":
        rows.append(
            ("Check", "structure " + ("passed" if smoke["passed"] else "failed: " + "; ".join(smoke["problems"])))
        )
    else:
        rows.append(("Smoke test", f"{'passed' if smoke.get('passed') else 'failed'}: {smoke.get('reply', '')!r}"))
        if smoke.get("template_error"):
            rows.append(("Chat template", f"could not be rendered, completion prompt used: {smoke['template_error']}"))
    title = manifest.BACKEND_TITLES[backend] + (f" {manifest.target(report)}" if report.get("target") else "")
    lines = [f"### {title}: {report['output_repo']}", "", "| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines) + "\n"
