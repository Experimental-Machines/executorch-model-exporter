"""Helpers every backend's export uses: host facts, hashing, side files, report pieces."""

from __future__ import annotations

import hashlib
import os
import shutil
from importlib import metadata as pkg_metadata
from pathlib import Path

from pipeline import hub, manifest, naming

TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model")


class ExportError(Exception):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def host_info() -> dict:
    info = {"nproc": os.cpu_count()}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        fields = {}
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            fields[key] = int(rest.split()[0]) * 1024
        info["mem_total_bytes"] = fields.get("MemTotal")
        info["swap_total_bytes"] = fields.get("SwapTotal")
    return info


def host_budget(info: dict, reserve: int = 1_000_000_000) -> int | None:
    if info.get("mem_total_bytes") is None:
        return None
    return info["mem_total_bytes"] + (info.get("swap_total_bytes") or 0) - reserve


def children_peak_rss() -> int | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    # ru_maxrss is in kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024


def self_peak_rss() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def toolchain(*extra: str) -> dict:
    versions = {}
    for package in ("executorch", "torch", "torchao", *extra):
        try:
            versions[package] = pkg_metadata.version(package)
        except pkg_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def copy_side_files(source: hub.SourceModel, src_dir: Path, out_dir: Path) -> tuple[str, list[str]]:
    """Tokenizer, license files and NOTICE into the repo root; returns (tokenizer, licenses).

    The tokenizer only ever goes to the root: if any backend folder carried its own, the
    app would stop lending the root one to the other folders (HuggingFaceClient.tokenizerFor).
    """
    for name in TOKENIZER_FILES:
        if (src_dir / name).exists():
            shutil.copy2(src_dir / name, out_dir / name)
            tokenizer = name
            break
    else:
        raise ExportError("source repo has neither tokenizer.json nor tokenizer.model")
    licenses = []
    for name in source.license_files:
        if (src_dir / name).exists():
            shutil.copy2(src_dir / name, out_dir / name)
            licenses.append(name)
    token = naming.app_family(naming.source_name(source.id))
    if token in manifest.NOTICES:
        (out_dir / "NOTICE").write_text(manifest.NOTICES[token], encoding="utf-8")
    return tokenizer, licenses


def source_report(source: hub.SourceModel, family: str | None, variant: str, licenses: list[str]) -> dict:
    return {
        "id": source.id,
        "sha": source.sha,
        "created_at": source.created_at.isoformat() if source.created_at else None,
        "total_params": source.total_params,
        "family": family,
        "variant": variant,
        "license": source.card_license,
        "license_files": licenses,
    }
