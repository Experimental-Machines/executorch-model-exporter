"""Loads config/pipeline.yaml and config/versions.env."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


@dataclass(frozen=True)
class XnnpackRecipe:
    qmode: str
    group_size: int
    embedding_quantize: str


@dataclass(frozen=True)
class Settings:
    hub_org: str
    repo_suffix: str
    hub_tags: tuple[str, ...]
    watch_orgs: tuple[str, ...]
    org_families: dict[str, tuple[str, ...]]
    limit_per_org: int
    max_dispatch_per_run: int
    pipeline_tag: str
    max_nominal_billions: float
    max_params: int
    name_exclude: tuple[str, ...]
    context_tiers: tuple[int, ...]
    device_budget_bytes: int
    runtime_overhead_bytes: int
    prefill_chunk: int
    xnnpack: XnnpackRecipe
    executorch_version: str


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


@lru_cache(maxsize=1)
def load() -> Settings:
    raw = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text(encoding="utf-8"))
    versions = read_env_file(CONFIG_DIR / "versions.env")
    hub, watch, export = raw["hub"], raw["watch"], raw["export"]
    tiers = tuple(sorted((int(t) for t in export["context_tiers"]), reverse=True))
    return Settings(
        hub_org=hub["org"],
        repo_suffix=hub["repo_suffix"],
        hub_tags=tuple(hub["tags"]),
        watch_orgs=tuple(watch["org_families"]),
        org_families={org: tuple(tokens) for org, tokens in watch["org_families"].items()},
        limit_per_org=int(watch["limit_per_org"]),
        max_dispatch_per_run=int(watch["max_dispatch_per_run"]),
        pipeline_tag=watch["pipeline_tag"],
        max_nominal_billions=float(watch["max_nominal_billions"]),
        max_params=int(watch["max_params"]),
        name_exclude=tuple(s.lower() for s in watch["name_exclude"]),
        context_tiers=tiers,
        device_budget_bytes=int(export["device_budget_bytes"]),
        runtime_overhead_bytes=int(export["runtime_overhead_bytes"]),
        prefill_chunk=int(export["prefill_chunk"]),
        xnnpack=XnnpackRecipe(
            qmode=export["xnnpack"]["qmode"],
            group_size=int(export["xnnpack"]["group_size"]),
            embedding_quantize=str(export["xnnpack"]["embedding_quantize"]),
        ),
        executorch_version=versions["EXECUTORCH_VERSION"],
    )
