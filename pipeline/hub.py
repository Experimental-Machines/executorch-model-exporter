"""Reading source models from the Hugging Face Hub."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Files a checkpoint download needs. Weights are the only large ones.
WEIGHT_PATTERNS = ["*.safetensors", "model.safetensors.index.json"]
SIDE_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "README.md",
]
LICENSE_PREFIXES = ("license", "use_policy", "notice")


@dataclass
class SourceModel:
    """What the pipeline needs to know about a source repo at one revision."""

    id: str
    sha: str
    created_at: datetime | None
    pipeline_tag: str | None
    tags: list[str]
    gated: str | bool | None
    total_params: int | None
    config: dict
    generation_config: dict = field(default_factory=dict)
    tokenizer_config: dict = field(default_factory=dict)
    card_license: dict = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    # Set when the repo is gated and the token has not been granted access.
    access_error: str | None = None

    @property
    def license_files(self) -> list[str]:
        return [f for f in self.files if "/" not in f and f.lower().startswith(LICENSE_PREFIXES)]


def api(token: str | None = None):
    from huggingface_hub import HfApi

    return HfApi(token=token or os.environ.get("HF_TOKEN") or None)


def _json_file(hf, repo_id: str, revision: str, filename: str, files: list[str]) -> dict:
    from huggingface_hub import hf_hub_download

    if filename not in files:
        return {}
    path = hf_hub_download(repo_id, filename, revision=revision, token=hf.token)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fetch(repo_id: str, revision: str = "main", token: str | None = None) -> SourceModel:
    hf = api(token)
    info = hf.model_info(
        repo_id,
        revision=revision,
        expand=["safetensors", "createdAt", "sha", "gated", "tags", "pipeline_tag", "cardData", "siblings"],
    )
    from huggingface_hub.errors import GatedRepoError

    sha = info.sha
    card = info.card_data.to_dict() if info.card_data else {}
    files = [s.rfilename for s in info.siblings or []]
    source = SourceModel(
        id=info.id,
        sha=sha,
        created_at=info.created_at,
        pipeline_tag=info.pipeline_tag,
        tags=list(info.tags or []),
        gated=info.gated,
        total_params=info.safetensors.total if info.safetensors else None,
        config={},
        card_license={k: card[k] for k in ("license", "license_name", "license_link") if k in card},
        files=files,
    )
    try:
        source.config = _json_file(hf, repo_id, sha, "config.json", files)
        source.generation_config = _json_file(hf, repo_id, sha, "generation_config.json", files)
        source.tokenizer_config = _json_file(hf, repo_id, sha, "tokenizer_config.json", files)
    except GatedRepoError:
        source.access_error = (
            f"gated repo and the HF_TOKEN account has no access: accept the license at "
            f"https://huggingface.co/{repo_id} with that account"
        )
    return source


def download(source: SourceModel, local_dir: Path, token: str | None = None) -> Path:
    """Weights, tokenizer and license files for ``source.sha`` into ``local_dir``."""
    from huggingface_hub import snapshot_download

    patterns = WEIGHT_PATTERNS + SIDE_FILES + source.license_files
    snapshot_download(
        source.id,
        revision=source.sha,
        local_dir=str(local_dir),
        allow_patterns=patterns,
        token=token or os.environ.get("HF_TOKEN") or None,
    )
    return local_dir


def special_token_ids(source: SourceModel) -> tuple[int | None, list[int]]:
    """BOS id and EOS ids, preferring generation_config.json as the runtime stop set."""

    def as_list(value) -> list[int]:
        if value is None:
            return []
        return [int(v) for v in value] if isinstance(value, list) else [int(value)]

    gen, cfg = source.generation_config, source.config
    bos = gen.get("bos_token_id", cfg.get("bos_token_id"))
    eos = as_list(gen.get("eos_token_id")) or as_list(cfg.get("eos_token_id"))
    seen: list[int] = []
    for token in eos:
        if token not in seen:
            seen.append(token)
    return (int(bos) if bos is not None else None), seen
