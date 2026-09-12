"""Whether a source model should be exported, and to which backends."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pipeline import families, naming
from pipeline.hub import SourceModel
from pipeline.settings import Settings

# Which app template token each family's names must carry.
FAMILY_APP_TOKENS = {
    "qwen3": {"qwen3"},
    "qwen2_5": {"qwen25"},
    "llama": {"llama32", "smollm2"},
    "gemma3": {"gemma3"},
    "smollm3": {"smollm3"},
}
_INSTRUCT_TOKENS = {"instruct", "it", "chat"}


@dataclass
class Verdict:
    model_id: str
    sha: str
    family: str | None
    variant: str
    # Reasons the model is skipped entirely; empty when it is eligible.
    reasons: list[str] = field(default_factory=list)
    # Backend → None when it will be exported, else why not.
    backends: dict[str, str | None] = field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return not self.reasons and any(v is None for v in self.backends.values())

    @property
    def export_backends(self) -> list[str]:
        return [b for b, why in self.backends.items() if why is None] if not self.reasons else []

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "sha": self.sha,
            "family": self.family,
            "variant": self.variant,
            "eligible": self.eligible,
            "reasons": self.reasons,
            "backends": self.backends,
        }


def variant(model_id: str, family: str | None) -> str:
    tokens = set(re.split(r"[-_.]", naming.source_name(model_id).lower()))
    if tokens & _INSTRUCT_TOKENS:
        return "instruct"
    # Qwen3 ships its chat models without a suffix and marks the others "-Base".
    if family == "qwen3" and "base" not in tokens:
        return "instruct"
    return "base"


def evaluate(source: SourceModel, settings: Settings) -> Verdict:
    name = naming.source_name(source.id)
    family = families.family_for(source.config) if source.config else None
    family_key = family.key if family else None
    verdict = Verdict(source.id, source.sha, family_key, variant(source.id, family_key))
    reasons = verdict.reasons

    if source.pipeline_tag not in (None, settings.pipeline_tag):
        reasons.append(f"pipeline tag {source.pipeline_tag!r} is not {settings.pipeline_tag!r}")
    normalised = naming.normalise(name)
    hits = [token for token in settings.name_exclude if token in normalised]
    if hits:
        reasons.append(f"name matches excluded marker(s) {hits}")
    if source.access_error:
        reasons.append(source.access_error)
    elif not source.config:
        reasons.append("repo has no config.json")
    elif families.is_moe(source.config):
        reasons.append("mixture-of-experts checkpoint")
    elif family is None:
        reasons.append(f"architecture {source.config.get('architectures')} has no recipe")
    nominal = naming.nominal_billions(name)
    if nominal is not None and nominal > settings.max_nominal_billions:
        reasons.append(f"named size {nominal:g}B is above {settings.max_nominal_billions:g}B")
    if source.total_params is None:
        reasons.append("no safetensors parameter count on the Hub")
    elif source.total_params >= settings.max_params:
        reasons.append(f"{source.total_params:,} parameters is not below {settings.max_params:,}")

    output_repo = naming.output_repo(source.id, settings.hub_org, settings.repo_suffix)
    app_token = naming.app_family(naming.app_model_name(output_repo, "x.pte"))
    if app_token is None:
        reasons.append(f"the app has no chat template for {name!r} (or refuses it by name)")
    elif family is not None and app_token not in FAMILY_APP_TOKENS.get(family.key, set()):
        reasons.append(f"name reads as app family {app_token!r}, architecture is {family.key!r}")

    for backend in families.BACKENDS:
        if family is None:
            verdict.backends[backend] = "no family"
        elif not family.supports(backend):
            verdict.backends[backend] = family.unsupported[backend]
        elif backend == "xnnpack":
            try:
                families.xnnpack_plan(family, source.config)
                verdict.backends[backend] = None
            except families.UnsupportedModel as error:
                verdict.backends[backend] = str(error)
        else:
            verdict.backends[backend] = None
    return verdict
