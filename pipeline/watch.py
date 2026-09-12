"""Watch Hugging Face orgs for new models and dispatch the export workflows.

State lives in one JSON file (committed to the ``state`` branch by the workflow):

    {"version": 1, "seeded_at": ..., "models": {"<model id>": {
        "status": "seeded" | "skipped" | "pending" | "dispatched" | "retry",
        "sha": ..., "created_at": ..., "first_seen": ..., "reasons": [...],
        "backends": {"xnnpack": {"status": "pending" | "dispatched" | "skipped", ...}}}}}

The first run records every listed model as ``seeded`` and exports nothing; existing
models are exported by backfill. After that, each model not yet in the state is checked
once: by name first (no network), then against its config. Eligible models get their
backends dispatched, at most ``max_dispatch`` workflow runs per watcher run; the rest stay
``pending`` for the next run.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pipeline import eligibility, hub, naming
from pipeline.settings import Settings

STATE_VERSION = 1
# Backend → the workflow that exports it (one run covers every target chip). MediaTek
# joins in phase 4.
WORKFLOWS = {"xnnpack": "export-xnnpack.yml", "qnn": "export-qnn.yml"}
# A model whose metadata could not be read is retried this many runs before it is skipped.
MAX_ATTEMPTS = 5


@dataclass(frozen=True)
class Listed:
    id: str
    sha: str | None
    created_at: str | None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_org(org: str, limit: int) -> list[Listed]:
    """The org's newest ``limit`` models. Sorted by creation, so older repos never move up."""
    models = hub.api().list_models(author=org, sort="created_at", limit=limit, expand=["createdAt", "sha"])
    return [Listed(m.id, m.sha, m.created_at.isoformat() if m.created_at else None) for m in models]


def gh_dispatch(workflow: str, model_id: str, revision: str) -> None:
    command = ["gh", "workflow", "run", workflow, "-f", f"model_id={model_id}", "-f", f"revision={revision}"]
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        command += ["-R", repo]
    subprocess.run(command, check=True)


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != STATE_VERSION:
        raise ValueError(f"{path} has state version {state.get('version')}, expected {STATE_VERSION}")
    return state


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def org_reason(model_id: str, org: str, settings: Settings) -> str | None:
    """Why a watched org's repo is not one of that org's own families, if it is not."""
    own = settings.org_families.get(org, ())
    token = naming.app_family(naming.source_name(model_id))
    if own and token not in own:
        return f"not one of {org}'s own families ({', '.join(own)})"
    return None


def _evaluate(
    model_id: str,
    revision: str,
    settings: Settings,
    fetch: Callable[[str, str], hub.SourceModel],
    entry: dict,
    org: str | None = None,
) -> None:
    """Fill ``entry`` with the model's verdict: skipped, pending, or retry on a fetch error."""
    reasons = eligibility.name_reasons(model_id, settings)
    if org is not None and not reasons:
        reason = org_reason(model_id, org, settings)
        reasons = [reason] if reason else []
    if reasons:
        entry.update(status="skipped", reasons=reasons, backends={})
        return
    try:
        source = fetch(model_id, revision)
    except Exception as error:  # network or Hub errors: try again next run
        attempts = entry.get("attempts", 0) + 1
        if attempts >= MAX_ATTEMPTS:
            entry.update(status="skipped", reasons=[f"metadata unreadable after {attempts} runs: {error}"])
        else:
            entry.update(status="retry", attempts=attempts, reasons=[f"metadata fetch failed: {error}"])
        return
    verdict = eligibility.evaluate(source, settings)
    entry["sha"] = source.sha
    entry["family"] = verdict.family
    entry["reasons"] = verdict.reasons
    entry["backends"] = {}
    for backend, why in verdict.backends.items():
        if not verdict.reasons and why is None and backend in WORKFLOWS:
            entry["backends"][backend] = {"status": "pending"}
        else:
            entry["backends"][backend] = {"status": "skipped", "reason": why or "no export workflow yet"}
    pending = any(b["status"] == "pending" for b in entry["backends"].values())
    entry["status"] = "pending" if pending else "skipped"
    entry.pop("attempts", None)


def run(
    state_path: Path,
    settings: Settings,
    *,
    dispatch: bool = True,
    backfill: tuple[str, ...] = (),
    limit_per_org: int = 200,
    max_dispatch: int = 6,
    lister: Callable[[str, int], list[Listed]] = list_org,
    fetch: Callable[[str, str], hub.SourceModel] = hub.fetch,
    dispatcher: Callable[[str, str, str], None] = gh_dispatch,
) -> dict:
    """One watcher pass. Returns a summary; the caller saves ``summary["state"]``."""
    state = load_state(state_path)
    stamp = now()
    summary = {"seeded": 0, "new": [], "backfilled": [], "dispatched": [], "state": None}

    seeding = state is None
    if seeding:
        # Everything that exists today is recorded, not exported; a backfill in the same run
        # still exports the models it names.
        state = {"version": STATE_VERSION, "seeded_at": stamp, "models": {}}
        for org in settings.watch_orgs:
            for m in lister(org, limit_per_org):
                state["models"][m.id] = {
                    "status": "seeded",
                    "sha": m.sha,
                    "created_at": m.created_at,
                    "first_seen": stamp,
                }
        summary["seeded"] = len(state["models"])
    models = state["models"]

    for org in settings.watch_orgs if not seeding else ():
        for m in lister(org, limit_per_org):
            entry = models.get(m.id)
            if entry is not None and entry["status"] != "retry":
                continue
            entry = entry or {"first_seen": stamp, "created_at": m.created_at}
            _evaluate(m.id, m.sha or "main", settings, fetch, entry, org=org)
            models[m.id] = entry
            summary["new"].append(m.id)

    for model_id in backfill:
        entry = {"first_seen": models.get(model_id, {}).get("first_seen", stamp), "backfill": stamp}
        _evaluate(model_id, "main", settings, fetch, entry)
        models[model_id] = entry
        summary["backfilled"].append(model_id)

    budget = max_dispatch
    for model_id, entry in sorted(models.items(), key=lambda kv: kv[1].get("first_seen") or ""):
        if entry.get("status") != "pending":
            continue
        for backend, info in entry["backends"].items():
            if info["status"] != "pending" or budget == 0:
                continue
            if dispatch:
                dispatcher(WORKFLOWS[backend], model_id, entry["sha"])
                info.update(status="dispatched", at=stamp)
            summary["dispatched"].append(f"{model_id} -> {backend}")
            budget -= 1
        if all(info["status"] != "pending" for info in entry["backends"].values()):
            entry["status"] = "dispatched"

    summary["state"] = state
    return summary


def markdown(summary: dict) -> str:
    """Job summary for $GITHUB_STEP_SUMMARY."""
    state = summary["state"]
    lines = ["### Hugging Face watcher", ""]
    if summary["seeded"]:
        lines += [
            f"First run: recorded {summary['seeded']} existing models from the watched orgs without "
            "exporting them. Export any of them with the backfill input.",
            "",
        ]
    checked = summary["new"] + summary["backfilled"]
    lines.append(f"Checked {len(checked)} model(s); dispatched {len(summary['dispatched'])} export run(s).")
    if checked:
        lines += ["", "| Model | Result | Why |", "|---|---|---|"]
        for model_id in checked:
            entry = state["models"][model_id]
            why = "; ".join(entry.get("reasons") or [])
            if not why:
                skipped = [f"{b}: {i['reason']}" for b, i in entry.get("backends", {}).items() if i.get("reason")]
                why = "; ".join(skipped)
            lines.append(f"| `{model_id}` | {entry['status']} | {why.replace('|', '/')} |")
    if summary["dispatched"]:
        lines += ["", "Dispatched:", *[f"- {d}" for d in summary["dispatched"]]]
    return "\n".join(lines) + "\n"
