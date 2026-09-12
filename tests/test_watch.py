import json

import pytest
from conftest import hf_config, make_source

from pipeline import settings, watch

CFG = settings.load()
QWEN3 = hf_config("Qwen/Qwen3-1.7B")


class FakeHub:
    """Listings per org, and the config each model id resolves to."""

    def __init__(self, listings, configs=None, failing=()):
        self.listings = listings
        self.configs = configs or {}
        self.failing = set(failing)
        self.fetched = []
        self.dispatched = []

    def lister(self, org, limit):
        return [watch.Listed(m, f"sha-{m}", "2026-09-01T00:00:00+00:00") for m in self.listings.get(org, [])][:limit]

    def fetch(self, model_id, revision):
        self.fetched.append(model_id)
        if model_id in self.failing:
            raise RuntimeError("503 from the Hub")
        return make_source(model_id, config=self.configs.get(model_id, QWEN3), sha=f"sha-{model_id}")

    def dispatch(self, workflow, model_id, revision):
        self.dispatched.append((workflow, model_id, revision))


def run(tmp_path, hub, **kwargs):
    summary = watch.run(
        tmp_path / "seen.json",
        CFG,
        lister=hub.lister,
        fetch=hub.fetch,
        dispatcher=hub.dispatch,
        **kwargs,
    )
    if kwargs.get("dispatch", True):
        watch.save_state(tmp_path / "seen.json", summary["state"])
    return summary


def test_first_run_seeds_and_exports_nothing(tmp_path):
    hub = FakeHub({"Qwen": ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"], "google": ["google/gemma-3-1b-it"]})
    summary = run(tmp_path, hub)
    assert summary["seeded"] == 3
    assert hub.fetched == [] and hub.dispatched == []
    state = json.loads((tmp_path / "seen.json").read_text())
    assert {e["status"] for e in state["models"].values()} == {"seeded"}


def test_new_models_are_checked_once_and_dispatched(tmp_path):
    hub = FakeHub({"Qwen": ["Qwen/Qwen3-0.6B"]})
    run(tmp_path, hub)  # seed
    hub.listings["Qwen"] = [
        "Qwen/Qwen3-1.7B-Instruct-2609",
        "Qwen/Qwen3-1.7B-FP8",
        "Qwen/Qwen3.8-1B",
        "Qwen/Qwen3-0.6B",
    ]
    summary = run(tmp_path, hub)
    models = summary["state"]["models"]
    # Excluded by name: never fetched.
    assert "Qwen/Qwen3-1.7B-FP8" not in hub.fetched
    assert models["Qwen/Qwen3-1.7B-FP8"]["status"] == "skipped"
    # Same architecture class, later generation: refused by the family name pattern.
    assert models["Qwen/Qwen3.8-1B"]["status"] == "skipped"
    assert "not named like a qwen3 release" in models["Qwen/Qwen3.8-1B"]["reasons"][0]
    # The eligible one goes to XNNPACK at the revision the watcher saw.
    assert hub.dispatched == [
        ("export-xnnpack.yml", "Qwen/Qwen3-1.7B-Instruct-2609", "sha-Qwen/Qwen3-1.7B-Instruct-2609")
    ]
    entry = models["Qwen/Qwen3-1.7B-Instruct-2609"]
    assert entry["status"] == "dispatched"
    assert entry["backends"]["xnnpack"]["status"] == "dispatched"
    assert entry["backends"]["qnn"]["status"] == "skipped"
    # Seeded models are not re-checked.
    assert "Qwen/Qwen3-0.6B" not in summary["new"]
    # A second pass with nothing new does nothing.
    hub.dispatched.clear()
    assert run(tmp_path, hub)["new"] == [] and hub.dispatched == []


def test_orgs_are_watched_for_their_own_families_only(tmp_path):
    # Seen in a real dry run: HuggingFaceTB's GSM8K fine-tune of Qwen3 passes every other rule.
    hub = FakeHub({"HuggingFaceTB": []})
    run(tmp_path, hub)
    hub.listings["HuggingFaceTB"] = ["HuggingFaceTB/qwen3-1.7b-gsm8k-sft"]
    entry = run(tmp_path, hub)["state"]["models"]["HuggingFaceTB/qwen3-1.7b-gsm8k-sft"]
    assert entry["status"] == "skipped"
    assert entry["reasons"] == ["not one of HuggingFaceTB's own families (smollm2, smollm3)"]
    assert hub.fetched == [] and hub.dispatched == []
    # Asking for it explicitly still works.
    run(tmp_path, hub, backfill=("HuggingFaceTB/qwen3-1.7b-gsm8k-sft",))
    assert [d[1] for d in hub.dispatched] == ["HuggingFaceTB/qwen3-1.7b-gsm8k-sft"]


def test_dispatch_budget_carries_over(tmp_path):
    hub = FakeHub({"Qwen": []})
    run(tmp_path, hub)
    hub.listings["Qwen"] = ["Qwen/Qwen3-0.6B-A", "Qwen/Qwen3-0.6B-B", "Qwen/Qwen3-0.6B-C"]
    run(tmp_path, hub, max_dispatch=2)
    assert len(hub.dispatched) == 2
    summary = run(tmp_path, hub, max_dispatch=2)
    assert len(hub.dispatched) == 3
    assert {e["status"] for e in summary["state"]["models"].values()} == {"dispatched"}


def test_fetch_errors_are_retried_then_given_up(tmp_path):
    hub = FakeHub({"Qwen": []}, failing={"Qwen/Qwen3-0.6B-X"})
    run(tmp_path, hub)
    hub.listings["Qwen"] = ["Qwen/Qwen3-0.6B-X"]
    for attempt in range(1, watch.MAX_ATTEMPTS):
        entry = run(tmp_path, hub)["state"]["models"]["Qwen/Qwen3-0.6B-X"]
        assert (entry["status"], entry["attempts"]) == ("retry", attempt)
    entry = run(tmp_path, hub)["state"]["models"]["Qwen/Qwen3-0.6B-X"]
    assert entry["status"] == "skipped"
    assert len(hub.fetched) == watch.MAX_ATTEMPTS


def test_backfill_exports_a_seeded_model(tmp_path):
    hub = FakeHub({"Qwen": ["Qwen/Qwen3-1.7B"]})
    run(tmp_path, hub)
    summary = run(tmp_path, hub, backfill=("Qwen/Qwen3-1.7B",))
    assert summary["backfilled"] == ["Qwen/Qwen3-1.7B"]
    assert hub.dispatched == [("export-xnnpack.yml", "Qwen/Qwen3-1.7B", "sha-Qwen/Qwen3-1.7B")]


def test_backfill_on_the_first_run_still_seeds_everything_else(tmp_path):
    hub = FakeHub({"Qwen": ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"]})
    summary = run(tmp_path, hub, backfill=("Qwen/Qwen3-1.7B",))
    assert summary["seeded"] == 2
    assert [d[1] for d in hub.dispatched] == ["Qwen/Qwen3-1.7B"]
    assert summary["state"]["models"]["Qwen/Qwen3-0.6B"]["status"] == "seeded"


def test_dry_run_dispatches_nothing(tmp_path):
    hub = FakeHub({"Qwen": []})
    run(tmp_path, hub)
    hub.listings["Qwen"] = ["Qwen/Qwen3-0.6B-A"]
    summary = run(tmp_path, hub, dispatch=False)
    assert hub.dispatched == []
    assert summary["dispatched"] == ["Qwen/Qwen3-0.6B-A -> xnnpack"]
    assert "Checked 1 model(s)" in watch.markdown(summary)


def test_state_version_is_checked(tmp_path):
    (tmp_path / "seen.json").write_text(json.dumps({"version": 99, "models": {}}))
    with pytest.raises(ValueError, match="state version"):
        watch.load_state(tmp_path / "seen.json")
