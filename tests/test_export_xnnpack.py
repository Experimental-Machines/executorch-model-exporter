import pytest
from conftest import make_source

from pipeline import export_xnnpack, exporting, settings


class Downloaded(Exception):
    """Raised by the fake download: the run passed every pre-download check."""


@pytest.fixture
def qwen3_1_7b(monkeypatch):
    source = make_source("Qwen/Qwen3-1.7B")
    monkeypatch.setattr(export_xnnpack.hub, "fetch", lambda model_id, revision: source)
    monkeypatch.setattr(export_xnnpack.hub, "download", lambda *a, **k: (_ for _ in ()).throw(Downloaded()))
    # A 16.8 GB + 24 GiB hosted runner.
    monkeypatch.setattr(
        export_xnnpack,
        "host_info",
        lambda: {"nproc": 4, "mem_total_bytes": 16_766_414_848, "swap_total_bytes": 25_769_803_776},
    )
    return source


def run(tmp_path, context):
    return export_xnnpack.run("Qwen/Qwen3-1.7B", "main", tmp_path / "out", tmp_path / "work", context=context)


def test_windows_over_the_phone_budget_are_exported_anyway(tmp_path, qwen3_1_7b):
    # Qwen3-1.7B at 16k needs about 5.6 GB resident (over the 5 GB phone budget): exported,
    # the app and the benchmarker decide what fits (PLAN, "Context auto-fit").
    assert settings.load().device_budget_bytes == 5_000_000_000
    with pytest.raises(Downloaded):
        run(tmp_path, 16384)


def test_a_window_the_runner_cannot_build_is_a_skip_not_a_failure(tmp_path, qwen3_1_7b):
    # 28 layers x 32k^2 causal masks: 30 GB of masks alone, more than RAM + swap.
    with pytest.raises(exporting.SkipExport, match="causal masks alone"):
        run(tmp_path, 32768)
    assert issubclass(exporting.SkipExport, exporting.ExportError)


def test_no_window_given_means_the_largest_the_host_can_build(tmp_path, qwen3_1_7b, capsys):
    with pytest.raises(Downloaded):
        run(tmp_path, None)
    assert "window 16384" in capsys.readouterr().out
