import pytest
from conftest import make_source

from pipeline import export_xnnpack, settings


def test_a_forced_window_over_the_phone_budget_is_refused_before_any_download(tmp_path, monkeypatch):
    cfg = settings.load()
    source = make_source("Qwen/Qwen3-1.7B")
    monkeypatch.setattr(export_xnnpack.hub, "fetch", lambda model_id, revision: source)
    monkeypatch.setattr(export_xnnpack.hub, "download", lambda *a, **k: pytest.fail("downloaded"))
    # Qwen3-1.7B at 16k is over the 5 GB phone budget (PLAN: auto-fit picks 8k).
    assert cfg.device_budget_bytes == 5_000_000_000
    with pytest.raises(export_xnnpack.ExportError, match="resident on the phone"):
        export_xnnpack.run("Qwen/Qwen3-1.7B", "main", tmp_path / "out", tmp_path / "work", context=16384)
    # Far past the host too: the phone check fires first and says so.
    with pytest.raises(export_xnnpack.ExportError, match="resident on the phone"):
        export_xnnpack.run("Qwen/Qwen3-1.7B", "main", tmp_path / "out", tmp_path / "work", context=32768)
