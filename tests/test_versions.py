import re

from pipeline import settings


def pins(path):
    found = {}
    for line in (settings.ROOT / path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line.strip())
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def test_executorch_pin_matches_the_app_runtime_version():
    assert pins("requirements/export-xnnpack.txt")["executorch"] == settings.load().executorch_version


def test_torch_pin_matches_executorch_1_4_0():
    # ExecuTorch v1.4.0 torch_pin.py: TORCH_VERSION = "2.13.0"; install_requirements.py: torchao 0.18.0.
    export = pins("requirements/export-xnnpack.txt")
    assert export["torch"] == "2.13.0"
    assert export["torchao"] == "0.18.0"


def test_every_export_environment_pins_the_same_runtime():
    xnnpack = pins("requirements/export-xnnpack.txt")
    for path in ("requirements/export-qnn.txt", "requirements/mtk-tools.txt"):
        other = pins(path)
        for package in ("executorch", "torch", "torchao"):
            assert other[package] == xnnpack[package], (path, package)


def test_mediatek_tools_keep_transformers_4():
    # examples/mediatek imports transformers.tokenization_utils (4.x), which caps
    # huggingface_hub below 1.0: hence a separate environment from the pipeline's.
    tools = pins("requirements/mtk-tools.txt")
    assert tools["transformers"].startswith("4.")
    assert tools["huggingface_hub"].startswith("0.")
    versions = settings.read_env_file(settings.CONFIG_DIR / "versions.env")
    assert versions["MTK_PYTHON_VERSION"] == "3.10"  # mtk_converter is cp310-only
    assert re.fullmatch(r"[0-9a-f]{64}", versions["NEUROPILOT_SDK_SHA256"])


def test_huggingface_hub_pin_is_shared():
    assert pins("requirements/dev.txt")["huggingface_hub"] == pins("requirements/export-xnnpack.txt")["huggingface_hub"]
