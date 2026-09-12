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


def test_huggingface_hub_pin_is_shared():
    assert pins("requirements/dev.txt")["huggingface_hub"] == pins("requirements/export-xnnpack.txt")["huggingface_hub"]
