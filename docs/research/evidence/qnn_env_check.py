"""Lower a tiny model to QNN HTP in a child process, without and with pipeline.export_qnn.qnn_env."""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/repo")
from pipeline import export_qnn  # noqa: E402

CHILD = """
import torch
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec, generate_qnn_executorch_compiler_spec, to_edge_transform_and_lower_to_qnn)
m = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU()).eval()
spec = generate_qnn_executorch_compiler_spec(
    soc_model=QcomChipset.SM8650, backend_options=generate_htp_compiler_spec(use_fp16=True))
prog = to_edge_transform_and_lower_to_qnn(m, (torch.randn(1, 16),), spec).to_executorch()
print("LOWERED", len(prog.buffer), b"QnnBackend" in prog.buffer, flush=True)
"""

sdk = export_qnn.qnn_sdk()
print("SDK", sdk, flush=True)

print("=== fresh child, inherited environment (run 5's setup)", flush=True)
plain = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True)
print("exit", plain.returncode)
print("\n".join(l for l in (plain.stdout + plain.stderr).splitlines() if "LOWERED" in l or "QnnSystem" in l or "Error" in l)[-2000:])

print("=== child with qnn_env", flush=True)
env = export_qnn.qnn_env(sdk, Path("/tmp/qnn-libs"), dict(os.environ))
print("LD_LIBRARY_PATH", env["LD_LIBRARY_PATH"], flush=True)
fixed = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True)
print("exit", fixed.returncode)
out = fixed.stdout + fixed.stderr
print("\n".join(l for l in out.splitlines() if "LOWERED" in l or "ERROR" in l or "Error" in l)[-3000:])
if fixed.returncode != 0:
    print(out[-5000:])
