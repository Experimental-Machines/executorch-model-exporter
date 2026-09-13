# exe-expo

Exports small open-weight LLMs from Hugging Face to ExecuTorch `.pte` files for the
[openweights](https://github.com/alpharomercoma/openweights) Android app, on standard
GitHub-hosted runners. Design and decisions: [docs/PLAN.md](docs/PLAN.md).

| Backend | Status |
|---|---|
| XNNPACK (CPU) | Qwen3, Qwen2.5, Llama 3.2, SmolLM2 |
| Qualcomm QNN (SM8650, SM8750) | checkpoints in ExecuTorch 1.4.0's Qualcomm registry (Qwen3, Qwen2.5 base, Gemma 3 1B, SmolLM2 135M, SmolLM3 3B, Llama 3.2) |
| MediaTek NeuroPilot (MT6989, MT6991) | Qwen3, Qwen2.5 (phase 4, first export pending); Llama 3.2 and Gemma 3 not validated yet |
| HF watcher (auto-dispatch) | every 6 hours; state on the `state` branch |

## Running an export

Actions → **Export XNNPACK** (or **Export QNN**, **Export MediaTek**) → Run workflow, with a
model id such as `Qwen/Qwen3-1.7B`. The run exports, checks the result (XNNPACK: a smoke
test with ExecuTorch's `TextLLMRunner`, the runner the app uses; NPU backends: a structural
check, as there is no host NPU runtime), uploads an artifact, and publishes to
`experimentalmachines/<model>-ExecuTorch` on Hugging Face and to a GitHub release.

The QNN and MediaTek workflows download Qualcomm's and MediaTek's SDKs from their publishers
on each run, which accepts their license terms: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Repository secrets:

- `HF_TOKEN`: a Hugging Face token with write access to the output org
  (`hub.org` in [config/pipeline.yaml](config/pipeline.yaml)). Its account must have
  accepted the licenses of gated source models (Llama, Gemma).

**Watch Hugging Face** runs every 6 hours. Its first run records the models that already
exist without exporting them; to export existing ones, run it with `backfill` set to a
comma-separated list of model ids (or run **Export XNNPACK** directly). `dry_run` reports
what it would do without dispatching or saving state.

**Probe runner** exports one small model at a list of windows (default 2k and 16k) without publishing, to
measure what the runner really has and calibrate the sizing estimates.

## Locally

```sh
pip install -r requirements/dev.txt
python -m pipeline plan Qwen/Qwen3-1.7B        # eligibility and window choice, no download
pytest
```

A full export needs Linux x86_64 (the executorch wheel's LLM runner is not built for
Windows); see the install steps in
[.github/actions/setup-export/action.yml](.github/actions/setup-export/action.yml), then
`python -m pipeline export-xnnpack <model> --out out --work work`.

## Layout

- `config/pipeline.yaml`: output org, watched orgs, size limits, context tiers, phone memory budget, quantization recipe.
- `config/versions.env`: ExecuTorch release every backend must match (the app's runtime).
- `pipeline/`: eligibility, family recipes, checkpoint conversion, export, smoke test, publishing.
- `tests/`: unit tests; `tests/fixtures` holds real HF `config.json` files and ExecuTorch's own params files.
