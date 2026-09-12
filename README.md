# exe-expo

Exports small open-weight LLMs from Hugging Face to ExecuTorch `.pte` files for the
[openweights](https://github.com/alpharomercoma/openweights) Android app, on standard
GitHub-hosted runners. Design and decisions: [docs/PLAN.md](docs/PLAN.md).

| Backend | Status |
|---|---|
| XNNPACK (CPU) | Qwen3, Qwen2.5, Llama 3.2, SmolLM2 |
| Qualcomm QNN (SM8650, SM8750) | planned (phase 3) |
| MediaTek NeuroPilot (MT6989, MT6991) | planned (phase 4) |
| HF watcher (auto-dispatch) | every 6 hours; state on the `state` branch |

## Running an export

Actions → **Export XNNPACK** → Run workflow, with a model id such as `Qwen/Qwen3-1.7B`.
The run exports, smoke-tests the `.pte` with ExecuTorch's `TextLLMRunner` (the runner the
app uses), uploads an artifact, and publishes to
`experimentalmachines/<model>-ExecuTorch` on Hugging Face and to a GitHub release.

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
