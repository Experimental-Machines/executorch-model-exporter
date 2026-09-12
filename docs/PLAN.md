# exe-expo: Hugging Face → ExecuTorch export pipeline

Watches Hugging Face for new small dense LLMs and exports them to ExecuTorch `.pte` for
the [openweights](https://github.com/alpharomercoma/openweights) Android app: XNNPACK
(CPU), Qualcomm QNN (Snapdragon HTP) and MediaTek NeuroPilot (Dimensity APU).

## Decisions

| Topic | Decision |
|---|---|
| Models | Dense (no MoE) text LLMs of the 4B class and smaller (size in the name ≤ 4B; real count < 4.5B, since Qwen3-4B is 4,022,468,096), instruct and base, original bf16/fp16 weights only |
| Watched orgs | `Qwen`, `google`, `meta-llama`, `HuggingFaceTB` |
| Trigger | Scheduled watcher; a new eligible model auto-dispatches all three backend workflows |
| First run | Seeds state without exporting; existing models are backfilled by manual dispatch |
| Runners | Standard GitHub-hosted `ubuntu-latest` (public repo: 4 vCPU, 16 GB RAM), one workflow run per backend |
| Chips | QNN: SM8650 (8 Gen 3), SM8750 (8 Elite). MediaTek: MT6989 (D9300), MT6991 (D9400) |
| Outputs | Hugging Face Hub, GitHub Releases (files ≤ 2 GiB), Actions artifacts |
| HF layout | One repo per model, backend folders; NPU exports published even though the app can't load them yet |
| Context window | Auto-fit per model: largest of 32k/16k/8k/4k/2k that fits both the phone budget and the runner |
| Runtime pin | ExecuTorch **1.4.0** everywhere; QAIRT **2.37.0** (what `executorch-android-qnn:1.4.0` depends on) |
| Vendor SDKs | Downloaded from the vendor at run time, cached only in this repo's Actions cache, never re-hosted |

## Hugging Face repo layout

```
experimentalmachines/Qwen3-1.7B-ExecuTorch
├── README.md                 # generated from every backend's export-report.json
├── LICENSE…                  # upstream license files, copied verbatim
├── tokenizer.json            # root only: nested weights borrow it (HuggingFaceClient.kt)
├── xnnpack/Qwen3-1.7B-8da4w-8k.pte, config.json, export-report.json
├── qnn/sm8650/…, qnn/sm8750/…
└── mtk/mt6989/…, mtk/mt6991/…
```

Constraints taken from the app source:

- Repo carries the `executorch` tag (Discover searches `filter=executorch`).
- Size stays in the repo name (`1.7B`): the size filter reads it.
- Repo name must not contain `xnnpack`: `CompiledBackend.of(repoId + path)` checks `xnnpack`
  first, which would make every QNN/MTK file in the repo read as XNNPACK.
- Normalised `repo name + file stem` must contain a family token the app has a template
  for (`qwen3`, `qwen25`, `smollm2`, `smollm3`, `llama32`, `phi4mini`, `gemma3`, `lfm25`)
  and none of `vl`, `vision`, `coder`, `guard`, `qwen35`. `tests/test_naming.py` ports
  these rules and checks every generated name.
- `config.json` next to each `.pte`, in the `variants[].methods` form Discover reads.

## Context auto-fit

ExecuTorch fixes the window at export and allocates the full fp32 KV cache at load. The
openweights window matrix found Qwen3-1.7B at 32k resident at 6.1-8.5 GB and killed by
Samsung's and MIUI's 6 GB memory guards, so the window is fitted to a phone budget, not
just to what the runner can export:

- KV cache bytes = `n_layers × 2 × n_kv_heads × head_dim × window × 4`.
  Qwen3-1.7B at 32k: 28 × 2 × 8 × 128 × 32,768 × 4 = 7,516,192,768 bytes.
- Resident on the phone ≈ `.pte` + KV cache + 0.5 GB (measured in the app's research).
- `.pte` estimate for 8da4w/g32 + int8 embeddings ≈ `embedding params × 1 B + linear
  params × 0.625 B` (tied embeddings add a separate 4-bit output projection).
- Pick the largest tier with resident ≤ `device_budget_bytes` (default 5.0 GB) and
  estimated export peak ≤ runner RAM + swap.

## Backends

### XNNPACK (phase 1)

`export_llm` from the pip wheel, the recipe the app was measured with: 8-bit dynamic
activations and 4-bit weights in groups of 32, int8 per-channel embeddings, XNNPACK with
extended ops, prefill chunk 2048, fp32 KV cache, the family's BOS/EOS ids in metadata.
`export_llm`'s `model_class` only chooses the example directory; the params file defines
the architecture, so it is generated from the HF `config.json`, and the HF safetensors are
converted to ExecuTorch's checkpoint layout by `pipeline/convert.py`.

| Family | HF architecture | XNNPACK in 1.4.0 |
|---|---|---|
| Qwen3 | `Qwen3ForCausalLM` | yes |
| Qwen2.5 | `Qwen2ForCausalLM` | yes |
| Llama 3.2, SmolLM2 | `LlamaForCausalLM` | yes (q/k un-permuted for Meta RoPE) |
| Gemma 3, SmolLM3 | `Gemma3ForCausalLM`, `SmolLM3ForCausalLM` | not yet: not in `export_llm`'s model list; needs a validated path (optimum-executorch or params support) |

Smoke test: the Linux wheel ships `TextLLMRunner`, the same C++ runner the app calls
through `LlmModule`. Greedy generation of a short prompt must produce non-degenerate text
(and "Paris" for models ≥ 500M parameters); the `.pte`'s metadata methods are read back
and written to `config.json`. The wheel's runner only links portable and XNNPACK kernels,
so `portable_lib`, `custom_ops` and `kernels.quantized` are imported first to register
`llama::custom_sdpa`, `update_cache` and `embedding_byte` (the app's AAR links them).

First end-to-end run (local Docker, 8 GB, SmolLM2-135M-Instruct at 2k): 106,018,048-byte
`.pte` (estimate 112,383,432), export peak RSS 2,748,440,576 B, 31 min of mostly
single-threaded lowering, reply "The capital of France is Paris." Gated repos the token
cannot read are reported as a skip reason, not a crash.

### QNN (phase 3)

QAIRT 2.37.0 + ExecuTorch v1.4.0 built from source with QNN host bindings, cached per
version. `examples/qualcomm/oss_scripts/llama/llama.py --compile_only --model_mode hybrid`
per chip. Only models with a `--decoder_model` entry in ExecuTorch 1.4.0 are exportable.

### MediaTek (phase 4)

Python 3.10 (the `mtk_converter` wheel is cp310), NeuroPilot SDK from MediaTek's URL (the
one ExecuTorch's own CI uses), `examples/mediatek` export scripts per chip. Before any
MediaTek code: read the license bundled in the SDK archive and write
`THIRD_PARTY_NOTICES.md` and the model-card attribution from its exact text.

## Phases

0. **Probe** (`probe-runner.yml`, written, not yet run on GitHub): measure the runner's
   real RAM/disk/swap; export Qwen3-0.6B at 2k and 32k to calibrate the export-memory
   estimate and the `.pte` size estimate.
1. **XNNPACK end to end** (`export-xnnpack.yml`, validated locally in Docker; publishing
   not yet exercised): manual dispatch → export → smoke test → HF + release + artifact.
2. **Watcher**: cron, state on a `state` branch, seeding, auto-dispatch.
3. **QNN.**
4. **MediaTek.**

## Known limits

- QNN and MediaTek only export models ExecuTorch has hard-coded; a new family waits for an
  ExecuTorch release, and the app's AAR must move with it.
- QNN/MediaTek exports of 3-4B models may run out of memory or hit the 6-hour job limit
  on 16 GB runners; they fail independently of the other backends.
- NPU context windows are fixed at compile time and will be far smaller than 32k.
- Upstream licenses differ (e.g. Qwen2.5-3B is under the Qwen Research License); cards and
  `LICENSE` files copy upstream's terms exactly.
