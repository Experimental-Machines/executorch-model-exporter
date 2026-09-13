# exe-expo: Hugging Face → ExecuTorch export pipeline

Watches Hugging Face for new small dense LLMs and exports them to ExecuTorch `.pte` for
the [openweights](https://github.com/alpharomercoma/openweights) Android app: XNNPACK
(CPU), Qualcomm QNN (Snapdragon HTP) and MediaTek NeuroPilot (Dimensity APU).

## Decisions

| Topic | Decision |
|---|---|
| Models | Dense (no MoE) text LLMs of the 4B class and smaller (size in the name ≤ 4B; real count < 4.5B, since Qwen3-4B is 4,022,468,096), instruct and base, original bf16/fp16 weights only |
| Watched orgs | `Qwen`, `google`, `meta-llama`, `HuggingFaceTB`, each for its own families |
| Trigger | Scheduled watcher; a new eligible model auto-dispatches every backend workflow that supports it |
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
- `.pte` estimate for 8da4w/g32 + int8 embeddings = (`embedding params × 1 B + linear
  params × 0.5625 B + window × head_dim × 16 B` of RoPE tables) × 1.01, with linear
  params counted from the architecture (the output projection always gets its own 4-bit
  copy). Within 1% of every measured file.
- Export peak ≈ fp32 weights + KV cache + `n_layers × window²` bytes of causal masks +
  2.5 GB.
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

No source build: the executorch 1.4.0 Linux x86_64 wheel ships the Qualcomm backend
(`PyQnnManagerAdaptor`, `libqnn_executorch_backend.so`) and downloads QAIRT 2.37.0.250724
plus a libc++ into `~/.cache/executorch/qnn` on first import; the workflow caches that
directory per QAIRT version. `export-qnn.yml` runs one job per chip in `qnn.socs`, each
calling ExecuTorch's own script in compile-only mode:

`python -m executorch.examples.qualcomm.oss_scripts.llama.llama --decoder_model <key>
--soc_model <SoC> --compile_only --model_mode hybrid --max_seq_len 2048 --max_context_len
2048 --prefill_ar_len 128 --calib_tasks wikitext --calib_limit 1`

- Only checkpoints in the script's registry (`SUPPORTED_LLM_MODELS`) can be exported; from
  the watched orgs that is Qwen3-0.6B/1.7B, Qwen2.5-0.5B/1.5B (base), gemma-3-1b-it,
  SmolLM2-135M-Instruct, SmolLM3-3B and Llama-3.2-1B/3B-Instruct (`families.QNN_DECODERS`).
  Gemma 3 and SmolLM3 are QNN-only for now (no XNNPACK path yet).
- Each registry entry carries its own quantization recipe; the script downloads the weights
  from the entry's repo at `main`. Llama 3.2 entries have no repo, so Meta's original
  checkpoint, params and tokenizer (`original/` in meta-llama's repos) are passed in.
- The other entries point at params `.json` files in ExecuTorch's source tree, which the
  wheel does not package; copies from v1.4.0 live in `third_party/executorch/` and are passed
  with `--params` (`families.QNN_PARAMS`). Tests check them against the HF configs.
- The wheel's import-time SDK setup edits `LD_LIBRARY_PATH` after the loader has read it, so
  QNN cannot open `libQnnSystem.so` by name. The script is started with `QNN_SDK_ROOT` and
  `LD_LIBRARY_PATH` (SDK libs + soname links to the wheel's libc++) already set
  (`export_qnn.qnn_env`), ExecuTorch's documented manual setup.
- Calibration dependencies are ExecuTorch's example pins: `transformers==5.0.0rc1`,
  `datasets==3.6.0`, `lm_eval==0.4.5` (`requirements/export-qnn.txt`).
- No host runtime for HTP binaries, so the check is structural: the program loads, has
  `prefill_forward` and `kv_forward`, and delegates to `QnnBackend`.
- The window is fixed at compile time: `qnn.max_context_len` (2048 to start), or the
  workflow's `context` input (`--context`) for one run.
- QAIRT's license: `THIRD_PARTY_NOTICES.md`.

### MediaTek (phase 4)

`export-mtk.yml` runs one job per chip in `mtk.socs` (MT6989 = DX3, MT6991 = DX4):

- **SDK:** NeuroPilot Express build 20250327, the archive ExecuTorch 1.4.0's CI installs,
  downloaded from MediaTek on every run, checked against `NEUROPILOT_SDK_SHA256`, and
  deleted once `mtk_converter` 8.13.0 and `mtk_neuron` 8.2.19 are installed. Its license
  (read 2026-09-13, accepted for ExperimentalMachines) is summarised in
  `THIRD_PARTY_NOTICES.md`; the agreement is marked MediaTek Confidential, so it is
  paraphrased, not copied.
- **Two Python environments:** `mtk_converter` is cp310-only and `examples/mediatek`
  imports transformers 4.x internals, which cap `huggingface_hub` below the 1.x the pipeline
  uses. The pipeline keeps its own environment (`requirements/export-mtk.txt`) and drives a
  Python 3.10 virtualenv (`requirements/mtk-tools.txt` + MediaTek's wheels) as `MTK_PYTHON`.
- **Scripts:** the LLM export scripts are in ExecuTorch's source, not the wheel; the
  workflow sparse-checks-out `examples/mediatek` at `EXECUTORCH_COMMIT`.
- **Recipe** (MediaTek's own, `shell_scripts/export_qwen.sh`): A16W4, the model cut into up
  to 4 chunks of equal layer counts, a 128-token prompt graph and a one-token generation
  graph over a 512-token cache, calibrated on MediaTek's `alpaca.txt` prompts (8, up to 9
  generated tokens each) in the family's chat template. The cache is MediaTek's default
  because calibration keeps a full fp32 KV cache per prompt and step: the first run, at
  2048, exhausted the runner's 16.8 GB + 24 GB swap while preparing calibration inputs
  (estimate 75 GB, `export_mtk.calibration_bytes`, now checked before any download).
- **Families:** the scripts build the model from `config.json`'s `model_type`, so any Qwen3
  or Qwen2.5 size works, not a fixed list. Llama 3.2 (the scripts read
  `rope_scaling['type']`, its config has `rope_type: llama3`), SmolLM2 (tokenizer class) and
  Gemma 3 (`gemma3_text` vs `gemma3`) wait for validation.
- **Output** per chip: the chunk `.pte` files, the fp32 token embedding table the runner
  reads from disk, and `config.json` with the flags MediaTek's LLM runner
  (`examples/mediatek/executor_runner`) needs. They do not run on `TextLLMRunner`.
- **Check:** structural; every chunk loads, has its two methods, and delegates to
  `NeuropilotBackend`.
- The watcher does not dispatch `export-mtk.yml` until its first export passes.

## Phases

0. **Probe** (`probe-runner.yml`, done 2026-09-12): `ubuntu-latest` on this public repo is
   4 vCPU (AMD EPYC 9V74), 16,766,414,848 B RAM, one 160 GB NVMe (~103 GB free after
   cleanup, no `/mnt`), plus the 24 GiB swap file. Qwen3-0.6B exports:

   | Window | `.pte` | Peak RSS | Export | Smoke |
   |---|---|---|---|---|
   | 2k | 496,570,368 B | 5,836,587,008 B | 600 s | "Paris", 83 tok/s decode |
   | 16k | 525,932,032 B | 15,781,117,952 B | 721 s | "Paris", 83 tok/s decode |
   | 32k | — | runner killed (causal masks) | — | — |

   `pipeline/sizing.py` is calibrated on these: estimates within 1% of every measured
   `.pte` and 4-7% above the measured export peaks.
1. **XNNPACK end to end** (`export-xnnpack.yml`, done 2026-09-12): first publish
   [experimentalmachines/Qwen3-0.6B-ExecuTorch](https://huggingface.co/experimentalmachines/Qwen3-0.6B-ExecuTorch)
   (16k window, smoke test "Paris") and GitHub release `Qwen3-0.6B-xnnpack-c1899de`.
2. **Watcher** (`watch-hf.yml`, every 6 hours at :17, plus manual dispatch with `backfill`
   and `dry_run` inputs). Lists each org's 200 newest repos; a repo not yet in
   `seen.json` (on the `state` branch) is checked once, by name first and then by config,
   and each org only for its own families (`org_families`). Eligible models go to
   `export-xnnpack.yml` at the revision seen, at most 6 runs per watcher run. A dry run over
   the 40 newest real repos (2026-09-12) skipped all of them correctly: Qwen3.8 multimodal
   and FP8, Qwen3-ASR, Llama 4, Llama Guard, SmolLM3 (no XNNPACK recipe yet), and
   HuggingFaceTB's GSM8K fine-tune of Qwen3 (not HuggingFaceTB's own family).
3. **QNN** (`export-qnn.yml`; the watcher dispatches it for registry checkpoints). First
   publish 2026-09-12: Qwen3-0.6B at 2k into the same repo (`qnn/sm8650/`, `qnn/sm8750/`)
   and releases `Qwen3-0.6B-qnn-<chip>-c1899de`, both passing the structural check:

   | Chip | `.pte` | Peak RSS | Export |
   |---|---|---|---|
   | SM8650 | 665,070,592 B | 15,657,545,728 B | 3,637 s (calibration + quantize 2,042 s, compile 1,538 s) |
   | SM8750 | 664,296,448 B | 15,644,954,624 B | 2,434 s |

   A 4k probe (`context` input, SM8750, not published) decides whether 4k becomes the
   default window.
4. **MediaTek** (`export-mtk.yml`, built 2026-09-13; first export pending).

## Known limits

- Export memory grows with the square of the window: every attention layer of ExecuTorch
  1.4.0's transformer builds its own window × window causal mask (not stored in the
  `.pte`). Qwen3-0.6B at 32k needs 30,064,771,072 bytes of masks alone and killed a
  16.8 GB + 24 GB swap runner; `sizing.export_peak_bytes` counts it and a forced window
  that cannot fit is refused.

- QNN and MediaTek only export models ExecuTorch has hard-coded; a new family waits for an
  ExecuTorch release, and the app's AAR must move with it.
- QNN/MediaTek exports of 3-4B models may run out of memory or hit the 6-hour job limit
  on 16 GB runners; they fail independently of the other backends.
- NPU context windows are fixed at compile time and will be far smaller than 32k. Every
  decode step attends over the whole window, so a larger one slows every token, and the
  16-bit KV cache costs 114,688 B per token for Qwen3-0.6B (28 layers × K,V × 8 heads × 128
  × 2 B: 234,881,024 B at 2k).
- QNN `.pte` metadata carries `get_bos_id` 1 and `get_eos_id` 2 whatever the model:
  ExecuTorch 1.4.0 hard-codes them (`static_llama.py`), and its Qualcomm runner takes stop
  tokens from the tokenizer per family instead (`<|im_end|>` for Qwen). The app has to do
  the same when it loads QNN models.
- Upstream licenses differ (e.g. Qwen2.5-3B is under the Qwen Research License); cards and
  `LICENSE` files copy upstream's terms exactly.
