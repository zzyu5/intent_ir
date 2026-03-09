# IntentIR Agent Handoff (2026‑03‑09)

This document is meant to be handed to a **new agent** so they can pick up work without re‑discovering repo structure, goals, invariants, and the latest evidence.

Snapshot:

- Date: **2026‑03‑09**
- Branch: `compiler-cleanup-v1`
- HEAD: `9029451d75bcaa63e69e7e68bbd7db35d7c3d789`
- Working tree: clean

---

## 1) What is IntentIR (current definition, not historical)

IntentIR is a compiler stack + workflow that:

1. **Lifts kernel behavior into an Intent representation** (carrier form is `intentir.intent_json_b64` on a MLIR module).
2. Lowers that intent through a **Real‑MLIR pipeline** (std/SCF/etc → LLVM IR → target code).
3. Runs **strict, evidence‑driven validation**:
   - correctness: real execution + compare against baseline provider outputs
   - performance: measured ratios vs provider baselines
   - auditability: per‑kernel contract JSON (schema v2) describing exactly what was compiled and run

Two key principles that are treated as “non‑negotiable invariants”:

- **Strict execution**: no hidden runtime fallback (must see `runtime_fallback=false` in contracts; suite summary must show `runtime_fallback_kernel_count=0`).
- **CUDA PTX provenance**: must be produced by LLVM `llc` (“real MLIR → LLVM IR → llc”), i.e. `cuda_ptx_origin=llvm_llc` in contracts/perf graphs.

LLM usage is **deliberate and cache‑driven**:

- “Auto” / LLM is only used to **create a cache** once (intent seeds / carriers).
- After cache is present, all regression/perf runs use **`--intentir-mode force_cache`** so we do **not** keep re‑calling LLM when nothing changed.

---

## 2) Repo structure (what lives where)

This is the “map” you need to navigate quickly.

### Core compiler & IR
- `intent_ir/`
  - `intent_ir/mlir/pipelines/`: MLIR pipeline YAMLs (python stack + cpp_plugin stack)
    - `intent_ir/mlir/pipelines/downstream_cuda_std_cpp_llvm.yaml`
    - `intent_ir/mlir/pipelines/downstream_rvv_std_llvm_cpp.yaml`
  - `intent_ir/mlir/pass_manager.py`: wrapper around `mlir-opt` etc; injects `-load-pass-plugin` when `INTENTIR_MLIR_PASS_PLUGIN` is set.
  - `intent_ir/mlir/passes/`: legacy + python stack passes (still exist, but direction is cpp_plugin).

### “Real mature compiler stack”: MLIR C++ pass plugin
- `compiler/intentir_mlir_plugin/`
  - `compiler/intentir_mlir_plugin/IntentIRPasses.cpp`: **the C++ lowering** used by the `cpp_plugin` compiler stack.
  - `compiler/intentir_mlir_plugin/CMakeLists.txt`: out‑of‑tree plugin build.

Passes currently registered in the plugin (important for debugging):

- `intentir-apply-tuning-db-cuda-v1`
- `intentir-lower-cuda-focus-v1`
- `intentir-lower-cuda-full196-v1`
- `intentir-extract-gpu-module-llvm-v1`
- `intentir-lower-rvv-cpu-loops-v1` (limited scope; see RVV section)

### Orchestration (“driver”)
- `pipeline/`: reusable orchestration library (compile, run, evidence, contracts)
  - `pipeline/mlir_contract_artifacts.py`: LLVM IR → llc → PTX, and **PTX cache** (`artifacts/ptx_cache`) with auditable contract fields.
  - `pipeline/common/evidence_mode.py`: `INTENTIR_EVIDENCE_MODE={on,off}`.
  - `pipeline/triton/core.py`, `pipeline/cuda/core.py`, `pipeline/tilelang/core.py`: backend selection & compiler stack routing.

### Kernel sources / providers (baselines)
- `kernels/`
  - `kernels/triton/`: Triton kernels + baseline runner (“native”)
  - `kernels/tilelang/`: TileLang kernels (separate lane)
  - `kernels/cuda/`: CUDA snapshots (used for comparisons / exports)

### User-facing CLIs / workflows
- `scripts/intentir.py`: main entrypoint for suites, kernel runs, **measured tune**, env checks, etc.
- `scripts/rvv_remote_run.py`, `scripts/rvv_remote_suite.py`: remote RVV compile/run + evidence.
- `scripts/flaggems/*`: coverage batches, perf graphs, workflow state updates.

### Workflow state and “truth ledger”
- `workflow/flaggems/state/`
  - `progress_log.jsonl`: append‑only log of evidence runs (“what happened, where is proof”).
  - `handoff.md`: short “last session” handoff (auto‑refreshed per session).
  - `current_status.json`: summary dashboard (currently stale; see “Known issues”).
  - allowlists:
    - `cuda_real_mlir_wave25_kernels.json` (**184 kernels**)
    - `rvv_real_mlir_wave22_kernels.json` (**114 kernels**, currently frozen)
    - `compiler_cpp_wave4_kernels.json` (cpp_plugin wave selection for routing)
  - tuning DB:
    - `tuning_db/cuda.jsonl`: arch/shape/compiler_stack‑aware overrides.

### Evidence outputs
- `artifacts/`
  - `artifacts/validation_rounds/<YYYYMMDD>/...`: local validated runs.
  - `artifacts/remote_import/<YYYYMMDD>/<machine>/...`: imported remote evidence.
  - `artifacts/ptx_cache/`: PTX reuse cache keyed by LLVM IR + toolchain fingerprint + arch.
  - `artifacts/mlir_plugins/intentir/`: built plugin + `plugin_manifest.json`.

---

## 3) “Compiler stacks” (python vs cpp_plugin)

### Python stack (legacy/compat)
- Lowering logic lives in Python passes under `intent_ir/mlir/passes/`.
- Still useful for quick iteration, but user direction is: **do not deepen** python‑lowering; use C++ plugin.

### `cpp_plugin` stack (current direction)
Goal: “a real compiler stack”: transforms happen in **`mlir-opt` + C++ pass plugin**.

Routing is controlled by env vars:

- `INTENTIR_COMPILER_STACK=cpp_plugin|python`
- `INTENTIR_COMPILER_CPP_WAVE=wave4` (controls which kernels are routed to cpp stack; see `workflow/flaggems/state/compiler_cpp_wave4_kernels.json`)
- `INTENTIR_MLIR_PASS_PLUGIN=$PWD/artifacts/mlir_plugins/intentir/libIntentIRPasses.so`

Pipeline YAMLs used by the cpp stack:

- CUDA: `intent_ir/mlir/pipelines/downstream_cuda_std_cpp_llvm.yaml`
  - runs: `intentir-apply-tuning-db-cuda-v1`, `intentir-lower-cuda-focus-v1`, `intentir-lower-cuda-full196-v1`
  - then: `convert-gpu-to-nvvm`, `convert-nvgpu-to-nvvm`, `mlir-translate`, `llvm-as`, `opt -O3`
- RVV: `intent_ir/mlir/pipelines/downstream_rvv_std_llvm_cpp.yaml`

---

## 4) Strict mode invariants & required evidence fields

For any “real” run we care about, set:

- `INTENTIR_REAL_MLIR=1`
- `INTENTIR_FALLBACK_POLICY=strict`
- `INTENTIR_CUDA_REQUIRE_LLVM_PTX=1`
- (recommended) `INTENTIR_EVIDENCE_MODE=off` for perf/remote (keep only contract+ptx+summaries)
- (recommended) `INTENTIR_CUDA_PTX_CACHE=1` (enabled by default)
- **arch pinning**: `INTENTIR_CUDA_SM=sm_89|sm_90|sm_120` per machine

Evidence must show (in contract/perf graph entry):

- `runtime_fallback=false`
- `cuda_ptx_origin=llvm_llc`
- `compiler_stack=cpp_plugin` (when using plugin)
- `lowering_kind=...` (e.g. `cuda_focus_v1` or `cuda_full196_v1`)
- `intentir_tuning_source` + `intentir_tuning_applied` (if tuning_db matched)
- `cuda_sm` matches the machine (`sm_89`/`sm_90`/`sm_120`)

---

## 5) Status: CUDA correctness/coverage (cpp_plugin Real‑MLIR)

### Coverage denominators
Repo truth (as of HEAD above):

- `coverage_batches.json`: **159 total**, **158 unique kernels** (7 batches)
- CUDA real‑MLIR allowlist: `workflow/flaggems/state/cuda_real_mlir_wave25_kernels.json` has **184 kernels**

### Full196 (CUDA‑only correctness gate) — cpp_plugin stack

We have **real machine evidence** that CUDA full196 is green under cpp_plugin + strict + llc PTX:

1) Local sm89 (4080S)
- Evidence: `artifacts/validation_rounds/20260306/full196_cuda_cpp_plugin_sm89_v1/run_summary.json`
  - `ok=true`
  - `runtime_fallback_kernel_count=0`
  - `mlir_llvm_chain_ok=true`

2) Remote H100 sm90
- Evidence: `artifacts/remote_import/20260306/h100/h100_cuda_cpp_full196_sm90_v3/run_summary.json`
  - `ok=true`
  - `runtime_fallback_kernel_count=0`
  - `mlir_llvm_chain_ok=true`
- Example per‑kernel contract proving cpp_plugin stack:
  - `artifacts/remote_import/20260306/h100/h100_cuda_cpp_full196_sm90_v3/family_reduction/chunk_001/pipeline_reports/argmax2d.intentir.intentdialect.downstream_cuda_std_cpp_llvm.contract.json`
    - `artifacts.compiler_stack="cpp_plugin"`
    - `artifacts.lowering_kind="cuda_full196_v1"`
    - `artifacts.cuda_ptx_origin="llvm_llc"`
    - `artifacts.cuda_sm="sm_90"`

3) Remote 5090D sm120
- Evidence: `artifacts/remote_import/20260306/sm120/sm120_cuda_cpp_full196_sm120_v1/run_summary.json`
  - `ok=true`
  - `runtime_fallback_kernel_count=0`
  - `mlir_llvm_chain_ok=true`

Important nuance:

- The cpp_plugin stack currently achieves “full196 correctness” mainly via
  - optimized “focus” lowering for a small set of performance‑critical kernels, plus
  - `intentir-lower-cuda-full196-v1` correctness‑first generic lowering (`cuda_full196_graph_v1`) for the long tail.
- This is intentional: **correctness/coverage first**, then performance is concentrated on focus kernels.

---

## 6) Status: CUDA focus performance (goal: beat Triton)

Focus kernel set used for perf evidence:

- attention: `flash_attention2d`, `masked_attention2d`, `_attn_fwd`
- matmul: `ai_bench_matmul`, `matmul_fused_epilogue2d`
- norm: `rms_norm2d`

All perf evidence below is: **cpp_plugin + strict + `cuda_ptx_origin=llvm_llc` + real execution**.

### H100 sm90 (needs work: flash_attention2d)
- Evidence: `artifacts/remote_import/20260306/h100/h100_cuda_cpp_triton_native_focus_covperf_sm90_4da9dd3_v1/gpu_perf_graph.json`
- Ratios vs triton-native baseline:
  - `_attn_fwd` ≈ **1.098**
  - `masked_attention2d` ≈ **1.020**
  - `ai_bench_matmul` ≈ **1.081**
  - `matmul_fused_epilogue2d` ≈ **1.002**
  - `rms_norm2d` ≈ **1.002**
  - `flash_attention2d` ≈ **0.972**  ← current worst focus item on H100

### 5090D sm120 (already ≥ Triton on focus)
- Evidence: `artifacts/remote_import/20260306/sm120/sm120_cuda_cpp_triton_native_focus_covperf_sm120_4da9dd3_v1/gpu_perf_graph.json`
- Ratios:
  - `ai_bench_matmul` ≈ **1.499**
  - `flash_attention2d` ≈ **1.0002**
  - `matmul_fused_epilogue2d` ≈ **1.015**
  - others > 1.0

### Local sm89 (near parity; tiny gaps remain)
- Evidence: `artifacts/remote_perf/20260306/local_triton_native_cpp_focus_covperf_sm89_4da9dd3_v1/gpu_perf_graph.json`
- Ratios:
  - `flash_attention2d` ≈ **0.990**
  - `masked_attention2d` ≈ **0.996**
  - `rms_norm2d` ≈ **0.993**
  - `ai_bench_matmul` ≈ **1.034**
  - `matmul_fused_epilogue2d` ≈ **1.006**

Takeaway:

- The “超越 Triton” performance story is already true on **sm120**, mostly true on **sm89**, and **one remaining attention hotspot** exists on **H100 sm90** (`flash_attention2d`).

---

## 7) Tuning system (what it is, why it exists, what’s real today)

### Why tuning_db exists (and why it’s not “just a script hack”)

We distinguish:

- **Where parameters come from** (source of truth / provenance): `tuning_db`
- **Where parameters take effect** (actual compilation): MLIR lowering pass (C++ plugin)

`workflow/flaggems/state/tuning_db/cuda.jsonl` stores:

- `(kernel, arch, compiler_stack)` selection
- optional shape guards (`when: {M:..., N:..., ...}`)
- `kernel_kind` (variant selection)
- `bindings` (tile sizes, warps, async_copy, etc)
- audit notes

Then `intentir-apply-tuning-db-cuda-v1` (C++ pass) applies it into module attrs so
lowering passes can pick the best variant deterministically.

### Recent tuning work (and why it’s trustworthy now)

Measured tuning is run via `scripts/intentir.py tune`.

Recent hardening ensures a candidate cannot “look good” while silently failing to
produce the required strict artifacts:

- coverage stage must validate:
  - `diff.ok == true`
  - `mlir.llvm_emit_ok == true`
  - downstream contract exists + says `cuda_ptx_origin=llvm_llc`
  - referenced `*.kernel.ptx` exists

Key commits:

- `41b8a55` tuning: harden measured tune artifact validation
- `26a8ef7` tuning: add sm90 shape‑scoped winner; plugin build generator fallback
- `1f901dd` workflow: log sm90 measured tune winner + plugin rebuild evidence

---

## 8) RVV status (frozen coverage, not expanding)

Policy (locked): **RVV kernel list is frozen at wave22**.

- Allowlist: `workflow/flaggems/state/rvv_real_mlir_wave22_kernels.json` (**114 kernels**)
- RVV remote evidence chain exists historically (see `workflow/flaggems/state/progress_log.jsonl` for the incremental wave runs).

Important nuance:

- The **cpp_plugin RVV lowering pass** `intentir-lower-rvv-cpu-loops-v1` currently supports a small subset (6 kernels):
  - `add2d`, `row_sum`, `gather2d`, `cat2d`, `diag2d`, `flip2d`
- The broader RVV wave22 coverage is primarily from the python stack / existing RVV backend lanes.

Remote RVV evidence requirement (when we do run it):

- `compile_rc=0`, `run_rc=0`
- `objdump -d | rg vsetvli` hits > 0
- compare OK (at least main output)

---

## 9) Current work (“what we are doing right now”)

1) **CUDA perf on H100 sm90**
   - Bring `flash_attention2d` from ~0.972 to **≥ 1.00** vs triton-native.
   - Then push `matmul_fused_epilogue2d` from ~1.00 towards **≥ 1.05** (target “超越”).

2) **Operational maturity**
   - Keep using `INTENTIR_EVIDENCE_MODE=off` for perf/remote to avoid I/O blowups.
   - Keep `PTX cache` on and auditable (contract already records hit/key/path).
   - Avoid repeated LLM usage in regressions: `--intentir-mode force_cache` + `--intentir-miss-policy strict`.

3) **Workflow truth freshness**
   - `workflow/flaggems/state/current_status.json` currently does not reflect the latest full196 cpp_plugin evidence; needs refresh.

---

## 10) Known pain points / pitfalls

1) **Remote repo drift**
   - Previously, H100 had an older `scripts/intentir.py`, breaking tune runs (missing artifacts / wrong validation).
   - Fix: always sync or pin commit; ensure remote runs use a git clone, not a partial rsync.

2) **Remote build environment differences**
   - Some remotes don’t have `ninja`. Build helper now falls back to `Unix Makefiles`.
   - Evidence: `artifacts/remote_import/20260306/h100/plugin_rebuild_sm90_v1/plugin_manifest.json`

3) **Workflow status is stale**
   - `workflow/flaggems/state/current_status.json` still points at older validated commits and does not capture the newer cpp_plugin full196 runs.
   - This is a bookkeeping problem, not a correctness failure, but it can confuse new agents.

4) **Perf targets require meaningful shapes**
   - Some canonical shapes (e.g. tiny matmul `32^3`) are noisy and can hide real wins/losses.
   - For “超越” claims, prefer a large, throughput‑relevant bucket (e.g. `M=256,N=512,K=256`) in tune sweeps.

---

## 11) “How to run the important things” (commands)

### Build the MLIR plugin (local or remote)
```bash
python3 scripts/intentir/build_intentir_mlir_plugin.py --clean
export INTENTIR_MLIR_PASS_PLUGIN=$PWD/artifacts/mlir_plugins/intentir/libIntentIRPasses.so
```

### CUDA full196 correctness (cpp_plugin, strict, real execution)
```bash
export INTENTIR_COMPILER_STACK=cpp_plugin
export INTENTIR_COMPILER_CPP_WAVE=wave4
export INTENTIR_REAL_MLIR=1
export INTENTIR_FALLBACK_POLICY=strict
export INTENTIR_CUDA_REQUIRE_LLVM_PTX=1
export INTENTIR_EVIDENCE_MODE=off
export INTENTIR_CUDA_PTX_CACHE=1
export INTENTIR_CUDA_SM=sm_89   # or sm_90 / sm_120 on the target machine

python scripts/intentir.py suite --suite flaggems-full196 \
  --backend-target cuda_5090d --cases-limit 1 \
  --intentir-mode force_cache --intentir-miss-policy strict \
  --cuda-runtime-backend nvrtc \
  --out-root artifacts/validation_rounds/$(date +%Y%m%d)/full196_cuda_cpp_plugin_sm89_vN
```

### CUDA focus perf (cpp_plugin, strict, triton-native baseline)
```bash
python scripts/intentir.py suite --suite gpu-perf-triton-native \
  --backend-target cuda_5090d \
  --kernel flash_attention2d --kernel masked_attention2d --kernel _attn_fwd \
  --kernel ai_bench_matmul --kernel matmul_fused_epilogue2d --kernel rms_norm2d \
  --perf-warmup 10 --perf-iters 50 --perf-repeats 3 \
  --intentir-mode force_cache --intentir-miss-policy strict \
  --out-root artifacts/remote_perf/$(date +%Y%m%d)/local_cuda_cpp_focus_perf_sm89_vN
```

### Measured tune (one kernel, one arch)
```bash
python scripts/intentir.py tune \
  --backend-target cuda_h100 --arch sm90 \
  --kernel flash_attention2d \
  --intentir-mode force_cache --intentir-miss-policy strict \
  --candidate 'kernel_kind=attn2d_causal_softmax_v6,ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6,FLASH_ATTN_ASYNC_COPY=1' \
  --candidate 'kernel_kind=attn2d_causal_softmax_v7,ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=8,FLASH_ATTN_ASYNC_COPY=1' \
  --out-root artifacts/tuning_runs/$(date +%Y%m%d)/tune_flash_sm90_vN
```

---

## 12) Next steps plan (what the new agent should do)

### Step 1 — Refresh workflow “truth”
- Run `python scripts/flaggems/build_workflow_state.py` so `workflow/flaggems/state/current_status.json` reflects:
  - latest CUDA full196 cpp_plugin evidence (sm89 + remote imports)
  - latest focus perf evidence (sm89/sm90/sm120)
- Update `workflow/flaggems/state/handoff.md` to point at this file and the latest evidence dirs.

### Step 2 — Fix the only remaining H100 focus gap (`flash_attention2d`)
- Run measured tune sweep on H100 sm90 (cpp_plugin) over:
  - `kernel_kind ∈ {attn2d_causal_softmax_v6, attn2d_causal_softmax_v7}`
  - `ATTN_BLOCK_KV ∈ {32,64}`
  - `ATTN_SCORE_WARPS ∈ {4,6,8}`
  - `FLASH_ATTN_ASYNC_COPY ∈ {0,1}`
- Persist winner into `workflow/flaggems/state/tuning_db/cuda.jsonl` with a tight `when` guard (e.g. `{Q_CTX:64, KV_CTX:64, HEAD_DIM:64}`) so it won’t destabilize other shapes.
- Re-run H100 focus perf evidence and confirm `flash_attention2d ratio >= 1.00`.

### Step 3 — Move `matmul_fused_epilogue2d` from parity to “超越”
- Add a larger, meaningful shape bucket for tuning (avoid only `32^3`).
- Tune `kernel_kind ∈ {matmul_fused_epilogue_mma_tf32_v2, v3, global_v2}` + tile params.
- If still stuck on sm90: evaluate implementing a Hopper‑specific WGMMA path in the plugin (requires `nvgpu` ops lowering support).

### Step 4 — Keep LLM usage correct
- Default regression/perf runs: `--intentir-mode force_cache --intentir-miss-policy strict`
- Only use `--intentir-mode auto` when:
  - there is a true cache miss, and
  - you intend to persist the new cache artifacts as part of the change.

---

## Appendix: Remote machines (SSH)

- H100 (source): `kingdom@211.87.236.70`
  - repo used previously: `/home/kingdom/intentir_remote_20260302`
- 5090D (target): `aii-works@211.87.236.79`
- RVV: `ubuntu@192.168.8.72`
- Local target: RTX 4080 SUPER (sm89)
