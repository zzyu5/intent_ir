# IntentIR Agent Handoff (Update 2026‑03‑10)

This is an incremental update to `workflow/flaggems/state/AGENT_HANDOFF_20260309.md` with the new progress made on **2026‑03‑10**.

## 0) Repo Truth snapshot (do not guess)

- Branch: `compiler-cleanup-v1`
- HEAD: `96393bcc887ef9cc8480d9533c932ed807f571f1`
- Evidence is imported under: `artifacts/remote_import/20260310/h100/`

Hard invariants still apply:

- `INTENTIR_FALLBACK_POLICY=strict` and `runtime_fallback=false` (kernel contracts + suite summaries)
- `cuda_ptx_origin=llvm_llc` (real MLIR → LLVM IR → `llc` PTX)

## 1) What was “the current problem” and what changed

### Problem: H100 focus-perf multi-kernel variance

On H100(sm90), multi-kernel focus perf (`gpu-perf-triton-native` on focus6) showed **run-to-run ratio swings** for micro-kernels (e.g. `masked_attention2d`, `rms_norm2d`) even when `cuda_ptx_cache_key` / `kernel_kind` were unchanged, indicating **measurement drift/order bias** rather than compilation changes.

### Fix: paired benchmark + ratio stabilization in the perf runner

`scripts/flaggems/run_gpu_perf_graph.py` now benches native + intent in a **paired, alternating-order** loop and reports:

- `paired_bench` metadata per entry (repeat counts + per-repeat ratio stats)
- `ratio` uses `paired_bench.ratio_by_repeat_median` when available (more robust vs outliers)
- `ratio_from_qps` is kept for audit/debug (ratio derived from `qps_intentir/qps_native`)
- graph retiming (`retimed_iters`) preserves the paired bench methodology (no silent regression to one-sided benches)

This eliminates the systematic “native-first vs intent-first” drift bias and makes focus-perf evidence reproducible.

## 2) New H100(sm90) focus6 evidence (strict + llvm_llc)

Latest imported focus-perf evidence:

- `artifacts/remote_import/20260310/h100/h100_triton_native_focus_perf_sm90_sw14_v10_paired_median_ratio/gpu_perf_graph.json`

Key facts in that evidence:

- 6/6 focus kernels measured; `min_ratio >= 1.0`
- `flash_attention2d` uses the tuned shape-scoped winner (`ATTN_SCORE_WARPS=14`, `FLASH_ATTN_ASYNC_COPY=1`) via tuning_db

## 3) What to do next (pragmatic)

1) **Refresh workflow dashboard truth** (bookkeeping):
   - Ensure `workflow/flaggems/state/current_status.json` and `session_context.json` reflect the latest full196 + focus perf evidence.

2) **Optional: meaningful perf shapes**
   - `matmul_fused_epilogue2d` already uses a policy shape override (`M=256,N=512,K=256`).
   - If needed, consider adding policy shape overrides for `masked_attention2d` / `rms_norm2d` to reduce “tiny-shape” noise and better reflect real workloads.

3) **Commit hygiene**
   - Split “engineering changes” (perf runner + plugin + pipeline) from “workflow state updates” (handoff/current_status/progress logs) into separate commits when ready.

