# ORG Perf Scale-up Report — 2026-03-09

## Scope
- Target machine: local 5090D (`sm120`)
- Toolchain: full LLVM/MLIR20 path for CUDA lowering (`effective_sm = sm_120`, no downlevel on the reported guided runs)
- ORG mode: `apply`
- Purpose: measure real guided performance, then expand to a broader kernel suite and identify I/H/P blind spots

## A. Immediate Flash Result
- Current effective planner head is `attn2d_causal_softmax_v8:ATTN_BLOCK_KV=32`. Candidate file: `/tmp/intentir_org_scale_suite_20260309/flash_attention2d.org_candidates.txt`
- Real compare output: `/tmp/intentir_org_scale_cmp_20260309/flash_attention2d/comparison.txt`
- Guided result on 5090D:
  - `guided_best_ratio = 0.667265`
  - `guided_best_qps_intentir = 162471.052`
  - `guided_best_qps_native = 243519.901`
  - `effective_sm = sm_120`, `downleveled = False`
- Outcome: still **below native Triton**. It does **not** reach `>= 1.0` on `flash_attention2d`.
- Important detail: source replay and target oracle both require `cluster_variant_shift` repair and converge to the same `v8@32` portable candidate. The old `v6@64,6` / `v9@32` frontier no longer wins the full compare.

Top of the current flash candidate list:
```text
attn2d_causal_softmax_v8:ATTN_BLOCK_KV=32
attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=4
attn2d_causal_softmax_v6:ATTN_BLOCK_KV=32,ATTN_SCORE_WARPS=4
attn2d_causal_softmax_v6:ATTN_BLOCK_KV=32,ATTN_SCORE_WARPS=6
attn2d_causal_softmax_v6:ATTN_BLOCK_KV=16,ATTN_SCORE_WARPS=6
attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=2
attn2d_causal_softmax_v6:ATTN_BLOCK_KV=32,ATTN_SCORE_WARPS=2
attn2d_causal_softmax_v8:ATTN_BLOCK_KV=64
```

## B. Real Performance Summary

| Kernel | Category | ORG status | Guided best candidate | Guided QPS | Guided Ratio vs Triton | Native QPS | effective_sm | Note |
|---|---|---:|---|---:|---:|---:|---|---|
| `flash_attention2d` | attention | compared | `attn2d_causal_softmax_v8:ATTN_BLOCK_KV=32` | 162471.1 | 0.6673 | 243519.9 | `sm_120` | large gap |
| `matmul_fused_epilogue2d` | matmul | compared | `matmul_mma_tf32_v1:MMA_BK=16,MMA_BM=32,MMA_BN=32` | 254273.6 | 1.0413 | 244184.1 | `sm_120` | beats Triton |
| `_attn_fwd` | attention | compared | `attn_fwd_softmax_v2` | 37565.3 | 0.4618 | 81342.3 | `sm_120` | large gap |
| `softmax_inner` | row-softmax | compared | `row_softmax_axis1_v1` | 311929.6 | 1.0294 | 303029.7 | `sm_120` | beats Triton |
| `masked_softmax2d` | masked-row-softmax | compared | `row_masked_softmax_axis1_v1` | 307330.8 | 1.0311 | 298063.0 | `sm_120` | beats Triton |
| `add2d` | expanded suite | deferred | — | — | — | — | `sm120` | `org_kernel_deferred` |
| `ai_bench_matmul` | expanded suite | deferred | — | — | — | — | `sm120` | `org_kernel_deferred` |
| `ai_bench_softmax` | expanded suite | deferred | — | — | — | — | `sm120` | `org_kernel_deferred` |
| `exp2d` | expanded suite | deferred | — | — | — | — | `sm120` | `org_kernel_deferred` |
| `group_norm_kernel` | expanded suite | deferred | — | — | — | — | `sm120` | `org_kernel_deferred` |
| `layer_norm_persistent` | expanded suite | deferred | — | — | — | — | `sm120` | `org_kernel_deferred` |
| `masked_attention2d` | expanded suite | deferred | — | — | — | — | `sm120` | `org_kernel_deferred` |
| `row_max` | expanded suite | deferred | — | — | — | — | `sm120` | `org_kernel_deferred` |
| `row_sum` | expanded suite | deferred | — | — | — | — | `sm120` | `org_kernel_deferred` |

### Winners / Losers
- **Beats Triton**: `matmul_fused_epilogue2d`, `softmax_inner`, `masked_softmax2d`
- **Loses badly**: `_attn_fwd`, `flash_attention2d`
- **Not mapped yet**: `add2d`, `exp2d`, `row_sum`, `row_max`, `group_norm_kernel`, `layer_norm_persistent`, `masked_attention2d`, `ai_bench_softmax`, `ai_bench_matmul`

## C. Expanded Suite Observations
- For all expanded kernels above, ORG extraction itself succeeded under `INTENTIR_ORG_SEED_POLICY=auto` and produced `org.json` artifacts.
- The failure mode is **not** “LLM could not produce rationale”. The dominant failure mode is `apply_reason = org_kernel_deferred`: ORG can describe them, but the CUDA planner/catalog cannot map them into `BackendPlan + candidates` yet.

| Deferred kernel | Source oracle available? | Example extracted goals | Example extracted mechanisms | Immediate gap |
|---|---:|---|---|---|
| `add2d` | False | `resident_working_set, avoid_materialization, latency_hiding` | `blocked_register_layout, tile_load_direct, 2d_grid_mapping, warp_parallel_execution, masked_edge_handling` | planner/catalog missing |
| `ai_bench_matmul` | True | `operand_reuse, resident_working_set, latency_hiding, mma_acceleration` | `operand_tile_stage, blocked_layout, async_copy_pipeline, dot_op, output_layout_convert` | planner/catalog missing |
| `ai_bench_softmax` | True | `resident_working_set, streaming_softmax_state, avoid_materialization, latency_hiding` | `row_tile_resident, online_softmax_reduce, warp_shuffle_reduce, row_parallel_mapping, vector_row_path` | planner/catalog missing |
| `exp2d` | False | `resident_working_set, avoid_materialization, latency_hiding` | `blocked_register_layout, tile_load_direct, elementwise_exp_primitive, two_axis_grid_mapping, masked_edge_handling` | planner/catalog missing |
| `group_norm_kernel` | False | `resident_working_set, streaming_softmax_state, avoid_materialization, fused_epilogue_avoid_writeback` | `group_tile_resident, warp_reduction, online_normalization, affine_fused_epilogue, blocked_layout` | planner/catalog missing |
| `layer_norm_persistent` | False | `resident_working_set, streaming_softmax_state, avoid_materialization, latency_hiding` | `row_tile_resident, warp_reduction, register_staging, warp_parallel_execution, row_program_axis` | planner/catalog missing |
| `masked_attention2d` | True | `resident_working_set, streaming_softmax_state, avoid_materialization, latency_hiding` | `q_resident_state, kv_tile_load, online_softmax_reduce, mask_causal_apply, row_tile_resident` | planner/catalog missing |
| `row_max` | False | `resident_working_set, streaming_softmax_state, avoid_materialization, latency_hiding` | `row_tile_resident, tile_load_stage, warp_reduction_tree, row_parallel_axis, block_synchronization` | planner/catalog missing |
| `row_sum` | False | `resident_working_set, streaming_softmax_state, avoid_materialization, latency_hiding` | `row_tile_resident, vector_row_path, row_reduction, warp_parallel_rows, shared_staging` | planner/catalog missing |

## D. Gap Analysis

### I-space gaps (Intent / rationale vocabulary)
- `row_sum` / `row_max` / `layer_norm_persistent` still get attention-flavored goals like `streaming_softmax_state`. That is a mismatch: the current intent vocabulary is overfit to attention/row-softmax and under-specifies generic reduction-tree choices, persistent reduction strategy, and norm-specific staging.
- `add2d` / `exp2d` only expose very generic goals (`resident_working_set`, `avoid_materialization`, `latency_hiding`). They do not express the real optimization axes that Triton exploits for simple bandwidth kernels: vector width, transaction coalescing, mask cost, and launch geometry.
- `group_norm_kernel` and `layer_norm_persistent` need explicit intent concepts for multi-phase normalization (`mean/var`, affine epilogue, persistent row residency). Those are only weakly approximated right now.
- `ai_bench_matmul` exposes `mma_acceleration` and tile dims, but there is no downstream ORG vocabulary yet for epilogue scheduling, pipeline depth tradeoffs, or accumulator/writeback pressure.

### H-space gaps (Hardware / toolchain model)
- Current `HardwareModel` is still coarse. It sees `cluster`, shared memory, async/MMA/shuffle support, and `effective_sm`, but it does not model occupancy, register spills, memory transaction width, bank conflicts, or reduction crossover points.
- That is visible in `flash_attention2d`: even after fixing `effective_sm = sm_120`, the planner still needs empirical correction, and the winning candidate (`v8@32`) only delivers `~0.667x` of native Triton.
- `_attn_fwd` is the clearest warning sign: the ORG path compiles cleanly on `sm120`, but only reaches `0.4618x` native Triton. This is no longer a toolchain blocker; it is missing hardware/performance modeling.

### P-space / backend gaps (Planner / realization)
- The largest blind spot is still P-space coverage. For the nine expanded kernels above, ORG rationale exists, but there is **no CUDA planner/catalog implementation**, so the system stops at `org_kernel_deferred`.
- This means the current architecture has proven `I -> rationale` on those kernels, but has **not** proven `I x H -> P` for them yet.
- The mapped set is still effectively: `flash_attention2d`, `_attn_fwd`, `softmax_inner`, `masked_softmax2d`, `matmul_fused_epilogue2d`.

## E. Hard Conclusions
- The current system **does not** yet support the project’s end goal “Guided >= Native Triton” uniformly.
- It already works on some kernels:
  - `matmul_fused_epilogue2d`: `1.0413x`
  - `softmax_inner`: `1.0294x`
  - `masked_softmax2d`: `1.0311x`
- It still fails materially on attention kernels that matter most:
  - `flash_attention2d`: `0.6673x`
  - `_attn_fwd`: `0.4618x`
- Therefore the next research bottleneck is no longer basic ORG plumbing. It is:
  1. better performance-aware H modeling for attention
  2. broader planner/catalog coverage for deferred kernels
  3. better I-space vocabulary for generic reductions, norms, and bandwidth kernels

## F. Data Locations
- Flash compare: `/tmp/intentir_org_scale_cmp_20260309/flash_attention2d/comparison.txt`
- Supported kernel compares: `/tmp/intentir_org_scale_cmp_20260309/`
- Deferred kernel auto-apply runs: `/tmp/intentir_org_scale_gaps_auto_20260309/` and `/tmp/intentir_org_gap_*_20260309/`
- Batch pipeline reports: `/tmp/intentir_org_scale_suite_20260309/`
