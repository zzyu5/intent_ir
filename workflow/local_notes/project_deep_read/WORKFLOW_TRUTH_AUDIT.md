# WORKFLOW_TRUTH_AUDIT

更新时间（UTC）：`2026-02-25T23:54:00Z`

证据优先级（冲突时）：
`run_summary + status_converged（同 commit） > current_status.json > handoff/progress 文本`

## 1. 快照一致性

- git HEAD：`34e839652470d4d9e8fc49b1e6c47c7206b40b47`
- `workflow/flaggems/state/current_status.json:head_commit`：`34e839652470d4d9e8fc49b1e6c47c7206b40b47`
- 判定：一致

## 2. freshness 状态

| 维度 | 记录值 | 判定 |
|---|---|---|
| full196_validated_commit | `34e839652470d4d9e8fc49b1e6c47c7206b40b47` | fresh |
| full196_commits_since_validated | `0` | fresh |
| gpu_perf_validated_commit | `34e839652470d4d9e8fc49b1e6c47c7206b40b47` | fresh |
| gpu_perf_commits_since_validated | `0` | fresh |
| coverage_integrity_phase | `recomputed_ok` | fresh |
| gpu_perf_phase | `recomputed_ok` | fresh |

## 3. HEAD 级证据链

- full196 strict（2026-02-26）：
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v23_strict_o3/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v23_strict_o3/status_converged.json`
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v23_strict_o3/coverage_integrity.json`
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v23_strict_o3/ci_gate.json`
- gpu_perf strict（2026-02-26）：
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v14_strict_policy_refresh4/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v14_strict_policy_refresh4/status_converged.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v14_strict_policy_refresh4/gpu_perf_graph.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v14_strict_policy_refresh4/stage_timing_breakdown.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v14_strict_policy_refresh4/ci_gate.json`
- mlir_migration strict waves（2026-02-26）：
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_wave13_v1/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_wave13_v1/status_converged.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_attention3_v1/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_attention3_v1/status_converged.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_wave24_v1/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_wave24_v1/status_converged.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_wave40_v1/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_wave40_v1/status_converged.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_wave64_v1/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_wave64_v1/status_converged.json`

## 4. 当前关键状态

- `full196`：`run_summary.ok=true`，`coverage_batches=7/7`，`status_converged.counts_global.dual_pass=196`
- `gpu_perf`：`run_summary.ok=true`，`gpu_perf_kernel_measured=159`，`status_converged.counts_global.dual_pass=159`
- `mlir`：`current_status.mlir_llvm_chain_ok=true`，`mlir_cutover_level=mlir_primary`
- `gate`：coverage/gpu_perf 两个 profile 的 `ci_gate` 均 `ok=true`

## 5. 收敛说明

1. `gpu_perf_key_kernel_baseline` 子门禁已通过（min_relative=1.0425）。
2. `scaled_dot_product_attention_bhsd`、`flash_attn_varlen_func_bhsd`、`unique2d` 在 strict 路径下均为 `reason_code=ok`，且 `runtime_fallback=false`。
3. scoped kernel 子集的 `status_converged.counts_global` 已按 scope 投影，避免全 registry 的 `provider_report_missing` 噪声干扰波次结论。
4. strict matrix_wave64（64 kernels）已完成，`counts_scoped.dual_pass=75`，`runtime_fallback_kernel_count=0`。
5. `current_status` freshness 字段保持归零，未出现 commit 漂移。

## 6. 审计结论

- freshness 已闭环（full196/gpu_perf validated_commit 均与 HEAD 对齐，gap=0）。
- strict full196/gpu_perf 双主门禁通过（196/196 与 159/159）。
- workflow 状态与 artifact 证据一致，可继续推进 RVV strict 最终收口（去 compat 默认路径并保持 hard-fail 语义）。
