# WORKFLOW_TRUTH_AUDIT

更新时间（UTC）：`2026-02-26T05:38:29Z`

证据优先级（冲突时）：
`run_summary + status_converged（同 commit） > current_status.json > handoff/progress 文本`

## 1. 快照一致性

- git HEAD：`4d90df5f4ea47440ce94b35c57923f2bbe296590`
- `workflow/flaggems/state/current_status.json:head_commit`：`4d90df5f4ea47440ce94b35c57923f2bbe296590`
- 判定：一致

## 2. freshness 状态

| 维度 | 记录值 | 判定 |
|---|---|---|
| full196_validated_commit | `4d90df5f4ea47440ce94b35c57923f2bbe296590` | fresh |
| full196_commits_since_validated | `0` | fresh |
| gpu_perf_validated_commit | `4d90df5f4ea47440ce94b35c57923f2bbe296590` | fresh |
| gpu_perf_commits_since_validated | `0` | fresh |
| coverage_integrity_phase | `recomputed_ok` | fresh |
| gpu_perf_phase | `recomputed_ok` | fresh |

## 3. HEAD 级证据链

- full196 strict（HEAD refresh）
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v24_strict_o3/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v24_strict_o3/status_converged.json`
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v24_strict_o3/coverage_integrity.json`
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v24_strict_o3/ci_gate.json`
- gpu_perf strict（HEAD refresh）
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v17_strict_normed_fix/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v17_strict_normed_fix/status_converged.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v17_strict_normed_fix/gpu_perf_graph.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v17_strict_normed_fix/stage_timing_breakdown.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v17_strict_normed_fix/ci_gate.json`

## 4. 当前关键状态

- `full196`：`run_summary.ok=true`，`coverage_batches=7/7`，`mlir_llvm_artifact_complete=true`，`runtime_fallback_kernel_count=0`
- `gpu_perf`：`run_summary.ok=true`，`kernel_measured=159`，`dual_pass=159`，`blocked_backend=0`，`runtime_fallback=0`
- 性能阈值：`min_ratio=0.8275962766613206`（阈值 `0.8`）
- MLIR 切入：`mlir_backend_contract_ready=true`，`mlir_llvm_chain_ok=true`，`mlir_cutover_level=mlir_primary`
- gate：coverage/gpu_perf 两个 profile 的 `ci_gate` 均 `ok=true`

## 5. 本轮新增动作

1. RVV strict 收口：`scripts/rvv_remote_run.py` 新增 execution mode 显式校验，移除隐式 `compat_c_src` 默认。
2. RVV compat C 导入收紧：`pipeline/mlir_contract_artifacts.py` 仅在 `INTENTIR_RVV_EMIT_COMPAT_C_SRC=1` 时导入 compat C++ codegen。
3. artifacts 生命周期清理执行：`workflow/flaggems/state/cleanup_reports/20260226/summary.json` 显示删除 `168` 路径，容量从 `21,006,689,322` 降至 `120,985,167` 字节，保留 validated 证据与 toolchains。
4. `gpu_perf_policy` baseline 更新为保留产物：`workflow/flaggems/state/gpu_perf_policy.json` 指向 `.../gpu_perf_head_refresh_v17_strict_normed_fix/gpu_perf_graph.json`。

## 6. 审计结论

- 当前 HEAD 上 freshness 闭环成立（full196/gpu_perf 均 gap=0）。
- strict 主门禁通过（full196 196/196，gpu_perf 159/159，fallback=0）。
- 可继续推进 RVV strict 兼容分支降级与多前端统一 strict 语义收口。
