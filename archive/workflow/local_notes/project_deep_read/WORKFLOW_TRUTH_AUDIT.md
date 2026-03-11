# WORKFLOW_TRUTH_AUDIT

更新时间（UTC）：`2026-02-26T06:46:35Z`

证据优先级（冲突时）：
`run_summary + status_converged（同 commit） > current_status.json > handoff/progress 文本`

## 1. 快照一致性

- 证据 HEAD（workflow 对齐点）：`aeacf5b3ff3b7c7f02fbfdd76929ece5d1c9ce34`
- `workflow/flaggems/state/current_status.json:head_commit`：`aeacf5b3ff3b7c7f02fbfdd76929ece5d1c9ce34`
- 判定：一致

## 2. freshness 状态

| 维度 | 记录值 | 判定 |
|---|---|---|
| full196_validated_commit | `aeacf5b3ff3b7c7f02fbfdd76929ece5d1c9ce34` | fresh |
| full196_commits_since_validated | `0` | fresh |
| gpu_perf_validated_commit | `aeacf5b3ff3b7c7f02fbfdd76929ece5d1c9ce34` | fresh |
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

- `full196`：`run_summary.ok=true`，`coverage_batches=7/7`，`dual_pass=196`，`mlir_llvm_artifact_complete=true`，`runtime_fallback_kernel_count=0`
- `full196` strict 字段：`strict_mode=true`，`fallback_policy=strict`，`contract_schema_version=intent_mlir_backend_contract_v2`
- `gpu_perf`：`run_summary.ok=true`，`kernel_measured=159`，`dual_pass=159`，`blocked_backend=0`，`runtime_fallback=0`
- 性能阈值：`min_ratio=0.8275962766613206`（阈值 `0.8`）
- MLIR 切入：`mlir_backend_contract_ready=true`，`mlir_llvm_chain_ok=true`，`mlir_cutover_level=mlir_primary`
- gate：coverage/gpu_perf 两个 profile 的 `ci_gate` 均 `ok=true`

## 5. 本轮新增动作

1. `run_coverage_batches.py` 的 family/run-batch summary 增加 strict 字段：`strict_mode`、`fallback_policy`、`contract_schema_version`。
2. `aggregate_coverage_batches.py` 的 full196 聚合 summary 也已对齐 strict 字段。
3. `test_flaggems_coverage_batches.py` 增加字段回归断言，固定 schema 行为。
4. full196 与 gpu_perf 再次在 aeac HEAD 复刷，freshness 归零并通过双 gate。

## 6. 审计结论

- 当前 workflow 真相与 artifact 证据一致，freshness 闭环成立。
- strict 主门禁保持 green（196/196 与 159/159）。
- 可继续推进 RVV compat 分支进一步收口与多前端 strict schema 统一。
