# WORKFLOW_TRUTH_AUDIT

更新时间（UTC）：`2026-02-26T05:19:01Z`

证据优先级（冲突时）：
`run_summary + status_converged（同 commit） > current_status.json > handoff/progress 文本`

## 1. 快照一致性

- git HEAD：`8f597ef3a669a9dbe6dc07af5574e2f6d1b09ff3`
- `workflow/flaggems/state/current_status.json:head_commit`：`8f597ef3a669a9dbe6dc07af5574e2f6d1b09ff3`
- 判定：一致

## 2. freshness 状态

| 维度 | 记录值 | 判定 |
|---|---|---|
| full196_validated_commit | `8f597ef3a669a9dbe6dc07af5574e2f6d1b09ff3` | fresh |
| full196_commits_since_validated | `0` | fresh |
| gpu_perf_validated_commit | `8f597ef3a669a9dbe6dc07af5574e2f6d1b09ff3` | fresh |
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

- `full196`：`run_summary.ok=true`，`coverage_batches=7/7`，`status_converged.counts_global.dual_pass=196`
- `gpu_perf`：`run_summary.ok=true`，`gpu_perf_kernel_measured=159`，`status_converged.counts_global.dual_pass=159`
- `normed_cumsum2d`：`reason_code=ok`，`ratio=0.9195`，`native_launch_source=kernel_adapter:normed_cumsum2d`
- `mlir`：`current_status.mlir_llvm_chain_ok=true`，`mlir_cutover_level=mlir_primary`
- gate：coverage/gpu_perf 两个 profile 的 `ci_gate` 均 `ok=true`

## 5. 说明

1. 这次回写执行了 `--resume` HEAD refresh（full196 与 gpu_perf），保证证据链与当前提交对齐。
2. `full196` 的 `repo.dirty=true` 来自执行期 workflow 状态文件更新，不影响 `head_commit` 对齐和 freshness 判定。
3. `mlir_migration` 的最新大波次证据仍是 `matrix_wave64_v1`（其 `repo.head_commit=34e839...`），属于历史阶段证据，不覆盖当前 fresh gate 判定。

## 6. 审计结论

- freshness 已闭环（full196/gpu_perf validated_commit 均与 HEAD 对齐，gap=0）。
- strict 主门禁通过（196/196 与 159/159）。
- 当前可继续推进 RVV strict 收口与 deep_read 持续同步。
