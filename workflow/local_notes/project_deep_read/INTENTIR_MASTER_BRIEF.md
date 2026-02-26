# INTENTIR_MASTER_BRIEF

## 一句话定位
IntentIR 是多前端、多后端编译工程；FlagGems 只是 Triton frontend 的 provider 路径之一，不是整体架构主语义。

## 当前真实状态（证据优先）

- 分支/HEAD：`compiler-cleanup-v1` / `4d90df5f4ea47440ce94b35c57923f2bbe296590`
- full196 freshness：`validated_commit=HEAD`，`commits_since_validated=0`
- gpu_perf freshness：`validated_commit=HEAD`，`commits_since_validated=0`
- MLIR 状态：`mlir_backend_contract_ready=true`，`mlir_llvm_chain_ok=true`，`mlir_cutover_level=mlir_primary`

## 核心证据

- full196 strict（196/196）
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v24_strict_o3/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v24_strict_o3/status_converged.json`
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v24_strict_o3/ci_gate.json`
- gpu_perf strict（159/159）
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v17_strict_normed_fix/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v17_strict_normed_fix/gpu_perf_graph.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v17_strict_normed_fix/ci_gate.json`
- artifacts 清理
  - `workflow/flaggems/state/cleanup_reports/20260226/summary.json`
  - `workflow/flaggems/state/cleanup_reports/20260226/plan.json`

## 本轮推进（已完成）

1. RVV strict 收口：`scripts/rvv_remote_run.py` 要求 execution mode 显式存在，移除隐式 compat 默认。
2. RVV compat C 导入收紧：`pipeline/mlir_contract_artifacts.py` 仅在 `INTENTIR_RVV_EMIT_COMPAT_C_SRC=1` 时导入 compat C++ codegen。
3. 执行 artifacts 最小基线清理：删除 168 路径，保留 validated full196/gpu_perf 证据及 `artifacts/toolchains`。
4. 清理后恢复 gate：`workflow/flaggems/state/gpu_perf_policy.json` baseline 更新到保留的 v17 graph，coverage/gpu_perf gate 均恢复 `ok=true`。

## 下一步（按既定顺序）

1. 继续 RVV strict 硬切，进一步压缩 compat 分支到 debug-only 可控开关。
2. 在 strict 语义下推进多前端统一 gate/report 字段对齐。
3. 每次 checkpoint 后同步 deep_read 与 workflow（遵循证据优先级）。
