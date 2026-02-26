# INTENTIR_MASTER_BRIEF

## 一句话定位
IntentIR 是多前端、多后端编译工程；FlagGems 是 Triton frontend 的 provider 路径之一，不是架构主语义。

## 当前真实状态（证据优先）

- 证据对齐 HEAD：`aeacf5b3ff3b7c7f02fbfdd76929ece5d1c9ce34`
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

## 本轮推进（已完成）

1. family/full196 两层 coverage summary 全部补齐 strict 字段：`strict_mode`、`fallback_policy`、`contract_schema_version`。
2. 覆盖批处理测试新增 strict 字段断言并通过。
3. full196/gpu_perf 在 aeac HEAD 完成复刷，双 gate 通过，freshness 归零。

## 下一步（按既定顺序）

1. 继续 RVV strict 收口，把 compat 路径压缩到 debug-only 的最小可控面。
2. 推进多前端 report/gate strict 字段一致性（Triton/TileLang/CUDA frontend）。
3. 每次 checkpoint 后继续回写 workflow 与 deep_read 真相包。
