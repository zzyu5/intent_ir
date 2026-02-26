# INTENTIR_MASTER_BRIEF

## 一句话定位
IntentIR 是一个“语义抽取 -> MLIR/backend 合同 -> 后端执行 -> workflow 门禁”的多前端、多后端编译器工程；FlagGems 只是 Triton 前端下的一条 provider 路径。

## 当前真实状态（以证据为准）

- 分支/HEAD：`compiler-cleanup-v1` / `8f597ef3a669a9dbe6dc07af5574e2f6d1b09ff3`
- full196 freshness：`validated_commit=HEAD`，`commits_since_validated=0`
- gpu_perf freshness：`validated_commit=HEAD`，`commits_since_validated=0`
- MLIR 状态：`mlir_backend_contract_ready=true`，`mlir_llvm_chain_ok=true`，`mlir_cutover_level=mlir_primary`

## 关键证据

- full196 strict（196/196）
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v24_strict_o3/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v24_strict_o3/status_converged.json`
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v24_strict_o3/ci_gate.json`
- gpu_perf strict（159/159）
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v17_strict_normed_fix/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v17_strict_normed_fix/status_converged.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v17_strict_normed_fix/gpu_perf_graph.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v17_strict_normed_fix/ci_gate.json`
- 关键修复点
  - `scripts/flaggems/run_gpu_perf_graph.py`
  - `tests/frontends/triton/test_flaggems_gpu_perf_graph_native_launch.py`

## 这轮关键推进

1. 使用同一 out-root `--resume` 完成 full196/gpu_perf 的当前 HEAD 证据刷新。
2. 修复 `normed_cumsum2d` 的 native baseline 调用路径，避免 heuristic 误配导致性能门禁失真。
3. 重新收敛 `gpu_perf` 到 `dual_pass=159`，并确保 `normed_cumsum2d` 达标（ratio `0.9195`）。
4. coverage 与 gpu_perf 的 `ci_gate` 在当前证据下均通过。

## 当前剩余问题

1. RVV strict 兼容路径（`compat_cpp_codegen`）的彻底下线仍需后续阶段收口。
2. mlir_migration 的 64-kernel wave 证据仍停在 `34e839...`，尚未做当前 HEAD 的迁移波次复刷。

## 接力建议

1. 进入 RVV strict 收口：默认禁 compat，保留显式 debug 开关。
2. 完成后再跑一次 strict wave（mlir_migration lane）并回写 deep_read 与 workflow。
