# INTENTIR_MASTER_BRIEF

## 一句话定位
IntentIR 是一个“语义抽取 -> MLIR/backend 合同 -> 后端执行 -> workflow 门禁”的编译器工程。

## 当前真实状态（以证据为准）

- 分支/HEAD：`compiler-cleanup-v1` / `34e839652470d4d9e8fc49b1e6c47c7206b40b47`
- full196 freshness：`validated_commit=HEAD`，`commits_since_validated=0`
- gpu_perf freshness：`validated_commit=HEAD`，`commits_since_validated=0`
- MLIR 状态：`mlir_backend_contract_ready=true`，`mlir_llvm_chain_ok=true`，`mlir_cutover_level=mlir_primary`

## 关键证据

- full196 strict（196/196）：
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v23_strict_o3/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v23_strict_o3/status_converged.json`
- gpu_perf strict（159/159）：
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v14_strict_policy_refresh4/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v14_strict_policy_refresh4/status_converged.json`
- 双 profile gate：
  - `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v23_strict_o3/ci_gate.json`
  - `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v14_strict_policy_refresh4/ci_gate.json`
- mlir_migration strict waves：
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_wave13_v1/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_attention3_v1/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_wave24_v1/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_wave40_v1/run_summary.json`
  - `artifacts/flaggems_matrix/daily/20260226/mlir_contract_wave_v8_strict/matrix_wave64_v1/run_summary.json`

## 这轮关键推进

1. 刷新并固定了 `gpu_perf` key-kernel baseline policy 到 strict 基线（2026-02-26）。
2. 修复 `run_gpu_perf_graph` native 路径中关键 kernel 的 callable 误选问题（新增 adapter）。
3. `unique2d` 在 strict perf 路径下恢复可测且通过阈值（ratio>0.8，当前约 2.42）。
4. `gpu_perf` 重跑到 `v14_strict_policy_refresh4`，恢复 `159 measured / 159 dual_pass`。
5. 覆盖与性能两个 profile 的 `ci_gate` 均为 `ok=true`。
6. strict matrix 已从 4 -> 8 -> 13 kernels 扩展，跨 family 保持 `runtime_fallback=0`。
7. strict attention 子族（`embedding2d`/`scaled_dot_product_attention_bhsd`/`flash_attn_varlen_func_bhsd`）已实现双后端 `dual_pass`。
8. strict matrix 已扩展到 64 kernels（跨 family），当前 scoped `dual_pass=75` 且 `runtime_fallback=0`。
9. `wave64` 的 RVV 执行计划为 `remote_llvm`，未出现 `compat_cpp_codegen` runtime fallback。

## 当前剩余问题

1. RVV strict 仍需继续去除 `compat_cpp_codegen` 旧路径（属于后续波次，不影响当前 strict 主门禁收敛）。

## 接力建议

1. 以当前 `v23 + v14 + matrix_wave64` 证据作为稳定基线，继续推进 RVV strict 切换与对应 gate。
2. 保持 `run_summary + status_converged` 同 commit 复核规则，不再回退到文本口径。
