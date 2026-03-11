# CODE_OWNERSHIP_MAP

## 1. 项目主目标（当前版本）
- 统一通过 `scripts/intentir.py` 运行 suite/kernel。
- 前端（Triton/TileLang/CUDA）提取语义 -> IntentIR/MLIR pass -> backend 合约执行（CUDA/RVV）-> verify diff -> workflow gate。
- workflow 维护 full196 / gpu_perf / mlir_migration 的可审计状态。

## 2. 核心入口与职责

| 入口 | layer | consumes | produces | downstream | main_path |
|---|---|---|---|---|---|
| `scripts/intentir.py:suite` | `cli` | `suite args`, `workflow/flaggems/state/coverage_batches.json` | `artifacts/intentir_suite/*`, `artifacts/flaggems_matrix/**` | `scripts/flaggems/run_coverage_batches.py`, `scripts/flaggems/run_multibackend_matrix.py`, `scripts/flaggems/run_gpu_perf_graph.py` | `true` |
| `scripts/intentir.py:kernel` | `cli` | `frontend/provider/kernel args` | `artifacts/intentir_kernel/*` | `scripts/flaggems/run_multibackend_matrix.py`, `scripts/tilelang/full_pipeline_verify.py`, `scripts/cuda/full_pipeline_verify.py` | `true` |
| `pipeline/triton/core.py:run_pipeline_for_spec` | `pipeline_core` | `Triton spec`, `provider hooks`, `execution policy` | `pipeline report json`, `intentdialect *.mlir`, `llvm *.ll` | `pipeline/mlir_contract_artifacts.py`, `verify/diff_runner.py`, `backends/*/pipeline/driver.py` | `true` |
| `pipeline/tilelang/core.py:run_pipeline_for_spec` | `pipeline_core` | `TileLang spec` | `tilelang pipeline reports`, `intentdialect *.mlir`, `llvm *.ll` | `pipeline/mlir_contract_artifacts.py`, `verify/diff_runner.py`, `backends/*/pipeline/driver.py` | `true` |
| `pipeline/cuda/core.py:run_pipeline_for_spec` | `pipeline_core` | `CUDA frontend spec` | `cuda pipeline reports`, `intentdialect *.mlir`, `llvm *.ll` | `pipeline/mlir_contract_artifacts.py`, `verify/diff_runner.py`, `backends/*/pipeline/driver.py` | `true` |
| `backends/cuda/pipeline/driver.py:run_cuda_pipeline` | `backend_pipeline` | `MlirBackendContract | IntentMLIRModule | mlir text`, `shape bindings` | `stage results (legalize/shape_infer/schedule/emit/compile/launch)`, `timings`, `reason_code` | `backends/cuda/codegen/cpp_driver.py`, `backends/cuda/runtime.py` | `true` |
| `backends/spmd_rvv/pipeline/driver.py:run_rvv_pipeline` | `backend_pipeline` | `MlirBackendContract | IntentMLIRModule | mlir text`, `shape bindings` | `stage results (legalize/shape_infer/schedule/emit_cpp/compile/run)`, `timings`, `reason_code` | `backends/spmd_rvv/codegen/cpp_driver.py`, `backends/spmd_rvv/runtime/*` | `true` |
| `verify/diff_runner.py:run_diff` | `verification` | `IntentFunction`, `reference runner outputs`, `generated cases` | `DiffResult list`, `Counterexample list` | `pipeline/*/core.py reports`, `scripts/flaggems/run_multibackend_matrix.py` | `true` |
| `scripts/flaggems/build_workflow_state.py` | `workflow_state` | `feature_list/current_status/session_context/progress/handoff/artifacts` | `workflow/flaggems/state/current_status.json`, `workflow/flaggems/state/session_context.json` | `scripts/flaggems/ci_gate.py`, `workflow/flaggems/init.sh`, `workflow/flaggems/nightly.sh` | `true` |
| `scripts/flaggems/ci_gate.py` | `workflow_gate` | `run_summary`, `status_converged`, `workflow state` | `ci_gate.json` | `nightly maintenance`, `handoff decisions` | `true` |

## 3. 目录级职责映射（文件级入口）

- `frontends/`
  - Triton: `frontends/triton/adapter.py`, `frontends/triton/llm_intent.py`, `frontends/triton/facts.py`, `frontends/triton/contract.py`
  - TileLang: `frontends/tilelang/adapter.py`, `frontends/tilelang/llm_intent.py`, `frontends/tilelang/facts.py`
  - CUDA: `frontends/cuda/adapter.py`, `frontends/cuda/llm_intent.py`, `frontends/cuda/facts.py`
- `pipeline/`
  - 跨前端编排：`pipeline/triton/core.py`, `pipeline/tilelang/core.py`, `pipeline/cuda/core.py`
  - provider 插件：`pipeline/triton/providers/*`（含 `flaggems`）
  - backend contract 产物：`pipeline/mlir_contract_artifacts.py`
- `intent_ir/`
  - IR 基础：`intent_ir/ir/ir_types.py`
  - parser：`intent_ir/parser/parser_llm.py`
  - opset/specs：`intent_ir/ops/specs.py`, `intent_ir/ops/opset.py`
  - MLIR 子系统：`intent_ir/mlir/*`
- `backends/`
  - CUDA pipeline: `backends/cuda/pipeline/driver.py` + `backends/cuda/codegen/cpp_driver.py` + `backends/cuda/cpp_codegen/*`
  - RVV pipeline: `backends/spmd_rvv/pipeline/driver.py` + `backends/spmd_rvv/codegen/cpp_driver.py` + `backends/spmd_rvv/cpp_codegen/*`
  - shared contract: `backends/common/mlir_contract.py`
- `verify/`
  - diff 主逻辑：`verify/diff_runner.py`
  - case 生成：`verify/gen_cases.py`
  - 解释器：`verify/interpreter.py`
- `workflow/` + `scripts/flaggems/`
  - 状态快照：`scripts/flaggems/build_workflow_state.py`
  - 覆盖批次：`build_coverage_batches.py` + `run_coverage_batches.py` + `aggregate_coverage_batches.py`
  - 门禁：`check_batch_gate.py` + `ci_gate.py`

## 4. 主执行链（Triton/TileLang/CUDA）

1. `scripts/intentir.py suite|kernel` 解析参数与 suite 选择。
2. 调用对应 frontend pipeline core，构建 Intent/MLIR 工件并运行 pass。
3. `pipeline/mlir_contract_artifacts.py` 从 MLIR 模块发射 CUDA/RVV contract JSON。
4. backend driver (`run_cuda_pipeline` / `run_rvv_pipeline`) 执行 legalize->shape_infer->schedule->emit->compile->launch/run。
5. `verify/diff_runner.py` 完成与基线输出的 diff。
6. `scripts/flaggems/*gate*.py` 汇总 run_summary/status_converged 并刷新 workflow 状态。