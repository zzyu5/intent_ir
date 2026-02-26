# MLIR_POSITION_REPORT

## 1. `intent_ir/mlir/` 模块清单

- 核心类型与转换：`module.py`, `types.py`, `convert_from_intent.py`, `convert_to_intent.py`
- pass 执行器：`pass_manager.py`
- pass 集合：`passes/attach_provider_meta.py`, `normalize_symbols.py`, `canonicalize_intent.py`, `cse_like.py`, `expand_macros_intent.py`, `backend_legalize.py`, `emit_cuda_contract.py`, `emit_rvv_contract.py`, `lower_intent_to_llvm_dialect.py`, `ensure_llvm_ir_text.py`
- pipeline 配置：`pipelines/upstream.yaml`, `midend.yaml`, `downstream_cuda.yaml`, `downstream_cuda_llvm.yaml`, `downstream_rvv.yaml`, `downstream_rvv_llvm.yaml`
- 工具链探测：`toolchain.py`

## 2. MLIR 在系统中的真实位置（证据化）

### 上游（frontend/pipeline）
- `pipeline/triton/core.py`, `pipeline/tilelang/core.py`, `pipeline/cuda/core.py` 均显式调用 `to_mlir()` 与 `run_mlir_pipeline()`。
- 三个 core 都会输出 `intentdialect` 系列工件与 pass trace，并写入 report 的 `mlir` 字段。

### 中游（pass）
- `upstream -> midend -> downstream_*` pipeline 由 `intent_ir/mlir/pass_manager.py` 统一驱动。
- 额外存在 `downstream_*_llvm` 管线，生成 LLVM 相关工件与 `llvm_ir_path`。

### 下游（backend contract）
- `pipeline/mlir_contract_artifacts.py` 使用 `emit_cuda_contract` / `emit_rvv_contract` 生成 `MlirBackendContract`。
- `backends/cuda/pipeline/driver.py` 与 `backends/spmd_rvv/pipeline/driver.py` 输入契约支持 `MlirBackendContract | IntentMLIRModule | mlir text`。
- `backends/common/mlir_contract.py` 是共享 contract 类型。

## 3. “主表示 vs 工件化”结论

- 结论：**MLIR 已是主执行工件与 backend contract 生成来源，但前端提取入口仍从 IntentFunction 起步**。
- 也就是说当前是“IntentFunction 作为入口数据结构 + MLIR 作为主 pass/contract 执行介质”的双层形态，不是只做打印工件。
- `backends/common/mlir_bridge.py` 源文件已不在主代码目录（仅 pycache 残留），说明主路径已明显向 contract-first 收敛。

## 4. 仍存在的兼容痕迹

- `pipeline/*/core.py` 仍在多个分支写出 `*.intentir.fallback*.mlir` 兼容工件，说明迁移未完全去掉历史命名与回退语义。
- workflow freshness 未对齐 HEAD 时，会出现“MLIR 状态看起来完成但证据不是当前 commit”的认知偏差。