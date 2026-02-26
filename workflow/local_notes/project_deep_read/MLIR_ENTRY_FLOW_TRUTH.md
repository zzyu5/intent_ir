# MLIR 入口与执行链路真相文档

## 1. 文档目的

这份文档只回答三个问题：

1. 入口到底在哪里。
2. 现在代码实际走的执行逻辑是什么。
3. 为什么 backends 里还保留 cpp 相关路径，以及当前是否能删。

本文件仅基于仓库静态代码和已落盘 artifact，不引入“计划中但未实现”的推断。

## 2. 当前快照与口径

- 分支：`compiler-cleanup-v1`
- 当前 git HEAD：`71ab9429887e65306d00853e0a1fb4a66982c1da`（命令：`git rev-parse HEAD`）
- `workflow/flaggems/state/current_status.json` 记录的 `head_commit` 仍是 `34e839...`：`workflow/flaggems/state/current_status.json:5`

说明：workflow state 目前不是最新 HEAD 的新鲜快照；下面的“实现逻辑”以代码为准，“通过率/性能”以对应 artifact commit 为准。

## 3. 顶层入口（你问的“入口在哪里”）

### 3.1 主入口脚本

- 顶层 CLI 入口：`scripts/intentir.py`
- `suite` 分发入口：`scripts/intentir.py:76`
- `kernel` 分发入口：`scripts/intentir.py:296`

### 3.2 MLIR-only 强制点

- 默认执行 IR 固定 `mlir`：`scripts/intentir.py:51`
- `--execution-ir` 只允许 `mlir`：`scripts/intentir.py:606`、`scripts/intentir.py:683`
- 覆盖批次 runner 显式拒绝非 mlir：`scripts/flaggems/run_coverage_batches.py:537`
- Triton core 也忽略非 mlir 并强制 mlir：`pipeline/triton/core.py:194`
- 矩阵 runner 同样强制 mlir：`scripts/flaggems/run_multibackend_matrix.py:122`

结论：从脚本入口层面，已经是 MLIR-only。

## 4. 三条主执行链路

## 4.1 full196（coverage）链路

### 4.1.1 入口分发

`scripts/intentir.py suite --suite flaggems-full196` 会按顺序执行：

1. 生成覆盖批次：`scripts/flaggems/build_coverage_batches.py`（`scripts/intentir.py:82-90`）
2. 执行批次：`scripts/flaggems/run_coverage_batches.py`（`scripts/intentir.py:91-138`）

### 4.1.2 family/chunk 执行

`run_coverage_batches.py` 里每个 chunk 实际调用：

- `scripts/flaggems/run_multibackend_matrix.py`：`scripts/flaggems/run_coverage_batches.py:776-828`

这一步不是单后端；会继续拆成 pipeline + rvv/cuda + converge。

### 4.1.3 聚合与 full196 证据

- family/chunk 输出再进入聚合脚本：`scripts/flaggems/aggregate_coverage_batches.py`
- 聚合写入 full196 关键字段：
  - `mlir_llvm_artifact_complete`：`scripts/flaggems/aggregate_coverage_batches.py:1057`
  - `runtime_fallback_kernel_count`：`scripts/flaggems/aggregate_coverage_batches.py:1062`
  - `stages.mlir_llvm_artifacts`：`scripts/flaggems/aggregate_coverage_batches.py:1088-1094`

## 4.2 gpu_perf graph 链路

### 4.2.1 入口分发

`scripts/intentir.py suite --suite gpu-perf-graph` 会执行：

1. 先 build coverage batches（复用 family 定义）：`scripts/intentir.py:140-148`
2. 再跑 `scripts/flaggems/run_gpu_perf_graph.py`：`scripts/intentir.py:149-184`

### 4.2.2 intentir 侧执行

`run_gpu_perf_graph.py` 中 IntentIR 路径核心是：

- `lower_cuda_contract_to_kernel(...)`：`scripts/flaggems/run_gpu_perf_graph.py:997`
- 执行引擎标记 `mlir_native`：`scripts/flaggems/run_gpu_perf_graph.py:1107`
- runtime fallback 标记来自 `cuda_ptx_origin != llvm_llc`：`scripts/flaggems/run_gpu_perf_graph.py:1099-1111`

### 4.2.3 native baseline 侧

- native callable 构建入口：`scripts/flaggems/run_gpu_perf_graph.py:796`
- 已有 kernel adapter 显式适配（比如 add2d/clamp2d/attention/unique 等）：`scripts/flaggems/run_gpu_perf_graph.py:630-791`
- adapter 优先于启发式签名绑定：`scripts/flaggems/run_gpu_perf_graph.py:846-861`

## 4.3 multibackend matrix 链路（smoke/coverage/mlir wave）

`scripts/flaggems/run_multibackend_matrix.py` 的阶段顺序：

1. pipeline（生成每 kernel provider 报告）：`scripts/flaggems/run_multibackend_matrix.py:769-807`
2. mlir_llvm_artifacts 聚合：`scripts/flaggems/run_multibackend_matrix.py:830-853`
3. rvv_local（可选）：`scripts/flaggems/run_multibackend_matrix.py:887-920`
4. rvv_remote（可选）：`scripts/flaggems/run_multibackend_matrix.py:922-971`
5. cuda_local：`scripts/flaggems/run_multibackend_matrix.py:973+`
6. converge_status 汇总：`scripts/flaggems/run_multibackend_matrix.py:1069+`

## 5. 后端执行到底在哪（不是 common 执行器）

## 5.1 `backends/common` 的定位

- `backends/common/mlir_contract.py` 是 contract schema（`MlirBackendContract`）：`backends/common/mlir_contract.py:51`
- 这里只定义数据结构，不负责 CUDA/RVV 执行。

## 5.2 CUDA 实际执行点

- 入口：`backends/cuda/pipeline/driver.py:404` `lower_cuda_contract_to_kernel`
- strict LLVM PTX 门禁：`INTENTIR_CUDA_REQUIRE_LLVM_PTX`，拒绝非 `llvm_llc`：`backends/cuda/pipeline/driver.py:419`、`428-432`、`476-480`
- 返回执行引擎：`execution_engine=mlir_native`：`backends/cuda/pipeline/driver.py:469`

## 5.3 RVV 实际执行点

- contract executable 解析：`backends/spmd_rvv/pipeline/driver.py:102`
- 仅接受 `rvv_elf/elf`（legacy cpp-driver fallback removed）：`backends/spmd_rvv/pipeline/driver.py:129-133`
- remote 侧执行计划解析：`scripts/rvv_remote_run.py:675`
  - `prebuilt_elf`：`scripts/rvv_remote_run.py:717-726`
  - `remote_llvm`：`scripts/rvv_remote_run.py:727-736`
  - compat 仅显式开启：`scripts/rvv_remote_run.py:738-747`、`scripts/rvv_remote_run.py:1098`

结论：当前是“common contract + backend-specific driver”，不是统一后端执行器。

## 6. 为什么还有 cpp 代码，能不能删

## 6.1 还在被调用的 cpp/fallback 路径

### CUDA 侧

- LLVM->PTX 失败时可退回 cpp codegen + nvcc/nvrtc（非 strict 才允许）：
  - `pipeline/mlir_contract_artifacts.py:868-903`

### RVV 侧

- 兼容 C 产物仅在 `INTENTIR_RVV_EMIT_COMPAT_C_SRC=1` 时写出：
  - `pipeline/mlir_contract_artifacts.py:986-1006`

### LLVM dialect pass

- CUDA 分支失败会 fallback 到 C->LLVM：`intent_ir/mlir/passes/lower_intent_to_llvm_dialect.py:121-137`
- RVV/默认路径仍是 C->LLVM：`intent_ir/mlir/passes/lower_intent_to_llvm_dialect.py:139-165`

## 6.2 结论

- 现在不能“直接删除全部 cpp 相关代码”。
- 当前状态是“硬切主路径已建立，compat/fallback 仍保留为受控兜底”。
- 删除动作应以 strict gate 在当前 HEAD 完整复验通过为前提，并分 backend 逐段移除。

## 7. 当前 gate/状态逻辑怎么判定

## 7.1 converge_status 对 fallback 的处理

- `compat_cpp_codegen` 被定义为 forbidden fallback：`scripts/flaggems/converge_status.py:280-297`
- 即使 runtime 双后端通过，若命中 forbidden fallback 仍打回 `blocked_backend`：`scripts/flaggems/converge_status.py:400-405`

## 7.2 ci_gate/check_batch_gate 对 mlir_native 的要求

- `execution_engine` 必须是 `mlir_native`：
  - `scripts/flaggems/ci_gate.py:467-487`
  - `scripts/flaggems/check_batch_gate.py:179-199`
- 仍会扫描 `cpp_codegen/compat_cpp_codegen/nvrtc_fallback_from_llvm` 等 fallback 标记：
  - `scripts/flaggems/ci_gate.py:523-533`
  - `scripts/flaggems/check_batch_gate.py:235-246`

## 7.3 workflow state 的 MLIR 判定

- `mlir_llvm_chain_ok` 取值来自 run_summary 的 artifact 证据，不再靠 timing 推断：`scripts/flaggems/build_workflow_state.py:278-318`
- `mlir_cutover_level` 判定顺序：`scripts/flaggems/build_workflow_state.py:845-856`

## 8. 现阶段“到底切到什么地步”

1. 入口层：MLIR-only（已完成）。
2. 执行层：CUDA/RVV 均走 backend driver 的 contract-first 主路径（已完成）。
3. 兼容层：cpp/fallback 仍在代码里，默认尽量不走，部分由显式 env 开关控制（未完全移除）。
4. workflow freshness：当前 `current_status.json` 与真实 HEAD 存在 1 个 commit 差（需重建 state）。

## 9. 我们现在到底在测什么（静态对应）

1. gpu_perf native adapter 与分母补齐：`tests/frontends/triton/test_flaggems_gpu_perf_graph_native_launch.py:122`
2. RVV remote 硬切执行计划（prebuilt_elf/remote_llvm 优先）：`tests/backends/spmd_rvv/test_rvv_remote_run_bytes.py:101`
3. compat fallback 禁止计入双通过：`tests/frontends/triton/test_flaggems_converge_status.py:100`
4. strict artifact 收集规则（pipeline-specific contract）：`tests/frontends/triton/test_flaggems_run_multibackend_matrix_artifacts.py:20`

## 10. 与现有证据的一致性说明

- full196 196/196 与 gpu_perf 159/159 的强证据在 `20260226` 目录内成立，但绑定 commit 是 `34e839...`（见各 run_summary 的 `repo.head_commit`）。
- 当前代码 HEAD 是 `71ab942...`，因此要回答“当前 HEAD 是否已验证”必须重新刷新 state/或重跑最小验证。

## 11. FlagGems 到底是怎么调用起来的（端到端）

### 11.1 从 suite 到每个 kernel 的真实调用链

1. `scripts/intentir.py suite --suite flaggems-full196` 进入 coverage 批次执行，最终每个 chunk 调 `scripts/flaggems/run_multibackend_matrix.py`：`scripts/flaggems/run_coverage_batches.py:776-828`。
2. `run_multibackend_matrix.py` 的 pipeline 阶段固定调用 `scripts/triton/flaggems_full_pipeline_verify.py`：`scripts/flaggems/run_multibackend_matrix.py:769-807`。
3. `flaggems_full_pipeline_verify.py` 选择 specs（smoke/coverage），并对每个 spec 调统一 Triton pipeline：`run_pipeline_for_spec(...)`：`scripts/triton/flaggems_full_pipeline_verify.py:40-49`、`scripts/triton/flaggems_full_pipeline_verify.py:201-208`。

### 11.2 original vs intentir 两条路径如何分流

1. `--flaggems-path original|intentir` + `--intentir-mode` 先由 `resolve_flaggems_execution(...)` 解析成执行策略：`pipeline/triton/providers/flaggems/execution.py:39-88`。
2. 若 `original`，core 在 stage5 明确走 `traditional_provider_path`，不构造 IntentIR：`pipeline/triton/core.py:2951-2981`。
3. 若 `intentir`，core 在 stage5 走“seed cache/LLM/确定性 fallback -> IntentIR -> MLIR -> backend contract”：`pipeline/triton/core.py:2987-3255`。

结论：FlagGems 不是“旁路”；它是通过 Triton 统一 pipeline 的 provider 分支接入，按 `original|intentir` 明确分流。

## 12. FlagGems 是怎么“拿到 Triton kernel”的

### 12.1 kernel source 从哪里来

1. 每个 FlagGems kernel 都是 `KernelSpec(name,module,attr,runner,...)`，例如 `add2d`：`pipeline/triton/providers/flaggems/specs.py:5875-5890`。
2. `attr` 指向 `FLAGGEMS_*_SRC`，而这类对象由 `_LazyModuleSource` 按需 import `flag_gems.ops.*` 并读取模块源码文本：`pipeline/triton/providers/flaggems/specs.py:58-83`、`pipeline/triton/providers/flaggems/specs.py:85-90`。
3. Triton adapter 在 stage2 直接 `str(_import_attr(spec.module, spec.attr))` 装入 descriptor 的 `source_text`：`frontends/triton/adapter.py:85-99`。

### 12.2 TTIR 是怎么来的

1. FlagGems runner 在 `with flag_gems.use_gems(include=[...])` 环境内执行 torch op（示例 add2d）：`pipeline/triton/providers/flaggems/specs.py:258-275`。
2. core stage3 会先实际跑一次 runner 触发 Triton 编译与 dump，再扫描 TTIR：`pipeline/triton/core.py:2857-2864`、`pipeline/triton/core.py:2876`。
3. adapter 也有兜底：若没拿到 TTIR，再补跑一次 runner 后重试：`frontends/triton/adapter.py:151-164`。

结论：我们拿到 kernel 的方式是“源码上下文 + 一次真实运行触发的 TTIR dump”，不是手工直接抓某个预编译 kernel 对象。

## 13. 与其他 triton/tilelang 路径到底有什么不同

### 13.1 Triton 内部：native vs flaggems

1. provider 注册只有 `native` 和 `flaggems`：`pipeline/triton/providers/registry.py:8-11`。
2. 两者共用 `pipeline/triton/core.py`，区别在 spec/provider plugin：
   - native spec 主要指向 `kernels.triton.ops.*`：`pipeline/triton/core.py:2275-2325`、`pipeline/triton/core.py:2393-2431`。
   - flaggems spec 指向 `flag_gems.ops.*` 源码 lazy loader + FlagGems runner：`pipeline/triton/providers/flaggems/specs.py:5-10`、`pipeline/triton/providers/flaggems/specs.py:58-87`。
3. flaggems 还加了 provider 级 canonical deterministic 逻辑（seed/normalize/repair），native 没这层专用策略：`pipeline/triton/providers/flaggems/plugin.py:166-167`、`pipeline/triton/providers/flaggems/plugin.py:231-249`。
4. semantic op -> e2e spec 的映射由 FlagGems registry 维护：`pipeline/triton/providers/flaggems/registry.py:63-66`。

### 13.2 TileLang 与 Triton 的差异

1. TileLang 入口脚本是 `scripts/tilelang/full_pipeline_verify.py`，走 `pipeline.tilelang.core.run_pipeline_for_spec`，不是 Triton core：`scripts/tilelang/full_pipeline_verify.py:18`、`scripts/tilelang/full_pipeline_verify.py:44`。
2. TileLang adapter 输入是 `tvm.tir.PrimFunc`，descriptor `source_kind=ir`，并可落盘 TVM JSON IR：`frontends/tilelang/adapter.py:43-51`、`frontends/tilelang/adapter.py:90-101`。
3. TileLang 还可直接导出 CUDA/PTX（生成器路径）：`frontends/tilelang/cuda_export.py:59-76`。
4. 但 TileLang core 仍会走 IntentIR -> MLIR -> backend contract 同类链路：`pipeline/tilelang/core.py:2714-2721`、`pipeline/tilelang/core.py:2770-2778`、`pipeline/tilelang/core.py:2872-2883`。

结论：FlagGems 与 native Triton 的差别在 provider/spec/seed 策略；TileLang 的差别在 frontend IR 获取方式。后端 contract 执行面仍是后端分叉驱动，不是“统一执行器”。

## 14. “我们的 MLIR”具体长什么样（文件形态 + 真实样例）

### 14.1 产物命名与阶段

`pipeline/triton/core.py` 的 `_emit_mlir_shadow_artifacts` 会产出以下核心文件：`pipeline/triton/core.py:238-267`、`pipeline/triton/core.py:280-330`、`pipeline/triton/core.py:406-415`。

1. `*.intentir.intentdialect.mlir`：Intent dialect 初始模块。
2. `*.intentir.intentdialect.midend.mlir`：midend 后模块。
3. `*.intentir.intentdialect.downstream_{cuda|rvv}.mlir`：后端下游语义模块。
4. `*.intentir.intentdialect.downstream_{cuda|rvv}_llvm.mlir` 与 `.ll`：LLVM 方言/文本 IR。
5. `*.intentir.intentdialect.downstream_*_llvm.contract.json`：可执行 contract（schema v2）。

contract 写出逻辑在 `emit_backend_contract_artifacts(...)`：`pipeline/mlir_contract_artifacts.py:1164-1189`、`pipeline/mlir_contract_artifacts.py:1222-1278`，文件名规则为 `*.intentir.intentdialect.{suffix}.contract.json`：`pipeline/mlir_contract_artifacts.py:1160`。

### 14.2 真实样例（2026-02-26 full196 artifact）

样例路径（vector_norm2d）：

- `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v23_strict_o3/family_norm_activation/chunk_002/pipeline_reports/vector_norm2d.intentir.intentdialect.mlir`
- `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v23_strict_o3/family_norm_activation/chunk_002/pipeline_reports/vector_norm2d.intentir.intentdialect.downstream_cuda_llvm.mlir`
- `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v23_strict_o3/family_norm_activation/chunk_002/pipeline_reports/vector_norm2d.intentir.intentdialect.downstream_cuda_llvm.contract.json`

`intentdialect.mlir` 开头就是 Intent op（`intent.mul/reduce_sum/sqrt`）：

```mlir
module attributes {intent.dialect_version = "intent_dialect_v0"} {
  intent.func @vector_norm2d() {
    %sq = intent.mul(%inp, %inp) : !intent.tensor<?Mx?Nxf32>{layout="row_major"}
    %sum_sq = intent.reduce_sum(%sq) {dims=[1], keepdims=false} : !intent.tensor<?Mxf32>{layout="row_major"}
    %out = intent.sqrt(%sum_sq) : !intent.tensor<?Mxf32>{layout="row_major"}
    intent.return %out
  }
}
```

`downstream_cuda_llvm.mlir` 在当前链路下是 LLVM IR 文本（示例含 `target triple = "nvptx64-nvidia-cuda"`）：

- `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v23_strict_o3/family_norm_activation/chunk_002/pipeline_reports/vector_norm2d.intentir.intentdialect.downstream_cuda_llvm.mlir`

对应 contract 会写出执行来源与可执行产物路径，例如：

1. `schema_version = intent_mlir_backend_contract_v2`
2. `artifacts.cuda_ptx_origin = llvm_llc`
3. `executable.format = cuda_ptx`

见：

- `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v23_strict_o3/family_norm_activation/chunk_002/pipeline_reports/vector_norm2d.intentir.intentdialect.downstream_cuda_llvm.contract.json`
