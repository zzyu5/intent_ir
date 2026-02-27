# IntentIR MLIR 切换现状与后端执行真相（2026-02-27）

## 1. 一句话结论

- 从“主执行路径 + 现有 gate 证据”看，CUDA 与 RVV 都已走 `MLIR -> LLVM -> backend executable` strict 路径，且当前 HEAD 证据为 fresh。  
- 但“全部 kernel 都覆盖”不成立：当前主 gate 分母是 `196 semantic ops`（correctness）和 `159 kernels`（gpu perf），不是 source registry 全量 `302`。  
- `cpp codegen` 运行路径已物理移除（目录和入口模块已不存在），但仍保留了少量“检测 legacy tag 的 gate/测试代码”和 RVV 运行时 C harness 文件（这是运行时壳，不是旧 codegen）。

---

## 2. 当前 HEAD 与运行证据（实际执行面）

### 2.1 HEAD / freshness

- 当前 HEAD：`84a0aca4ed94968713480a740955f891b1000ac8`（`git rev-parse HEAD`）
- `workflow/flaggems/state/current_status.json` 显示：
  - `full196_validated_commit = 84a0aca...`
  - `gpu_perf_validated_commit = 84a0aca...`
  - `full196_commits_since_validated = 0`
  - `gpu_perf_commits_since_validated = 0`
  - `mlir_cutover_level = mlir_primary`
  - `mlir_llvm_chain_ok = true`
  - `mlir_backend_contract_ready = true`

证据文件：
- `workflow/flaggems/state/current_status.json`

### 2.2 full196（correctness）

- `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v30_strict_cache_o3_auto_h2/run_summary.json`
  - `ok = true`
  - `coverage_batches_completed = 7`
  - `coverage_batches_expected = 7`
  - `mlir_llvm_artifact_complete = true`
  - `runtime_fallback_kernel_count = 0`
  - `repo.head_commit = 84a0aca...`
- 对应 `status_converged.json`
  - `counts_global.dual_pass = 196`
  - `strict_mode = true`
  - `fallback_policy = strict`
  - `execution_engine = mlir_native`
  - `runtime_fallback_kernel_count = 0`

证据文件：
- `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v30_strict_cache_o3_auto_h2/run_summary.json`
- `artifacts/flaggems_matrix/daily/20260226/full196_head_refresh_v30_strict_cache_o3_auto_h2/status_converged.json`

### 2.3 gpu_perf（performance）

- `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v18_strict_postfix/run_summary.json`
  - `ok = true`
  - `gpu_perf_kernel_measured = 159`
  - `gpu_perf_categories_complete = true`
  - `gpu_perf_per_device_ok = true`
  - `repo.head_commit = 84a0aca...`
- 对应 `status_converged.json`
  - `counts_global.dual_pass = 159`
  - `strict_mode = true`
  - `fallback_policy = strict`
  - `execution_engine = mlir_native`
  - `runtime_fallback_kernel_count = 0`
- `gpu_perf_graph.json` 设备级最小比值：
  - `min_ratio = 0.8173812494935464`（>= 0.80）

补充：`p50` 未直接写在该版本 schema 顶层，我从 `entries[*].ratio` 计算得到：
- `p50 = 0.9046554254288934`

证据文件：
- `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v18_strict_postfix/run_summary.json`
- `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v18_strict_postfix/status_converged.json`
- `artifacts/flaggems_matrix/daily/20260226/gpu_perf_head_refresh_v18_strict_postfix/gpu_perf_graph.json`

---

## 3. 你的问题逐条回答

### Q1：MLIR 路径“全部走通”了吗？

结论：在当前 gate 覆盖范围内，走通了。

- 运行证据显示 strict + no fallback：见上面 full196/gpu_perf 两套 `run_summary + status_converged`。
- 静态代码也收紧为 MLIR contract 可执行物：
  - CUDA 只接受 `executable.format in {cuda_ptx, ptx}`，否则报错“fallback removed”：`backends/cuda/pipeline/driver.py:848`、`backends/cuda/pipeline/driver.py:961`
  - RVV 只接受 `executable.format in {rvv_elf, elf}`，否则报错“legacy cpp-driver fallback removed”：`backends/spmd_rvv/pipeline/driver.py:101`、`backends/spmd_rvv/pipeline/driver.py:128`
  - LLVM lowering 不再接受 C/CUDA-C fallback：`intent_ir/mlir/passes/lower_intent_to_llvm_dialect.py:8`、`intent_ir/mlir/passes/lower_intent_to_llvm_dialect.py:32`

### Q2：kernel 都覆盖了吗？

结论：主 gate 覆盖完成，但不是全 registry 全量覆盖。

- correctness 主分母：`196 semantic ops`（`dual_pass=196`）
- perf 主分母：`159 measured kernels`（`dual_pass=159`）
- source registry 总量：`302`，过滤后 `212`，主语义集 `196`

证据文件：
- `workflow/flaggems/state/coverage_batches.json`（`summary.semantic_ops_total=196`）
- `artifacts/flaggems_matrix/daily/20260227/registry_gap_report_v1.json`（`302/212/196`）
- `artifacts/flaggems_matrix/daily/20260227/registry_expansion_wave1_candidates_v1.json`（额外分母扩展候选）

### Q3：cpp 路径都删除了吗（CUDA+RVV）？

结论：执行路径上的 cpp codegen 已删除；遗留的是检测逻辑和少量注释/测试文字，不是运行入口。

- 目录层面：
  - `backends/cuda/cpp_codegen` 不存在
  - `backends/spmd_rvv/codegen` 不存在
- 测试显式检查入口已删除：
  - `tests/backends/cuda/test_cuda_pipeline_driver.py:210`
  - `tests/backends/spmd_rvv/test_rvv_remote_run_bytes.py:37`
- 仍存在 `compat_cpp_codegen` 字样的位置主要是：
  - gate/strict_policy 用于识别并阻断 legacy 标签
  - 测试用例构造 forbidden fallback 场景

注意：RVV 目录里有 `intentir_runtime.c/intentir_driver.c/intentir_ops.c`（`backends/spmd_rvv/runtime`），这是 remote 执行时的运行时壳代码，不是旧 cpp codegen。

---

## 4. FlagGems 到底怎么被调用、怎么拿到 Triton kernel

## 4.1 suite 到 runner 的调度

- 顶层入口：`scripts/intentir.py`
  - `--suite flaggems-full196` -> `build_coverage_batches.py` + `run_coverage_batches.py`：`scripts/intentir.py:82`
  - `--suite gpu-perf-graph` -> `build_coverage_batches.py` + `run_gpu_perf_graph.py`：`scripts/intentir.py:140`

- coverage 里每个 family/chunk 再调用：
  - `scripts/flaggems/run_multibackend_matrix.py`：`scripts/flaggems/run_coverage_batches.py:806`

- matrix 的 pipeline 阶段调用：
  - `scripts/triton/flaggems_full_pipeline_verify.py`：`scripts/flaggems/run_multibackend_matrix.py:769`

## 4.2 FlagGems 与 Triton native 的关系

- provider 只分 `native` / `flaggems`：`pipeline/triton/providers/registry.py:8`
- FlagGems 不引入新 frontend，仍复用 Triton core：`pipeline/triton/providers/flaggems/specs.py:5`
- `--flaggems-path original|intentir` 只决定是否走 IntentIR：`pipeline/triton/providers/flaggems/execution.py:39`

## 4.3 “拿 kernel” 的方式

- spec 中 `module + attr` 定位到具体 Triton kernel 源：
  - `frontends/triton/adapter.py:85`（`build_descriptor`）
  - `frontends/triton/adapter.py:88`（读 `source_text`）
- FlagGems 源是 lazy 读取 `flag_gems.ops.*` 模块文件：`pipeline/triton/providers/flaggems/specs.py:77`
- stage3 会真实跑一次 runner 触发 Triton dump TTIR：`pipeline/triton/core.py:2881`
- adapter 会把 TTIR 拷贝入 artifact：`frontends/triton/adapter.py:151`

---

## 5. 后端“现在到底在干什么”

## 5.1 Contract 是统一边界，不是统一执行器

- 统一 schema：`MlirBackendContract`（JSON）在 `backends/common/mlir_contract.py:51`
- 真正执行仍是 backend 分叉 driver：
  - CUDA：`backends/cuda/pipeline/driver.py:848`
  - RVV：`backends/spmd_rvv/pipeline/driver.py:101`

## 5.2 CUDA 后端链路

1. 下游 LLVM pipeline：`intent_ir/mlir/pipelines/downstream_cuda_llvm.yaml`
2. materialize：LLVM IR -> PTX（`llc nvptx`），并写 `cuda_ptx_origin=llvm_llc`：
   - `pipeline/mlir_contract_artifacts.py:829`
   - `pipeline/mlir_contract_artifacts.py:860`
3. driver 读取 contract executable（PTX）并 launch：
   - `backends/cuda/pipeline/driver.py:864`
4. strict 下若不是 `llvm_llc` 直接拒绝：`backends/cuda/pipeline/driver.py:876`

## 5.3 RVV 后端链路

1. 下游 LLVM pipeline：`intent_ir/mlir/pipelines/downstream_rvv_llvm.yaml`
2. materialize：LLVM IR -> ELF（`llc + clang`），写 `rvv_elf_origin=llvm_llc` 与 `rvv_compat_removed=true`：
   - `pipeline/mlir_contract_artifacts.py:892`
   - `pipeline/mlir_contract_artifacts.py:916`
   - `pipeline/mlir_contract_artifacts.py:923`
3. remote 执行计划只允许两种模式：
   - `prebuilt_elf`
   - `remote_llvm`
   解析在 `scripts/rvv_remote_run.py:598`
4. 若不满足（无 RVV ELF 且无 RVV triple LLVM），直接 fail：`scripts/rvv_remote_run.py:652`

说明：RVV remote 仍会上传并编译 `intentir_runtime.c/intentir_driver.c/intentir_ops.c` 作为远端运行时壳，见 `scripts/rvv_remote_run.py:1227`；这不是 legacy cpp codegen 回退。

---

## 6. 你看到“还像旧路径”的几个点，分别是什么

1. `compat_cpp_codegen` 文本仍存在：主要在 gate/strict_policy/测试里用于“识别并拦截旧标签”，不是运行调用点。  
2. `backends/cuda/opset.py` 顶部注释仍写了 `cpp_driver.py`：这是注释过时，不代表真实依赖（建议单独清理）。  
3. `run_gpu_perf_graph.py` 里还 import `compile_cuda_extension`：当前 strict contract 降级为 PTX executable，正常路径不会走到该分支；该分支更像保守兼容壳。

---

## 7. 现阶段可操作的后续清理（不改变主结论）

1. 文案清理：修 `backends/cuda/opset.py` 的过时注释。  
2. 死分支清理：评估 `run_gpu_perf_graph.py` 与 `scripts/cuda_backend_smoke.py` 中 `cuda_src` 分支是否可直接删。  
3. 扩分母推进：按 `302 -> 212 -> 196` gap 报告推进 wave，保持“主 gate 稳定 + 扩展 lane 增量引入”。

