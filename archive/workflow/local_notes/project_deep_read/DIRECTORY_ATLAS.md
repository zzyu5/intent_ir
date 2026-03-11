# DIRECTORY_ATLAS

## 1. 全仓顶层目录总览（13/13）

| 顶层目录 | category | file_count(recursive) | size | 说明 |
|---|---:|---:|---:|---|
| `archive` | `archive_history` | 24641 | 14.14 GB | 历史论文/实验/旧脚本与旧测试归档区，不作为主运行路径。 |
| `artifacts` | `generated_artifacts` | 582331 | 14.72 GB | 运行工件与中间产物目录，体量最大，需按证据索引读取。 |
| `backends` | `runtime_code` | 139 | 2.28 MB | CUDA 与 RVV 后端实现及 pipeline 驱动。 |
| `docs` | `docs` | 8 | 19.66 KB | 维护文档与架构说明。 |
| `frontends` | `runtime_code` | 83 | 777.88 KB | Triton/TileLang/CUDA 前端提取与约束逻辑。 |
| `intent_ir` | `runtime_code` | 101 | 740.06 KB | IntentIR 核心类型、parser、macro 与 MLIR 子系统。 |
| `kernels` | `runtime_code` | 280 | 1.28 MB | 三类前端的样例 kernel 源。 |
| `pipeline` | `runtime_code` | 50 | 2.33 MB | 跨前端统一编排层，衔接 verify 与 backends。 |
| `requirements` | `config` | 6 | 2.62 KB | 环境依赖与锁文件。 |
| `scripts` | `runtime_code` | 66 | 1.34 MB | 统一 CLI 与 lane/gate 运行脚本。 |
| `tests` | `tests` | 72 | 1.15 MB | 门禁与契约验证测试集合。 |
| `verify` | `runtime_code` | 17 | 464.49 KB | 差分验证、解释器、测试样例生成。 |
| `workflow` | `workflow_state` | 99 | 689.97 KB | 长任务状态、nightly 调度、会话接力上下文。 |

## 2. 三级结构树（depth<=3）

### `archive`

- `archive/data`  (files=1, size=5.88 KB)
- `archive/doc`  (files=100, size=16.89 MB)
  - `archive/doc/backend` (files=2, size=14.22 KB)
  - `archive/doc/old_version(not_used)` (files=18, size=199.43 KB)
  - `archive/doc/paper` (files=66, size=16.37 MB)
- `archive/experiment`  (files=24451, size=14.12 GB)
  - `archive/experiment/AI-Benchmark` (files=84, size=5.45 MB)
  - `archive/experiment/FlagGems` (files=1824, size=24.80 MB)
  - `archive/experiment/triton-cpu` (files=22543, size=14.09 GB)
- `archive/scripts`  (files=28, size=575.42 KB)
  - `archive/scripts/cuda` (files=0, size=0.00 B)
  - `archive/scripts/experiments` (files=15, size=488.19 KB)
  - `archive/scripts/legacy` (files=11, size=64.60 KB)
  - `archive/scripts/tilelang` (files=0, size=0.00 B)
  - `archive/scripts/tools` (files=2, size=22.64 KB)
- `archive/tests`  (files=60, size=302.97 KB)
  - `archive/tests/backends` (files=13, size=122.89 KB)
  - `archive/tests/frontends` (files=30, size=128.14 KB)
  - `archive/tests/golden` (files=7, size=20.89 KB)
  - `archive/tests/intentir` (files=1, size=5.54 KB)

> 注：`archive` 体量巨大，本节只列代表性高体量子域；完整机器索引见 `directory_index.json`。

### `artifacts`

- `artifacts/_tmp_fig_view`  (files=3, size=4.89 MB)
- `artifacts/cuda_full_pipeline`  (files=356, size=4.41 MB)
- `artifacts/experiments`  (files=425, size=34.52 MB)
  - `artifacts/experiments/E1` (files=1, size=24.58 KB)
  - `artifacts/experiments/E1E3` (files=32, size=2.86 MB)
  - `artifacts/experiments/E2` (files=22, size=396.14 KB)
  - `artifacts/experiments/E4` (files=33, size=22.04 MB)
  - `artifacts/experiments/E5` (files=277, size=6.09 MB)
  - `artifacts/experiments/E6` (files=32, size=2.72 MB)
  - `artifacts/experiments/paper` (files=18, size=206.26 KB)
  - `artifacts/experiments/summaries` (files=8, size=187.88 KB)
- `artifacts/flaggems_matrix`  (files=515591, size=11.38 GB)
  - `artifacts/flaggems_matrix/daily` (files=515544, size=11.38 GB)
  - `artifacts/flaggems_matrix/gather_index_batch_v1` (files=4, size=343.79 KB)
  - `artifacts/flaggems_matrix/gather_index_batch_v2_cuda` (files=5, size=346.00 KB)
  - `artifacts/flaggems_matrix/isfinite_batch_v1` (files=4, size=372.68 KB)
  - `artifacts/flaggems_matrix/isfinite_batch_v2` (files=4, size=364.34 KB)
  - `artifacts/flaggems_matrix/norm_lerp_batch` (files=5, size=337.33 KB)
  - `artifacts/flaggems_matrix/ops_batch_v3` (files=4, size=372.55 KB)
  - `artifacts/flaggems_matrix/smoke_mlir_default_v1` (files=3, size=515.27 KB)
- `artifacts/flaggems_triton_full_pipeline`  (files=37977, size=868.95 MB)
  - `artifacts/flaggems_triton_full_pipeline/_triton_cache` (files=12376, size=273.68 MB)
  - `artifacts/flaggems_triton_full_pipeline/_triton_dump` (files=5825, size=169.59 MB)
  - `artifacts/flaggems_triton_full_pipeline/batch_active10_sort_topk_var_v1` (files=1764, size=49.38 MB)
  - `artifacts/flaggems_triton_full_pipeline/batch_conv_family_v1` (files=1530, size=189.85 MB)
  - `artifacts/flaggems_triton_full_pipeline/batch_isin_kron_lin_log_masked_v1` (files=829, size=18.49 MB)
  - `artifacts/flaggems_triton_full_pipeline/batch_isin_kron_lin_log_masked_v2` (files=829, size=18.49 MB)
  - `artifacts/flaggems_triton_full_pipeline/cpp_modsplit_wave2_validation_v1` (files=1184, size=15.66 MB)
  - `artifacts/flaggems_triton_full_pipeline/reduction_index_split_validation_v2` (files=1253, size=12.83 MB)
- `artifacts/flaggems_triton_full_pipeline_conv_family_v2`  (files=1973, size=166.58 MB)
  - `artifacts/flaggems_triton_full_pipeline_conv_family_v2/_triton_cache` (files=1278, size=100.01 MB)
  - `artifacts/flaggems_triton_full_pipeline_conv_family_v2/_triton_dump` (files=595, size=65.48 MB)
- `artifacts/flaggems_triton_full_pipeline_qr4_batch_v3`  (files=569, size=4.17 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr4_batch_v3/_triton_cache` (files=290, size=1.99 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr4_batch_v3/_triton_dump` (files=195, size=1.56 MB)
- `artifacts/flaggems_triton_full_pipeline_qr4_v4_nvrtc`  (files=569, size=4.17 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr4_v4_nvrtc/_triton_cache` (files=290, size=1.99 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr4_v4_nvrtc/_triton_dump` (files=195, size=1.56 MB)
- `artifacts/flaggems_triton_full_pipeline_qr5_v1_nvrtc`  (files=684, size=4.81 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr5_v1_nvrtc/_triton_cache` (files=354, size=2.24 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr5_v1_nvrtc/_triton_dump` (files=240, size=1.78 MB)
- `artifacts/flaggems_triton_full_pipeline_qr5_v2_nvrtc`  (files=684, size=4.81 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr5_v2_nvrtc/_triton_cache` (files=354, size=2.24 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr5_v2_nvrtc/_triton_dump` (files=240, size=1.78 MB)
- `artifacts/flaggems_triton_full_pipeline_qr5_v3_nvrtc`  (files=684, size=4.81 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr5_v3_nvrtc/_triton_cache` (files=354, size=2.24 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr5_v3_nvrtc/_triton_dump` (files=240, size=1.78 MB)
- `artifacts/flaggems_triton_full_pipeline_qr6_v1_nvrtc`  (files=894, size=15.97 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr6_v1_nvrtc/_triton_cache` (files=482, size=7.90 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr6_v1_nvrtc/_triton_dump` (files=325, size=7.23 MB)
- `artifacts/flaggems_triton_full_pipeline_qr6_v2_nvrtc`  (files=844, size=13.92 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr6_v2_nvrtc/_triton_cache` (files=452, size=6.92 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr6_v2_nvrtc/_triton_dump` (files=305, size=6.29 MB)
- `artifacts/flaggems_triton_full_pipeline_qr6_v3_nvrtc`  (files=844, size=13.92 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr6_v3_nvrtc/_triton_cache` (files=452, size=6.92 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr6_v3_nvrtc/_triton_dump` (files=305, size=6.29 MB)
- `artifacts/flaggems_triton_full_pipeline_qr6_v4_nvrtc`  (files=844, size=13.92 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr6_v4_nvrtc/_triton_cache` (files=452, size=6.92 MB)
  - `artifacts/flaggems_triton_full_pipeline_qr6_v4_nvrtc/_triton_dump` (files=305, size=6.29 MB)
- `artifacts/flaggems_triton_full_pipeline_twave_p0check_nvrtc`  (files=688, size=5.84 MB)
  - `artifacts/flaggems_triton_full_pipeline_twave_p0check_nvrtc/_triton_cache` (files=400, size=3.11 MB)
  - `artifacts/flaggems_triton_full_pipeline_twave_p0check_nvrtc/_triton_dump` (files=195, size=1.90 MB)
- `artifacts/flaggems_triton_full_pipeline_twave_v1_nvrtc`  (files=987, size=8.39 MB)
  - `artifacts/flaggems_triton_full_pipeline_twave_v1_nvrtc/_triton_cache` (files=575, size=4.40 MB)
  - `artifacts/flaggems_triton_full_pipeline_twave_v1_nvrtc/_triton_dump` (files=320, size=3.15 MB)
- `artifacts/flaggems_triton_full_pipeline_twave_v2_nvrtc`  (files=987, size=8.39 MB)
  - `artifacts/flaggems_triton_full_pipeline_twave_v2_nvrtc/_triton_cache` (files=575, size=4.40 MB)
  - `artifacts/flaggems_triton_full_pipeline_twave_v2_nvrtc/_triton_dump` (files=320, size=3.15 MB)
- `artifacts/flaggems_triton_full_pipeline_twave_v3_nvrtc`  (files=733, size=6.06 MB)
  - `artifacts/flaggems_triton_full_pipeline_twave_v3_nvrtc/_triton_cache` (files=423, size=3.23 MB)
  - `artifacts/flaggems_triton_full_pipeline_twave_v3_nvrtc/_triton_dump` (files=210, size=1.98 MB)
- `artifacts/flaggems_triton_full_pipeline_twave_v4_nvrtc`  (files=733, size=6.06 MB)
  - `artifacts/flaggems_triton_full_pipeline_twave_v4_nvrtc/_triton_cache` (files=423, size=3.23 MB)
  - `artifacts/flaggems_triton_full_pipeline_twave_v4_nvrtc/_triton_dump` (files=210, size=1.98 MB)
- `artifacts/flaggems_triton_full_pipeline_twave_v5_nvrtc`  (files=733, size=6.07 MB)
  - `artifacts/flaggems_triton_full_pipeline_twave_v5_nvrtc/_triton_cache` (files=423, size=3.23 MB)
  - `artifacts/flaggems_triton_full_pipeline_twave_v5_nvrtc/_triton_dump` (files=210, size=1.98 MB)
- `artifacts/full_pipeline_verify`  (files=1103, size=25.35 MB)
  - `artifacts/full_pipeline_verify/_triton_cache` (files=469, size=5.95 MB)
  - `artifacts/full_pipeline_verify/_triton_dump` (files=290, size=4.52 MB)
- `artifacts/tilelang_full_pipeline`  (files=270, size=2.73 MB)
- `artifacts/toolchains`  (files=20, size=121.90 MB)
  - `artifacts/toolchains/mlir-14` (files=20, size=121.90 MB)
  - `artifacts/toolchains/mlir-current` (files=20, size=121.90 MB)
- `artifacts/torch_extensions`  (files=11613, size=2.01 GB)
  - `artifacts/torch_extensions/py312` (files=11613, size=2.01 GB)

> 注：`artifacts` 体量巨大，本节只列代表性高体量子域；完整机器索引见 `directory_index.json`。

### `backends`

- `backends/__pycache__`  (files=2, size=6.29 KB)
- `backends/common`  (files=9, size=35.51 KB)
  - `backends/common/__pycache__` (files=5, size=22.50 KB)
- `backends/cuda`  (files=61, size=755.29 KB)
  - `backends/cuda/__pycache__` (files=3, size=3.41 KB)
  - `backends/cuda/codegen` (files=4, size=24.87 KB)
  - `backends/cuda/cpp_codegen` (files=28, size=601.72 KB)
  - `backends/cuda/pipeline` (files=6, size=35.31 KB)
  - `backends/cuda/runtime` (files=16, size=85.83 KB)
- `backends/spmd_rvv`  (files=64, size=1.50 MB)
  - `backends/spmd_rvv/__pycache__` (files=3, size=8.13 KB)
  - `backends/spmd_rvv/analysis` (files=13, size=144.76 KB)
  - `backends/spmd_rvv/codegen` (files=4, size=10.89 KB)
  - `backends/spmd_rvv/cpp_codegen` (files=25, size=1.13 MB)
  - `backends/spmd_rvv/experiments` (files=3, size=16.11 KB)
  - `backends/spmd_rvv/pipeline` (files=6, size=39.17 KB)
  - `backends/spmd_rvv/runtime` (files=6, size=155.12 KB)

### `docs`


### `frontends`

- `frontends/__pycache__`  (files=1, size=463.00 B)
- `frontends/common`  (files=16, size=215.17 KB)
  - `frontends/common/__pycache__` (files=8, size=119.24 KB)
- `frontends/cuda`  (files=24, size=227.59 KB)
  - `frontends/cuda/__pycache__` (files=9, size=88.89 KB)
  - `frontends/cuda/ptx` (files=6, size=70.07 KB)
- `frontends/tilelang`  (files=17, size=119.62 KB)
  - `frontends/tilelang/__pycache__` (files=7, size=63.49 KB)
  - `frontends/tilelang/shims` (files=1, size=497.00 B)
- `frontends/triton`  (files=23, size=214.47 KB)
  - `frontends/triton/__pycache__` (files=10, size=111.68 KB)

### `intent_ir`

- `intent_ir/__pycache__`  (files=2, size=5.36 KB)
- `intent_ir/ir`  (files=10, size=319.43 KB)
  - `intent_ir/ir/__pycache__` (files=5, size=179.42 KB)
- `intent_ir/llm`  (files=8, size=80.83 KB)
  - `intent_ir/llm/__pycache__` (files=4, size=41.22 KB)
- `intent_ir/macros`  (files=14, size=74.80 KB)
  - `intent_ir/macros/__pycache__` (files=3, size=13.11 KB)
  - `intent_ir/macros/macro_lowering` (files=8, size=51.79 KB)
- `intent_ir/mlir`  (files=44, size=124.83 KB)
  - `intent_ir/mlir/__pycache__` (files=8, size=47.32 KB)
  - `intent_ir/mlir/passes` (files=22, size=44.54 KB)
  - `intent_ir/mlir/pipelines` (files=6, size=815.00 B)
- `intent_ir/ops`  (files=10, size=37.64 KB)
  - `intent_ir/ops/__pycache__` (files=5, size=19.49 KB)
- `intent_ir/parser`  (files=4, size=89.04 KB)
  - `intent_ir/parser/__pycache__` (files=2, size=42.78 KB)
- `intent_ir/utils`  (files=4, size=2.75 KB)
  - `intent_ir/utils/__pycache__` (files=2, size=1.73 KB)

### `kernels`

- `kernels/__pycache__`  (files=1, size=285.00 B)
- `kernels/cuda`  (files=136, size=951.78 KB)
  - `kernels/cuda/__pycache__` (files=1, size=451.00 B)
  - `kernels/cuda/ops` (files=134, size=951.05 KB)
- `kernels/tilelang`  (files=67, size=121.87 KB)
  - `kernels/tilelang/__pycache__` (files=1, size=342.00 B)
  - `kernels/tilelang/ops` (files=64, size=120.94 KB)
- `kernels/triton`  (files=74, size=241.07 KB)
  - `kernels/triton/__pycache__` (files=1, size=266.00 B)
  - `kernels/triton/ops` (files=54, size=219.64 KB)
  - `kernels/triton/support` (files=14, size=12.98 KB)
  - `kernels/triton/tools` (files=3, size=7.81 KB)

### `pipeline`

- `pipeline/__pycache__`  (files=5, size=14.30 KB)
- `pipeline/cuda`  (files=4, size=192.33 KB)
  - `pipeline/cuda/__pycache__` (files=2, size=107.75 KB)
- `pipeline/tilelang`  (files=4, size=285.75 KB)
  - `pipeline/tilelang/__pycache__` (files=2, size=154.37 KB)
- `pipeline/triton`  (files=31, size=1.83 MB)
  - `pipeline/triton/__pycache__` (files=3, size=206.68 KB)
  - `pipeline/triton/providers` (files=24, size=1.01 MB)

### `requirements`

- `requirements/lock`  (files=2, size=1.52 KB)

### `scripts`

- `scripts/__pycache__`  (files=5, size=235.04 KB)
- `scripts/cuda`  (files=2, size=5.53 KB)
  - `scripts/cuda/__pycache__` (files=1, size=3.45 KB)
- `scripts/flaggems`  (files=36, size=819.58 KB)
  - `scripts/flaggems/__pycache__` (files=13, size=392.54 KB)
- `scripts/intentir`  (files=10, size=60.15 KB)
  - `scripts/intentir/__pycache__` (files=5, size=35.07 KB)
- `scripts/tilelang`  (files=3, size=16.76 KB)
  - `scripts/tilelang/__pycache__` (files=1, size=3.63 KB)
- `scripts/triton`  (files=1, size=12.00 KB)

### `tests`

- `tests/__pycache__`  (files=8, size=186.76 KB)
- `tests/backends`  (files=13, size=148.31 KB)
  - `tests/backends/__pycache__` (files=3, size=9.51 KB)
  - `tests/backends/cuda` (files=5, size=91.19 KB)
  - `tests/backends/spmd_rvv` (files=4, size=46.94 KB)
- `tests/frontends`  (files=28, size=658.38 KB)
  - `tests/frontends/common` (files=0, size=0.00 B)
  - `tests/frontends/cuda` (files=0, size=0.00 B)
  - `tests/frontends/tilelang` (files=0, size=0.00 B)
  - `tests/frontends/triton` (files=28, size=658.38 KB)
- `tests/intentir`  (files=13, size=97.43 KB)
  - `tests/intentir/__pycache__` (files=7, size=77.16 KB)

### `verify`

- `verify/__pycache__`  (files=7, size=252.68 KB)

### `workflow`

- `workflow/flaggems`  (files=99, size=689.97 KB)
  - `workflow/flaggems/state` (files=93, size=677.58 KB)
- `workflow/local_notes`  (files=0, size=0.00 B)
  - `workflow/local_notes/project_deep_read` (files=0, size=0.00 B)
