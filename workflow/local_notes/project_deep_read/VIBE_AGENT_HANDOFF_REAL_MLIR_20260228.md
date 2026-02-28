# IntentIR Real-MLIR Cutover Handoff (2026-02-28)

目标读者：下一位接手的工程代理（vibe coding）。

本文件是“静态+证据”可核对的交接，不依赖当前对话上下文。

---

## 0. 当前仓库基线

- Repo: `/home/kingdom/intentir`
- Branch: `compiler-cleanup-v1`
- HEAD: `c4088ecb42878ecc1e301fe74a43297372c7cbe0`
- 关键提交（按时间倒序）：
  - `c4088ec` `workflow: log triton-native real-mlir perf smoke v7`
  - `d27db08` `refactor(workflow): only derive full196/gpu_perf from their lanes`
  - `35b4b5f` `workflow: rebuild state after attn warp reduction`
  - `2213387` `perf(cuda-real-mlir): warp-sized attention dot reductions`

---

## 1. “真正 MLIR 化”现在到底做到哪一步了

这里的“真正 MLIR”指：生成的 IR 能被 `mlir-opt/mlir-translate` 直接处理，并由 MLIR->LLVM IR->`llc` 产出 PTX/ELF，且 strict 下无隐式 fallback 到历史缓存。

### 1.1 真实入口（Triton-native path）

真实入口在 Triton frontend 的 pipeline orchestration：

- real-MLIR 开关：`pipeline/triton/core.py::_real_mlir_enabled()` 读取 `INTENTIR_REAL_MLIR=1`
- CUDA wave allowlist：`pipeline/triton/core.py::_load_cuda_real_mlir_wave_kernels()` 读取
  - `workflow/flaggems/state/cuda_real_mlir_wave9_kernels.json`（当前 `kernels` 数量为 125）
- downstream pipeline 选择：`pipeline/triton/core.py::_downstream_llvm_pipeline()`
  - 当 `INTENTIR_REAL_MLIR=1` 且 `INTENTIR_CUDA_REAL_MLIR_WAVE=wave9` 且 kernel 在 allowlist 内：
    - 走 `intent_ir/mlir/pipelines/downstream_cuda_std_llvm.yaml`（名字：`downstream_cuda_std_llvm`）
  - 否则在 real-MLIR 模式下 **不允许** 走 cached LLVM IR（返回 `None, None`）

### 1.2 CUDA real-MLIR pipeline 的关键组成

- Pipeline 文件：`intent_ir/mlir/pipelines/downstream_cuda_std_llvm.yaml`
  - `python:lower_intent_to_cuda_gpu_kernel`：`intent_ir/mlir/passes/lower_intent_to_cuda_gpu_kernel.py`
  - `mlir-opt: convert-gpu-to-nvvm ... finalize-memref-to-llvm`
  - `mlir-translate:mlir-to-llvmir`（生成文本 LLVM IR）
  - `llvm-as` / `opt:-O0`
- PTX materialization：由 `pipeline/mlir_contract_artifacts.py` 驱动（最终 `cuda_ptx_origin=llvm_llc`）

当前这条链路在证据层面 **已验证可运行**（见第 2 节）。

---

## 2. 当前已覆盖的 kernel / 证据（真实执行 + 真实 PTX）

### 2.1 Triton-native coverage（38 kernels，真实执行，严格 real-MLIR）

证据目录：
- `artifacts/validation_rounds/20260228/triton_native_coverage_full_wave9_realmlir_v4_attn_warp32/`

可核对结论（从该目录内 38 个 `<kernel>.json` 汇总）：
- `diff_ok = 38/38`
- `mlir.llvm_emit_ok = 38/38`
- 每个 kernel 均产出 `*.downstream_cuda_std_llvm.ll` 与 `*.downstream_cuda_std_llvm.kernel.ptx`

复现实跑命令：
```bash
cd /home/kingdom/intentir
OUT=artifacts/validation_rounds/20260228/triton_native_coverage_full_wave9_realmlir_v4_attn_warp32
INTENTIR_REAL_MLIR=1 \
INTENTIR_CUDA_REAL_MLIR_WAVE=wave9 \
INTENTIR_FALLBACK_POLICY=strict \
INTENTIR_CUDA_REQUIRE_LLVM_PTX=1 \
python scripts/triton/full_pipeline_verify.py \
  --suite coverage \
  --backend-target cuda_5090d \
  --cases-limit 1 \
  --no-stage-c \
  --no-mutation-kill \
  --out-dir "$OUT"
```

### 2.2 Triton-native perf（38 kernels，真实 PTX 执行，对比 Triton baseline）

证据目录：
- `artifacts/flaggems_matrix/daily/20260228/gpu_perf_triton_native_wave9_realmlir_smoke_v7_attn_warp32/`

证据要点（`run_summary.json`）：
- measured：`38/38`
- `gpu_perf_min_ratio = 0.28662192678800236`
- `gpu_perf_p50_ratio = 0.9980739114458275`
- 设备：RTX 4080 SUPER（见 `gpu_perf_graph.json.devices[0].gpu_name`）
- `dirty=false`，`repo.head_commit=c4088ec...`

阈值未达标的 kernel（来自 `gpu_perf_graph.json.entries[]`，`ratio < 0.8`）：
- `_attn_fwd`：ratio `0.2866`（intentir `0.0553ms` vs native `0.0158ms`）
- `flash_attention2d`：ratio `0.3364`（intentir `0.0344ms` vs native `0.0116ms`）
- `ai_bench_matmul`：ratio `0.5365`（intentir `0.02135ms` vs native `0.01078ms`）

复现 perf smoke 命令：
```bash
cd /home/kingdom/intentir
DATE=20260228
OUT=artifacts/flaggems_matrix/daily/${DATE}/gpu_perf_triton_native_wave9_realmlir_smoke_v7_attn_warp32
INTENTIR_REAL_MLIR=1 \
INTENTIR_CUDA_REAL_MLIR_WAVE=wave9 \
INTENTIR_FALLBACK_POLICY=strict \
INTENTIR_CUDA_REQUIRE_LLVM_PTX=1 \
python scripts/intentir.py suite \
  --suite gpu-perf-triton-native \
  --out-root "$OUT" \
  --gpu-perf-threshold 0.80 \
  --gpu-perf-p50-threshold 0.90 \
  --perf-warmup 10 --perf-iters 50 --perf-repeats 3 \
  --cuda-runtime-backend nvrtc \
  --progress-style chunk \
  --stream --resume \
  --gpu-perf-intent-artifact-dir \
    artifacts/validation_rounds/20260228/triton_native_coverage_full_wave9_realmlir_v4_attn_warp32
```

---

## 3. “196 / 159 / 38” 分母真相（为什么 correctness 和 perf 不同）

这几个数字属于不同门禁/套件：

1. `38 kernels`：Triton-native coverage/perf（`scripts/triton/full_pipeline_verify.py --suite coverage`）。
2. `159 kernels`：FlagGems gpu_perf graph 的 perf 分母（coverage_batches 口径）。
3. `196 semantic ops`：FlagGems “语义算子覆盖” correctness 分母（full196）。

对应的“分母证据”入口：
- `workflow/flaggems/state/coverage_batches.json`（语义分组与 kernel 列表来源）
- `workflow/flaggems/state/kernel_denominators.json`（聚合后的分母/缺口报告）

核心区别：
- correctness(full196) 更偏“语义覆盖/差分正确性”；perf(gpu_perf) 更偏“可测性能 kernel 集”（需要 baseline 可 launch、参数绑定完整等）。
- Triton-native 38 是单独的 frontend 套件，不等价于 full196 的语义集。

---

## 4. workflow(vibe coding) 工具当前怎么用，以及本轮做了什么改进

### 4.1 本轮把 triton-native perf 证据写进 workflow（但不污染 gate）

- 通过 `scripts/flaggems/end_session.py --lane workflow` 记录了 v7 perf smoke：
  - `workflow/flaggems/state/progress_log.jsonl`（新增一条）
  - `workflow/flaggems/state/handoff.md`（更新）
- `workflow/flaggems/state/current_status.json.latest_artifacts` 现在指向 v7 的 `run_summary/status_converged`：
  - `workflow/flaggems/state/current_status.json`

### 4.2 修复 build_workflow_state 的“lane 污染”问题（重要）

变更：
- `scripts/flaggems/build_workflow_state.py`
  - full196 只从 `lane=coverage` 推导
  - gpu_perf 只从 `lane=backend_compiler` 推导

目的：
避免把 monitor-only 的 perf 实验误算进 gate 状态（否则会把 `gpu_perf_validated_commit` 错误覆盖成失败 commit）。

---

## 5. 当前最关键的短板（下一位 agent 的首要攻坚）

### 5.1 性能短板集中在 attention + matmul（重核）

证据（v7）：
- `artifacts/flaggems_matrix/daily/20260228/gpu_perf_triton_native_wave9_realmlir_smoke_v7_attn_warp32/gpu_perf_graph.json`

需要达到的目标（你之前锁定的 SLO）：
- `all >= 0.80` 且 `p50 >= 0.90`

当前状态：
- p50 已接近 1.0，但 min_ratio 仍被 attention/matmul 拉低（0.28/0.33/0.53）。

### 5.2 evidence/tuning 还没有真正进入 codegen 决策

观察点：
- Intent JSON 中已经包含 schedule hints / access witness 等 meta（例如 `_attn_fwd` 的 `intentir.intent_json_b64`）
- 但 CUDA real-MLIR 的 lowering 仍主要是硬编码/启发式（尤其 attention）

下一步需要把“证据驱动调优”做成可落地的接口：
- 形状/stride witness -> tile space -> 选择/缓存（tuning DB）-> codegen 生效 -> perf evidence 回写

---

## 6. 建议的下一步执行计划（面向接手 agent，可直接照做）

### Step 1：补齐 perf 可观测性（先做，成本低，收益高）

问题：当前 perf entry 的 `shape` 基本为空 `{}`，不利于调参定位。

建议改造：
- `scripts/flaggems/run_gpu_perf_graph.py`
  - triton_native kernel_source 下，把 `KernelSpec.canonical_shapes` 与本次实际 bindings 写入 entry 的 `shape`
  - 对 `_attn_fwd/flash_attention2d/masked_attention2d/ai_bench_matmul` 也写入其 constexpr/meta（BLOCK_*）

验收：
- 新跑一次 perf smoke（v8），`gpu_perf_graph.json.entries[].shape` 非空且可用于复现。

### Step 2：attention 性能（先把 min_ratio 拉上来）

目标文件：
- `intent_ir/mlir/passes/lower_intent_to_cuda_gpu_kernel.py`

当前实现瓶颈（推断，需用 Step1 的 shape telemetry 验证）：
- dot/softmax/reduction 的 barrier/共享内存访问粒度偏粗
- 缺少 Q/K/V 分块重用与向量化 load

建议路线（按优先级）：
1. 先对 coverage canonical shape（例如 `_attn_fwd`: Q_CTX=128, KV_CTX=128, HEAD_DIM=64）做专用优化：
   - K/V tile（例如 KV tile=64 或 128），每 block 处理 1 个 q
   - warp-level reduce（避免全 block barrier）
   - shared memory coalesced load（按 HEAD_DIM 向量化）
2. 把 tile 参数显式化并可从 schedule/evidence 注入（后续接 tuning DB）。

验收：
- v9 perf smoke：`_attn_fwd` 与 `flash_attention2d` ratio >= 0.8

### Step 3：matmul 性能（ai_bench_matmul）

目标文件：
- `intent_ir/mlir/passes/lower_intent_to_cuda_gpu_kernel.py`（`kernel_kind=matmul_tile_v1` 分支）

建议路线：
1. 增加更合理的 tile/warp mapping（当前固定 BM/BN/BK 组合可能不匹配 sm89）
2. 增加 vectorized load/store（例如 128-bit）并减少 barrier
3. 长期：评估引入 `nvgpu.mma.sync` / tensorcore lowering（需兼顾 correctness/tolerance）

验收：
- v9 perf smoke：`ai_bench_matmul` ratio >= 0.8

### Step 4：把 real-MLIR 覆盖从 38 扩到更大的分母

目标：
- 从 Triton-native 38 -> 覆盖 batches(159) -> 最终 full196(196 semantic ops)

机制：
- 继续扩充 wave 文件（例如 wave10/wave11），并用 strict 禁止 cache fallback 强制暴露缺口：
  - `workflow/flaggems/state/cuda_real_mlir_wave10_kernels.json`（新增）
  - 逐步把 coverage_batches 中常用 kernel 加入 allowlist

验收：
- 每次 wave 扩张都必须产出：`coverage + perf` 的可读证据目录，并在 workflow lane 中记录（但不污染 gate）。

---

## 7. 快速导航（下一位 agent 常用入口）

- Triton-native coverage runner（38）：`scripts/triton/full_pipeline_verify.py`
- Triton-native perf runner（38）：`scripts/intentir.py suite --suite gpu-perf-triton-native`
- CUDA real-MLIR lowering：`intent_ir/mlir/passes/lower_intent_to_cuda_gpu_kernel.py`
- CUDA downstream pipeline：`intent_ir/mlir/pipelines/downstream_cuda_std_llvm.yaml`
- strict policy：`pipeline/common/strict_policy.py`
- wave allowlist：`workflow/flaggems/state/cuda_real_mlir_wave9_kernels.json`
- workflow 状态：`workflow/flaggems/state/current_status.json`

