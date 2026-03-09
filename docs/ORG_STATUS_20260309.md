# ORG / IntentIR 当前状态总结（2026-03-09）

## 1. 一句话结论

- ORG 已经不是概念原型，而是一个独立子系统，能在 `IntentIR` 主路径旁边做第二条 LLM/后端迁移路径。
- 当前最稳的结果不是 `flash_attention2d`，而是：
  - `_attn_fwd`
  - `softmax_inner`
  - `masked_softmax2d`
  - `matmul_fused_epilogue2d`
- `flash_attention2d` 还没有完全收敛，但已经从“工具链 ceiling 卡死”推进到了“可在 `sm120` 上做真实前沿候选比较”的阶段。

## 2. 我们现在到底在做什么

### 2.1 两条科研路径分开

当前系统明确分成两条路径：

1. `IntentIR` 主路径  
   `frontend/triton -> IntentIR semantic IR -> 验证 -> lowering/run`

2. `ORG` 迁移路径  
   `frontend artifacts + source oracle + hardware/toolchain -> ORG rationale -> BackendPlan -> candidates -> compare/tune`

这里有几个原则已经固定：

- LLM 只负责恢复优化意图，不直接给后端参数。
- 后端 mapping 是确定性的，在 ORG planner 中完成。
- ORG 是独立子系统，不再嵌在 `intent_ir/intent_ir/org` 里。

### 2.2 ORG 现在的真实输入

ORG 不是只看源码，而是吃多层证据：

- `TTGIR facts`
- `PTX facts`
- `source oracle facts`
- `hardware model`
- `toolchain model`

其中：

- `TTGIR` 负责恢复布局、staging、mapping、pipeline 的高层机制
- `PTX` 负责确认这些机制是否真的在低层落实
- `source oracle` 负责告诉系统“源 GPU 上最优落点是什么”
- `hardware model` 负责 target GPU 的 shared memory / async copy / MMA 等能力
- `toolchain model` 负责判断当前编译链是否真的能把目标 `sm` 物化出来

## 3. 代码现在怎么组织

### 3.1 ORG 主代码

主要代码都在：

- `ORG-Migrate/org/schema.py`
- `ORG-Migrate/org/backend_plan.py`
- `ORG-Migrate/org/backend_model/toolchain_model.py`
- `ORG-Migrate/org/facts/ttgir.py`
- `ORG-Migrate/org/facts/ptx.py`
- `ORG-Migrate/org/facts/source_oracle.py`
- `ORG-Migrate/org/mapping/cuda/*.py`

当前已经有 planner 的 kernel：

- `flash_attention2d`
- `matmul_fused_epilogue2d`
- `_attn_fwd`
- `softmax_inner`
- `masked_softmax2d`

### 3.2 IntentIR 侧桥接

IntentIR 侧只保留了薄桥接：

- `pipeline/triton/org_bridge.py`

它现在做的事情是：

1. 读取 `TTGIR/PTX/source_oracle/hardware/toolchain`
2. 调 ORG 生成 `OrgDoc`
3. 调对应 CUDA planner 生成 `BackendPlan`
4. 写出：
   - `*.org.json`
   - `*.org_plan.json`
   - `*.org_candidates.txt`
   - `*.org_*_facts.json`
5. 可选做 compile-backed checks

## 4. 现在的执行逻辑（实际运行时）

### 4.1 `ORG_MODE=apply` 时

对于已接入的 kernel，当前完整路径是：

1. Triton 跑 baseline，一次 dump 出：
   - `ttir`
   - `ttgir`
   - `ptx`
   - `llir`
   - `cubin`

2. IntentIR 主路径拿到 semantic IR / report

3. ORG bridge 读取：
   - `shape_bindings`
   - `ttgir_facts`
   - `ptx_facts`
   - `source_oracle_facts`
   - `hardware_model`
   - `toolchain_model`

4. ORG planner 输出：
   - `BackendPlan`
   - candidate list

5. compare / tune 再用这些 candidate 去做真实 coverage + perf

### 4.2 compare 现在怎么做

`ORG-Migrate/tools/compare_source_oracle_vs_guided.py` 现在会同时跑：

- `guided`
- `source_replay`
- `target_oracle`

并输出：

- raw ratio
- portable ratio
- qps
- shared-native ratio
- requested/effective sm
- downleveled diagnostics

最近又补了一层：

- flash 的 cluster repair 现在不只会做 `variant_shift`
- 也能做 `param_shift`
- repair 的选择现在优先看 `qps_intentir`，不再只看 raw ratio

## 5. LLVM / toolchain 现在是什么状态

### 5.1 已经解决的问题

最初最大的 blocker 是：

- target 是 `sm120`
- 但本地 LLVM/NVPTX ceiling 只到 `sm86`

这个问题现在已经被拆开并解决到可用状态：

- repo 里已经有官方 LLVM20 预编译包
- `llc` 走 LLVM20
- `mlir-opt/mlir-translate/llvm-as/opt` 走 MLIR14 fallback
- 复合 toolchain 通过 `mlir-current` 暴露给系统

也就是说现在不是“半手动切环境变量”，而是 repo-local composite toolchain。

### 5.2 现在的实际配置

当前复合 toolchain 的策略是：

- `llc`：LLVM20
- `mlir-opt`：MLIR14
- `mlir-translate`：MLIR14
- `llvm-as`：MLIR14
- `opt`：MLIR14

这么做的原因很实际：

- LLVM20 的 `llc` 需要来支持 `sm120`
- 但 LLVM20 的 `mlir-translate` 和当前 MLIR14 生成的 LLVM dialect 在 `flash` 上并不完全兼容

所以现在不是“全链都升级到 LLVM20”，而是“后端 PTX 物化升级，前端 MLIR 工具保持兼容”。

## 6. 每个 kernel 现在到什么程度

### 6.1 `matmul_fused_epilogue2d`

这是目前最像“真正迁移优化”的一条线。

状态：

- source async MMA 不可移植这件事，已经不是单纯失败了
- planner 能自动 repair 到 target-side fallback
- compare 里能解释 preserve / replace

最近稳定结果：

- `guided_best_ratio ≈ 1.0268`
- `target_oracle_portable_ratio ≈ 0.9991`

含义：

- guided 已经略优于 portable target oracle
- 这条线科研叙事是成立的

### 6.2 `_attn_fwd`

这是 Attention 族里目前最好的一个。

状态：

- ORG apply 主路径已接入
- TTGIR facts / planner / compare 都已打通

已有结果：

- `guided_best_ratio ≈ 1.498`

这是目前 Attention 族里最强的结果。

### 6.3 `softmax_inner`

状态：

- 最初 row-softmax triton path 在 CUDA runtime 上不稳
- 后来修了 row-softmax portability
- 现在 compare 已经能跑通

结果：

- `guided_best_ratio ≈ 1.007`

这条线已经稳定可用。

### 6.4 `masked_softmax2d`

状态：

- 最初卡在 `__nv_expf` / PTX load fail
- 现在已经修通

结果：

- `guided_best_ratio ≈ 0.992`
- `guided_best_qps_intentir ≈ 305k`

这条线已经从“runtime fail”推进到“真实可测”。

### 6.5 `flash_attention2d`

这是当前最难、也最需要你继续指导的那条线。

#### 已经解决的部分

1. toolchain ceiling  
   已不再 downlevel 到 `sm86`

2. planner 看不到 toolchain 的问题  
   现在 planner 能看到 `effective_sm=sm120`

3. frontier 候选空间太窄  
   现在已正式展开：
   - `v6`
   - `v8`
   - `v9`

4. compare repair 只会做 variant repair  
   现在已经能做 `param_shift`

#### 目前最新实测

我专门做了一个 focused sweep：

- 文件：`/tmp/flash_frontier_tune_v2/summary.json`

结果最关键的是：

- `v6@32,6`：`ratio ≈ 0.8786`
- `v9@32`：`ratio ≈ 0.8700`
- `v6@32,4`：`ratio ≈ 0.7147`
- `v8@32`：`ratio ≈ 0.6678`
- `v7` 全系仍然很差：大约 `0.20~0.21`

如果看绝对 `qps_intentir`，当前最好的是：

- `v9@32`

如果只看 raw ratio，则最好的是：

- `v6@32,6`

所以 flash 现在真正的问题已经变成：

- **ratio-best 和 qps-best 不是同一个 candidate**

这也是我最近继续改 compare 的原因。

#### flash 当前的真实判断

目前对 `flash_attention2d` 最准确的说法是：

- 已经从“工具链问题”推进到“真实前沿候选比较”阶段
- 但还没有收敛成单一无争议的最终最佳实现
- 当前最强的两个候选是：
  - `v6@32,6`
  - `v9@32`

而不是以前的：

- `v6@64,6`

## 7. 当前最重要的代码逻辑变化

### 7.1 toolchain-aware planning

以前 `flash` planner 只知道：

- target cluster 是 `cuda_tc_mid_smem`

现在 `flash` planner 还知道：

- `requested_sm`
- `effective_sm`
- `downleveled`

所以 planner 现在不是“只按 cluster 猜”，而是“按真实 toolchain 能力展开 frontier”。

### 7.2 qps-aware compare / repair

以前 compare 很依赖 raw ratio。

现在已经补到两层：

1. `shared_native_qps`
2. repair 选候选时优先参考 `qps_intentir`

这一步对 flash 很关键，因为 flash 的多个 candidate 经常：

- `qps_intentir` 很接近
- 但 `native qps` 差异很大

如果只看 raw ratio，很容易得出误导结论。

## 8. 当前最明确的 open problems

### 8.1 flash 还没 paper-ready

flash 现在虽然已经有了：

- `sm120` toolchain
- frontier candidates
- portable repair
- qps-aware compare

但还缺：

1. 一个真正稳定的 final best candidate 结论
2. 新 compare 全量跑完后的最终 guided/source/target 对照
3. 对 `v6@32,6` 和 `v9@32` 的最终取舍

### 8.2 compare 仍然在收敛

我已经启动了新的 compare run，但还没把所有旧 run 统一重跑完。

当前你看到的旧 compare 文件里，有些结论已经过时：

- 特别是早期 `flash` compare 还停留在旧 planner / 旧 target oracle / 旧 repair 逻辑上

所以 flash 相关最可信的当前依据，应该优先看：

- `flash_frontier_tune_v2`
- 最新生成的 `flash_attention2d.org_plan.json`

## 9. 我建议你现在怎么指导我

我认为你现在最值得拍板的不是“要不要继续做 ORG”，而是下面 3 个方向里优先哪一个：

### 方向 A：把 flash compare 全量重跑完

目标：

- 用新 planner
- 用新 tuning_db oracle
- 用 qps-aware repair

拿出一版新的：

- `guided`
- `source_replay`
- `target_oracle`

完整结论

这是最直接的实验收敛路线。

### 方向 B：继续攻 flash kernel/codegen 本体

目标：

- 不再只是改 planner
- 直接改 `flash v6/v9` 的 lowering / kernel quality

这个方向更激进，但收益也可能更大。

### 方向 C：先把 ORG 的论文叙事文档化

目标：

- 把当前已有结果整理成一份系统设计 + 状态 + 实验结论文档
- 明确什么已经成立，什么还没成立

如果你想先“指导我而不是继续盲改”，这个方向最合适。

## 10. 我当前的建议

如果让我自己排优先级，我建议：

1. 先走 **方向 A**
2. 如果 flash compare 仍然显示 `v6@32,6` / `v9@32` 很接近，再走 **方向 B**
3. 同时我可以把这份状态文档继续扩成更正式的 `design/status md`

---

如果你愿意，我下一步就直接按这份状态文档继续：

- 要么我继续把 **flash 的新 compare 全量跑完并总结**
- 要么我把这份状态文档改成更正式、结构更强的一版给你审阅
