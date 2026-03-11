# ORG Agent Handoff

## 1. 核心科研思想

- ORG 的目标不是恢复数学语义的 `What`，而是恢复优化的 `Why & How`。这里的核心对象不是旧论文意义上的 Intent，而是 Rationale。
- 当前系统的核心映射公式是：
  - `f: R x H -> P`
  - `R` = Rationale Space：LLM 从 `TTGIR/PTX/source-oracle` 证据中恢复的优化目标、机制和参数候选
  - `H` = Hardware / Toolchain Space：目标硬件资源、指令机制能力、编译链真实能力与 ABI/entry 约束
  - `P` = Backend Plan / Parameter Space：后端模块、替代机制、参数空间、compile-backed realizations
- ORG 的科研贡献不是“让 LLM 直接吐性能最优参数”，而是：
  1. 把优化思想从具体模板和具体 GPU 上解耦出来
  2. 让后端在 `R x H` 条件下确定性生成 `P`
  3. 再由真实 compile/perf 验证 `P` 是否成立

### 绝对红线

- 禁止在后端通过 **Kernel Name** 或固定模板名直接写死优化。
- 所有底层优化机制，例如：
  - shared staging
  - vectorized global I/O
  - warp shuffle reduction
  - persistent row / tile resident
  - async copy / mma path
  必须由：
  - `R` 空间的 `goals/mechanisms/dims`
  - 和 `H` 空间的资源/工具链约束
  共同触发。
- `flash_attention2d` 的关键消融已经证明了这条红线必须遵守：
  - `v8@32` **无 ORG 机制绑定**（不带 `FLASH_KV_SHARED_STAGE=1`）：
    - `ratio = 0.8833872246908309`
    - `qps_intentir = 165698.60378862807`
  - `v8@32` **有 ORG 机制绑定**（`resident_working_set + kv_streamed_tiles + H constraints -> FLASH_KV_SHARED_STAGE=1`）：
    - `ratio = 1.0049737738769526`
    - `qps_intentir = 243566.57771551298`
- 这说明：底层优化必须服务于 ORG 机制，而不是服务于 kernel 名字。

## 2. 真实系统架构现状

### R 空间（Rationale）

- 当前 `R` 的主 schema 在：
  - `ORG-Migrate/org/schema.py`
- 已落地的主字段：
  - `goals[]`
  - `mechanisms[]`
  - `dims[]`
  - `source_context`
  - `source_oracle`
  - `evidence`
- 当前 `R` 仍然偏浅，但已经足够驱动真实 planner：
  - `goals`：例如 `resident_working_set`、`streaming_softmax_state`
  - `mechanisms`：例如 `scratchpad_staging`、`warp_reduction_tree`、`vector_global_io`
  - `dims`：例如 block threads、vector width、score warps、shared stage 开关

### H 空间（Hardware / Toolchain）

- `HardwareModel` 在：
  - `ORG-Migrate/org/mapping/hardware_model.py`
- `ToolchainModel` 在：
  - `ORG-Migrate/org/backend_model/toolchain_model.py`
- `H` 目前已经能真实暴露：
  - `arch / arch_cluster`
  - `shared_mem_kb`
  - `warp_size`
  - `supports_async_copy`
  - `supports_mma`
  - `supports_ldmatrix`
  - `supports_shuffle`
  - `requested_sm`
  - `effective_sm`
  - `downleveled`
  - `supported_sms`
  - `mlir_version / llvm_version / compiler_stack`
- LLVM/MLIR20 相关现状：
  - 现在已经完成整链切换到 `/usr/lib/llvm-20/bin`
  - `mlir-opt / mlir-translate / llvm-as / opt / llc` 全部已经走 LLVM/MLIR20
  - out-of-tree MLIR plugin 也已经按 LLVM20 API 重新编译
  - 当前系统不再依赖“LLVM20 llc + MLIR14 前端”的旧复合折中方案

### P 空间（Backend Plan）

- 主结构在：
  - `ORG-Migrate/org/backend_plan.py`
- 当前 `P` 已经不是“候选字符串列表”，而是显式结构：
  - `selected_modules`
  - `module_edges`
  - `param_space`
  - `constraints`
  - `substitutions`
  - `candidates`
  - `toolchain_model`
  - `effective_target`
  - `compile_checks`
  - `realizations`
- 当前 planner 仍然是 **heuristic planner**，不是通用约束求解器，但已经能通过 module catalog 组装多类 kernel 的有效 `P`。

### 当前已覆盖 kernel 家族

- Attention
  - `flash_attention2d`
  - `_attn_fwd`
  - `masked_attention2d`
  - `masked_softmax2d`
  - `softmax_inner`
  - `ai_bench_softmax`
- Matmul / MMA
  - `matmul_fused_epilogue2d`
  - `ai_bench_matmul`
- Reduction / Row kernels
  - `row_sum`
  - `row_max`
- Elementwise
  - `add2d`
  - `exp2d`
- Norm
  - `layer_norm_persistent`
  - `group_norm_kernel`

### 当前最关键的工程结论

- `flash_attention2d` 已经通过 ORG rationale 绑定拿到 `~1.00x` native Triton。
- `row_sum / row_max / layer_norm_persistent` 在大 shape memory-bound 压力下，通过 ORG 驱动向量化访存拿到：
  - `row_sum`: `1.0468320728200828`
  - `row_max`: `1.0456710463387116`
  - `layer_norm_persistent`: `1.0018236517641885`
- `group_norm_kernel` 在 ORG 驱动的 persistent + vec4 + warp reduction 路径下拿到：
  - `ratio = 2.5043780397710305`
- `ai_bench_softmax / ai_bench_matmul / masked_attention2d` 最后一批 deferred kernel 也已经打通并满足 `>= 1.0x`

## 3. 铁血工程纪律

### 只认证实机性能

- 不接受只靠 `pytest` 或 IR 文本对比来宣称系统成功。
- 一切性能结论以 **5090D** 上的真实大 shape 压力测试为准。
- 对 memory-bound kernel，必须放大到能真正施加显存带宽压力的 shape；不能用小 shape 的 1.00x 巧合结果充数。
- 评价标准固定为：
  - `Guided QPS`
  - `Native Triton QPS`
  - `Guided / Native Ratio`

### 长链条自治与强制 Commit

- 遇到编译失败、Runtime Error、PTX load fail、性能不达标时，必须自行排查并修复，不得中途停下来等待指令。
- 每一个形成闭环的实质性修改都必须单独 `git commit`。
- 汇报时必须给出：
  - commit hash
  - 极简 technical changelog
  - 真实 perf table

### 严禁反向指导

- 不得在汇报末尾擅自建议“下一步做什么”或给出选项。
- 由架构师决定下一步方向。

### 严禁擅自生成无用文档

- 除非明确要求，否则不要自动生成额外的 Markdown 报告。
- 核心结果应直接写在对话汇报里。
- 本 handoff 文档是明确要求生成的特例。

### ORG 触发权不可回退

- 所有底层优化都必须通过 `R x H -> P` 闭环落地。
- 任何“先按 kernel 名写特判，后补一个理由”的做法都视为违反架构红线。
- 已经发生过一次的错误示例：
  - `flash_attention2d` 的 shared staging 最初被写死在 `v8` lowering 里
  - 后来被重构为由 `resident_working_set + kv_streamed_tiles + H constraints` 驱动的 `FLASH_KV_SHARED_STAGE`
  - 这是后续所有 kernel 的模板

## 4. 明确的技术债务与演进方向

### R 空间最致命的 4 个缺失

1. **张量生命周期 / 驻留区间表达缺失**
   - 现在只能粗粒度表达 resident/staging
   - 不能精确表达 producer-consumer 的驻留边界、失效时机、cache 生命周期

2. **并行拓扑 / reduction topology 表达缺失**
   - 现在对 warp 数、subwarp 切分、tree reduction 形状、pipeline 深度还缺少显式 rationale 表达
   - 很多仍在 planner 内部靠启发式猜

3. **传输机制表达缺失**
   - `cp.async`
   - direct global path
   - shared stage
   - vector width
   - multicast / broadcast
   这些还没有统一成可组合的 rationale 维度

4. **复杂融合流 / 多阶段数据流表达缺失**
   - 对多阶段 epilogue、online normalization、mask+bias+activation 融合、统计量重用的表达仍然不够细
   - 面对更复杂融合 kernel 时，现有 `goals/mechanisms/dims` 可能不够用

### H 空间最致命的 4 个缺失

1. **精确寄存器压力 / occupancy 模型缺失**
   - 当前只有粗粒度 cluster 与经验阈值
   - 还没有把寄存器使用、occupancy、CTA 驻留数做成可计算的第一类约束

2. **shared-memory bank conflict / layout cost 模型缺失**
   - 当前 planner 还不能精确估算 shared layout、bank conflict、multicast 代价
   - 对 shared-memory 优化仍有不少 heuristic 成分

3. **async 资源模型缺失**
   - 还缺少对 scoreboard、commit/wait 深度、copy engine 饱和点、async stage 数的结构化建模
   - 导致 async transport 选择还不够精确

4. **ABI / entry / vector legality 约束建模缺失**
   - 目前虽然 `ToolchainModel` 已能感知 `effective_sm`、`downleveled`、LLVM20 ABI
   - 但编译器 ABI、entry 可见性、向量化合法性、symbol/contract 约束仍没有完全成为 `H` 的第一类字段

## 5. 给下一个 Agent 的硬指令

- 不要回退到“Intent = What”的旧语境，统一使用 ORG / Rationale 语境。
- 不要把 `flash_attention2d` 的成功简化成某个 variant 的 hardcode；关键在于 ORG 绑定机制触发了 shared staging。
- 不要再把性能工作退化成纯 planner 调分；必要时直接改 lowering/pass/codegen，但触发权必须仍在 `R x H -> P`。
- 不要用小 shape 的 1.00x 作为性能成功标准；必须做大 shape 极限施压。
- 任何新增 kernel，先问三个问题：
  1. `R` 里需要新增什么 mechanism / dim？
  2. `H` 里缺不缺这个 kernel 所需的资源约束？
  3. `P` 是否真的把这些机制落成了 compile-backed、perf-validated codegen？
