# 真实的“优化思想”是什么：以 FlashAttention 为例（给 ORG-Migrate 的研究指导稿）

> 这份文档是对 `doc/org.md` 的补充：只讲“真实优化思想（why）”与它在 ORG 中应该如何被表达与迁移。  
> 重点：**ORG 的“硬件无关”不等于“去掉内存层次/靠近数据”**。恰恰相反——“在更近的存储里跑（SRAM/scratchpad residency）”就是最典型、最核心、也最难从代码表面直接读出的优化思想之一。

---

## 1. 先把概念钉死：思想（why）≠ 机制（how）≠ 参数（numbers）

很多系统最后看起来像“调参”，根因是只留下了 numbers（tile/threads/stages），而没有把 why 与 how 分离出来。

我们希望 ORG 真正承载的是三层：

1) **优化思想 / 目标（Why）**：我要达到什么性能目的？（减少远端访问、提高复用、隐藏延迟、降低写回、减少同步域……）  
2) **实现机制 / 手段（How）**：我用什么机制去实现这个目的？（分块、staging、双缓冲、流式归约、布局编码、并行映射、通信图……）  
3) **参数化维度（Numbers / Dims）**：为了让机制生效，需要调哪些维度？它们的范围/约束是什么？

LLM 的不可替代性主要在第 1) 与第 2) 的“结构理解”：
- 代码/IR 往往能告诉你“发生了什么”（有 shared、有 async copy、有 barrier…）  
- 但很难自动总结出“为什么要这么组织”（比如“把工作集压进 SRAM，避免 materialize 到 HBM；并用 streaming softmax 维持数值稳定”）。

---

## 2. “在 SRAM 里跑”到底指什么（可迁移、非 GPU 专属）

这里的 SRAM 是**抽象意义**：更近、更快、容量更小的局部存储（GPU shared / AMD LDS / TPU scratchpad / DSA local SRAM / 甚至 CPU 的 L1/L2 或软件管理的 buffer）。

“在 SRAM 里跑”的思想通常包含三类可迁移陈述：

### 2.1 工作集驻留（Working-Set Residency）
- 目标：把**热数据**（会被多次复用的数据）尽可能驻留在近存储里，避免反复触达远端内存。
- 可迁移的核心不是“用 shared”，而是：
  - 哪些张量/子块是热的（Q/K/V/partial sums/weights/activations…）
  - 热数据的驻留窗口（在什么循环层级内复用）
  - 驻留预算（近存储容量/带宽/银行冲突/端口数等约束）

### 2.2 流式计算（Streaming / Online）
- 目标：避免生成/写回中间大张量（尤其是 O(n²) 的 attention matrix），改为**边读边算边归约**。
- 核心思想：把“需要全局知道”的量变成可在线维护的状态（例如 softmax 的 max 与 sum）。

### 2.3 近存储内迭代（Iterate-in-SRAM）
- 目标：把主循环组织成“**外层流式加载远端块**，内层在近存储上完成多步计算/复用/归约”，减少远端往返。
- 这是一类非常“专家化”的结构：它不是某条指令，而是 loop nest + buffer life-time 的设计。

---

## 3. FlashAttention 的“真实思想”（你强调的那句：在 SRAM 里跑）

把 FlashAttention 的思想压缩成一句话：
> **不 materialize 注意力矩阵；让 softmax 在线计算；把关键工作集（Q block 与归约状态）驻留在 SRAM/寄存器里，对 K/V 进行分块流式扫描，并在近存储内完成累积。**

拆成 ORG 应该能表达的“why / how / dims”：

### 3.1 Why（思想/目标）
1) **IO 主导问题 → IO-aware**：注意力瓶颈不是算，而是 IO（尤其是中间矩阵写回与重复读）。  
2) **避免 O(n²) 中间物化**：attention matrix 不落地，只做在线归约。  
3) **工作集驻留 + 近存储内迭代**：让 Q 子块、softmax 状态、输出累积尽量留在近存储，K/V 以块为单位流式经过。  
4) **延迟隐藏**：远端加载与计算重叠（prefetch / pipeline / double-buffer）。

### 3.2 How（机制/手段）
1) **分块（tiling）**：Q/KV 按块组织；块大小由“近存储预算 + 算子结构”决定。  
2) **staging 到近存储（scratchpad staging）**：把当前需要复用/计算的块放到 scratchpad。  
3) **在线 softmax 归约（streaming reduction）**：维护 max/sum 状态，分块更新。  
4) **双缓冲/流水（double buffering / pipeline stages）**：下一块 KV 预取时，当前块计算。  
5) **并行映射与通信域选择**：warp/subgroup 内归约 vs block 归约，取决于工作集大小与通信代价。

### 3.3 Dims（维度/约束）
FlashAttention 真正关键的维度不是“一个 BLOCK_KV 数字”，而是这些维度与约束的组合：
- `tile_kv`：KV 分块长度（直接决定 SRAM 占用与复用机会）
- `block_threads` / `subgroup_count`：并行域大小（决定 reduction 域、寄存器压力与 occupancy）
- `pipeline_stages`：流水深度（决定预取距离与在途数据量）
- 以及结构性约束（表达思想所需）：
  - `resident_bytes(Q_block + state + partial_out + staged_KV) <= scratchpad_budget`
  - 在线 softmax 的数值稳定性约束（需要维护 max/sum 的更新次序）

> 注意：这里的“scratchpad_budget”不是写死 GPU shared 容量；它来自 HardwareModel。  
> ORG 只表达“要驻留”和“驻留对象/窗口/预算关系”。

---

## 4. 为什么这些“思想”很难从代码直接分析出来（也正是 LLM 价值）

从 PTX/IR 你通常能看见：
- 有 shared / 有 async copy / 有 barrier / 有 reduction 指令

但你很难“自动总结”出：
- 为什么要把 loop 组织成某种扫描次序（Q 外、KV 内 vs 反过来）
- 为什么要在线 softmax（避免中间物化）并且如何保持数值稳定
- 哪些数据是工作集、驻留窗口是什么、驻留的真正目的是什么（复用/带宽/延迟隐藏）

这类总结往往依赖“专家范式”，LLM 更适合生成这类**结构化解释**，但必须被证据约束，避免变成幻觉故事。

---

## 5. 对 ORG 的要求：必须能表达“近存储内迭代”的高级思想

为了让 ORG 真正承载“优化思想迁移”，ORG 至少需要有一类信息能表达：

1) **驻留目标**：哪些数据希望驻留在近存储（抽象地叫 scratchpad/local)  
2) **驻留窗口**：在哪个循环层级/阶段内复用  
3) **流式扫描结构**：谁外层、谁内层；哪些量在线归约维护状态  
4) **预算约束**：驻留对象总字节与硬件预算关系（由 HardwareModel 提供预算值）

这些信息不等同于“cp.async present”这种机制证据；它们是更高层的“why/how”结构。

建议把 ORG 视为：
- **Mechanism 层**：抽象机制存在性（可从证据归一化而来，稳定）  
- **Rationale/Intent 层**：LLM 输出的高级思想（例如 iterate-in-scratchpad、streaming-softmax、avoid-materialization），并用 Evidence 做支撑或做“best-effort 证据代理”（shared bytes hint、在线归约模式、buffer life-time 片段等）

---

## 6. 迁移时如何保持“思想不变，机制可替代”

以 FlashAttention 的核心思想为例：

- 思想（ORG）写的是：  
  - `iterate_in_local_scratchpad`  
  - `streaming_softmax_state`  
  - `resident_working_set`（Q/state/out in local, KV streamed in tiles）

- 目标硬件的落地（HardwareMap + 后端）决定的是：  
  - local scratchpad 是 shared / LDS / local SRAM / cache-tiling  
  - async 预取是 cp.async / DMA / software pipelining  
  - reduction 的实现是 subgroup shuffle / block reduce / NoC reduce

如果目标硬件不支持某机制（比如没有显式 scratchpad），后端就进行 substitution：
- preserve-first：保留“驻留/流式/在线归约”的思想  
- 用最接近的机制替代（例如 cache-tiling + prefetch + vectorized load），并记录替代原因与预期损失

---

## 7. 这份文档对你“删代码重写”的直接指导

当你从零重写时，优先保证两件事：

1) ORG 不仅能列出“机制标签”，还必须能表达 **“在更近存储里跑/近存储内迭代/流式在线归约”** 这类高级思想；  
2) LLM 抽取出来的高级思想必须能驱动后端映射（不是只写在报告里好看），至少要影响：
   - 模块选择（是否需要 scratchpad-staging 家族）
   - pass 拼接（是否需要 streaming reduction pass）
   - 参数维度（必须出现驻留预算相关的维度/约束）

否则系统就会退化成“模板 + 调参”，失去你的科研目标。

---

## 8. 后端落地：你说的“模块 + pass 拼接”应该怎么做（最小可行但研究味道足）

> 这一节只回答一个问题：**ORG 的高级思想如何驱动“后端拼接”，并最终生成 kernel？**  
> 关键：LLM 不参与“怎么拼接”，它只产出 ORG；拼接逻辑是后端确定性的、可解释的。

### 8.1 三个层次的后端产物（从 ORG 到代码）

把后端落地拆成三层对象，避免混成“模板调参”：

1) **BackendModule（模块库）**：可组合的后端构件（每个模块实现/近似实现一个抽象机制或一段结构）  
2) **BackendPass（拼接与降级的 pass）**：决定“怎么把模块拼起来”、怎么处理不支持机制的 substitution  
3) **BackendPlan（后端计划）**：模块图 + 参数空间 + 约束 + substitution trace（调优与 codegen 的唯一输入）

> Phase-1 可以允许 modules≈templates（工程最省），但接口必须按“可组合模块”设计。  
> Phase-2 再把模板拆到更细粒度模块，不改变上层 ORG/Mapper 的契约。

### 8.2 BackendModule 最小接口（你关心的“拼接”本体）

每个模块至少要声明这些信息（否则无法可靠拼接与替代）：
- `provides`：它实现/提供的 abstract mechanisms（例如 `abstract.scratchpad_staging`）  
- `requires`：它依赖的 abstract mechanisms 或前置条件（例如需要 `abstract.barrier` 或需要 `scratchpad_budget>0`）  
- `params`：它暴露的参数维度（例如 `tile_kv`, `pipeline_stages`, `block_threads`）  
- `constraints`：本模块对参数的结构性约束（例如 `resident_bytes <= scratchpad_budget`、`tile_kv multiple_of 16`）  
- `lowering`：落到目标后端 IR/代码的实现（phase-1 可直接调用 template codegen）

模块的目标不是“更快”，而是：**把优化思想落地成可组合的结构单元**。

### 8.3 BackendPass：把 ORG “拼接成一条可执行 pipeline”

建议后端拼接最少包含这些 pass（按顺序）：

1) **Mechanism-to-Module Selection Pass**（机制→模块选择）
   - 输入：ORG.mechanisms + ORG.rationale/intent + HardwareModel.features  
   - 输出：模块集合（允许多实现候选：例如 async-copy 模块家族 vs sync-load 家族）

2) **Residency & Buffer Lifetime Pass**（驻留与生命周期）
   - 把“在 SRAM 里跑”变成硬约束：  
     - 选择哪些 buffer 在 local/scratchpad 驻留  
     - 定义驻留窗口（在哪个 loop 阶段有效）  
     - 生成 `resident_bytes(...) <= scratchpad_budget` 约束并注入 BackendPlan

3) **Pipeline Scheduling Pass**（流水/双缓冲）
   - 若 ORG 需要 overlap：插入 prefetch、commit/wait（或其抽象等价）  
   - 生成与 stages 相关的结构约束（例如 stage 数、inflight tile 数）

4) **Communication Lowering Pass**（通信域与归约落地）
   - 根据 HardwareModel 的 subgroup 特性与成本模型，选择：subgroup reduce / block reduce / 分层 reduce  
   - 这一步是“思想迁移”的关键：思想是“需要归约”，机制可替代

5) **Layout/Encoding Pass**（布局与编码）
   - 将 ORG 的“需要避免冲突/需要连续访问/需要某种 operand packing”映射成目标布局选择  
   - 这类东西往往最依赖硬件模型（bank、vector width、load/store 粒度）

6) **Codegen Pass**（生成）
   - Phase-1：直接落到模板族（模板=粗粒度模块）  
   - Phase-1：直接落到模板族（模板=粗粒度模块）  
   - Phase-2：按模块图生成更可组合的 IR

### 8.4 Tuner：调参不是 LLM 的工作，而是 BackendPlan 的工作

当 BackendPlan 已经给出：
- 模块拼接结构
- 参数空间（dims）
- 结构约束（resident_bytes、alignment、stage 合法性…）

调优器要做的只是：
1) 在约束内采样/枚举小预算候选（或解析模型导向）  
2) 以目标硬件的成本模型/真实 bench 选择最优

这一步天然属于后端：你要把“专家怎么调”写进模块约束与成本模型，而不是写进 prompt。

### 8.5 substitution trace：科研报告必须解释“思想保留但机制替代”

当目标硬件缺少某个机制（例如没有显式 scratchpad 或没有 async copy），mapper/pass 必须：
- 记录 substitution：`abstract.async_copy → sync_prefetch` / `scratchpad_staging → cache_tiling` 等  
- 记录原因：来自 HardwareModel 的 feature/cost/limit  
- 记录预期损失：吞吐、带宽、同步开销、occupancy 等（可粗略）

> 这条 trace 是你论文里“迁移”部分最关键的可验收证据：我们迁移的是思想，不是硬件细节。

---

## 9. FlashAttention：从 ORG 到“模块+pass 拼接”的一个具体例子（不绑定 CUDA）

> 这节给一个“最像你脑子里专家工作流”的示例：  
> LLM 抽出 ORG 的高级思想；后端按 HardwareModel 拼接模块，最终生成 kernel。

### 9.1 ORG（LLM 抽取）应该至少明确这些点

思想/意图（why）：
- `resident_working_set`：Q block + softmax state + partial out 尽量驻留近存储
- `streaming_softmax_state`：在线维护 max/sum，避免 attention matrix 物化
- `iterate_in_scratchpad`：KV 分块流式扫描，内层在近存储完成累积

机制与维度（how/numbers）：
- mechanisms：`abstract.scratchpad_staging`, `abstract.tiling`, `abstract.pipeline_stages>=2`, `abstract.(subgroup|block)_reduce`, `abstract.double_buffering`（若证据支持）
- dims：`tile_kv`, `block_threads`, `subgroup_count`, `pipeline_stages`
- 结构约束：`resident_bytes(Q_block + state + partial_out + staged_KV) <= scratchpad_budget`

### 9.2 Mapper 选择模块（由 HardwareModel 驱动，不由 LLM）

给定 HardwareModel（目标硬件）：
- 若 `has_scratchpad=true`：允许 `ScratchpadStage(KV)` / `ScratchpadResidentState` 模块族  
- 若 `has_async_copy=true`：允许 `AsyncPrefetchPipeline` 模块族；否则 fallback 到 `SyncPrefetch`  
- 若 `has_subgroup_shuffle=true` 且 subgroup 成本低：优先 `SubgroupReduce`；否则 `BlockReduce` 或分层 reduce  
- 若 `has_matrix_accel=true` 且 ORG 指出矩阵加速：选择 `MatrixPrimitive` 模块族；否则 `SIMD/Scalar` 族

### 9.3 Pass 拼接出结构（把“在 SRAM 里跑”变成硬事实）

1) Residency pass 选定驻留对象与窗口，并生成预算约束  
2) Pipeline pass 选择 stages 与双缓冲策略（将 `pipeline_stages` 变成结构约束而非仅数字）  
3) Communication pass 选择归约域（subgroup vs block）并插入必要同步  
4) Layout pass 选择 staged_KV 的布局/对齐/pack（避免冲突/保证向量化）  
5) Codegen pass 生成最终 kernel（模板或组合 IR）

### 9.4 输出 BackendPlan（给 tuner 与 report 的唯一输入）

BackendPlan 至少要包含：
- `modules[]`：模块清单（以及模块图的连接关系）  
- `param_space`：`tile_kv/block_threads/subgroup_count/pipeline_stages/...`  
- `constraints`：驻留预算、alignment、stage 合法性、通信域一致性  
- `trace.substitutions[]`：任何机制替代与原因（来自 HardwareModel）

这样你就能在报告里回答：
- ORG 抽出了“在 SRAM 里跑/在线 softmax/流式扫描”的思想  
- 后端如何把这个思想映射为模块结构  
- 哪些机制因为硬件不支持而替代  
- 调优只在该结构内搜索参数

