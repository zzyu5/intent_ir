# 任务文档 0：项目蓝图（GPU Kernel “优化思想/理据”迁移）

> **用途**：这份文档是给 vibecoding 的“总蓝图/总需求说明书（PRD+Research Plan）”。  
> **范围**：第一阶段只做 **GPU→GPU**（H100 → 本机 5090D），**不做 DSA**；RVV 先做接口预留。

---

## 0. 研究目标（一句话）

给定一个已经为源 GPU（如 H100）高度优化的 kernel（Triton/TileLang/CUDA 前端），自动恢复其 **“为什么这么写才能快”** 的优化思想（rationale），并把这些思想在目标 GPU（如 本机 5090D）上用**等价机制**重新实例化（得到一个小而强的候选空间），用少量测量预算在目标 GPU 上恢复接近最优的性能。

---

## 1. 为什么这是科研（而不重复 TVM / 不重复 IntentIR）

### 1.1 不重复 TVM/Ansor（我们迁移的是“高质量空间从哪里来”）
TVM/Ansor 的强点是“在给定 schedule space 内搜索”。但现实痛点是：
- 工业/开源高性能 kernel 往往是 **黑盒实现**（Triton/CUDA/TileLang），其中优化思想被埋在低级代码习语里；
- “可搜索空间”本身并不会自动出现，需要人手工设计模板/空间；
- GPU→GPU（H100→消费级 GPU）同 ISA 仍存在明显性能不可移植性，原因是微结构与内存层次差异导致最优策略变化。

我们的研究问题是：
> **如何从复杂高性能实现中自动抽取“优化理据（rationale）”，生成可迁移的策略骨架与参数维度，并据此生成目标硬件上的高质量候选空间。**

### 1.2 不强绑定 IntentIR（但可兼容其视角）
IntentIR（A/B/C）强调：语义、结构、schedule hints。  
本项目核心新增的是：**Rationale Layer（思想层）**——解释 schedule 选择背后的 “why/how”，并用于机制替代与候选空间生成。

> 实现上允许：仅用 Triton AST/IR + 运行时 profile 做语义锚点；  
> 若未来 IntentIR 发表，可作为更强的语义锚点来源，但不作为必需前置。

---

## 2. 核心概念与形式化（必须写进论文/报告的“公式+对象”）

### 2.1 基本对象
- Kernel 源代码：`c`（Triton/TileLang/CUDA）  
- 源硬件：`h_s`（H100）  
- 目标硬件：`h_t`（本机 5090D）  
- 一个可执行实现配置（Triton meta-params）：`θ`，例如 `{BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, ...}`  
- 性能测量：`T(h, θ; s)`，对工作负载 shape `s` 的 median latency（ms）或 throughput（TFLOPS/GB/s）

### 2.2 ORG：Optimization Rationale Graph（优化理据图）
定义 ORG 为一个带类型属性的有向图：
\[
\mathcal{G}=(V,E)
\]
- 节点 `v∈V` 是“优化思想原语”，不是具体指令。第一阶段固定 6 类：  
  1) `tiling`（分块/循环结构）  
  2) `staging`（把数据放到更近存储：shared/register）  
  3) `overlap_pipeline`（搬运/计算 overlap：num_stages、prefetch distance）  
  4) `parallel_mapping`（迭代空间→并行单元：num_warps、program_id axes）  
  5) `communication`（归约/置换图：reduction tree / shuffle intent）  
  6) `special_primitive`（微原语：dot/mma/ldmatrix 等抽象）

每个节点携带：
- `params.dims`：**参数维度列表**（不是具体值）  
- `attrs`：结构属性（是否双缓冲、通信域类型等）  
- `evidence`：来自事实抽取器的证据引用（AST片段/IR片段/统计）

### 2.3 迁移任务的优化目标（一个公式足够）
我们要做的是：给定 ORG，在目标硬件上选择机制映射与参数实例化，使预测/实测时间最小。第一阶段只做 GPU→GPU，所以机制映射主要体现为“策略保持 + 参数重定向”。

定义：
- `m`：策略到目标硬件的机制映射（GPU→GPU 时很多是 identity，但允许替代，例如不同 mma variant、不同 pipeline 深度策略）
- `θ`：目标配置参数

目标：
\[
(\theta^\*, m^\*)=\arg\min_{\theta, m}\; T(h_t, \theta; s)
\quad \text{s.t.}\; \theta\in\Theta(h_t, m, \mathcal{G})
\]
其中 `Θ` 是可行域（shared/reg 限制、对齐、编译约束等）。

实际做法：我们用 **小预算候选集** `C_B`（|C_B|=B）替代全空间搜索：
\[
\theta^\* \approx \arg\min_{\theta \in C_B(\mathcal{G}, h_s, h_t, \theta_s)} T(h_t,\theta;s)
\]

---

## 3. 系统总流水线（你要实现的 end-to-end）

### 输入
1) 源 kernel `c`（优先 Triton）  
2) 源硬件 `h_s` 上 oracle 最优配置 `θ_s`（通过 autotune 获得）  
3) 目标硬件 `h_t` 的 device summary（nvidia-smi + torch props）  
4) 预算 `B`（例如 32）

### 输出
- 一个候选配置集合 `C_B`（JSONL），以及最优候选 `θ_best`  
- 实验报告：迁移损失、恢复比例、预算节省

### 流程（阶段）
1) **Benchmark & Oracle**：在 H100 上对每个 kernel/shape 得到 `θ_s` 与 `T_s^*`  
2) **Facts Extract**：从 `c` 抽取结构事实 `facts`  
3) **ORG Build**：由 facts 构建 ORG（LLM 可插拔 + rule fallback）  
4) **Candidate Gen**：根据 ORG + `θ_s` + `h_t` 生成候选 `C_B`  
5) **Measure**：在目标 GPU 上只测试 `C_B`，得到 `T_t^{guided}`  
6) 对比目标 oracle `T_t^*`，计算指标

---

## 4. 评价指标（必须固定，方便写论文）

对每个 kernel/shape/目标 GPU：
- **Transfer Loss（直接照搬损失）**：
  \[
  \text{Loss}=\frac{\text{Perf}(h_t,\theta_s)}{\text{Perf}(h_t,\theta_t^*)}
  \]
- **Recovery Ratio（恢复比例）**：
  \[
  \text{Recovery}=\frac{\text{Perf}(h_t,\theta_{guided})}{\text{Perf}(h_t,\theta_t^*)}
  \]
- **Budget Saving**：
  \[
  \text{Saving}=\frac{B}{|\mathcal{X}_{oracle}|}\quad \text{or}\quad \frac{time_{guided}}{time_{oracle}}
  \]

目标（第一阶段可设定的成功线）：
- `B=32` 时 Recovery ≥ 0.85（多数 kernel/shape），且显著优于直接照搬（Guided Gain > 1.2）。

---

## 5. 实际执行环境（固定到代码里）

### 机器（免密 SSH）
- `kingdom 211.87.236.70`：H100（source，远程）  
- `kingdom 192.168.8.72`：RVV（可选扩展，远程）  
- Local（本机）：5090D（target）

### 建议 host alias（~/.ssh/config）

配置免密登录步骤（首次需要）：
```bash
# 若本机尚无 SSH 密钥，先生成
ssh-keygen -t ed25519 -C "org-migrate"

# 将公钥推送到各远程机器
ssh-copy-id h100      # H100 (211.87.236.70)
ssh-copy-id rvv-jump  # RVV 跳板机 (211.87.236.75)
ssh-copy-id rvv       # RVV (经跳板机到 192.168.8.72)
```

在 `~/.ssh/config` 中追加：
```ssh
Host h100
  HostName 211.87.236.70
  User kingdom
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 60

Host rvv-jump
  HostName 211.87.236.75
  User kingdom
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 60

Host rvv
  HostName 192.168.8.72
  User ubuntu
  ProxyJump rvv-jump
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 60
```

> `rvv` 走 `211.87.236.75` 跳板机一步透明跳转，`ssh rvv` 即可直达 `192.168.8.72`。

验证免密是否生效：
```bash
ssh h100     hostname
ssh rvv-jump hostname
ssh rvv      hostname
```

---

## 6. 里程碑（对应后续 4 个任务文档）

1) **Infra + Benchmark Harness**：能在 H100（远程）与本机 5090D 上跑 matmul/softmax/layernorm 的 oracle + frozen + candidates 模式  
2) **ORG schema + Facts extractor**：能稳定从 Triton kernel 抽出事实并生成 ORG skeleton  
3) **ORG Builder（LLM可插拔）**：能输出包含 rationale 的 ORG（或 fallback）  
4) **GPU→GPU Candidate Generator**：给出预算 B 的候选空间，并能显著提升迁移性能  
5) **End-to-end Experiments + Report**：自动跑完整实验并生成可发表的图表与表格

---

## 7. 输出物清单（项目完成时应该得到什么）
- 可复现代码仓库 + 结果 artifacts  
- 一套 benchmark kernel 集（≥3 类）  
- ORG schema + 抽取器 + 迁移器  
- 实验报告（自动生成）
- 可复用：未来可扩到 RVV / DSA（但第一阶段不要求）
