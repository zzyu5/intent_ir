# IntentIR 项目实现深度分析报告

✅ **TileLang Golden Tests 已补齐**：Triton/TileLang 均有 golden tests 锁 semantic_facts（避免 semantic drift）  
✅ **O6（structured sync）已实现（MVP）**：obligations 不再恒为 UNKNOWN，能给 PASS/FAIL/UNKNOWN  
✅ **Cost Model 已加入实测验证 harness**：支持在真实 RVV 机器上做 predicted vs measured 对比（含 Spearman）

> 注：上述 3 项已在实现中落地；本文件保留其余中长期改进项作为 roadmap。

**证据强度**：⭐⭐⭐⭐☆（4.5/5）

| 论文要求 | IntentIR 实现 | 强度 |
|---------|--------------|------|
| Formal Specification | Recoverability Contract (FULL/PARTIAL/OUT_OF_SCOPE) + Assumptions | ⭐⭐⭐⭐ |
| Per-Translation Certificate | SemanticCertificateV2 + Obligations | ⭐⭐⭐⭐⭐ |
| Soundness/Falsification | SMT O3 (bounded) + Mutation-Kill (empirical) | ⭐⭐⭐⭐ |
| Scalability | Golden Tests + Deterministic Evidence | ⭐⭐⭐⭐ |

### 4.3 局限性与改进空间

#### 🟢 4.3.1 Cost Model 实测验证（已落地）

**当前状态（已落地）**：
- 已提供实测验证 harness：在真实 RVV 设备上对多个 tile 配置做预测 vs 实测对比
- 支持输出 Spearman rank correlation（用于验证“排序信号”是否可靠）

**下一步建议（论文强化）**：
- 把“实测对比”扩展到更多 kernel/shape（不仅 GEMM），形成 case study 表格
- 在论文中报告：Spearman 相关系数 + top-k 命中率（例如 top-1/top-3）

#### ⚠️ 4.3.2 Cost Model 仅支持 GEMM

**当前实现**：`GEMMCostModel` 的公式假设 GEMM workload（2MNK FLOPs）

**覆盖率不足**：
- Softmax / LayerNorm（reduce + exp）无法使用此 model
- Attention（多 matmul + reduce）需要复合 model

**建议**：
- 扩展为 `OpCostModel` 基类 + `GEMMCostModel` / `ReduceCostModel` 子类
- 或参考 TVM cost model（支持 conv/reduce/elemwise）

#### 🟡 4.3.3 Hardware Profile（已支持远程 probe，但仍需扩展）

**当前状态**：
- 已支持远程 probe：通过 SSH 在目标 RVV 机器上读取并返回 `RVVHardwareProfile`
- 仍保留 JSON / preset 路径（用于离线/无 SSH 场景）

**风险**：
- 用户可能不知道 L1/L2 cache size
- 不同 RISC-V 芯片（如 T-Head C920 vs StarFive JH7110）参数差异大

**下一步建议**：
- 增强 probe 的覆盖：更可靠地拿到 cache/topology/bandwidth（必要时用 microbench）
- 增加更多“设备 preset”（例如常见 C9xx/JH7110 等）

---

### 4.4 结论：Cost Model 真实有用，且已具备“可验证证据链”

**证据强度**：⭐⭐⭐⭐☆（4/5）

**真实有用的证据**：
1. ✅ `tuning.py` Line 274 真实调用 `GEMMCostModel.search_best_tile(...)`
2. ✅ Cost model 返回的 tile 被写入 `ScheduleSketch`
3. ✅ Roofline 公式有学术基础（非 placeholder）

**仍可强化的点**：
1. ⚠️ 仅支持 GEMM（覆盖率有限）
2. ⚠️ hardware profile 的 probe/preset 仍需更完善（避免人为填参）

**论文发表建议**：
- **系统会议（如 CGO）**：必须补充实测实验（至少 3 个 kernels 在真实硬件上）
- **Workshop**：当前实现可直接使用（强调 "analytical model" 而非 "learned model"）

---

## 5. 对标 NextSteps 文档的总体完成度

### 5.1 PR 完成度矩阵

| PR | 目标 | 实现状态 | 完成度 |
|----|------|---------|--------|
| PR#1 | 接口骨架（KernelDescriptor/FrontendAdapter/registry） | ✅ `pipeline/interfaces.py` + `pipeline/registry.py` | 100% |
| PR#2 | Triton Adapter 化 | ✅ `frontends/triton/adapter.py` | 100% |
| PR#3 | LLMIntentHub | ✅ `intent_ir/llm/llm_hub.py` | 100% |
| PR#4 | Canonical Evidence + CertificateV2 | ✅ `frontends/common/evidence.py` + `certificate_v2.py` | 100% |
| PR#5 | Obligations 规则化 + Contract.assumptions | ✅ `frontends/common/obligations.py` + `contract_v2.py` | 100% |
| PR#6 | gen_cases 吃 assumptions + out-of-contract probing | ✅ `verify/gen_cases.py` 的 `GeneratedCases` | 100% |
| PR#7 | SMT(O3) MVP | ✅ `frontends/common/smt_o3.py` (bounded model search) | 95% (无 Z3) |
| PR#8 | Golden tests（只锁 semantic_facts） | ✅ Triton/TileLang 均覆盖（含更新后的 golden） | 100% |
| PR#9 | TileLang MVP 前端 | ✅ `frontends/tilelang/adapter.py` + pipeline | 100% |

### 5.2 关键设计原则遵守度

| 原则 | NextSteps 要求 | 实际实现 | 评分 |
|-----|--------------|---------|------|
| **前端解耦** | semantic_facts 不依赖 TTIR 细节 | ✅ TileLang/Triton 都用 `CanonicalEvidence` | ⭐⭐⭐⭐⭐ |
| **Contract 规则化** | assumptions 机器可读 | ✅ `["N % 128 == 0"]` 格式统一 | ⭐⭐⭐⭐⭐ |
| **Obligations 跨前端** | O1-O7 不依赖前端 | ✅ `evaluate_obligations` 输入仅 CertificateV2 | ⭐⭐⭐⭐⭐ |
| **SMT 产出 witness** | FAIL 时给出 counterexample | ✅ `O3Report` 包含 `counterexample.assignments` | ⭐⭐⭐⭐⭐ |
| **Golden 锁 semantic_facts** | schedule_hints 不参与 golden 对比 | ⚠️ Triton 实现，TileLang 未覆盖 | ⭐⭐⭐☆☆ |
| **Schema versioned** | descriptor/cert/report 都有 version | ✅ `schema_version: "cert_v2.0"` | ⭐⭐⭐⭐⭐ |

---

## 6. 风险与改进建议

### 6.1 高优先级（影响论文发表）

#### 🟢 1. TileLang Golden Tests（已修复）

**问题（已修复）**：
- Triton/TileLang 均已有 semantic_facts golden tests

**影响**：
- 论文 reviewer 可能质疑："你如何保证 TileLang extractor 的稳定性？"

**状态**：已落地（并补齐 TileLang golden files，同时更新 Triton golden 以匹配新锚点）

#### 🟢 2. Cost Model 实测验证（已修复）

**问题（已修复）**：
- 已提供实测对比脚本：输出 predicted vs measured + Spearman

**影响**：
- 系统会议（如 CGO）reviewer 必然要求："predicted vs measured 误差多少？"

**状态**：已落地（并支持从远端 host probe profile + 解析 bench JSON）

#### 🟢 3. O6 (structured sync)（已修复）

**问题（已修复）**：
- `O6_STRUCTURED_SYNC` 不再恒 UNKNOWN；有锚点时会给出 PASS/FAIL（无锚点则 UNKNOWN）

**影响**：
- 对于使用 shared memory barrier 的 kernel（如 Flash Attention），无法检测 sync 正确性

**修复**：
```python
# 添加 O6 的 MVP 检查：
# - 检测 tl.atomic_cas / __syncthreads / tir.tvm_thread_allreduce 等 sync ops
# - 如果存在但无法证明 structured，返回 FAIL + reason
```

---

### 6.2 中优先级（增强鲁棒性）

#### 🟡 4. Metamorphic Relations 覆盖率低

**问题**：
- `verify/metamorphic.py` 仅支持 3 种 relation（permutation/shift/zero）
- 许多 kernel 返回 "not applicable"

**建议**：
- 添加 `scale_invariance`（如 LayerNorm 的 input 缩放应保持 normalized output）
- 添加 `associativity`（如 reduce_sum 的分块求和应等价）

#### 🟡 5. Bounded Model Search 的枚举范围固定

**问题**：
- `smt_o3.py` Line 221-245 的 bounded search 固定范围（如 `r0 ∈ [0, 8)`）

**风险**：
- 对于大 tile size（如 N=128），小范围枚举可能漏掉反例

**建议**：
- 根据 `shape_hints` 动态调整枚举范围（如 `N_hint=128 → 枚举 [0, min(128, 32))`）

---

### 6.3 低优先级（长期改进）

#### 🟢 6. Cost Model 扩展到非 GEMM

**当前**：`GEMMCostModel` 仅适用于 matmul

**建议**：
- 添加 `ReduceCostModel`（适用 softmax/layernorm）
- 添加 `ConvCostModel`（适用 conv2d）

#### 🟢 7. Hardware Profile Auto-Detection

**当前**：`RVVHardwareProfile` 需手动配置

**建议**：
- 添加 `detect_rvv_profile()` 函数（读取 Linux sysfs）
- 提供 preset profiles（如 `profiles.C920`, `profiles.JH7110`）

---

## 7. 总结：项目是否按设计实现？

### 7.1 总体评价

| 维度 | 评分 | 说明 |
|-----|------|------|
| **TileLang 解耦** | ⭐⭐⭐⭐⭐ (5/5) | 完全复用跨前端通用层，无 fallback |
| **通用验证** | ⭐⭐⭐⭐⭐ (5/5) | obligations + diff + metamorphic + mutation 全部运行 |
| **论文级思路** | ⭐⭐⭐⭐☆ (4.5/5) | Translation Validation 范式完整，SMT/Mutation-Kill 到位 |
| **Cost Model** | ⭐⭐⭐⭐☆ (4/5) | 真实调用 + Roofline 实现，缺实测验证 |


⚠️ **需补充的实验**：
- TileLang Golden Tests（保证 semantic extraction 稳定性）
- Cost Model 实测验证（证明 predicted GFLOPs 有效性）

---


### 8.2 必须补充的实验（Timeline: 2-3 周）

| 实验 | 目的 | 工作量 |
|-----|------|--------|
| TileLang Golden Tests | 证明 semantic extraction 稳定性 | 2 天 |
| Cost Model 实测验证 | 证明 predicted vs measured 误差 < 20% | 5 天 |
| Mutation-Kill Ablation | 证明各 verification stage 的独立贡献 | 3 天 |
| End-to-End Case Study | 展示 Triton/TileLang → RVV 完整流程 | 3 天 |

