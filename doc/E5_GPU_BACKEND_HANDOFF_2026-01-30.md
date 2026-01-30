# IntentIR · E5 GPU（Triton vs IntentIR-GPU Backend）交接文档

> 生成时间：2026-01-30  
> 目的：**把目前 E5 GPU 实验“做到哪了 / 数据怎么来的 / ablation 到底在关什么 / 现在的最佳版本固定在哪 / 下一个 AI 应该从哪里继续优化”**一次性讲清楚。  
> 备注：仓库的全局架构请先读 `doc/INTENTIR_PROJECT_HANDOFF.md`，本文只聚焦 **E5 GPU 子实验 + GPU 后端实现**。

---

## 0. TL;DR（你要交给下一个 AI 的最关键内容）

### 0.1 固定的“最佳已测版本”

- **性能数据（paper pinned）对应的 revision：`391daae`**
  - 这两个 pinned JSON 文件就是从该 revision 跑出来的：
    - `artifacts/experiments/E5/e5_cuda_h100_ablation_391daae.json`
    - `artifacts/experiments/E5/e5_cuda_5090d_ablation_391daae.json`
- 由于后续曾做过一些不稳定的实验性改动，目前仓库已经通过一次 revert **把 GPU 后端代码恢复回 `391daae` 的稳定实现**：
  - 当前仓库中对应 revert commit：`1305438`
  - 图脚本修正（纵坐标从 0 开始 + 默认优先读 pinned JSON）：`46bf8d9`

### 0.2 当前“图 & 数据”怎么拿到

- 生成两张 paper 图（H100 + 5090D）：
  - `python scripts/experiments/paper_figures.py --paper-json-dir artifacts/experiments/paper --paper-dir doc/paper/my-sigconf-paper`
  - 输出：
    - `doc/paper/my-sigconf-paper/fig/e5_cuda_gpu_h100.pdf`
    - `doc/paper/my-sigconf-paper/fig/e5_cuda_gpu_5090d.pdf`
- 这两张图使用 **speedup over Triton** 的纵坐标，且 **y-axis 从 0 开始**（避免“截断纵坐标”造成误读）。

### 0.3 这组 ablation 三条柱子分别是什么

在图里每个 kernel 有三条（如果有 ablation 数据）：

1. **IntentIR-GPU (quick)**：正常路径（证书/合同开启，且 host-dispatch 会做 variant 选择）
2. **Host dispatch off**：关闭 host-dispatch 的“选择逻辑”（但仍走 host-dispatch wrapper），用于隔离“选择策略”的贡献
3. **Contract off**：把 contract_v2 强制降到 OUT_OF_SCOPE，并关闭维度 specialize，观察失去 contract/evidence 后的性能退化

如果你看到 “Host dispatch off 反而更快”，通常属于：
- 选择逻辑在该 kernel 上本来就没收益（单 variant / 或 variant 差别小）
- 或者属于 **测量噪声（几 % 的差异）** / 选择误判（后面 §4 解释）

---

## 1. E5 GPU 实验：我们到底在测什么？

E5 GPU 的问题定义很简单：

> **同一组高层 kernel（AI-Bench8 suite），Triton JIT 版本 vs IntentIR lowering 到 GPU 后端版本，谁更快？**

我们把 **Triton GPU kernel** 当作 baseline（speedup=1.0×），报告我们后端的 speedup。

### 1.1 被测的 8 个 kernel（AI-Bench8）

脚本里固定顺序（画图也用这个顺序）：

- `ai_bench_matmul`
- `ai_bench_dropout`
- `ai_bench_softmax`
- `ai_bench_layernorm`
- `ai_bench_correlation`
- `ai_bench_resize`
- `ai_bench_rope`
- `ai_bench_warp`

对应 shape 在：`scripts/experiments/triton_gpu_vs_intentir_cuda.py` 的 `AI_BENCH_SHAPES`。

### 1.2 为什么要做 ablation？

论文需要回答的不是“我们写 CUDA 写得好”，而是：

> **IntentIR 的 contract/evidence + 后端机制**是否能系统化地产生高性能版本，而不是靠手工特例？

因此我们必须把“性能来自哪”拆开：
- 证书/合同（contract_v2 / canonical_shapes）是否真的在启用 fast path（specialize dims, 无 mask 等）
- host-dispatch 是否真的在“多版本”中选到了更快的实现

这就是 ablation 的意义。

---

## 2. 跑实验的入口、产物格式、以及复现方式

### 2.1 跑 E5 GPU 的脚本入口

- 主脚本：`scripts/experiments/triton_gpu_vs_intentir_cuda.py`

它输出一个 JSON：

```json
{
  "meta": {...},
  "summary": {...},
  "results": [
    {
      "kernel": "ai_bench_matmul",
      "triton": {"ns_per_iter": ..., "ns_per_iter_repeats": [...]},
      "ours": {"ns_per_iter": ..., "ns_per_iter_repeats": [...], "host_launch": true, "selected_tag": "..."},
      "ours_dispatch_off": {...},
      "ours_contract_off": {...},
      "speedup_ours_over_triton": 1.03,
      "speedup_ours_dispatch_off_over_triton": 0.79,
      "speedup_ours_contract_off_over_triton": 0.76
    }
  ]
}
```

### 2.2 “quick” vs “ablation” 文件命名约定

我们现在在 `artifacts/experiments/E5/` 里存两类：

- `e5_cuda_*_quick_<rev>.json`：仅 quick（ours vs triton）
- `e5_cuda_*_ablation_<rev>.json`：带 ablation（ours/dispatch_off/contract_off + triton）

注意：很多文件是探索历史残留；paper 默认优先使用 `*_391daae.json`。

### 2.3 图生成脚本与 pinned 输入

- 图脚本：`scripts/experiments/paper_figures.py`
  - 函数：`fig_e5_cuda_triton_vs_intentir(...)`
- 现在默认**优先读取**：
  - `e5_cuda_h100_{quick,ablation}_391daae.json`
  - `e5_cuda_5090d_{quick,ablation}_391daae.json`
  - 若不存在，才 fallback 到 “latest file”。

---

## 3. GPU 后端：代码路径（务必走 C++ codegen）

### 3.1 Python wrapper 只是壳：默认走 C++ codegen

- Python 入口：`backends/cuda/codegen/intentir_to_cuda.py`
  - 默认：`INTENTIR_CUDA_CODEGEN=cpp`
  - 如果设置 `INTENTIR_CUDA_CODEGEN=py` 会强制走 legacy Python 字符串拼接后端（不建议用于论文/长期）
  - 设置 `INTENTIR_CUDA_CODEGEN_STRICT=1` 可以禁止 fallback（用于确保“真后端路径”）

### 3.2 真正的“后端实现”在 C++

核心文件：

- C++ codegen：`backends/cuda/cpp_codegen/intentir_cuda_codegen.cpp`
  - 负责：从 IntentIR JSON（IntentFunction）生成 `.cu` 源码 + host-dispatch wrapper
  - 负责：根据 bindings/contract/schedule 决定是否 specialize dims、是否启用 host dispatch
- Runtime kernels（被 codegen include / 调用）：
  - `backends/cuda/runtime/intentir_cuda_ops.cuh`
  - `backends/cuda/runtime/kernels/*.cuh`（matmul/dropout/softmax/...）

这套结构是我们强调的“真正后端”的基础：  
Python 层只做 lowering 调用、拼装 bindings、编译 extension；核心策略和模板在 C++/CUDA。

---

## 4. “Host dispatch off / Contract off”到底关了什么？为什么可能出现“反直觉”的数据？

### 4.1 Host dispatch（我们现在做的是什么）

**Host dispatch** 的目标：对同一个 IntentIR kernel，生成一小组候选实现（variants），运行时选择一个。

实现上（概念层面）：

- codegen 会生成：
  - `variant_0(...)`、`variant_1(...)` ...（不同 block/tile/vec/pipeline 等）
  - 一个 `extern "C" launch(...)` wrapper
- wrapper 做两件事：
  1) 第一次调用时（或按 seed），跑一个轻量选择策略：决定使用哪个 variant  
  2) 后续调用直接 dispatch 到已选择的 variant

在 JSON 里你会看到：

- `ours.host_launch: true`
- `ours.selected_variant` / `ours.selected_tag`

### 4.2 Host dispatch off（ablation 的精确定义）

在 `scripts/experiments/triton_gpu_vs_intentir_cuda.py` 里：

- dispatch_off 通过 bindings 设置：
  - `CUDA_HOST_DISPATCH=1`（仍然使用 host-dispatch wrapper）
  - `CUDA_HOST_DISPATCH_SELECT=0`（关闭选择逻辑）

因此它隔离的是：

> **host-dispatch 的“选择策略”有多大贡献？**

它不是“完全不走 host-dispatch”，而是“走 host-dispatch 但不选，直接固定到默认 variant”。

#### 为什么 dispatch_off 有时会更快？

如果差异很小（例如 1%~3%），它可能来自：

- GPU 时钟/温度波动导致的统计噪声（即使我们用 median-of-repeats）
- 选择阶段偶发误判（测一次/测三次仍可能选错）
- kernel 本身只有一个 variant 或 variants 差异很小：选择策略没有信息增益

如果差异很大（例如 matmul/resize/warp 上 dispatch_off 明显更慢），反而是“科学的”：

- 说明 default variant 不适合当前 shape 或硬件
- 选择策略确实找到了更好的实现

### 4.3 Contract off（ablation 的精确定义）

contract_off 的目标是把“有合同/证书时可以做的激进优化”关掉，观察性能下降：

- 代码里会强制：
  - `contract_v2.level = OUT_OF_SCOPE`
  - 并设置 `CUDA_SPECIALIZE_DIMS=0`（关闭维度常量化）

这会导致：

- dims 无法成为 compile-time 常量
- fast path（例如无 mask / 无边界分支 / 更强对齐假设）无法启用
- 后端必须用更保守的通用实现

因此 contract_off 明显更慢是完全合理的：  
它证明“合同/证书不仅是 correctness 叙事，也能成为性能解耦的开关”。

---

## 5. 当前（pinned）结果应该怎么解读？（不要被个别点带偏）

### 5.1 5090D（pinned：`e5_cuda_5090d_ablation_391daae.json`）

- Geomean（quick）：约 **1.16×**
- matmul/dropout/softmax：基本 ~1.03×（接近 Triton）
- resize/warp：quick 明显更强，而 dispatch_off/contract_off 会显著掉速
  - 这恰好符合我们希望展示的：**多版本 + 证据/合同开关能影响性能**

### 5.2 H100（pinned：`e5_cuda_h100_ablation_391daae.json`）

- Geomean（quick）：约 **1.49×**
- matmul/rope/correlation：明显超过 Triton
- softmax：quick 仍低于 1×（这是后续真正值得优化的点）

### 5.3 为什么某些卡上看起来“更快/更慢”？

不要用“绝对速度”跨卡比较（例如 5090D vs H100），我们这里报告的是 **speedup over Triton**：

- Triton baseline 自身在不同 GPU 上的表现差异非常大（尤其是是否走到最佳 kernel/config）
- speedup 是“相对值”：如果 Triton 在某张卡上更强，我们的 speedup 可能变小；反之亦然

科学比较只在同一 GPU、同一 baseline 上做。

---

## 6. 下一个 AI Agent 该怎么继续优化（建议路线，非本次执行）

> 你现在要做的是“可写进论文的、可复用的后端能力”，不是“手工调参一次跑赢一次”。

### 6.1 优先级 1：把 softmax 在 H100 上拉到 ≥ 1×

现状：H100 softmax 仍 < 1×。  
对应代码集中在：

- `backends/cuda/runtime/kernels/softmax.cuh`

可做方向（示意）：

- 更合理的 warp/block 归约策略（减少同步、提升占用）
- vectorized load（对齐时用 float4/half2 等）
- 更激进的 fast-math / exp 近似（需与 Triton 对齐误差容忍）
- 如果保持“科研叙事”，建议做成“contract/evidence 驱动的 fast path + fallback”，而不是写死一条路径

### 6.2 优先级 2：让 host-dispatch “更可信/更稳健”

dispatch_off 偶尔更快不是致命问题，但会削弱叙事可信度。可做：

- 选择策略从 “median-of-3” 升级到更稳健（例如结合重复测量 + 方差阈值）
- 将选择结果缓存到（GPU, shape-signature, kernel-id）键上，避免重复抖动
- 在 benchmark 输出里记录 selection 的候选列表与得分（强化可解释性）

### 6.3 优先级 3：把“IR 优势”落到可复用机制（建议写论文时强调）

真正适合写进论文第四章末尾/后端章节的点不是“IR 自动变快”，而是：

1) **Contract/Certificate 把“激进优化的适用条件”显式化**  
   - 后端可以系统化生成 fast path（in-contract）与 fallback（out-of-contract）
2) **Canonical shapes / access pattern（evidence）让 specialize & vectorize 有依据**  
   - 不需要手工为每个 kernel 写死；可以抽象为编译规则
3) **Host dispatch = 让小搜索空间 multi-versioning 成为后端能力**  
   - 不是大 autotune；而是由证据/合同约束后的“小候选集选择”

在论文里，contract_off / host-dispatch-off 这类 ablation 就是证据。

---

## 7. 附：你要交付给下一个 AI 的“运行指令 & 注意事项”

### 7.1 强制走 C++ codegen（避免又走回 Python 拼字符串）

建议在运行前设置：

- `export INTENTIR_CUDA_CODEGEN=cpp`
- `export INTENTIR_CUDA_CODEGEN_STRICT=1`

### 7.2 E5 GPU 重新跑数据（下一个 AI 用）

只给出“典型命令形式”（具体 warmup/iters/repeats 可按机器调）：

```bash
python scripts/experiments/triton_gpu_vs_intentir_cuda.py \\
  --bench-mode graph --warmup 20 --iters 200 --repeats 5 \\
  --ablation --ablation-modes evidence_on,dispatch_off,contract_off \\
  --out artifacts/experiments/E5/e5_cuda_<gpu>_ablation_<rev>.json
```

### 7.3 生成 paper 图（H100 + 5090D）

```bash
python scripts/experiments/paper_figures.py \\
  --paper-json-dir artifacts/experiments/paper \\
  --paper-dir doc/paper/my-sigconf-paper
```

---

## 8. 本文没有做什么（避免误会）

- 本文 **不再继续性能调参/反复跑 benchmark**（你明确要求停下来固定版本）
- 本文 **不做 freeze vs retune 叙事**（你明确说这不是你要的点）
- 本文不写论文正文，只提供 “数据+图+可交接理解”

