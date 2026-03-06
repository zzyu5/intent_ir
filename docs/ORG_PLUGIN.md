# IntentIR <-> ORG-Migrate Integration

`ORG` 的研究设计、证据模型和迁移路线不再以 `IntentIR` 内部文档为主，主文档固定放在：
- `ORG-Migrate/doc/00_Project_Blueprint_Rationale_Transfer.md`
- `ORG-Migrate/doc/org.md`
- `ORG-Migrate/doc/08_Real_Optimization_Ideas_and_FlashAttention.md`

当前仓库里，`IntentIR` 只承担 ORG 的**桥接接入**：
- 唯一接入入口：`pipeline/triton/org_bridge.py`
- ORG 运行时代码：`ORG-Migrate/org/`
- ORG 开关仍沿用：`INTENTIR_ORG_MODE`, `INTENTIR_ORG_MODEL`, `INTENTIR_ORG_SEED_POLICY`, `INTENTIR_ORG_BUDGET`

Wave-1 的接入约束：
- 只有 `flash_attention2d` 走完整 ORG 主路径。
- `apply` 模式下，`flash_attention2d` 必须有 `TTGIR` 证据；缺失时明确报 `ttgir_missing`，不允许静默退回 `TTIR-only`。
- 其它当前 mapper 已迁到 `ORG-Migrate/org/mapping/cuda/`，但不接主路径，统一在 report 中标记为 `org_kernel_deferred`。

测试归属：
- ORG 代码层测试：`ORG-Migrate/tests/`
- IntentIR 集成测试：`tests/pipeline/`
