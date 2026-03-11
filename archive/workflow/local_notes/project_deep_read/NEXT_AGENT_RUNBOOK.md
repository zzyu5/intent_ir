# NEXT_AGENT_RUNBOOK

## 0. 先验规则
- 不要先相信 `done`，先核 freshness。
- 不要先改代码，先对齐“当前 HEAD 对应的证据链”。

## 1. 冷启动读取顺序（15 分钟）
1. `workflow/local_notes/project_deep_read/INTENTIR_MASTER_BRIEF.md`
2. `workflow/local_notes/project_deep_read/WORKFLOW_TRUTH_AUDIT.md`
3. `workflow/local_notes/project_deep_read/MLIR_POSITION_REPORT.md`
4. `workflow/local_notes/project_deep_read/CODE_OWNERSHIP_MAP.md`

## 2. 真相判定步骤（必须）
1. 读取 git HEAD。
2. 对比 `workflow/flaggems/state/current_status.json::head_commit`。
3. 对比 `full196_validated_commit/gpu_perf_validated_commit` 与 HEAD。
4. 若 mismatch，则状态判定为 stale，不得口头宣称“当前全绿”。

## 3. 主链路定位
- 用户入口：`scripts/intentir.py`
- 三前端 core：`pipeline/triton/core.py`, `pipeline/tilelang/core.py`, `pipeline/cuda/core.py`
- backend 驱动：`backends/cuda/pipeline/driver.py`, `backends/spmd_rvv/pipeline/driver.py`
- workflow 快照：`scripts/flaggems/build_workflow_state.py`

## 4. 常见误判排除
1. active_batch 为空 ≠ 没任务；需要看 lane next_focus 与 freshness。
2. run_summary ok=true 也要看其 `repo.head_commit` 是否等于当前 HEAD。
3. MLIR 默认开启 ≠ 所有链路都已彻底去兼容；要看 contract-first 实际调用点。

## 5. 本地索引入口
- 快照：`PROJECT_SNAPSHOT.json`
- 目录索引：`directory_index.json`
- 入口索引：`entrypoints.json`
- 证据索引：`evidence_index.json`