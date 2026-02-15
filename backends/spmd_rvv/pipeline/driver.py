"""
RVV compiler pipeline driver.

Current implementation is compatibility shim for incremental migration.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from .stages import RVV_PIPELINE_STAGES, RvvPipelineResult, RvvPipelineStage


def _stage(name: str, fn) -> RvvPipelineStage:
    t0 = perf_counter()
    try:
        detail = str(fn() or "")
    except Exception as e:  # pragma: no cover - defensive path
        return RvvPipelineStage(name=name, ok=False, ms=(perf_counter() - t0) * 1000.0, detail=str(e))
    return RvvPipelineStage(name=name, ok=True, ms=(perf_counter() - t0) * 1000.0, detail=detail)


def run_rvv_pipeline(intent_payload: Any) -> RvvPipelineResult:
    _ = intent_payload
    stages: list[RvvPipelineStage] = []
    for stage_name in RVV_PIPELINE_STAGES:
        stages.append(_stage(stage_name, lambda: "compat_shim"))
    ok = all(s.ok for s in stages)
    return RvvPipelineResult(ok=ok, stages=stages, reason_code=("ok" if ok else "runtime_fail"))

