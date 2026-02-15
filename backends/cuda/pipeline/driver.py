"""
CUDA compiler pipeline driver.

Current implementation is a compatibility shim that records stage boundaries.
It can be wired into runtime/codegen incrementally without breaking existing
script entrypoints.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from .stages import CUDA_PIPELINE_STAGES, CudaPipelineResult, CudaPipelineStage


def _stage(name: str, fn) -> CudaPipelineStage:
    t0 = perf_counter()
    try:
        detail = str(fn() or "")
    except Exception as e:  # pragma: no cover - defensive path
        return CudaPipelineStage(name=name, ok=False, ms=(perf_counter() - t0) * 1000.0, detail=str(e))
    return CudaPipelineStage(name=name, ok=True, ms=(perf_counter() - t0) * 1000.0, detail=detail)


def run_cuda_pipeline(intent_payload: Any) -> CudaPipelineResult:
    _ = intent_payload
    stages: list[CudaPipelineStage] = []
    for stage_name in CUDA_PIPELINE_STAGES:
        stages.append(_stage(stage_name, lambda: "compat_shim"))
    ok = all(s.ok for s in stages)
    return CudaPipelineResult(ok=ok, stages=stages, reason_code=("ok" if ok else "runtime_fail"))

