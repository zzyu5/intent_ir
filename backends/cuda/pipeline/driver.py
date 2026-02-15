"""
CUDA compiler pipeline driver.

This keeps a stable staged contract while legacy codegen entrypoints are still
used by runtime scripts.
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any, Mapping

from .stages import CUDA_PIPELINE_STAGES, CudaPipelineResult, CudaPipelineStage


def _stage(name: str, fn) -> CudaPipelineStage:
    t0 = perf_counter()
    try:
        out = fn()
        detail = ""
        artifacts: dict[str, Any] = {}
        if isinstance(out, tuple):
            detail = str(out[0] or "")
            if len(out) > 1 and isinstance(out[1], Mapping):
                artifacts = dict(out[1])
        else:
            detail = str(out or "")
    except Exception as e:  # pragma: no cover - defensive path
        return CudaPipelineStage(name=name, ok=False, ms=(perf_counter() - t0) * 1000.0, detail=str(e), artifacts={})
    return CudaPipelineStage(name=name, ok=True, ms=(perf_counter() - t0) * 1000.0, detail=detail, artifacts=artifacts)


def _dim_value(dim: Any) -> Any:
    try:
        return dim.value  # type: ignore[attr-defined]
    except Exception:
        return dim


def _shape_values(tensor: Any) -> list[Any]:
    try:
        shape = list(getattr(tensor, "shape") or [])
    except Exception:
        shape = []
    return [_dim_value(d) for d in shape]


def _collect_intent_info(intent_payload: Any) -> tuple[str, list[str], dict[str, list[Any]], dict[str, Any]]:
    name = str(getattr(intent_payload, "name", "intent"))

    ops_raw = list(getattr(intent_payload, "ops", []) or [])
    op_names: list[str] = []
    for op in ops_raw:
        op_name = str(getattr(op, "op", "") or "")
        if op_name:
            op_names.append(op_name)

    tensors_raw = getattr(intent_payload, "tensors", {})
    if not isinstance(tensors_raw, Mapping):
        tensors_raw = {}
    tensor_shapes: dict[str, list[Any]] = {}
    for key, tensor in tensors_raw.items():
        tensor_shapes[str(key)] = _shape_values(tensor)

    schedule = getattr(intent_payload, "schedule", None)
    schedule_info: dict[str, Any] = {}
    if schedule is not None:
        for field in ("tile_m", "tile_n", "tile_k", "vec_width", "pipeline_depth"):
            val = getattr(schedule, field, None)
            if val is not None:
                schedule_info[field] = val

    return name, op_names, tensor_shapes, schedule_info


def _classify_failure(detail: str) -> str:
    msg = str(detail).lower()
    if "unsupported" in msg or "missing op" in msg:
        return "lowering_missing_op"
    if "invalid" in msg or "empty" in msg:
        return "invalid_intent"
    return "runtime_fail"


def _legalize_rewrite_counts(op_names: list[str]) -> dict[str, int]:
    counts = {
        "identity_noop": 0,
        "layout_cast_noop": 0,
        "reshape_rewrite_candidates": 0,
        "transpose_rewrite_candidates": 0,
    }
    for op in op_names:
        if op == "identity":
            counts["identity_noop"] += 1
        elif op == "layout_cast":
            counts["layout_cast_noop"] += 1
        elif op == "reshape":
            counts["reshape_rewrite_candidates"] += 1
        elif op == "transpose":
            counts["transpose_rewrite_candidates"] += 1
    counts["total_rewrite_candidates"] = sum(int(v) for v in counts.values())
    return counts


def run_cuda_pipeline(intent_payload: Any) -> CudaPipelineResult:
    name, op_names, tensor_shapes, schedule_info = _collect_intent_info(intent_payload)
    stages: list[CudaPipelineStage] = []
    rewrite_counts = _legalize_rewrite_counts(op_names)

    def _legalize() -> tuple[str, dict[str, Any]]:
        if not op_names:
            raise ValueError("invalid intent: empty ops")
        if not tensor_shapes:
            raise ValueError("invalid intent: empty tensors")
        return (
            "validated intent payload",
            {
                "intent_name": name,
                "op_count": len(op_names),
                "tensor_count": len(tensor_shapes),
                "ops": op_names,
                "rewrite_counts": rewrite_counts,
            },
        )

    def _shape_infer() -> tuple[str, dict[str, Any]]:
        symbolic_dims: dict[str, list[str]] = {}
        for tensor_name, shape in tensor_shapes.items():
            syms = [str(d) for d in shape if not isinstance(d, int)]
            if syms:
                symbolic_dims[tensor_name] = syms
        return (
            "collected symbolic shape requirements",
            {
                "symbolic_tensor_count": len(symbolic_dims),
                "symbolic_dims": symbolic_dims,
            },
        )

    def _schedule() -> tuple[str, dict[str, Any]]:
        defaults = {"tile_m": 64, "tile_n": 128, "tile_k": 32}
        if not any(op in {"matmul", "conv1d", "conv2d", "conv3d"} for op in op_names):
            defaults = {"tile_m": 1, "tile_n": 256, "tile_k": 1}
        if int(rewrite_counts.get("total_rewrite_candidates", 0)) > 0:
            # Conservative scheduling when legalize has extra normalization work.
            defaults = dict(defaults)
            defaults["tile_n"] = min(int(defaults.get("tile_n", 256)), 128)
        merged = dict(defaults)
        merged.update({k: v for k, v in schedule_info.items() if v is not None})
        return (
            "resolved schedule hints",
            {
                "schedule_hints": merged,
                "rewrite_aware": bool(int(rewrite_counts.get("total_rewrite_candidates", 0)) > 0),
            },
        )

    def _emit() -> tuple[str, dict[str, Any]]:
        raw_codegen = os.getenv("INTENTIR_CUDA_CODEGEN", "cpp").strip().lower()
        codegen_mode = "py" if raw_codegen in {"0", "false", "no", "n", "py", "python"} else "cpp"
        return ("selected emit backend", {"codegen_mode": codegen_mode})

    def _compile() -> tuple[str, dict[str, Any]]:
        return ("deferred compile to runtime runner", {"compile_mode": "deferred"})

    def _launch() -> tuple[str, dict[str, Any]]:
        return ("deferred launch to runtime runner", {"launch_mode": "deferred"})

    stage_impls = {
        "legalize": _legalize,
        "shape_infer": _shape_infer,
        "schedule": _schedule,
        "emit": _emit,
        "compile": _compile,
        "launch": _launch,
    }
    failed = False
    fail_reason = "ok"
    fail_detail = ""
    for stage_name in CUDA_PIPELINE_STAGES:
        if failed:
            stages.append(
                CudaPipelineStage(
                    name=stage_name,
                    ok=False,
                    ms=0.0,
                    detail="skipped_after_failure",
                    artifacts={"skipped": True},
                )
            )
            continue
        stage = _stage(stage_name, stage_impls[stage_name])
        stages.append(stage)
        if not stage.ok:
            failed = True
            fail_detail = str(stage.detail)
            fail_reason = _classify_failure(fail_detail)

    ok = not failed and all(s.ok for s in stages)
    return CudaPipelineResult(ok=ok, stages=stages, reason_code=("ok" if ok else fail_reason), reason_detail=fail_detail)
