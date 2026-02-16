from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np


def run_stage(name: str, fn: Callable[[], Any], *, stage_factory: Callable[..., Any]) -> Any:
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
        return stage_factory(
            name=name,
            ok=False,
            ms=(perf_counter() - t0) * 1000.0,
            detail=str(e),
            artifacts={},
        )
    return stage_factory(
        name=name,
        ok=True,
        ms=(perf_counter() - t0) * 1000.0,
        detail=detail,
        artifacts=artifacts,
    )


def dim_value(dim: Any) -> Any:
    try:
        return dim.value  # type: ignore[attr-defined]
    except Exception:
        return dim


def shape_values(tensor: Any) -> list[Any]:
    try:
        shape = list(getattr(tensor, "shape") or [])
    except Exception:
        shape = []
    return [dim_value(d) for d in shape]


def collect_intent_info(intent_payload: Any) -> tuple[str, list[str], dict[str, list[Any]], dict[str, Any]]:
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
        tensor_shapes[str(key)] = shape_values(tensor)

    schedule = getattr(intent_payload, "schedule", None)
    schedule_info: dict[str, Any] = {}
    if schedule is not None:
        for field in ("tile_m", "tile_n", "tile_k", "vec_width", "pipeline_depth"):
            val = getattr(schedule, field, None)
            if val is not None:
                schedule_info[field] = val

    return name, op_names, tensor_shapes, schedule_info


def legalize_rewrite_counts(op_names: list[str]) -> dict[str, int]:
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


def op_family(op_names: list[str]) -> str:
    matmul_conv = {"matmul", "conv1d", "conv2d", "conv3d", "conv_depthwise2d"}
    reduction = {
        "reduce_sum",
        "reduce_prod",
        "reduce_max",
        "reduce_min",
        "reduce_any",
        "argmax",
        "argmin",
        "cumsum",
        "cummax",
        "cummin",
        "mean",
        "var",
        "std",
        "quantile",
        "softmax",
    }
    ops = set(op_names)
    if any(op in matmul_conv for op in ops):
        return "matmul_conv"
    if any(op in reduction for op in ops) or bool(ops):
        return "elementwise_reduction"
    return "other"


def env_int(*keys: str) -> int | None:
    import os

    for key in keys:
        raw = os.getenv(str(key))
        if raw is None or not str(raw).strip():
            continue
        try:
            return int(str(raw).strip())
        except Exception:
            continue
    return None


def normalize_bindings(shape_bindings: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in dict(shape_bindings or {}).items():
        key = str(k)
        if isinstance(v, bool):
            out[key] = int(v)
            continue
        if isinstance(v, int):
            out[key] = int(v)
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        out[key] = int(fv) if float(fv).is_integer() else float(fv)
    return out


def has_symbolic_dims(tensor_shapes: Mapping[str, list[Any]]) -> bool:
    for shape in tensor_shapes.values():
        for d in shape:
            if not isinstance(d, int):
                return True
    return False


def np_dtype(dt: str) -> Any:
    m = {
        "f16": np.float16,
        "bf16": np.float32,
        "f32": np.float32,
        "f64": np.float64,
        "i8": np.int8,
        "u8": np.uint8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "i1": np.bool_,
        "bool": np.bool_,
    }
    return m.get(str(dt), np.float32)


def resolve_dim_int(dim: Any, bindings: Mapping[str, Any]) -> int:
    raw = dim_value(dim)
    if isinstance(raw, int):
        return int(raw)
    key = str(raw)
    if key in bindings:
        try:
            return int(bindings[key])
        except Exception:
            return 1
    try:
        return int(key)
    except Exception:
        return 1
