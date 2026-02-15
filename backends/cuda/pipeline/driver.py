"""
CUDA compiler pipeline driver.

This driver now models a pure-compiler lifecycle:
legalize -> shape_infer -> schedule -> emit -> compile -> launch.
"""

from __future__ import annotations

import os
import numpy as np
from time import perf_counter
from typing import Any, Mapping

from backends.cuda.runtime import CudaLaunch, compile_cuda_extension, run_cuda_kernel

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
    if "compile_timeout" in msg or "launch_timeout" in msg:
        return msg
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


def _op_family(op_names: list[str]) -> str:
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


def _env_int(*keys: str) -> int | None:
    for key in keys:
        raw = os.getenv(str(key))
        if raw is None or not str(raw).strip():
            continue
        try:
            return int(str(raw).strip())
        except Exception:
            continue
    return None


def _schedule_overrides_from_env() -> tuple[dict[str, int], str]:
    overrides: dict[str, int] = {}
    tile_m = _env_int("INTENTIR_CUDA_TILE_M", "INTENTIR_TILE_M")
    tile_n = _env_int("INTENTIR_CUDA_TILE_N", "INTENTIR_TILE_N")
    tile_k = _env_int("INTENTIR_CUDA_TILE_K", "INTENTIR_TILE_K")
    if tile_m is not None:
        overrides["tile_m"] = int(tile_m)
    if tile_n is not None:
        overrides["tile_n"] = int(tile_n)
    if tile_k is not None:
        overrides["tile_k"] = int(tile_k)
    tag = str(
        os.getenv("INTENTIR_CUDA_SCHEDULE_PROFILE_TAG")
        or os.getenv("INTENTIR_SCHEDULE_PROFILE_TAG")
        or ""
    ).strip()
    return overrides, tag


def _normalize_bindings(shape_bindings: Mapping[str, Any] | None) -> dict[str, Any]:
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


def _has_symbolic_dims(tensor_shapes: Mapping[str, list[Any]]) -> bool:
    for shape in tensor_shapes.values():
        for d in shape:
            if not isinstance(d, int):
                return True
    return False


def _np_dtype(dt: str) -> Any:
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


def _resolve_dim_int(dim: Any, bindings: Mapping[str, Any]) -> int:
    if isinstance(dim, int):
        return int(dim)
    key = str(dim)
    if key in bindings:
        try:
            return int(bindings[key])
        except Exception:
            return 1
    try:
        return int(key)
    except Exception:
        return 1


def _parse_launch(launch_j: Mapping[str, Any]) -> CudaLaunch:
    grid = launch_j.get("grid")
    block = launch_j.get("block")
    shared_mem = launch_j.get("shared_mem", 0)
    if not (isinstance(grid, list) and len(grid) == 3 and isinstance(block, list) and len(block) == 3):
        raise ValueError("cuda pipeline emit returned invalid launch config")
    return CudaLaunch(
        grid=(int(grid[0]), int(grid[1]), int(grid[2])),
        block=(int(block[0]), int(block[1]), int(block[2])),
        shared_mem=int(shared_mem),
    )


def _build_dummy_inputs(*, io_spec: Mapping[str, Any], output_names: list[str], bindings: Mapping[str, Any]) -> dict[str, np.ndarray]:
    tensors = io_spec.get("tensors") if isinstance(io_spec.get("tensors"), Mapping) else {}
    out_set = {str(n) for n in output_names}
    inputs: dict[str, np.ndarray] = {}
    for name, spec in tensors.items():
        n = str(name)
        if n in out_set:
            continue
        if not isinstance(spec, Mapping):
            continue
        dtype = _np_dtype(str(spec.get("dtype") or "f32"))
        shape_spec = list(spec.get("shape") or [])
        shape = tuple(max(1, _resolve_dim_int(d, bindings)) for d in shape_spec)
        if len(shape) == 0:
            inputs[n] = np.array(1, dtype=dtype)
        else:
            inputs[n] = np.zeros(shape, dtype=dtype)
    return inputs


def run_cuda_pipeline(
    intent_payload: Any,
    *,
    shape_bindings: Mapping[str, Any] | None = None,
    execute_backend_stages: bool = True,
) -> CudaPipelineResult:
    name, op_names, tensor_shapes, schedule_info = _collect_intent_info(intent_payload)
    stages: list[CudaPipelineStage] = []
    rewrite_counts = _legalize_rewrite_counts(op_names)
    family = _op_family(op_names)
    bindings = _normalize_bindings(shape_bindings)
    has_symbolic_dims = _has_symbolic_dims(tensor_shapes)
    can_execute = bool(bindings) or (not has_symbolic_dims)
    state: dict[str, Any] = {"bindings": dict(bindings)}

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
                "can_execute": bool(can_execute),
                "bindings_count": len(bindings),
            },
        )

    def _schedule() -> tuple[str, dict[str, Any]]:
        defaults = {"tile_m": 64, "tile_n": 128, "tile_k": 32}
        if family != "matmul_conv":
            defaults = {"tile_m": 1, "tile_n": 256, "tile_k": 1}
        if int(rewrite_counts.get("total_rewrite_candidates", 0)) > 0:
            defaults = dict(defaults)
            defaults["tile_n"] = min(int(defaults.get("tile_n", 256)), 128)
        profile = "cuda_matmul_conv_v1" if family == "matmul_conv" else "cuda_elementwise_reduction_v1"
        merged = dict(defaults)
        merged.update({k: v for k, v in schedule_info.items() if v is not None})
        env_overrides, profile_tag = _schedule_overrides_from_env()
        if env_overrides:
            merged.update(env_overrides)
        if profile_tag:
            profile = f"{profile}_{profile_tag}"
        return (
            "resolved schedule hints",
            {
                "schedule_hints": merged,
                "rewrite_aware": bool(int(rewrite_counts.get("total_rewrite_candidates", 0)) > 0),
                "op_family": family,
                "schedule_profile": profile,
                "profile_tag": profile_tag,
                "overrides_applied": dict(env_overrides),
            },
        )

    def _emit() -> tuple[str, dict[str, Any]]:
        if not can_execute:
            return (
                "emit skipped: missing concrete shape bindings for symbolic dims",
                {"emit_backend": "cpp_pybind", "emit_mode": "skipped_missing_bindings"},
            )
        from backends.cuda.codegen.cpp_driver import lower_intent_to_cuda_kernel_cpp  # noqa: PLC0415

        lowered = lower_intent_to_cuda_kernel_cpp(intent_payload, bindings=bindings)
        state["lowered"] = dict(lowered)
        state["launch"] = _parse_launch(lowered.get("launch") if isinstance(lowered.get("launch"), Mapping) else {})
        kernel_name = str(lowered.get("kernel_name") or name)
        io_spec = lowered.get("io_spec") if isinstance(lowered.get("io_spec"), Mapping) else {}
        output_names = [str(x) for x in (lowered.get("output_names") or [])]
        state["kernel_name"] = kernel_name
        state["io_spec"] = dict(io_spec)
        state["output_names"] = output_names
        state["cuda_src"] = str(lowered.get("cuda_src") or "")
        return (
            "emitted CUDA kernel via C++ pybind compiler",
            {
                "emit_backend": "cpp_pybind",
                "emit_mode": "executed",
                "kernel_name": kernel_name,
                "cuda_src_bytes": len(state["cuda_src"]),
                "output_count": len(output_names),
            },
        )

    def _compile() -> tuple[str, dict[str, Any]]:
        if not bool(execute_backend_stages):
            return ("compile skipped: compatibility mode", {"compile_mode": "skipped_compatibility"})
        if not can_execute:
            return ("compile skipped: missing bindings", {"compile_mode": "skipped_missing_bindings"})
        if "kernel_name" not in state or "cuda_src" not in state or "io_spec" not in state:
            raise ValueError("emit stage artifacts missing for compile")
        compile_cuda_extension(
            kernel_name=str(state["kernel_name"]),
            cuda_src=str(state["cuda_src"]),
            io_spec=dict(state["io_spec"]),
        )
        return (
            "compiled CUDA extension",
            {
                "compile_mode": "executed",
                "kernel_name": str(state["kernel_name"]),
            },
        )

    def _launch() -> tuple[str, dict[str, Any]]:
        if not bool(execute_backend_stages):
            return ("launch skipped: compatibility mode", {"launch_mode": "skipped_compatibility"})
        if not can_execute:
            return ("launch skipped: missing bindings", {"launch_mode": "skipped_missing_bindings"})
        if "kernel_name" not in state or "cuda_src" not in state or "io_spec" not in state or "launch" not in state:
            raise ValueError("compile/emit artifacts missing for launch")
        output_names = list(state.get("output_names") or [])
        inputs_np = _build_dummy_inputs(io_spec=state["io_spec"], output_names=output_names, bindings=bindings)
        _ = run_cuda_kernel(
            kernel_name=str(state["kernel_name"]),
            cuda_src=str(state["cuda_src"]),
            io_spec=dict(state["io_spec"]),
            launch=state["launch"],
            bindings=dict(bindings),
            inputs_np=inputs_np,
            output_names=output_names,
        )
        return (
            "launched CUDA kernel with synthetic inputs",
            {
                "launch_mode": "executed",
                "input_tensor_count": len(inputs_np),
                "output_count": len(output_names),
            },
        )

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
