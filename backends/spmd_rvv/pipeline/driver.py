"""
RVV compiler pipeline driver.

Pure-compiler lifecycle:
legalize -> shape_infer -> schedule -> emit_cpp -> compile -> run.
"""

from __future__ import annotations

import os
import numpy as np
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from ..opset import SPMD_RVV_SUPPORTED_OPS
from .stages import RVV_PIPELINE_STAGES, RvvPipelineResult, RvvPipelineStage


def _stage(name: str, fn) -> RvvPipelineStage:
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
        return RvvPipelineStage(name=name, ok=False, ms=(perf_counter() - t0) * 1000.0, detail=str(e), artifacts={})
    return RvvPipelineStage(name=name, ok=True, ms=(perf_counter() - t0) * 1000.0, detail=detail, artifacts=artifacts)


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
    op_names = [str(getattr(op, "op", "") or "") for op in ops_raw if str(getattr(op, "op", "") or "")]
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
    tile_m = _env_int("INTENTIR_RVV_TILE_M", "INTENTIR_TILE_M")
    tile_n = _env_int("INTENTIR_RVV_TILE_N", "INTENTIR_TILE_N")
    tile_k = _env_int("INTENTIR_RVV_TILE_K", "INTENTIR_TILE_K")
    if tile_m is not None:
        overrides["tile_m"] = int(tile_m)
    if tile_n is not None:
        overrides["tile_n"] = int(tile_n)
    if tile_k is not None:
        overrides["tile_k"] = int(tile_k)
    tag = str(
        os.getenv("INTENTIR_RVV_SCHEDULE_PROFILE_TAG")
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


def _resolve_dim_int(dim: Any, bindings: Mapping[str, Any]) -> int:
    # Support IR Dim objects directly; `str(Dim(...))` is not a usable symbol key.
    raw = _dim_value(dim)
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


def _write_bin(path: Path, arr: np.ndarray, dtype: str) -> None:
    arr_np = np.asarray(arr)
    if arr_np.dtype.kind == "b" or dtype in {"bool", "i1"}:
        raw = np.asarray(arr_np, dtype=np.uint8).tobytes(order="C")
    elif dtype == "i8":
        raw = np.asarray(arr_np, dtype=np.int8).tobytes(order="C")
    elif dtype == "u8":
        raw = np.asarray(arr_np, dtype=np.uint8).tobytes(order="C")
    elif dtype == "i16":
        raw = np.asarray(arr_np, dtype=np.int16).tobytes(order="C")
    elif dtype == "i32":
        raw = np.asarray(arr_np, dtype=np.int32).tobytes(order="C")
    elif dtype == "i64":
        raw = np.asarray(arr_np, dtype=np.int64).tobytes(order="C")
    else:
        raw = np.asarray(arr_np, dtype=np.float32).tobytes(order="C")
    path.write_bytes(raw)


def _collect_external_inputs(intent_payload: Any) -> list[str]:
    produced = {str(getattr(op, "output", "")) for op in list(getattr(intent_payload, "ops", []) or []) if str(getattr(op, "output", ""))}
    used: list[str] = []
    for op in list(getattr(intent_payload, "ops", []) or []):
        for inp in list(getattr(op, "inputs", []) or []):
            s = str(inp)
            if s and s not in produced:
                used.append(s)
    seen: set[str] = set()
    out: list[str] = []
    for n in used:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _build_dummy_io_files(*, intent_payload: Any, bindings: Mapping[str, Any], out_dir: Path) -> tuple[int, int]:
    tensors = getattr(intent_payload, "tensors", {})
    if not isinstance(tensors, Mapping):
        tensors = {}
    external_inputs = _collect_external_inputs(intent_payload)
    outputs = [str(x) for x in list(getattr(intent_payload, "outputs", []) or [])]
    in_count = 0
    out_count = 0
    for name in external_inputs:
        t = tensors.get(name)
        if t is None:
            continue
        dtype = str(getattr(t, "dtype", "f32"))
        shape = tuple(max(1, _resolve_dim_int(d, bindings)) for d in list(getattr(t, "shape", []) or []))
        arr = np.array(1, dtype=_np_dtype(dtype)) if len(shape) == 0 else np.zeros(shape, dtype=_np_dtype(dtype))
        _write_bin(out_dir / f"{name}.bin", arr, dtype)
        in_count += 1
    for name in outputs:
        t = tensors.get(name)
        if t is None:
            continue
        dtype = str(getattr(t, "dtype", "f32"))
        shape = tuple(max(1, _resolve_dim_int(d, bindings)) for d in list(getattr(t, "shape", []) or []))
        arr = np.array(0, dtype=_np_dtype(dtype)) if len(shape) == 0 else np.zeros(shape, dtype=_np_dtype(dtype))
        _write_bin(out_dir / f"{name}_ref.bin", arr, dtype)
        out_count += 1
    return in_count, out_count


def _compile_local_rvv_program(*, workdir: Path) -> subprocess.CompletedProcess[str]:
    compile_cmd = [
        "gcc",
        "-O2",
        "-std=c11",
        "-D_POSIX_C_SOURCE=200809L",
        "-I.",
        "-o",
        str(workdir / "run"),
        str(workdir / "main.c"),
        str(workdir / "intentir_runtime.c"),
        str(workdir / "intentir_driver.c"),
        str(workdir / "intentir_ops.c"),
        "-lm",
        "-lrt",
    ]
    return subprocess.run(compile_cmd, cwd=workdir, capture_output=True, text=True)


def run_rvv_pipeline(
    intent_payload: Any,
    *,
    shape_bindings: Mapping[str, Any] | None = None,
    execute_backend_stages: bool = True,
) -> RvvPipelineResult:
    name, op_names, tensor_shapes, schedule_info = _collect_intent_info(intent_payload)
    stages: list[RvvPipelineStage] = []
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
        unsupported = sorted([op for op in op_names if op not in SPMD_RVV_SUPPORTED_OPS])
        if unsupported:
            raise ValueError(f"unsupported ops for rvv pipeline: {unsupported}")
        return (
            "validated rvv intent payload",
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
        defaults = {"tile_m": 1, "tile_n": 128, "tile_k": 1}
        if family == "matmul_conv":
            defaults = {"tile_m": 32, "tile_n": 64, "tile_k": 16}
        if int(rewrite_counts.get("total_rewrite_candidates", 0)) > 0:
            defaults = dict(defaults)
            defaults["tile_n"] = min(int(defaults.get("tile_n", 128)), 64)
        profile = "rvv_matmul_conv_v1" if family == "matmul_conv" else "rvv_elementwise_reduction_v1"
        merged = dict(defaults)
        merged.update({k: v for k, v in schedule_info.items() if v is not None})
        env_overrides, profile_tag = _schedule_overrides_from_env()
        if env_overrides:
            merged.update(env_overrides)
        if profile_tag:
            profile = f"{profile}_{profile_tag}"
        return (
            "resolved rvv schedule hints",
            {
                "schedule_hints": merged,
                "rewrite_aware": bool(int(rewrite_counts.get("total_rewrite_candidates", 0)) > 0),
                "op_family": family,
                "schedule_profile": profile,
                "profile_tag": profile_tag,
                "overrides_applied": dict(env_overrides),
            },
        )

    def _emit_cpp() -> tuple[str, dict[str, Any]]:
        if not can_execute:
            return ("emit skipped: missing bindings", {"emit_backend": "cpp", "emit_mode": "skipped_missing_bindings"})
        from backends.spmd_rvv.codegen.cpp_driver import lower_intent_to_c_with_files_cpp  # noqa: PLC0415

        src = lower_intent_to_c_with_files_cpp(intent_payload, shape_bindings=bindings, atol=1e-3, rtol=1e-3, mode="verify")
        state["c_src"] = str(src)
        return (
            "emitted standalone C via RVV C++ codegen",
            {"emit_backend": "cpp", "emit_mode": "executed", "c_src_bytes": len(str(src))},
        )

    def _compile() -> tuple[str, dict[str, Any]]:
        if not bool(execute_backend_stages):
            return ("compile skipped: compatibility mode", {"compile_mode": "skipped_compatibility"})
        if not can_execute:
            return ("compile skipped: missing bindings", {"compile_mode": "skipped_missing_bindings"})
        c_src = str(state.get("c_src") or "")
        if not c_src:
            raise ValueError("emit_cpp stage artifacts missing")
        tmp_ctx = tempfile.TemporaryDirectory(prefix=f"intentir_rvv_pipeline_{name}_")
        td = Path(tmp_ctx.name)
        state["tmp_ctx"] = tmp_ctx
        state["tmp_dir"] = td
        (td / "main.c").write_text(c_src, encoding="utf-8")
        runtime_dir = Path(__file__).resolve().parents[1] / "runtime"
        for fn in [
            "intentir_runtime.h",
            "intentir_runtime.c",
            "intentir_driver.h",
            "intentir_driver.c",
            "intentir_ops.h",
            "intentir_ops.c",
        ]:
            src_p = runtime_dir / fn
            if not src_p.exists():
                raise FileNotFoundError(f"missing RVV runtime file: {src_p}")
            shutil.copy(src_p, td / fn)
        cp = _compile_local_rvv_program(workdir=td)
        if int(cp.returncode) != 0:
            raise RuntimeError(f"rvv local compile failed rc={cp.returncode}: {cp.stderr or cp.stdout}")
        return (
            "compiled local RVV host program",
            {"compile_mode": "executed", "workdir": str(td)},
        )

    def _run() -> tuple[str, dict[str, Any]]:
        if not bool(execute_backend_stages):
            return ("run skipped: compatibility mode", {"run_mode": "skipped_compatibility"})
        if not can_execute:
            return ("run skipped: missing bindings", {"run_mode": "skipped_missing_bindings"})
        td = state.get("tmp_dir")
        if not isinstance(td, Path):
            raise ValueError("compile stage artifacts missing workdir")
        in_count, out_count = _build_dummy_io_files(intent_payload=intent_payload, bindings=bindings, out_dir=td)
        rp = subprocess.run([str(td / "run")], cwd=td, capture_output=True, text=True)
        if int(rp.returncode) != 0:
            raise RuntimeError(f"rvv local run failed rc={rp.returncode}: {rp.stderr or rp.stdout}")
        return (
            "executed local RVV host program with synthetic inputs",
            {
                "run_mode": "executed",
                "input_tensor_count": int(in_count),
                "output_count": int(out_count),
            },
        )

    stage_impls = {
        "legalize": _legalize,
        "shape_infer": _shape_infer,
        "schedule": _schedule,
        "emit_cpp": _emit_cpp,
        "compile": _compile,
        "run": _run,
    }
    failed = False
    fail_reason = "ok"
    fail_detail = ""
    try:
        for stage_name in RVV_PIPELINE_STAGES:
            if failed:
                stages.append(
                    RvvPipelineStage(
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
    finally:
        tmp_ctx = state.get("tmp_ctx")
        if tmp_ctx is not None:
            try:
                tmp_ctx.cleanup()
            except Exception:
                pass

    ok = not failed and all(s.ok for s in stages)
    return RvvPipelineResult(ok=ok, stages=stages, reason_code=("ok" if ok else fail_reason), reason_detail=fail_detail)
