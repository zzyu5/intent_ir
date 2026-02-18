"""
CUDA compiler pipeline driver.

This driver now models a pure-compiler lifecycle:
legalize -> shape_infer -> schedule -> emit -> compile -> launch.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Mapping

from backends.cuda.runtime import CudaLaunch, compile_cuda_extension, run_cuda_kernel
from backends.common.mlir_bridge import resolve_intent_payload_with_meta
from backends.common.pipeline_utils import (
    collect_intent_info,
    has_symbolic_dims,
    legalize_rewrite_counts,
    normalize_bindings,
    np_dtype,
    op_family,
    resolve_dim_int,
    run_stage,
    schedule_overrides_from_env,
)

from .stages import CUDA_PIPELINE_STAGES, CudaPipelineResult, CudaPipelineStage


def _stage(name: str, fn) -> CudaPipelineStage:
    return run_stage(name, fn, stage_factory=CudaPipelineStage)


def _classify_failure(detail: str) -> str:
    msg = str(detail).lower()
    if "unsupported" in msg or "missing op" in msg:
        return "lowering_missing_op"
    if "invalid" in msg or "empty" in msg:
        return "invalid_intent"
    if "compile_timeout" in msg or "launch_timeout" in msg:
        return msg
    return "runtime_fail"


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
        dtype = np_dtype(str(spec.get("dtype") or "f32"))
        shape_spec = list(spec.get("shape") or [])
        shape = tuple(max(1, resolve_dim_int(d, bindings)) for d in shape_spec)
        if len(shape) == 0:
            inputs[n] = np.array(1, dtype=dtype)
        else:
            inputs[n] = np.zeros(shape, dtype=dtype)
    return inputs


def run_cuda_pipeline(
    intent_payload: Any,
    *,
    shape_bindings: Mapping[str, Any] | None = None,
    pipeline_mode: str = "full",
) -> CudaPipelineResult:
    intent_payload, bridge_meta = resolve_intent_payload_with_meta(intent_payload)
    mode = str(pipeline_mode or "full").strip().lower()
    if mode not in {"full", "schedule_only"}:
        raise ValueError(f"unsupported cuda pipeline_mode: {pipeline_mode}")
    name, op_names, tensor_shapes, schedule_info = collect_intent_info(intent_payload)
    stages: list[CudaPipelineStage] = []
    rewrite_counts = legalize_rewrite_counts(op_names)
    family = op_family(op_names)
    bindings = normalize_bindings(shape_bindings)
    symbolic_dims_present = has_symbolic_dims(tensor_shapes)
    can_execute = bool(bindings) or (not symbolic_dims_present)
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
                "input_ir_kind": str(bridge_meta.source_kind),
                "mlir_parse_ms": float(bridge_meta.mlir_parse_ms),
                "mlir_bridge_used": bool(bridge_meta.used_mlir_bridge),
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
        env_overrides, profile_tag = schedule_overrides_from_env(backend_prefix="CUDA")
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
        if mode == "schedule_only":
            return ("compile skipped: schedule_only mode", {"compile_mode": "skipped_schedule_only", "pipeline_mode": mode})
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
        if mode == "schedule_only":
            return ("launch skipped: schedule_only mode", {"launch_mode": "skipped_schedule_only", "pipeline_mode": mode})
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
    return CudaPipelineResult(
        ok=ok,
        stages=stages,
        reason_code=("ok" if ok else fail_reason),
        reason_detail=fail_detail,
        input_ir_kind=str(bridge_meta.source_kind),
        mlir_parse_ms=float(bridge_meta.mlir_parse_ms),
    )
