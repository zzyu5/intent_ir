"""
RVV compiler pipeline driver.

Pure-compiler lifecycle:
legalize -> shape_infer -> schedule -> emit_cpp -> compile -> run.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from backends.common.mlir_contract import MlirBackendContract
from backends.common.pipeline_utils import (
    has_symbolic_dims,
    legalize_rewrite_counts,
    normalize_bindings,
    np_dtype,
    op_family,
    resolve_dim_int,
    run_stage,
    schedule_overrides_from_env,
)
from intent_ir.mlir.module import IntentMLIRModule
from intent_ir.mlir.passes.emit_rvv_contract import build_rvv_contract

from ..opset import SPMD_RVV_SUPPORTED_OPS
from .stages import RVV_PIPELINE_STAGES, RvvPipelineResult, RvvPipelineStage


def _stage(name: str, fn) -> RvvPipelineStage:
    return run_stage(name, fn, stage_factory=RvvPipelineStage)


def _classify_failure(detail: str) -> str:
    msg = str(detail).lower()
    if "unsupported" in msg or "missing op" in msg:
        return "lowering_missing_op"
    if "invalid" in msg or "empty" in msg:
        return "invalid_intent"
    return "runtime_fail"


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


def _build_dummy_io_files(*, contract: MlirBackendContract, bindings: Mapping[str, Any], out_dir: Path) -> tuple[int, int]:
    io_spec = contract.io_spec if isinstance(contract.io_spec, Mapping) else {}
    tensors = io_spec.get("tensors") if isinstance(io_spec.get("tensors"), Mapping) else {}
    outputs = [str(x) for x in list(io_spec.get("outputs") or [])]
    output_set = set(outputs)
    in_count = 0
    out_count = 0
    for name, spec in tensors.items():
        if not isinstance(spec, Mapping):
            continue
        n = str(name)
        dtype = str(spec.get("dtype") or "f32")
        shape_spec = list(spec.get("shape") or [])
        shape = tuple(max(1, resolve_dim_int(d, bindings)) for d in shape_spec)
        if n in output_set:
            arr = np.array(0, dtype=np_dtype(dtype)) if len(shape) == 0 else np.zeros(shape, dtype=np_dtype(dtype))
            _write_bin(out_dir / f"{n}_ref.bin", arr, dtype)
            out_count += 1
        else:
            arr = np.array(1, dtype=np_dtype(dtype)) if len(shape) == 0 else np.zeros(shape, dtype=np_dtype(dtype))
            _write_bin(out_dir / f"{n}.bin", arr, dtype)
            in_count += 1
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


@dataclass(frozen=True)
class _InputMeta:
    source_kind: str
    mlir_parse_ms: float
    mlir_backend_contract_used: bool


def _resolve_rvv_contract(
    payload: Any,
    *,
    shape_bindings: Mapping[str, Any] | None = None,
) -> tuple[MlirBackendContract, _InputMeta]:
    t0 = perf_counter()
    bindings = normalize_bindings(shape_bindings)

    if isinstance(payload, MlirBackendContract):
        contract = payload
        source_kind = "mlir_contract"
    elif isinstance(payload, Mapping) and str(payload.get("schema_version") or "") == "intent_mlir_backend_contract_v1":
        contract = MlirBackendContract.from_json_dict(dict(payload))
        source_kind = "mlir_contract"
    elif isinstance(payload, IntentMLIRModule):
        contract = build_rvv_contract(payload, source_kind="mlir_module")
        source_kind = "mlir_module"
    elif isinstance(payload, str):
        module = IntentMLIRModule(module_text=str(payload))
        contract = build_rvv_contract(module, source_kind="mlir_text")
        source_kind = "mlir_text"
    else:
        raise ValueError("invalid mlir payload for rvv pipeline")

    if bindings:
        reason_context = dict(contract.reason_context or {})
        reason_context["shape_bindings"] = dict(bindings)
        contract.reason_context = reason_context

    dt_ms = float((perf_counter() - t0) * 1000.0)
    return contract, _InputMeta(source_kind=source_kind, mlir_parse_ms=dt_ms, mlir_backend_contract_used=True)


def _contract_ops(contract: MlirBackendContract) -> list[str]:
    return [str(x) for x in list(contract.op_names or []) if str(x).strip()]


def _contract_tensor_shapes(contract: MlirBackendContract) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for k, v in dict(contract.tensor_shapes or {}).items():
        if isinstance(v, list):
            out[str(k)] = list(v)
    return out


def run_rvv_pipeline(
    intent_payload: Any,
    *,
    shape_bindings: Mapping[str, Any] | None = None,
    pipeline_mode: str = "full",
) -> RvvPipelineResult:
    try:
        contract, input_meta = _resolve_rvv_contract(intent_payload, shape_bindings=shape_bindings)
    except Exception as e:
        detail = str(e)
        fail_stage = RvvPipelineStage(
            name="legalize",
            ok=False,
            ms=0.0,
            detail=detail,
            artifacts={"input_ir_kind": "unknown"},
        )
        stages = [fail_stage]
        for stage_name in RVV_PIPELINE_STAGES[1:]:
            stages.append(
                RvvPipelineStage(
                    name=stage_name,
                    ok=False,
                    ms=0.0,
                    detail="skipped_after_failure",
                    artifacts={"skipped": True},
                )
            )
        return RvvPipelineResult(
            ok=False,
            stages=stages,
            reason_code=_classify_failure(detail),
            reason_detail=detail,
            input_ir_kind="unknown",
            mlir_parse_ms=0.0,
            mlir_backend_contract_used=False,
        )

    mode = str(pipeline_mode or "full").strip().lower()
    if mode not in {"full", "schedule_only"}:
        raise ValueError(f"unsupported rvv pipeline_mode: {pipeline_mode}")
    name = str(contract.kernel_name or "intent")
    op_names = _contract_ops(contract)
    tensor_shapes = _contract_tensor_shapes(contract)
    schedule_info = dict(contract.schedule or {})
    stages: list[RvvPipelineStage] = []
    rewrite_counts = legalize_rewrite_counts(op_names)
    family = op_family(op_names)
    bindings = normalize_bindings(shape_bindings or contract.reason_context.get("shape_bindings") or {})
    symbolic_dims_present = has_symbolic_dims(tensor_shapes)
    can_execute = bool(bindings) or (not symbolic_dims_present)
    state: dict[str, Any] = {"bindings": dict(bindings), "contract": contract}

    def _legalize() -> tuple[str, dict[str, Any]]:
        if not op_names:
            raise ValueError("invalid intent: empty ops")
        if not tensor_shapes:
            raise ValueError("invalid intent: empty tensors")
        unsupported = sorted([op for op in op_names if op not in SPMD_RVV_SUPPORTED_OPS])
        if unsupported:
            raise ValueError(f"unsupported ops for rvv pipeline: {unsupported}")
        return (
            "validated rvv mlir backend contract",
            {
                "intent_name": name,
                "op_count": len(op_names),
                "tensor_count": len(tensor_shapes),
                "ops": op_names,
                "rewrite_counts": rewrite_counts,
                "input_ir_kind": str(input_meta.source_kind),
                "mlir_parse_ms": float(input_meta.mlir_parse_ms),
                "mlir_backend_contract_used": bool(input_meta.mlir_backend_contract_used),
                "contract_backend": str(contract.backend or "rvv"),
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
        env_overrides, profile_tag = schedule_overrides_from_env(backend_prefix="RVV")
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
        from backends.spmd_rvv.codegen.cpp_driver import lower_intent_json_to_c_with_files_cpp  # noqa: PLC0415

        intent_json = contract.intent_json
        if not isinstance(intent_json, dict):
            raise ValueError("mlir backend contract missing intent_json for rvv emission")
        src = lower_intent_json_to_c_with_files_cpp(intent_json, shape_bindings=bindings, atol=1e-3, rtol=1e-3, mode="verify")
        state["c_src"] = str(src)
        return (
            "emitted standalone C via RVV C++ codegen",
            {"emit_backend": "cpp", "emit_mode": "executed", "c_src_bytes": len(str(src))},
        )

    def _compile() -> tuple[str, dict[str, Any]]:
        if mode == "schedule_only":
            return ("compile skipped: schedule_only mode", {"compile_mode": "skipped_schedule_only", "pipeline_mode": mode})
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
        if mode == "schedule_only":
            return ("run skipped: schedule_only mode", {"run_mode": "skipped_schedule_only", "pipeline_mode": mode})
        if not can_execute:
            return ("run skipped: missing bindings", {"run_mode": "skipped_missing_bindings"})
        td = state.get("tmp_dir")
        if not isinstance(td, Path):
            raise ValueError("compile stage artifacts missing workdir")
        in_count, out_count = _build_dummy_io_files(contract=contract, bindings=bindings, out_dir=td)
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
    return RvvPipelineResult(
        ok=ok,
        stages=stages,
        reason_code=("ok" if ok else fail_reason),
        reason_detail=fail_detail,
        input_ir_kind=str(input_meta.source_kind),
        mlir_parse_ms=float(input_meta.mlir_parse_ms),
        mlir_backend_contract_used=bool(input_meta.mlir_backend_contract_used),
    )
