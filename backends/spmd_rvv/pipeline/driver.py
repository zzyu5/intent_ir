"""
RVV compiler pipeline driver.

Pure-compiler lifecycle:
legalize -> shape_infer -> schedule -> emit_cpp -> compile -> run.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter
import numpy as np
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from backends.common.mlir_contract import CONTRACT_SCHEMA_V2, MlirBackendContract
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


def _resolve_repo_artifact_path(path_raw: str) -> Path:
    p = Path(str(path_raw).strip())
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parents[3] / p).resolve()


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


def _resolve_rvv_contract_executable(
    *,
    contract: MlirBackendContract,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    executable = contract.executable
    exe_format = str(executable.format or "").strip().lower()
    exe_target = str(executable.target or contract.backend or "rvv").strip().lower()
    if exe_target and exe_target != "rvv":
        raise ValueError(f"rvv contract executable target mismatch: {exe_target!r}")

    if exe_format in {"rvv_elf", "elf"}:
        exe_path_raw = str(executable.path or "").strip()
        if not exe_path_raw:
            raise ValueError("rvv contract executable.path is empty for ELF executable")
        exe_path = _resolve_repo_artifact_path(exe_path_raw)
        if not exe_path.is_file():
            raise FileNotFoundError(f"rvv ELF executable missing: {exe_path}")
        return {
            "executable_format": "rvv_elf",
            "executable_path": str(exe_path),
            "entry": str(executable.entry or contract.kernel_name or ""),
            "bindings": dict(bindings),
            "execution_engine": "mlir_native",
            "contract_schema_version": str(contract.schema_version or ""),
        }

    raise ValueError(
        "rvv contract executable unsupported or missing; expected executable.format in "
        "{rvv_elf,elf}. "
        "legacy cpp-driver runtime fallback has been removed."
    )


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
    elif isinstance(payload, Mapping):
        contract = MlirBackendContract.from_json_dict(dict(payload))
        source_kind = "mlir_contract"
    elif isinstance(payload, IntentMLIRModule):
        contract = build_rvv_contract(payload, source_kind="mlir_module")
        artifacts = dict(contract.artifacts or {})
        artifacts.setdefault("mlir_module_text", str(payload.module_text or ""))
        contract.artifacts = artifacts
        source_kind = "mlir_module"
    elif isinstance(payload, str):
        module = IntentMLIRModule(module_text=str(payload))
        contract = build_rvv_contract(module, source_kind="mlir_text")
        artifacts = dict(contract.artifacts or {})
        artifacts.setdefault("mlir_module_text", str(module.module_text or ""))
        contract.artifacts = artifacts
        source_kind = "mlir_text"
    else:
        raise ValueError("invalid mlir payload for rvv pipeline")

    if str(contract.schema_version or "") != CONTRACT_SCHEMA_V2:
        raise ValueError(
            f"unsupported rvv contract schema_version={contract.schema_version!r}; expected {CONTRACT_SCHEMA_V2}"
        )

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


def lower_rvv_contract_to_c_src(
    intent_payload: Any,
    *,
    shape_bindings: Mapping[str, Any] | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    mode: str = "verify",
) -> str:
    contract, _ = _resolve_rvv_contract(intent_payload, shape_bindings=shape_bindings)
    bindings = normalize_bindings(shape_bindings or contract.reason_context.get("shape_bindings") or {})
    if not bindings:
        raise ValueError("rvv contract lowering requires concrete shape bindings")
    resolved = _resolve_rvv_contract_executable(contract=contract, bindings=bindings)
    c_src = resolved.get("c_src")
    if isinstance(c_src, str) and c_src.strip():
        return str(c_src)
    # Compatibility path: when execution is hard-cut to prebuilt ELF, scripts may
    # still need C text for remote packaging and dtype parsing.
    allow_compat_c_src = str(os.getenv("INTENTIR_RVV_ALLOW_COMPAT_C_SRC", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if not allow_compat_c_src:
        raise ValueError(
            "rvv contract C-source compatibility access is disabled by default in hard-cut mode; "
            "set INTENTIR_RVV_ALLOW_COMPAT_C_SRC=1 to enable explicit compatibility fallback"
        )
    art_src = str((contract.artifacts or {}).get("rvv_kernel_src_path") or "").strip()
    if art_src:
        src_path = _resolve_repo_artifact_path(art_src)
        if src_path.is_file():
            return str(src_path.read_text(encoding="utf-8"))
    raise ValueError(
        "rvv contract is executable-only (rvv_elf) and has no readable rvv_kernel_src_path artifact "
        "for C-source compatibility access"
    )


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
            return ("emit skipped: missing bindings", {"emit_backend": "mlir_contract", "emit_mode": "skipped_missing_bindings"})
        resolved = _resolve_rvv_contract_executable(contract=contract, bindings=bindings)
        state["rvv_executable"] = dict(resolved)
        return (
            "resolved RVV executable via MLIR contract path",
            {
                "emit_backend": "mlir_contract",
                "emit_mode": "executed",
                "executable_format": str(resolved.get("executable_format") or ""),
                "prebuilt_elf_bytes": (
                    int(Path(str(resolved.get("executable_path") or "")).stat().st_size)
                    if str(resolved.get("executable_format") or "") == "rvv_elf"
                    and Path(str(resolved.get("executable_path") or "")).is_file()
                    else 0
                ),
                "contract_schema_version": str(contract.schema_version or ""),
                "execution_engine": "mlir_native",
            },
        )

    def _compile() -> tuple[str, dict[str, Any]]:
        if mode == "schedule_only":
            return ("compile skipped: schedule_only mode", {"compile_mode": "skipped_schedule_only", "pipeline_mode": mode})
        if not can_execute:
            return ("compile skipped: missing bindings", {"compile_mode": "skipped_missing_bindings"})
        resolved = state.get("rvv_executable")
        if not isinstance(resolved, Mapping):
            raise ValueError("emit_cpp stage artifacts missing executable metadata")
        exe_format = str(resolved.get("executable_format") or "")
        tmp_ctx = tempfile.TemporaryDirectory(prefix=f"intentir_rvv_pipeline_{name}_")
        td = Path(tmp_ctx.name)
        state["tmp_ctx"] = tmp_ctx
        state["tmp_dir"] = td
        if exe_format == "rvv_elf":
            src_elf = Path(str(resolved.get("executable_path") or ""))
            if not src_elf.is_file():
                raise FileNotFoundError(f"rvv prebuilt executable missing: {src_elf}")
            dst_elf = td / "run"
            shutil.copy(src_elf, dst_elf)
            dst_elf.chmod(dst_elf.stat().st_mode | 0o111)
            state["run_bin"] = dst_elf
            return (
                "staged prebuilt RVV ELF executable",
                {
                    "compile_mode": "prebuilt_elf_staged",
                    "workdir": str(td),
                },
            )
        raise ValueError("compile stage requires prebuilt rvv_elf executable from emit stage")

    def _run() -> tuple[str, dict[str, Any]]:
        if mode == "schedule_only":
            return ("run skipped: schedule_only mode", {"run_mode": "skipped_schedule_only", "pipeline_mode": mode})
        if not can_execute:
            return ("run skipped: missing bindings", {"run_mode": "skipped_missing_bindings"})
        td = state.get("tmp_dir")
        if not isinstance(td, Path):
            raise ValueError("compile stage artifacts missing workdir")
        run_bin = state.get("run_bin")
        if not isinstance(run_bin, Path):
            run_bin = td / "run"
        in_count, out_count = _build_dummy_io_files(contract=contract, bindings=bindings, out_dir=td)
        rp = subprocess.run([str(run_bin)], cwd=td, capture_output=True, text=True)
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
