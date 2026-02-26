"""
CUDA compiler pipeline driver.

Pure-compiler lifecycle:
legalize -> shape_infer -> schedule -> emit -> compile -> launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter
import numpy as np
from typing import Any, Mapping

from backends.common.mlir_contract import CONTRACT_SCHEMA_V2, MlirBackendContract
from backends.cuda.runtime import (
    CudaLaunch,
    load_cuda_ptx_module,
    run_cuda_kernel_ptx,
)
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
from intent_ir.mlir.passes.emit_cuda_contract import build_cuda_contract
from pipeline.common.strict_policy import cuda_require_llvm_ptx

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


def _resolve_repo_artifact_path(path_raw: str) -> Path:
    p = Path(str(path_raw).strip())
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parents[3] / p).resolve()

def _try_int(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def _base_expr_binding(bindings: Mapping[str, Any], *, base: str) -> int | None:
    # Heuristic for keys like "M + 1", "N + 3" emitted by shape materialization.
    pat = re.compile(rf"^{re.escape(str(base))}\s*\+\s*\d+$")
    out: list[int] = []
    for k, v in dict(bindings).items():
        key = str(k).strip()
        if not key:
            continue
        if not pat.match(key):
            continue
        iv = _try_int(v)
        if iv is not None:
            out.append(iv)
    if not out:
        return None
    return int(max(out))


def _augment_scalar_bindings_from_io_spec(
    *,
    bindings: Mapping[str, Any],
    io_spec: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(bindings or {})
    scalars = io_spec.get("scalars") if isinstance(io_spec.get("scalars"), Mapping) else {}
    if not isinstance(scalars, Mapping) or not scalars:
        return out

    def _has_binding(name: str) -> bool:
        if name not in out:
            return False
        value = out.get(name)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True

    def _set_if_missing(name: str, value: Any) -> None:
        if _has_binding(name):
            return
        iv = _try_int(value)
        if iv is not None:
            out[name] = int(iv)

    _set_if_missing("M0", out.get("M"))
    _set_if_missing("M1", out.get("M"))
    _set_if_missing("N0", out.get("N"))
    _set_if_missing("N1", out.get("N"))
    _set_if_missing("M_OUT", out.get("M_OUT"))
    _set_if_missing("N_OUT", out.get("N_OUT"))
    if not _has_binding("M_OUT"):
        _set_if_missing("M_OUT", _base_expr_binding(out, base="M"))
    if not _has_binding("N_OUT"):
        _set_if_missing("N_OUT", _base_expr_binding(out, base="N"))
    _set_if_missing("M_OUT", out.get("M"))
    _set_if_missing("N_OUT", out.get("N"))

    if "T" in scalars and not _has_binding("T"):
        m_out = _try_int(out.get("M_OUT"))
        n_out = _try_int(out.get("N_OUT"))
        m = _try_int(out.get("M"))
        n = _try_int(out.get("N"))
        l = _try_int(out.get("L"))
        if m_out is not None and n_out is not None:
            out["T"] = int(m_out * n_out)
        elif m is not None and n_out is not None:
            out["T"] = int(m * n_out)
        elif m is not None and n is not None:
            out["T"] = int(m * n)
        elif l is not None and n is not None:
            out["T"] = int(l * n)
        elif n_out is not None:
            out["T"] = int(n_out)
        elif n is not None:
            out["T"] = int(n)

    for name in [str(k) for k in scalars.keys()]:
        if _has_binding(name):
            continue
        m = re.match(r"^([A-Za-z_]+)\d+$", name)
        if m is not None:
            base = str(m.group(1))
            if base in out:
                _set_if_missing(name, out.get(base))
    return out


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


@dataclass(frozen=True)
class _InputMeta:
    source_kind: str
    mlir_parse_ms: float
    mlir_backend_contract_used: bool


def _resolve_cuda_contract(
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
        contract = build_cuda_contract(payload, source_kind="mlir_module")
        artifacts = dict(contract.artifacts or {})
        artifacts.setdefault("mlir_module_text", str(payload.module_text or ""))
        contract.artifacts = artifacts
        source_kind = "mlir_module"
    elif isinstance(payload, str):
        module = IntentMLIRModule(module_text=str(payload))
        contract = build_cuda_contract(module, source_kind="mlir_text")
        artifacts = dict(contract.artifacts or {})
        artifacts.setdefault("mlir_module_text", str(module.module_text or ""))
        contract.artifacts = artifacts
        source_kind = "mlir_text"
    else:
        raise ValueError("invalid mlir payload for cuda pipeline")

    if str(contract.schema_version or "") != CONTRACT_SCHEMA_V2:
        raise ValueError(
            f"unsupported cuda contract schema_version={contract.schema_version!r}; expected {CONTRACT_SCHEMA_V2}"
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


def lower_cuda_contract_to_kernel(
    intent_payload: Any,
    *,
    shape_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract, _ = _resolve_cuda_contract(intent_payload, shape_bindings=shape_bindings)
    bindings = normalize_bindings(shape_bindings or contract.reason_context.get("shape_bindings") or {})
    if not bindings:
        raise ValueError("cuda contract lowering requires concrete shape bindings")
    executable = contract.executable
    exe_format = str(executable.format or "").strip().lower()
    exe_entry = str(executable.entry or contract.kernel_name or "").strip()
    exe_target = str(executable.target or contract.backend or "cuda").strip().lower()
    if exe_target and exe_target != "cuda":
        raise ValueError(f"cuda contract executable target mismatch: {exe_target!r}")
    strict_llvm_ptx = bool(cuda_require_llvm_ptx())
    if exe_format in {"cuda_ptx", "ptx"}:
        exe_path_raw = str(executable.path or "").strip()
        if not exe_path_raw:
            raise ValueError("cuda contract executable.path is empty for cuda_ptx executable")
        exe_path = _resolve_repo_artifact_path(exe_path_raw)
        if not exe_path.is_file():
            raise FileNotFoundError(f"cuda ptx executable missing: {exe_path}")
        ptx_origin = str((contract.artifacts or {}).get("cuda_ptx_origin") or "").strip().lower()
        # Backward compatibility for older prebuilt PTX contracts that did not
        # stamp origin metadata. Strict mode treats these as LLVM-produced PTX.
        if not ptx_origin:
            ptx_origin = "llvm_llc"
        if strict_llvm_ptx and ptx_origin != "llvm_llc":
            raise ValueError(
                "cuda contract executable rejected under strict LLVM PTX mode: "
                f"cuda_ptx_origin={ptx_origin!r}"
            )
        invocation = dict(executable.invocation or {})
        merged_bindings = dict(bindings)
        inv_shape_bindings = invocation.get("shape_bindings")
        if isinstance(inv_shape_bindings, Mapping):
            for k, v in inv_shape_bindings.items():
                key = str(k).strip()
                if not key:
                    continue
                try:
                    merged_bindings.setdefault(key, int(v))
                except Exception:
                    continue
        io_spec = dict(contract.io_spec or {})
        inv_io = invocation.get("io_spec")
        if isinstance(inv_io, Mapping):
            # For prebuilt PTX materialized from CUDA codegen, runtime launch needs
            # arg_names/scalars metadata that semantic-level contracts may not carry.
            if not isinstance(io_spec.get("arg_names"), list):
                io_spec = dict(inv_io)
        merged_bindings = _augment_scalar_bindings_from_io_spec(bindings=merged_bindings, io_spec=io_spec)
        output_names = list((io_spec.get("outputs") if isinstance(io_spec, Mapping) else []) or [])
        inv_outputs = invocation.get("output_names")
        if (not output_names) and isinstance(inv_outputs, list):
            output_names = [str(x) for x in list(inv_outputs)]
        launch = dict(contract.launch or {})
        if (not launch) and isinstance(invocation.get("launch"), Mapping):
            launch = dict(invocation.get("launch") or {})
        return {
            "kernel_name": str(exe_entry or contract.kernel_name or "intent"),
            "io_spec": io_spec,
            "launch": launch,
            "output_names": [str(x) for x in output_names],
            "bindings": dict(merged_bindings),
            "cuda_ptx": exe_path.read_bytes(),
            "executable_format": exe_format,
            "executable_path": str(exe_path),
            "execution_engine": "mlir_native",
            "contract_schema_version": str(contract.schema_version or ""),
            "cuda_ptx_origin": ptx_origin,
            "runtime_fallback": bool(ptx_origin and ptx_origin != "llvm_llc"),
            "runtime_fallback_detail": (f"cuda_ptx_origin={ptx_origin}" if ptx_origin and ptx_origin != "llvm_llc" else ""),
        }
    raise ValueError(
        "cuda contract executable unsupported or missing; expected executable.format in "
        "{cuda_ptx,ptx}. "
        "mlir_module executable fallback is removed in strict hard-cut mode."
    )


def run_cuda_pipeline(
    intent_payload: Any,
    *,
    shape_bindings: Mapping[str, Any] | None = None,
    pipeline_mode: str = "full",
) -> CudaPipelineResult:
    try:
        contract, input_meta = _resolve_cuda_contract(intent_payload, shape_bindings=shape_bindings)
    except Exception as e:
        detail = str(e)
        fail_stage = CudaPipelineStage(
            name="legalize",
            ok=False,
            ms=0.0,
            detail=detail,
            artifacts={"input_ir_kind": "unknown"},
        )
        stages = [fail_stage]
        for stage_name in CUDA_PIPELINE_STAGES[1:]:
            stages.append(
                CudaPipelineStage(
                    name=stage_name,
                    ok=False,
                    ms=0.0,
                    detail="skipped_after_failure",
                    artifacts={"skipped": True},
                )
            )
        return CudaPipelineResult(
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
        raise ValueError(f"unsupported cuda pipeline_mode: {pipeline_mode}")

    name = str(contract.kernel_name or "intent")
    op_names = _contract_ops(contract)
    tensor_shapes = _contract_tensor_shapes(contract)
    schedule_info = dict(contract.schedule or {})
    stages: list[CudaPipelineStage] = []
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
        return (
            "validated mlir backend contract",
            {
                "intent_name": name,
                "op_count": len(op_names),
                "tensor_count": len(tensor_shapes),
                "ops": op_names,
                "rewrite_counts": rewrite_counts,
                "input_ir_kind": str(input_meta.source_kind),
                "mlir_parse_ms": float(input_meta.mlir_parse_ms),
                "mlir_backend_contract_used": bool(input_meta.mlir_backend_contract_used),
                "contract_backend": str(contract.backend or "cuda"),
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
                {"emit_backend": "mlir_contract", "emit_mode": "skipped_missing_bindings"},
            )
        lowered = lower_cuda_contract_to_kernel(contract, shape_bindings=bindings)
        state["lowered"] = dict(lowered)
        state["launch"] = _parse_launch(lowered.get("launch") if isinstance(lowered.get("launch"), Mapping) else {})
        kernel_name = str(lowered.get("kernel_name") or name)
        io_spec = lowered.get("io_spec") if isinstance(lowered.get("io_spec"), Mapping) else {}
        output_names = [str(x) for x in (lowered.get("output_names") or [])]
        state["kernel_name"] = kernel_name
        state["io_spec"] = dict(io_spec)
        state["output_names"] = output_names
        state["bindings"] = dict(lowered.get("bindings") or bindings)
        exe_format = str(lowered.get("executable_format") or "").strip()
        ptx_origin = str(lowered.get("cuda_ptx_origin") or "").strip()
        state["executable_format"] = exe_format
        state["cuda_ptx_origin"] = ptx_origin
        ptx_payload = lowered.get("cuda_ptx")
        if isinstance(ptx_payload, (bytes, bytearray)):
            state["cuda_ptx"] = bytes(ptx_payload)
        elif ptx_payload is not None:
            state["cuda_ptx"] = str(ptx_payload).encode("utf-8")
        else:
            raise ValueError("cuda emit stage requires prebuilt PTX executable (missing cuda_ptx payload)")
        return (
            "emitted CUDA kernel via MLIR contract executable path",
            {
                "emit_backend": "mlir_contract",
                "emit_mode": "executed",
                "kernel_name": kernel_name,
                "executable_format": exe_format,
                "cuda_ptx_origin": ptx_origin,
                "cuda_ptx_bytes": len(bytes(state.get("cuda_ptx") or b"")),
                "output_count": len(output_names),
                "mlir_backend_contract_used": True,
                "contract_schema_version": str(contract.schema_version or ""),
                "execution_engine": "mlir_native",
            },
        )

    def _compile() -> tuple[str, dict[str, Any]]:
        if mode == "schedule_only":
            return ("compile skipped: schedule_only mode", {"compile_mode": "skipped_schedule_only", "pipeline_mode": mode})
        if not can_execute:
            return ("compile skipped: missing bindings", {"compile_mode": "skipped_missing_bindings"})
        if "kernel_name" not in state or "io_spec" not in state:
            raise ValueError("emit stage artifacts missing for compile")
        if "cuda_ptx" in state:
            state["compiled_module"] = load_cuda_ptx_module(
                kernel_name=str(state["kernel_name"]),
                ptx=bytes(state["cuda_ptx"]),
                io_spec=dict(state["io_spec"]),
            )
            return (
                "loaded CUDA PTX module",
                {
                    "compile_mode": "executed_ptx",
                    "kernel_name": str(state["kernel_name"]),
                },
            )
        raise ValueError("compile stage requires PTX payload from emit stage")

    def _launch() -> tuple[str, dict[str, Any]]:
        if mode == "schedule_only":
            return ("launch skipped: schedule_only mode", {"launch_mode": "skipped_schedule_only", "pipeline_mode": mode})
        if not can_execute:
            return ("launch skipped: missing bindings", {"launch_mode": "skipped_missing_bindings"})
        if "kernel_name" not in state or "io_spec" not in state or "launch" not in state:
            raise ValueError("compile/emit artifacts missing for launch")
        output_names = list(state.get("output_names") or [])
        inputs_np = _build_dummy_inputs(io_spec=state["io_spec"], output_names=output_names, bindings=bindings)
        if "cuda_ptx" in state:
            _ = run_cuda_kernel_ptx(
                kernel_name=str(state["kernel_name"]),
                ptx=bytes(state["cuda_ptx"]),
                io_spec=dict(state["io_spec"]),
                launch=state["launch"],
                bindings=dict(state.get("bindings") or bindings),
                inputs_np=inputs_np,
                output_names=output_names,
                compiled_module=state.get("compiled_module"),
            )
        else:
            raise ValueError("launch stage requires PTX payload from emit stage")
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
        input_ir_kind=str(input_meta.source_kind),
        mlir_parse_ms=float(input_meta.mlir_parse_ms),
        mlir_backend_contract_used=bool(input_meta.mlir_backend_contract_used),
    )
