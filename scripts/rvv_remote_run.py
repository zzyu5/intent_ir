"""
Prototype: end-to-end RVV remote run for supported kernels.

Current support:
- strict MLIR backend contract execution via:
  - prebuilt RVV ELF
  - remote LLVM compile on RVV target

Usage:
  python scripts/rvv_remote_run.py --kernel any_kernel_dim --use-key
  INTENTIR_RVV_HOST=192.168.8.72 INTENTIR_RVV_USER=ubuntu python scripts/rvv_remote_run.py --kernel any_kernel_dim --use-key
  # or omit INTENTIR_SSH_PASSWORD and type it when prompted
Requires: `artifacts/<frontend>_full_pipeline/<kernel>.json` produced beforehand.
Artifact dirs:
  - triton (flaggems): artifacts/flaggems_triton_full_pipeline/ (active)
  - other frontends: legacy/archived (not part of current workflow)
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import re
import shlex
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable

import paramiko
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.spmd_rvv.analysis.device_query import load_profile, query_remote_device
from backends.spmd_rvv.analysis.tuning import (
    ScheduleCandidate,
    TuningRequest,
    parse_constraints,
    parse_locks,
    propose_schedule_candidates_from_intent_json,
    select_schedule_from_intent_json,
)
from backends.spmd_rvv.baseline_freeze_tile import freeze_tile_schedule_from_intent_json
from backends.common.mlir_contract import MlirBackendContract
from intent_ir.ir import ScheduleSketch
from intent_ir.macros import expand_macros_json
from intent_ir.mlir.convert_to_intent import to_intent
from verify.gen_cases import TestCase

DEFAULT_RVV_HOST = os.getenv("INTENTIR_RVV_HOST", "192.168.8.72")
DEFAULT_RVV_USER = os.getenv("INTENTIR_RVV_USER", "ubuntu")


def _normalize_io_name(name: str) -> str:
    s = str(name).strip().strip("_").lower()
    if s.startswith("ptr_"):
        s = s[4:]
    if s.endswith("_ptr"):
        s = s[:-4]
    if s.endswith("ptr") and len(s) > 3:
        s = s[:-3]
    # Single-letter IO names from some kernels/LLM outputs.
    if s == "i":
        s = "input"
    if s == "a":
        s = "input"
    if s == "o":
        s = "output"
    if s == "x":
        s = "input"
    if s == "y":
        s = "output"
    if s == "w":
        s = "weight"
    if s == "b":
        s = "bias"
    if s == "q":
        s = "query"
    if s == "k":
        s = "key"
    if s == "v":
        s = "value"
    if s == "s":
        s = "scale"
    if s == "sm_scale":
        s = "scale"
    if s == "smscale":
        s = "scale"
    if s == "in":
        s = "input"
    if s == "out":
        s = "output"
    # Common quantization naming aliases:
    # - y_q / yq / quantized -> primary output tensor
    # - y_s / ys / scales -> scale tensor
    if s in {"y_q", "yq", "quantized"}:
        s = "output"
    if s in {"y_s", "ys", "scales"}:
        s = "scale"
    s = s.replace("_", "")
    if s in {"yq", "quantized"}:
        s = "output"
    if s in {"ys", "scales"}:
        s = "scale"
    return s


def _with_io_aliases(wanted_names: set[str] | list[str], io: dict) -> dict:
    out = dict(io)
    norm_to_keys: dict[str, list[str]] = {}
    for k in io.keys():
        norm_to_keys.setdefault(_normalize_io_name(k), []).append(k)
    wanted = {str(x) for x in list(wanted_names or []) if str(x).strip()}
    for name in wanted:
        if name in out:
            continue
        keys = norm_to_keys.get(_normalize_io_name(name)) or []
        if keys:
            if len(keys) == 1:
                out[name] = io[keys[0]]
                continue
            # If collisions exist (e.g., both "Input" and "input"), pick a stable best match.
            lower_name = str(name).lower()
            norm = _normalize_io_name(name)
            preferred = None
            for k in keys:
                if str(k).lower() == lower_name:
                    preferred = k
                    break
            if preferred is None and norm in io:
                preferred = norm
            if preferred is None:
                preferred = keys[0]
            out[name] = io[preferred]
            continue
        norm = _normalize_io_name(name)
        # Avoid overly-short names (e.g., "N") accidentally matching "input".
        if len(norm) >= 3:
            candidates = [k for k in io.keys() if norm and (norm in _normalize_io_name(k))]
            if len(candidates) == 1:
                out[name] = io[candidates[0]]
    return out


def _np_dtype(dt: str) -> "np.dtype":
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


def _to_raw_bytes(arr: "np.ndarray", dt: str) -> bytes:
    arr_np = np.asarray(arr)
    dt = str(dt)
    # Match backend smoke path: declared tensor dtype is the source of truth.
    # Baseline arrays may be bool views even when intent tensors declare f32/i32.
    if dt in {"bool", "i1"}:
        return np.asarray(arr_np, dtype=np.uint8).tobytes(order="C")
    if dt == "i8":
        return np.asarray(arr_np, dtype=np.int8).tobytes(order="C")
    if dt == "i16":
        return np.asarray(arr_np, dtype=np.int16).tobytes(order="C")
    if dt == "u8":
        return np.asarray(arr_np, dtype=np.uint8).tobytes(order="C")
    if dt == "i32":
        return np.asarray(arr_np, dtype=np.int32).tobytes(order="C")
    if dt == "i64":
        return np.asarray(arr_np, dtype=np.int64).tobytes(order="C")
    if dt == "f16":
        return np.asarray(arr_np, dtype=np.float16).tobytes(order="C")
    if dt == "f64":
        return np.asarray(arr_np, dtype=np.float64).tobytes(order="C")
    return np.asarray(arr_np, dtype=np.float32).tobytes(order="C")


def _derive_scalar_input_array(name: str, *, dtype: str, bindings: dict) -> "np.ndarray" | None:
    """
    Derive a semantic scalar input (rank-0 tensor) that may be absent from the
    baseline runner output (because it was folded into constants).

    This keeps remote RVV correctness checks usable when IntentIR models such
    values explicitly (for verification / macro lowering).
    """
    value = None
    if name == "sm_scale":
        hd = bindings.get("HEAD_DIM")
        if hd is not None and int(hd) > 0:
            value = 1.0 / math.sqrt(float(hd))
    elif name in bindings:
        value = bindings.get(name)
    if value is None:
        return None
    return np.array(value, dtype=_np_dtype(dtype))


def _resolve_tensor_shape(shape_spec: list[Any], bindings: dict) -> tuple[int, ...] | None:
    shape = []
    for d in list(shape_spec or []):
        if isinstance(d, dict):
            kind = str(d.get("kind") or "").strip().lower()
            if kind == "sym":
                key = str(d.get("value") or "")
                if key not in bindings:
                    return None
                try:
                    shape.append(int(bindings[key]))
                except Exception:
                    return None
                continue
            if kind == "const":
                try:
                    shape.append(int(d.get("value")))
                except Exception:
                    return None
                continue
        if hasattr(d, "kind") and getattr(d, "kind") == "sym":
            key = str(getattr(d, "value"))
            if key not in bindings:
                return None
            try:
                shape.append(int(bindings[key]))
            except Exception:
                return None
            continue
        if hasattr(d, "kind") and getattr(d, "kind") == "const":
            try:
                shape.append(int(getattr(d, "value")))
            except Exception:
                return None
            continue
        if isinstance(d, int):
            shape.append(int(d))
            continue
        key = str(d)
        if key in bindings:
            try:
                shape.append(int(bindings[key]))
                continue
            except Exception:
                return None
        try:
            shape.append(int(key))
        except Exception:
            return None
    return tuple(shape)


def _tensor_shape_spec(tensor_spec: dict[str, Any]) -> list[Any]:
    shape = tensor_spec.get("shape")
    return list(shape) if isinstance(shape, list) else []


def _tensor_dtype(tensor_spec: dict[str, Any]) -> str:
    return str(tensor_spec.get("dtype") or "f32")


def _rvv_staging_dtype(dtype: str, *, execution_mode: str) -> str:
    """
    Normalize host-side staging dtype for strict RVV executable paths.

    Current RVV C codegen lowers f16/bf16 tensor buffers through f32 host buffers.
    For prebuilt/remote LLVM strict modes, match staging dtype to the lowered
    runtime expectation to avoid byte-size mismatches on remote load/compare.
    """
    dt = str(dtype or "f32").strip().lower()
    if str(execution_mode).strip().lower() in {"remote_llvm", "prebuilt_elf"}:
        if dt in {"f16", "bf16"}:
            return "f32"
    return dt or "f32"


def _tensor_specs_from_io_spec(io_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tensors = io_spec.get("tensors")
    if not isinstance(tensors, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in tensors.items():
        if isinstance(v, dict):
            out[str(k)] = dict(v)
    return out


def _tensor_specs_from_intent_json(intent_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tensors = intent_json.get("tensors")
    if not isinstance(tensors, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in tensors.items():
        if isinstance(v, dict):
            out[str(k)] = dict(v)
    return out


def _intent_outputs(intent_json: dict[str, Any]) -> list[str]:
    return [str(x) for x in list(intent_json.get("outputs") or [])]


def _intent_ops(intent_json: dict[str, Any]) -> list[dict[str, Any]]:
    ops = intent_json.get("ops")
    if not isinstance(ops, list):
        return []
    out: list[dict[str, Any]] = []
    for op in ops:
        if isinstance(op, dict):
            out.append(dict(op))
    return out


def _schedule_to_json(schedule: ScheduleSketch | dict | None) -> dict[str, Any]:
    if isinstance(schedule, ScheduleSketch):
        return {
            "tile_m": schedule.tile_m,
            "tile_n": schedule.tile_n,
            "tile_k": schedule.tile_k,
            "vec_width": schedule.vec_width,
            "pipeline_depth": schedule.pipeline_depth,
            "axis_bindings": dict(schedule.axis_bindings or {}),
            "vec_axis": schedule.vec_axis,
            "parallel_axes": list(schedule.parallel_axes or []),
            "memory_hint": dict(schedule.memory_hint or {}),
        }
    if isinstance(schedule, dict):
        return {
            "tile_m": schedule.get("tile_m"),
            "tile_n": schedule.get("tile_n"),
            "tile_k": schedule.get("tile_k"),
            "vec_width": schedule.get("vec_width"),
            "pipeline_depth": schedule.get("pipeline_depth"),
            "axis_bindings": dict(schedule.get("axis_bindings") or {}),
            "vec_axis": schedule.get("vec_axis"),
            "parallel_axes": [str(x) for x in (schedule.get("parallel_axes") or [])],
            "memory_hint": dict(schedule.get("memory_hint") or {}),
        }
    return {}


def _schedule_from_json(schedule: dict[str, Any] | None) -> ScheduleSketch:
    obj = dict(schedule or {})
    return ScheduleSketch(
        tile_m=obj.get("tile_m"),
        tile_n=obj.get("tile_n"),
        tile_k=obj.get("tile_k"),
        vec_width=obj.get("vec_width"),
        pipeline_depth=obj.get("pipeline_depth"),
        axis_bindings=dict(obj.get("axis_bindings") or {}),
        vec_axis=(obj.get("vec_axis") if isinstance(obj.get("vec_axis"), str) else None),
        parallel_axes=[str(x) for x in (obj.get("parallel_axes") or [])],
        memory_hint=dict(obj.get("memory_hint") or {}),
    )


def _augment_bindings_from_arrays(
    *,
    tensor_specs: dict[str, dict[str, Any]],
    bindings: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    out = dict(bindings)
    for name, arr in arrays.items():
        tensor_spec = tensor_specs.get(str(name))
        if not isinstance(tensor_spec, dict):
            continue
        arr_np = np.asarray(arr)
        spec_shape = _tensor_shape_spec(tensor_spec)
        if len(spec_shape) == 0 and arr_np.size == 1:
            # Scalar tensors often carry symbolic shape params (e.g., group_size).
            # Bind by tensor name so codegen can resolve reshape expressions.
            key = str(name)
            if key and key not in out:
                try:
                    out[key] = int(arr_np.reshape(()).item())
                except Exception:
                    pass
        arr_shape = tuple(int(v) for v in np.asarray(arr).shape)
        if len(spec_shape) != len(arr_shape):
            continue
        for dim_spec, dim_val in zip(spec_shape, arr_shape):
            key: str | None = None
            if isinstance(dim_spec, dict):
                kind = str(dim_spec.get("kind") or "").strip().lower()
                if kind == "sym":
                    key = str(dim_spec.get("value") or "")
            elif hasattr(dim_spec, "kind") and getattr(dim_spec, "kind") == "sym":
                key = str(getattr(dim_spec, "value"))
            elif isinstance(dim_spec, str):
                try:
                    int(dim_spec)
                except Exception:
                    key = str(dim_spec)
            if key and key not in out:
                out[key] = int(dim_val)
    return out


def _derive_optional_tensor_input_array(name: str, *, tensor_spec: dict[str, Any], bindings: dict) -> "np.ndarray" | None:
    # Attention kernels may omit mask from baseline exports when no mask is supplied.
    if str(name) != "attn_mask":
        return None
    shape = _resolve_tensor_shape(_tensor_shape_spec(tensor_spec), bindings)
    if shape is None:
        return None
    return np.zeros(shape, dtype=_np_dtype(_tensor_dtype(tensor_spec)))


def _sftp_mkdir_p(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = [p for p in path.split("/") if p]
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def _sftp_write_bytes(sftp: paramiko.SFTPClient, path: str, data: bytes) -> None:
    with sftp.file(path, "wb") as f:
        f.write(data)


def _artifact_dir_for_frontend(frontend: str, *, triton_provider: str = "native") -> str:
    if frontend == "triton":
        p = str(triton_provider)
        if p == "flaggems":
            return "flaggems_triton_full_pipeline"
        if p == "native":
            return "full_pipeline_verify"
        raise ValueError(f"unsupported triton provider: {triton_provider}")
    if frontend == "tilelang":
        return "tilelang_full_pipeline"
    if frontend == "cuda":
        return "cuda_full_pipeline"
    raise ValueError(f"unsupported frontend: {frontend}")


def _resolve_report_path(raw: object, *, artifact_root: Path) -> Path | None:
    p = Path(str(raw or "").strip())
    if not str(p):
        return None
    if p.is_absolute():
        return p if p.is_file() else None
    candidates = [p, (ROOT / p), (artifact_root / p)]
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        if resolved.is_file():
            return resolved
    return None


def _mlir_report_paths(report: dict, *, artifact_root: Path) -> list[Path]:
    mlir = report.get("mlir")
    if not isinstance(mlir, dict):
        return []
    out: list[Path] = []
    preferred: list[str] = []
    for key in sorted(mlir.keys()):
        if key.startswith("downstream_") and key.endswith("_module_path"):
            preferred.append(str(key))
    preferred += ["downstream_module_path", "midend_module_path", "module_path"]
    seen: set[str] = set()
    for key in preferred:
        if key in seen:
            continue
        seen.add(key)
        p = _resolve_report_path(mlir.get(key), artifact_root=artifact_root)
        if p is not None and p not in out:
            out.append(p)
    return out


def _contract_report_paths(report: dict, *, artifact_root: Path) -> list[Path]:
    mlir = report.get("mlir")
    if not isinstance(mlir, dict):
        return []
    preferred = [
        "downstream_rvv_llvm_contract_path",
        "downstream_rvv_contract_path",
        "downstream_cuda_llvm_contract_path",
        "downstream_cuda_contract_path",
        "downstream_llvm_contract_path",
        "downstream_contract_path",
        "midend_rvv_contract_path",
        "midend_cuda_contract_path",
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for key in preferred:
        if key in seen:
            continue
        seen.add(key)
        p = _resolve_report_path(mlir.get(key), artifact_root=artifact_root)
        if p is not None and p not in out:
            out.append(p)
    return out


def _resolve_contract_module_path(contract: MlirBackendContract, *, contract_path: Path) -> Path | None:
    artifacts = dict(contract.artifacts or {})
    module_path_raw = str(artifacts.get("mlir_module_path") or "").strip()
    if not module_path_raw and str(contract.executable.format or "").endswith("mlir_module"):
        module_path_raw = str(contract.executable.path or "").strip()
    if not module_path_raw:
        return None
    p = Path(module_path_raw)
    if p.is_absolute():
        return p if p.is_file() else None
    for cand in (contract_path.parent / p, ROOT / p):
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        if resolved.is_file():
            return resolved
    return None


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(str(name), "")
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "y", "on"}


def _resolve_contract_executable_path(contract: MlirBackendContract, *, contract_path: Path) -> Path | None:
    exe_path_raw = str((contract.executable.path or "")).strip()
    if not exe_path_raw:
        return None
    p = Path(exe_path_raw)
    if p.is_absolute():
        return p if p.is_file() else None
    for cand in (contract_path.parent / p, ROOT / p):
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        if resolved.is_file():
            return resolved
    return None


def _elf_machine(path: Path) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return "unknown"
    if len(data) < 20 or data[:4] != b"\x7fELF":
        return "unknown"
    ei_data = int(data[5]) if len(data) > 5 else 1
    endian = "<" if ei_data == 1 else ">"
    try:
        e_machine = int(struct.unpack_from(f"{endian}H", data, 18)[0])
    except Exception:
        return "unknown"
    if e_machine == 243:
        return "riscv"
    if e_machine == 62:
        return "x86_64"
    if e_machine == 183:
        return "aarch64"
    return f"machine_{e_machine}"


def _llvm_target_triple(llvm_ir_text: str) -> str:
    m = re.search(r'target\s+triple\s*=\s*"([^"]+)"', str(llvm_ir_text or ""))
    return str(m.group(1)).strip() if m is not None else ""


def _is_rvv_llvm_triple(triple: str) -> bool:
    t = str(triple or "").strip().lower()
    if not t:
        return False
    return ("riscv" in t) and ("linux" in t or "unknown" in t or "elf" in t)


def _resolve_rvv_execution_plan(
    *,
    contract_payload_json: dict[str, Any] | None,
    contract_artifact_path: str,
) -> dict[str, Any]:
    if not isinstance(contract_payload_json, dict):
        raise RuntimeError(
            "rvv hard-cut mode requires mlir backend contract; compatibility C-source path has been removed"
        )

    contract_path = Path(str(contract_artifact_path or ""))
    try:
        contract = MlirBackendContract.from_json_dict(dict(contract_payload_json))
    except Exception as e:
        raise RuntimeError(
            f"rvv hard-cut mode requires valid mlir backend contract, parse failed: {type(e).__name__}: {e}"
        ) from e

    exe = contract.executable
    exe_format = str(exe.format or "").strip().lower()
    exe_path = _resolve_contract_executable_path(contract, contract_path=contract_path)
    elf_machine = _elf_machine(exe_path) if isinstance(exe_path, Path) else "unknown"

    module_path = _resolve_contract_module_path(contract, contract_path=contract_path)
    llvm_ir_text = str((contract.artifacts or {}).get("mlir_module_text") or "")
    if not llvm_ir_text and isinstance(module_path, Path) and module_path.is_file():
        try:
            llvm_ir_text = module_path.read_text(encoding="utf-8")
        except Exception:
            llvm_ir_text = ""
    llvm_triple = _llvm_target_triple(llvm_ir_text)
    has_rvv_llvm = bool(llvm_ir_text) and _is_rvv_llvm_triple(llvm_triple)

    if exe_format in {"rvv_elf", "elf"} and isinstance(exe_path, Path) and exe_path.is_file() and elf_machine == "riscv":
        return {
            "mode": "prebuilt_elf",
            "reason": "contract_executable_rvv_elf",
            "local_elf_path": str(exe_path),
            "elf_machine": str(elf_machine),
            "llvm_triple": str(llvm_triple),
            "module_path": str(module_path) if isinstance(module_path, Path) else "",
            "llvm_ir_text": str(llvm_ir_text),
        }
    if has_rvv_llvm:
        return {
            "mode": "remote_llvm",
            "reason": "contract_downstream_rvv_llvm",
            "local_elf_path": str(exe_path) if isinstance(exe_path, Path) else "",
            "elf_machine": str(elf_machine),
            "llvm_triple": str(llvm_triple),
            "module_path": str(module_path) if isinstance(module_path, Path) else "",
            "llvm_ir_text": str(llvm_ir_text),
        }

    raise RuntimeError(
        "rvv hard-cut mode requires either riscv prebuilt ELF or RVV-target LLVM IR; "
        f"got executable.format={exe_format or '<empty>'}, elf_machine={elf_machine}, llvm_triple={llvm_triple or '<missing>'}"
    )


def _execution_mode_from_plan(execution_plan: dict[str, Any]) -> str:
    mode = str((execution_plan or {}).get("mode") or "").strip()
    if not mode:
        raise RuntimeError("rvv execution plan missing mode")
    return mode


def _synthesize_intent_from_contract(contract: MlirBackendContract) -> dict[str, Any]:
    io_spec = _runtime_io_spec_from_contract(contract)
    tensors_in = io_spec.get("tensors")
    tensors: dict[str, dict[str, Any]] = {}
    if isinstance(tensors_in, dict):
        for name, spec in tensors_in.items():
            if not isinstance(spec, dict):
                continue
            tensors[str(name)] = {
                "dtype": str(spec.get("dtype") or "f32"),
                "shape": list(spec.get("shape") or []),
                "layout": str(spec.get("layout") or "row_major"),
            }
    outputs = [str(x) for x in list(io_spec.get("outputs") or []) if str(x).strip()]
    schedule = dict(contract.schedule or {})
    parallel_axes = list(schedule.get("parallel_axes") or []) if isinstance(schedule, dict) else []
    return {
        "name": str(contract.kernel_name or "intent"),
        "tensors": tensors,
        "ops": [],
        "outputs": outputs,
        "schedule": dict(schedule),
        "parallel_axes": [str(x) for x in parallel_axes if str(x).strip()],
        "meta": {"source_kind": "contract_io_spec_synth"},
    }


def _runtime_io_spec_from_contract(contract: MlirBackendContract) -> dict[str, Any]:
    io_spec = dict(contract.io_spec or {})
    invocation = dict(contract.executable.invocation or {})
    inv_io = invocation.get("io_spec")
    if isinstance(inv_io, dict) and isinstance(inv_io.get("tensors"), dict):
        io_spec = dict(inv_io)
    if not isinstance(io_spec.get("outputs"), list):
        inv_out = invocation.get("output_names")
        if isinstance(inv_out, list):
            io_spec = dict(io_spec)
            io_spec["outputs"] = [str(x) for x in inv_out if str(x).strip()]
    return io_spec


def _intent_from_contract_path(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]] | None:
    try:
        payload_raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload_raw, dict):
        return None
    try:
        payload = MlirBackendContract.from_json_dict(payload_raw)
    except Exception:
        return None
    io_spec = _runtime_io_spec_from_contract(payload)
    payload_json = payload.to_json_dict()
    module_text = str((payload.artifacts or {}).get("mlir_module_text") or "").strip()
    if module_text:
        try:
            parsed = to_intent(module_text)
            return parsed.to_json_dict(), io_spec, payload_json
        except Exception:
            pass
    module_path = _resolve_contract_module_path(payload, contract_path=path)
    if module_path is not None:
        try:
            parsed = to_intent(module_path.read_text(encoding="utf-8"))
            return parsed.to_json_dict(), io_spec, payload_json
        except Exception:
            pass
    return _synthesize_intent_from_contract(payload), io_spec, payload_json


def run_remote(
    kernel: str,
    frontend: str,
    host: str,
    user: str,
    password: str | None,
    port: int = 22,
    case_index: int = 0,
    shape_overrides: dict | None = None,
    baseline_npz: str | None = None,
    prefer_live_baseline: bool = False,
    tune_request: TuningRequest | None = None,
    schedule_override: ScheduleSketch | dict | None = None,
    tune_profile: str | None = None,
    bench_iters: int = 0,
    bench_warmup: int = 1,
    bench_seed: int = 0,
    profile_ops: bool = False,
    bench_only: bool = False,
    omp_threads: int = 1,
    omp_proc_bind: str = "auto",
    omp_places: str = "cores",
    gomp_cpu_affinity: str | None = None,
    log: Callable[[str], None] | None = None,
    triton_provider: str = "native",
    artifact_dir: str | None = None,
    require_mlir_artifacts: bool = False,
):
    def _log(msg: str) -> None:
        if log is None:
            return
        try:
            log(str(msg))
        except Exception:
            pass

    artifact_rel = _artifact_dir_for_frontend(frontend, triton_provider=str(triton_provider))
    artifact_root = (Path(artifact_dir) if artifact_dir else (ROOT / "artifacts" / artifact_rel)).resolve()
    report_path = artifact_root / f"{kernel}.json"
    if not report_path.exists():
        if frontend == "triton" and str(triton_provider) == "flaggems":
            cmd_hint = f"python scripts/triton/flaggems_full_pipeline_verify.py --kernel {kernel}"
        else:
            cmd_hint = "python scripts/triton/flaggems_full_pipeline_verify.py --suite coverage"
        raise FileNotFoundError(
            f"artifact not found: {report_path}, please run `{cmd_hint}` first"
        )
    _log(f"[{frontend}:{kernel}] load artifact: {report_path}")
    report = json.loads(report_path.read_text())
    contract_artifact_path = ""
    contract_payload_json: dict[str, Any] | None = None
    mlir_artifact_path = ""
    intent_macro_json: dict[str, Any] | None = None
    intent_json: dict[str, Any] | None = None
    io_spec: dict[str, Any] = {}
    for contract_path in _contract_report_paths(report, artifact_root=artifact_root):
        parsed = _intent_from_contract_path(contract_path)
        if parsed is None:
            continue
        parsed_intent, io_spec, contract_payload_json = parsed
        if isinstance(parsed_intent, dict):
            intent_json = dict(parsed_intent)
            intent_macro_json = dict(parsed_intent)
        contract_artifact_path = str(contract_path)
        _log(f"[{frontend}:{kernel}] contract artifact selected: {contract_artifact_path}")
        break
    for mlir_path in _mlir_report_paths(report, artifact_root=artifact_root):
        if intent_json is not None:
            break
        try:
            parsed = to_intent(mlir_path.read_text(encoding="utf-8"))
            intent_json = parsed.to_json_dict()
            intent_macro_json = dict(intent_json)
            mlir_artifact_path = str(mlir_path)
            _log(f"[{frontend}:{kernel}] mlir artifact selected: {mlir_artifact_path}")
            break
        except Exception:
            continue
    if intent_json is None:
        if bool(require_mlir_artifacts):
            raise RuntimeError(
                "mlir_artifact_missing: no readable MLIR contract/module path in report: "
                f"{report_path}"
            )
        raw_macro_json = report.get("intent")
        if not isinstance(raw_macro_json, dict):
            raise ValueError(f"invalid artifact intent payload: {report_path}")
        intent_macro_json = dict(raw_macro_json)
        intent_expanded_json = report.get("intent_expanded")
        if not isinstance(intent_expanded_json, dict):
            intent_expanded_json = expand_macros_json(dict(raw_macro_json))
        intent_json = dict(intent_expanded_json)
    assert intent_json is not None
    assert intent_macro_json is not None
    tensor_specs = _tensor_specs_from_io_spec(io_spec) or _tensor_specs_from_intent_json(intent_json)
    if not tensor_specs:
        raise RuntimeError(f"invalid artifact intent payload: missing tensors in {report_path}")
    intent_json_base = dict(intent_json)
    intent_macro_json_base = dict(intent_macro_json)
    if schedule_override is not None:
        intent_json_base["schedule"] = _schedule_to_json(schedule_override)

    # Frontend-derived "tile-ish" constants (schedule hints). Triton CertificateV2
    # extracts these from TTIR; TileLang may leave empty (OK).
    cert_v2 = report.get("certificate_v2") or {}
    tile_hints: list[int] = []
    try:
        sh = cert_v2.get("schedule_hints") or {}
        th = sh.get("tile_hints")
        if isinstance(th, list):
            for x in th:
                try:
                    v = int(x)
                except Exception:
                    continue
                if v > 0:
                    tile_hints.append(v)
    except Exception:
        tile_hints = []
    if not tile_hints:
        try:
            # Legacy v1 certificate format (fallback).
            cert1 = report.get("certificate") or {}
            th = cert1.get("tile_hints")
            if isinstance(th, list):
                tile_hints = [int(x) for x in th if isinstance(x, (int, float, str)) and int(x) > 0]
        except Exception:
            tile_hints = []
    tile_hints = sorted(set(tile_hints))
    # Select shapes from cases (for binding).
    cases_raw = report.get("cases") or []
    cases: list[dict] = []
    if isinstance(cases_raw, dict):
        # v1.2 format: {"in_contract":[...], "out_of_contract":[...]}
        in_contract = cases_raw.get("in_contract")
        if isinstance(in_contract, list):
            cases = [c for c in in_contract if isinstance(c, dict)]
    elif isinstance(cases_raw, list):
        # legacy: list[dict]
        cases = [c for c in cases_raw if isinstance(c, dict)]
    case_idx = min(max(int(case_index), 0), len(cases) - 1) if cases else 0
    bindings = dict(cases[case_idx]) if cases else {}
    if shape_overrides:
        bindings.update(shape_overrides)
    # Common axis aliases (align kernel-signature symbols with user-friendly names).
    if "batch" in bindings and "Z" not in bindings:
        bindings["Z"] = bindings["batch"]
    if "Z" in bindings and "batch" not in bindings:
        bindings["batch"] = bindings["Z"]
    if "group" in bindings and "num_groups" not in bindings:
        bindings["num_groups"] = bindings["group"]
    if "num_groups" in bindings and "C" in bindings and "group_size" not in bindings:
        try:
            g = int(bindings["num_groups"])
            c = int(bindings["C"])
            if g > 0:
                bindings["group_size"] = c // g if (c % g == 0) else (c + g - 1) // g
        except Exception:
            pass
    if "group_size" in bindings and "HW" in bindings and "num_elements" not in bindings:
        try:
            bindings["num_elements"] = int(bindings["group_size"]) * int(bindings["HW"])
        except Exception:
            pass

    baseline = None
    if not bool(bench_only):
        npz_path = baseline_npz or (report.get("baseline") or {}).get("npz_path")
        if (not prefer_live_baseline) and npz_path:
            _log(f"[{frontend}:{kernel}] load baseline npz: {npz_path}")
            npz_path = str(npz_path)
            baseline_npz_path = Path(npz_path)
            if not baseline_npz_path.is_absolute():
                baseline_npz_path = (ROOT / baseline_npz_path).resolve()
            if not baseline_npz_path.exists():
                raise FileNotFoundError(f"baseline npz not found: {baseline_npz_path}")
            baseline = dict(np.load(baseline_npz_path, allow_pickle=False))

        # Fallback: if baseline is missing (or user forces live), try to re-launch Triton
        # to get baseline IO. This keeps Task6 usable on machines with CUDA.
        if baseline is None:
            _log(f"[{frontend}:{kernel}] baseline npz missing; try live baseline launch")
            try:
                if frontend == "triton":
                    if str(triton_provider) == "flaggems":
                        from pipeline.triton.providers.flaggems.specs import default_flaggems_kernel_specs

                        spec_map = {s.name: s for s in default_flaggems_kernel_specs()}
                    else:
                        from pipeline.triton.core import default_kernel_specs

                        spec_map = {s.name: s for s in default_kernel_specs()}
                elif frontend == "tilelang":
                    from pipeline.tilelang.core import default_kernel_specs, mvp_kernel_specs

                    spec_map = {s.name: s for s in (mvp_kernel_specs() + default_kernel_specs())}
                else:
                    from pipeline.cuda.core import coverage_kernel_specs

                    spec_map = {s.name: s for s in coverage_kernel_specs()}
                if kernel not in spec_map:
                    raise RuntimeError(f"unknown kernel {kernel}")
                spec = spec_map[kernel]
                if not bindings:
                    bindings = dict(spec.canonical_shapes)
                baseline = spec.runner(TestCase(shapes=bindings, dtypes={}, seed=0))
            except Exception as e:
                triton_cmd = "python scripts/triton/flaggems_full_pipeline_verify.py --suite coverage"
                raise RuntimeError(
                    "baseline not available: no cached baseline .npz in artifacts and live baseline launch failed. "
                    f"Run `{triton_cmd}` to produce pipeline artifacts (and baseline .npz when applicable), "
                    "or pass --baseline-npz.\n"
                    f"live launch error: {type(e).__name__}: {e}"
                ) from e

    # Prefer baseline shapes (the embedded inputs correspond to that launch).
    baseline_shapes = ((report.get("baseline") or {}).get("shapes") or {}) if isinstance(report.get("baseline"), dict) else {}
    if baseline_shapes:
        bindings = dict(baseline_shapes)
    if shape_overrides:
        bindings.update(dict(shape_overrides))

    # Common axis aliases (align kernel-signature symbols with user-friendly names).
    if "batch" in bindings and "Z" not in bindings:
        bindings["Z"] = bindings["batch"]
    if "Z" in bindings and "batch" not in bindings:
        bindings["batch"] = bindings["Z"]

    # Add naming aliases to baseline to match IntentIR tensor names.
    # Add naming aliases to baseline to match IntentIR tensor names.
    # Use contract / intent tensor names instead of class-bound lookups.
    if baseline is not None:
        baseline = _with_io_aliases(set(tensor_specs.keys()) | set(str(x) for x in list(intent_json.get("outputs") or [])), baseline)
        # Recover symbolic extents that appear only on outputs (e.g. nonzero count)
        # so backend lowering can resolve all output dimensions.
        baseline = dict(baseline)
        bindings = _augment_bindings_from_arrays(tensor_specs=tensor_specs, bindings=bindings, arrays=baseline)

    # Derive a few common implicit symbols.
    if "group" in bindings and "num_groups" not in bindings:
        bindings["num_groups"] = bindings["group"]
    if "num_groups" in bindings and "C" in bindings and "group_size" not in bindings:
        g = int(bindings["num_groups"])
        c = int(bindings["C"])
        if g > 0 and c % g == 0:
            bindings["group_size"] = c // g
    if "group_size" in bindings and "HW" in bindings and "num_elements" not in bindings:
        try:
            bindings["num_elements"] = int(bindings["group_size"]) * int(bindings["HW"])
        except Exception:
            pass
    # Generic grouped reductions: bind G = N / group_size when present.
    if "N" in bindings and "group_size" in bindings and "G" not in bindings:
        try:
            n = int(bindings["N"])
            gs = int(bindings["group_size"])
            if gs > 0 and n % gs == 0:
                bindings["G"] = n // gs
        except Exception:
            pass

    # Common derived symbols used by some frontends/LLM outputs (avoid "unbound symbol" failures).
    if "HEAD_DIM" in bindings:
        try:
            hd = int(bindings["HEAD_DIM"])
            if hd > 0:
                bindings.setdefault("HEAD_DIM_DIV2", hd // 2)
                bindings.setdefault("HEAD_DIM_DIV_2", hd // 2)
                bindings.setdefault("HEAD_DIM_HALF", hd // 2)
                bindings.setdefault("HEAD_DIM_MID", hd // 2)
        except Exception:
            pass

    execution_plan = _resolve_rvv_execution_plan(
        contract_payload_json=(dict(contract_payload_json) if isinstance(contract_payload_json, dict) else None),
        contract_artifact_path=str(contract_artifact_path),
    )
    execution_mode = _execution_mode_from_plan(execution_plan)
    _log(
        f"[{frontend}:{kernel}] execution plan: mode={execution_mode} "
        f"reason={execution_plan.get('reason') or ''}"
    )

    tuning_info: dict | None = None
    tune_candidates: list[ScheduleCandidate] | None = None
    selected_schedule = _schedule_from_json(
        intent_json_base.get("schedule") if isinstance(intent_json_base.get("schedule"), dict) else {}
    )
    can_tune = False
    if tune_request is not None:
        if not can_tune:
            reason = f"fixed_executable_mode:{execution_mode}"
            _log(f"[{frontend}:{kernel}] schedule selection skipped: {reason}")
            tuning_info = {
                "mode": str(tune_request.mode),
                "budget": int(getattr(tune_request, "budget", 0) or 0),
                "skipped": True,
                "reason": str(reason),
                "schedule": _schedule_to_json(selected_schedule),
            }
        else:
            _log(f"[{frontend}:{kernel}] schedule selection: mode={tune_request.mode} budget={getattr(tune_request,'budget',0)}")
            if tune_profile:
                prof = load_profile(str(tune_profile))
                prof_src = str(tune_profile)
            else:
                _log(f"[{frontend}:{kernel}] query remote RVV profile (probe)")
                prof = query_remote_device(host, user=user, password=password, port=port, timeout=20)
                prof_src = "remote"

            shape_bindings_int: dict[str, int] = {}
            for k, v in dict(bindings).items():
                try:
                    shape_bindings_int[str(k)] = int(v)
                except Exception:
                    continue

            budget = int(getattr(tune_request, "budget", 0) or 0)
            if budget > 1:
                if int(bench_iters) <= 0:
                    raise ValueError("tune-budget > 1 requires --bench-iters > 0 (measured autotune)")

                # Always include a "frozen" schedule candidate as a no-regression baseline:
                # resolve BLOCK_* style schedule symbols using frontend launch constexpr values.
                frozen_sched = None
                frozen_notes: list[str] = []
                try:
                    frozen = freeze_tile_schedule_from_intent_json(
                        intent_macro_json_base,
                        desc=(report.get("descriptor") if isinstance(report, dict) else None),
                    )
                    frozen_sched = frozen.schedule
                    frozen_notes = list(frozen.notes)
                except Exception:
                    frozen_sched = None
                    frozen_notes = []

                # Build a larger candidate pool, then pick a diverse top-K to benchmark.
                pool_limit = max(int(budget) * 4, int(budget))
                pool = propose_schedule_candidates_from_intent_json(
                    intent_json_base,
                    shape_bindings=shape_bindings_int,
                    profile=prof,
                    request=tune_request,
                    tile_hints=tile_hints,
                    limit=pool_limit,
                    evidence=cert_v2,
                )
                if not pool:
                    pool = propose_schedule_candidates_from_intent_json(
                        intent_json_base,
                        shape_bindings=shape_bindings_int,
                        profile=prof,
                        request=tune_request,
                        tile_hints=tile_hints,
                        limit=1,
                        evidence=cert_v2,
                    )

                def _sched_key(s: ScheduleSketch) -> tuple:
                    return (
                        s.tile_m,
                        s.tile_n,
                        s.tile_k,
                        s.vec_width,
                        s.pipeline_depth,
                        tuple(sorted((s.axis_bindings or {}).items())),
                        s.vec_axis,
                        tuple(s.parallel_axes or []),
                    )

                selected: list = []
                seen: set[tuple] = set()

                # Candidate 0: frozen baseline (if available), otherwise the top predicted one.
                if frozen_sched is not None:
                    freeze_cand = ScheduleCandidate(schedule=frozen_sched, score=0.0, tile_mnk=None, notes=(["freeze_baseline"] + frozen_notes))
                    k0 = _sched_key(freeze_cand.schedule)
                    selected.append(freeze_cand)
                    seen.add(k0)

                # Prefer diversity across vec_width, then tile_n, then fill by score.
                def _add_if_new(c) -> bool:
                    k = _sched_key(c.schedule)
                    if k in seen:
                        return False
                    selected.append(c)
                    seen.add(k)
                    return True

                # Pass 1: cover distinct vec_width values.
                seen_vw: set[int] = set()
                for c in pool:
                    if len(selected) >= int(budget):
                        break
                    vw = c.schedule.vec_width
                    if isinstance(vw, int) and vw > 0 and vw in seen_vw:
                        continue
                    if _add_if_new(c):
                        if isinstance(vw, int) and vw > 0:
                            seen_vw.add(int(vw))

                # Pass 2: cover distinct tile_n values.
                seen_tn: set[int] = set()
                for c in selected:
                    tn = c.schedule.tile_n
                    if isinstance(tn, int) and tn > 0:
                        seen_tn.add(int(tn))
                for c in pool:
                    if len(selected) >= int(budget):
                        break
                    tn = c.schedule.tile_n
                    if isinstance(tn, int) and tn > 0 and tn in seen_tn:
                        continue
                    if _add_if_new(c):
                        if isinstance(tn, int) and tn > 0:
                            seen_tn.add(int(tn))

                # Pass 3: fill remaining slots by predicted ranking.
                for c in pool:
                    if len(selected) >= int(budget):
                        break
                    _add_if_new(c)

                tune_candidates = selected or pool

                # Keep a deterministic default schedule in case benchmarking fails.
                if tune_candidates:
                    selected_schedule = tune_candidates[0].schedule
                tuning_info = {
                    "profile_source": prof_src,
                    "profile": prof.__dict__,
                    "mode": str(tune_request.mode),
                    "budget": int(budget),
                    "tile_hints": list(tile_hints),
                    "candidate_pool_limit": int(pool_limit),
                    "candidate_pool_size": int(len(pool)),
                    "candidates_pred": [
                        {
                            "score": float(c.score),
                            "tile_mnk": (list(c.tile_mnk) if c.tile_mnk is not None else None),
                            "notes": list(c.notes),
                            "schedule": {
                                "tile_m": c.schedule.tile_m,
                                "tile_n": c.schedule.tile_n,
                                "tile_k": c.schedule.tile_k,
                                "vec_width": c.schedule.vec_width,
                                "pipeline_depth": c.schedule.pipeline_depth,
                                "axis_bindings": dict(c.schedule.axis_bindings or {}),
                                "vec_axis": c.schedule.vec_axis,
                                "parallel_axes": list(c.schedule.parallel_axes or []),
                                "memory_hint": dict(c.schedule.memory_hint or {}),
                            },
                        }
                        for c in tune_candidates
                    ],
                }
            else:
                tuned = select_schedule_from_intent_json(
                    intent_json_base,
                    shape_bindings=shape_bindings_int,
                    profile=prof,
                    request=tune_request,
                    tile_hints=tile_hints,
                    evidence=cert_v2,
                )
                selected_schedule = tuned.schedule
                tuning_info = {
                    "profile_source": prof_src,
                    "profile": prof.__dict__,
                    "mode": str(tune_request.mode),
                    "budget": int(budget),
                    "tile_hints": list(tile_hints),
                    "notes": list(tuned.notes),
                    "schedule": _schedule_to_json(selected_schedule),
                }
                if getattr(tuned, "debug", None) is not None:
                    tuning_info["debug"] = tuned.debug

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    _log(f"[{frontend}:{kernel}] ssh connect: {user}@{host}:{port}")
    client.connect(hostname=host, port=port, username=user, password=password, timeout=20)
    sftp = client.open_sftp()
    remote_dir = f"/tmp/intentir_{kernel}_rvv"
    _sftp_mkdir_p(sftp, remote_dir)
    _log(f"[{frontend}:{kernel}] remote dir: {remote_dir}")

    # Prepare code + data according to kernel kind.
    remote_bin = f"{remote_dir}/run"

    # Upload the target-side runtime (shared helpers) once per kernel.
    runtime_dir = ROOT / "backends" / "spmd_rvv" / "runtime"
    runtime_h = runtime_dir / "intentir_runtime.h"
    runtime_c_local = runtime_dir / "intentir_runtime.c"
    driver_h = runtime_dir / "intentir_driver.h"
    driver_c_local = runtime_dir / "intentir_driver.c"
    ops_h = runtime_dir / "intentir_ops.h"
    ops_c_local = runtime_dir / "intentir_ops.c"
    if (
        not runtime_h.exists()
        or not runtime_c_local.exists()
        or not driver_h.exists()
        or not driver_c_local.exists()
        or not ops_h.exists()
        or not ops_c_local.exists()
    ):
        raise FileNotFoundError(
            f"missing RVV runtime: {runtime_h} / {runtime_c_local} / {driver_h} / {driver_c_local} / {ops_h} / {ops_c_local}"
        )
    _log(f"[{frontend}:{kernel}] upload runtime")
    with sftp.file(f"{remote_dir}/intentir_runtime.h", "w") as f:
        f.write(runtime_h.read_text(encoding="utf-8"))
    with sftp.file(f"{remote_dir}/intentir_runtime.c", "w") as f:
        f.write(runtime_c_local.read_text(encoding="utf-8"))
    with sftp.file(f"{remote_dir}/intentir_driver.h", "w") as f:
        f.write(driver_h.read_text(encoding="utf-8"))
    with sftp.file(f"{remote_dir}/intentir_driver.c", "w") as f:
        f.write(driver_c_local.read_text(encoding="utf-8"))
    with sftp.file(f"{remote_dir}/intentir_ops.h", "w") as f:
        f.write(ops_h.read_text(encoding="utf-8"))
    with sftp.file(f"{remote_dir}/intentir_ops.c", "w") as f:
        f.write(ops_c_local.read_text(encoding="utf-8"))

    # Generic lowering path: upload all external inputs and reference outputs, then lower ops list.
    intent_ops = _intent_ops(intent_json_base)
    produced: set[str] = set()
    used: set[str] = set()
    for op in intent_ops:
        out_name = str(op.get("output") or "").strip()
        if out_name:
            produced.add(out_name)
        for n in list(op.get("inputs") or []):
            name = str(n).strip()
            if name:
                used.add(name)
    if used:
        external_inputs = sorted([n for n in used if n in tensor_specs and n not in produced])
    else:
        declared_outputs = {str(n) for n in _intent_outputs(intent_json_base)}
        external_inputs = sorted([n for n in tensor_specs if n not in declared_outputs])
    intent_outputs = _intent_outputs(intent_json_base)
    has_baseline = isinstance(baseline, dict)
    if has_baseline:
        # Only verify outputs present in the baseline bundle. Some kernels
        # expose auxiliary outputs (e.g., indices) that are not exported.
        outputs = [n for n in list(intent_outputs) if n in baseline]
    else:
        outputs = [n for n in list(intent_outputs) if n in produced]
    if not outputs:
        outputs = [n for n in list(intent_outputs) if n in baseline] if has_baseline else list(intent_outputs)
    if not outputs:
        # Keep declared outputs to avoid constructing an invalid empty-output intent.
        outputs = list(intent_outputs)
    intent_codegen_json = dict(intent_json_base)
    if outputs != list(intent_outputs):
        intent_codegen_json["outputs"] = list(outputs)
    intent_codegen_json["schedule"] = _schedule_to_json(selected_schedule)
    backend_used = "mlir_contract"
    declared_dtypes = {
        str(name): _rvv_staging_dtype(_tensor_dtype(spec), execution_mode=execution_mode)
        for name, spec in dict(tensor_specs or {}).items()
        if isinstance(spec, dict)
    }

    if not bool(bench_only):
        # Upload inputs/refs for correctness runs.
        _log(f"[{frontend}:{kernel}] upload inputs/refs")
        assert baseline is not None
        for name in external_inputs:
            declared_dt = declared_dtypes.get(
                name,
                _rvv_staging_dtype(_tensor_dtype(tensor_specs.get(str(name), {})), execution_mode=execution_mode),
            )
            if name not in baseline:
                tt = tensor_specs.get(str(name))
                if isinstance(tt, dict):
                    if len(_tensor_shape_spec(tt)) == 0:
                        arr0 = _derive_scalar_input_array(str(name), dtype=_tensor_dtype(tt), bindings=bindings)
                        if arr0 is not None:
                            baseline[name] = arr0
                    else:
                        arrn = _derive_optional_tensor_input_array(str(name), tensor_spec=tt, bindings=bindings)
                        if arrn is not None:
                            baseline[name] = arrn
            if name not in baseline:
                raise RuntimeError(f"baseline missing input tensor {name} for {kernel}")
            raw = _to_raw_bytes(np.asarray(baseline[name]), str(declared_dt))
            _sftp_write_bytes(sftp, f"{remote_dir}/{name}.bin", raw)

        for name in outputs:
            if name not in baseline:
                raise RuntimeError(f"baseline missing output tensor {name} for {kernel}")
            declared_dt = declared_dtypes.get(
                name,
                _rvv_staging_dtype(_tensor_dtype(tensor_specs.get(str(name), {})), execution_mode=execution_mode),
            )
            raw = _to_raw_bytes(np.asarray(baseline[name]), str(declared_dt))
            _sftp_write_bytes(sftp, f"{remote_dir}/{name}_ref.bin", raw)

    def _compile_and_run(schedule: ScheduleSketch) -> dict:
        _log(f"[{frontend}:{kernel}] remote compile mode={execution_mode}")
        t_compile0 = time.perf_counter()
        comp_out = ""
        comp_err = ""
        compile_rc = 0
        q_remote_dir = shlex.quote(str(remote_dir))
        q_remote_bin = shlex.quote(str(remote_bin))
        q_runtime_c = shlex.quote(f"{remote_dir}/intentir_runtime.c")
        q_driver_c = shlex.quote(f"{remote_dir}/intentir_driver.c")
        q_ops_c = shlex.quote(f"{remote_dir}/intentir_ops.c")
        try:
            if execution_mode == "prebuilt_elf":
                local_elf_path = Path(str(execution_plan.get("local_elf_path") or ""))
                if not local_elf_path.is_file():
                    raise FileNotFoundError(f"prebuilt RVV ELF missing: {local_elf_path}")
                sftp.put(str(local_elf_path), str(remote_bin))
                chmod_cmd = f"chmod +x {q_remote_bin}"
                stdin, stdout, stderr = client.exec_command(chmod_cmd, timeout=20)
                comp_out = stdout.read().decode()
                comp_err = stderr.read().decode()
                compile_rc = stdout.channel.recv_exit_status()
                comp_out = (comp_out + "\n" if comp_out else "") + f"uploaded prebuilt elf: {local_elf_path}"
            elif execution_mode == "remote_llvm":
                llvm_ir_text = str(execution_plan.get("llvm_ir_text") or "")
                llvm_triple = str(execution_plan.get("llvm_triple") or "")
                if not llvm_ir_text.strip():
                    raise RuntimeError("remote_llvm mode requires non-empty LLVM IR text in contract artifacts")
                if not _is_rvv_llvm_triple(llvm_triple):
                    raise RuntimeError(
                        "remote_llvm mode requires RVV-target LLVM IR triple, "
                        f"got {llvm_triple or '<missing>'}"
                    )
                remote_ll = f"{remote_dir}/kernel.ll"
                remote_obj = f"{remote_dir}/kernel.o"
                q_remote_ll = shlex.quote(remote_ll)
                q_remote_obj = shlex.quote(remote_obj)
                q_remote_target = shlex.quote(f"--target={llvm_triple}")
                q_remote_mtriple = shlex.quote(f"-mtriple={llvm_triple}")
                with sftp.file(remote_ll, "w") as f:
                    f.write(llvm_ir_text)
                compile_cmd = (
                    "if command -v clang >/dev/null 2>&1; then "
                    f"clang -O3 -x ir {q_remote_target} -c -o {q_remote_obj} {q_remote_ll} && "
                    f"clang -O3 {q_remote_target} -fopenmp -std=c11 -D_POSIX_C_SOURCE=200809L -I{q_remote_dir} -o {q_remote_bin} {q_remote_obj} "
                    f"{q_runtime_c} {q_driver_c} {q_ops_c} -lm -lrt; "
                    "elif command -v llc >/dev/null 2>&1; then "
                    f"llc -O3 {q_remote_mtriple} -filetype=obj -o {q_remote_obj} {q_remote_ll} && "
                    f"gcc -O3 -std=c11 -D_POSIX_C_SOURCE=200809L -march=rv64gcv -fopenmp -I{q_remote_dir} -o {q_remote_bin} {q_remote_obj} "
                    f"{q_runtime_c} {q_driver_c} {q_ops_c} -lm -lrt; "
                    "else echo 'remote llvm toolchain missing: clang/llc' >&2; exit 127; fi"
                )
                stdin, stdout, stderr = client.exec_command(compile_cmd, timeout=120)
                comp_out = stdout.read().decode()
                comp_err = stderr.read().decode()
                compile_rc = stdout.channel.recv_exit_status()
            else:
                raise RuntimeError(f"unsupported rvv execution mode: {execution_mode}")
        except Exception as e:
            compile_rc = 1
            comp_err = f"{type(e).__name__}: {e}"

        compile_ms = float((time.perf_counter() - t_compile0) * 1000.0)
        if compile_rc != 0:
            return {
                "compile_rc": compile_rc,
                "compile_stdout": comp_out,
                "compile_stderr": comp_err,
                "compile_ms": compile_ms,
                "compile_mode": str(execution_mode),
                "run_rc": None,
                "stdout": "",
                "stderr": "",
                "bench": None,
                "profile_ops": None,
                "omp": None,
                "launch_ms": 0.0,
                "total_ms": compile_ms,
            }

        def _parse_bench(stdout_text: str) -> dict | None:
            try:
                for ln in str(stdout_text).splitlines():
                    if ln.startswith("INTENTIR_BENCH "):
                        return json.loads(ln[len("INTENTIR_BENCH ") :].strip())
            except Exception:
                return None
            return None

        def _parse_profile(stdout_text: str) -> dict | None:
            try:
                for ln in str(stdout_text).splitlines():
                    if ln.startswith("INTENTIR_PROFILE "):
                        return json.loads(ln[len("INTENTIR_PROFILE ") :].strip())
            except Exception:
                return None
            return None

        def _run_once(*, proc_bind: str | None, bench_iters_run: int, bench_warmup_run: int) -> dict:
            env_prefix = ""
            if int(omp_threads) > 0:
                t = int(omp_threads)
                env_prefix += f"INTENTIR_OMP_THREADS={t} OMP_NUM_THREADS={t} OMP_DYNAMIC=FALSE "
                if proc_bind:
                    env_prefix += f"OMP_PROC_BIND={str(proc_bind)} "
                if omp_places:
                    env_prefix += f"OMP_PLACES={str(omp_places)} "
                if gomp_cpu_affinity:
                    env_prefix += f"GOMP_CPU_AFFINITY={str(gomp_cpu_affinity)} "
            if bool(profile_ops):
                env_prefix += "INTENTIR_PROFILE_OPS=1 "

            cmd = f"cd {remote_dir} && {env_prefix}{remote_bin}"
            if int(bench_iters_run) > 0:
                bi = int(bench_iters_run)
                bw = int(bench_warmup_run)
                bs = int(bench_seed)
                if bw < 0:
                    bw = 0
                cmd = (
                    f"cd {remote_dir} && {env_prefix}"
                    f"INTENTIR_BENCH_ITERS={bi} INTENTIR_BENCH_WARMUP={bw} INTENTIR_BENCH_SEED={bs} {remote_bin}"
                )

            t_run0 = time.perf_counter()
            stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
            run_out = stdout.read().decode()
            run_err = stderr.read().decode()
            run_rc = stdout.channel.recv_exit_status()
            run_wall_ms = float((time.perf_counter() - t_run0) * 1000.0)
            return {
                "run_rc": run_rc,
                "stdout": run_out,
                "stderr": run_err,
                "bench": _parse_bench(run_out),
                "profile_ops": _parse_profile(run_out),
                "run_wall_ms": run_wall_ms,
            }

        _log(f"[{frontend}:{kernel}] remote run")

        proc_bind_used: str | None = None
        proc_bind_trials: list[dict] | None = None
        if int(omp_threads) > 0 and str(omp_proc_bind).lower() == "auto" and int(bench_iters) > 0:
            # Micro-autotune proc bind policy; compile is already the dominant cost.
            candidates = ["spread", "false"]
            it_small = min(max(int(bench_iters) // 4, 5), 30)
            wu_small = min(max(int(bench_warmup), 1), 5)
            proc_bind_trials = []
            best = None
            best_ns = None
            for cand in candidates:
                r = _run_once(proc_bind=cand, bench_iters_run=it_small, bench_warmup_run=wu_small)
                b = r.get("bench") or {}
                ns = None
                if isinstance(b, dict):
                    v = b.get("ns_per_iter")
                    if isinstance(v, (int, float)) and v > 0:
                        ns = float(v)
                proc_bind_trials.append({"proc_bind": cand, "run_rc": r.get("run_rc"), "bench": r.get("bench")})
                if r.get("run_rc") == 0 and ns is not None:
                    if best_ns is None or ns < best_ns:
                        best_ns = ns
                        best = cand
            proc_bind_used = best or "spread"
        else:
            s = str(omp_proc_bind).strip()
            # "auto" is our internal sentinel. Some OpenMP runtimes (libgomp)
            # reject OMP_PROC_BIND=auto, so choose a safe default.
            if s.lower() == "auto":
                proc_bind_used = ("spread" if int(omp_threads) > 1 else None)
            else:
                proc_bind_used = (s if s and s.lower() not in {"none"} else None)

        run = _run_once(proc_bind=proc_bind_used, bench_iters_run=int(bench_iters), bench_warmup_run=int(bench_warmup))

        return {
            "compile_rc": compile_rc,
            "compile_stdout": comp_out,
            "compile_stderr": comp_err,
            "compile_ms": compile_ms,
            "compile_mode": str(execution_mode),
            "run_rc": run.get("run_rc"),
            "stdout": run.get("stdout") or "",
            "stderr": run.get("stderr") or "",
            "bench": run.get("bench"),
            "profile_ops": run.get("profile_ops"),
            "launch_ms": float(run.get("run_wall_ms") or 0.0),
            "total_ms": float(compile_ms + float(run.get("run_wall_ms") or 0.0)),
            "omp": {
                "threads": int(omp_threads),
                "proc_bind": proc_bind_used,
                "places": str(omp_places),
                "gomp_cpu_affinity": gomp_cpu_affinity,
                "proc_bind_trials": proc_bind_trials,
            },
        }

    chosen_schedule = selected_schedule
    chosen = None
    if tune_candidates is not None and len(tune_candidates) > 0:
        # Measured autotune: benchmark top-K candidates and pick the best passing one.
        cand_runs: list[dict] = []
        best_idx = None
        best_ns = None
        for i, c in enumerate(tune_candidates):
            r = _compile_and_run(c.schedule)
            cand_runs.append(
                {
                    "idx": int(i),
                    "pred_score": float(c.score),
                    "tile_mnk": (list(c.tile_mnk) if c.tile_mnk is not None else None),
                    "notes": list(c.notes),
                    "schedule": {
                        "tile_m": c.schedule.tile_m,
                        "tile_n": c.schedule.tile_n,
                        "tile_k": c.schedule.tile_k,
                        "vec_width": c.schedule.vec_width,
                        "pipeline_depth": c.schedule.pipeline_depth,
                    },
                    "compile_rc": r.get("compile_rc"),
                    "run_rc": r.get("run_rc"),
                    "bench": r.get("bench"),
                }
            )
            if r.get("compile_rc") != 0 or r.get("run_rc") != 0:
                continue
            b = r.get("bench") or {}
            ns = b.get("ns_per_iter")
            if isinstance(ns, (int, float)) and ns > 0:
                if best_ns is None or float(ns) < float(best_ns):
                    best_ns = float(ns)
                    best_idx = int(i)
                    chosen = r
                    chosen_schedule = c.schedule
            elif best_idx is None:
                # No bench info (shouldn't happen if bench_iters>0), but keep the first passing run.
                best_idx = int(i)
                chosen = r
                chosen_schedule = c.schedule

        if tuning_info is None:
            tuning_info = {}
        tuning_info["measured_autotune"] = {
            "evaluated": cand_runs,
            "best_index": best_idx,
            "best_ns_per_iter": best_ns,
        }
        if chosen is None:
            # Fallback: run once with the current schedule (already set to the first candidate above).
            chosen = _compile_and_run(selected_schedule)
    else:
        chosen = _compile_and_run(selected_schedule)

    sftp.close()
    client.close()

    rc = int(chosen.get("compile_rc") or 0)
    run_rc = int(chosen.get("run_rc") or 0)
    run_out = str(chosen.get("stdout") or "")
    run_err = str(chosen.get("stderr") or "")
    bench = chosen.get("bench")
    prof = chosen.get("profile_ops")
    compile_ms = float(chosen.get("compile_ms") or 0.0)
    launch_ms = float(chosen.get("launch_ms") or 0.0)
    total_ms = float(chosen.get("total_ms") or (compile_ms + launch_ms))
    # Include a compact baseline summary for quick inspection (avoid huge blobs).
    baseline_summary = {}
    try:
        if baseline is None:
            baseline_summary = {}
        elif kernel == "any_kernel_dim" and "out" in baseline:
            out_ref = np.asarray(baseline["out"]).reshape(-1).astype(np.uint8)
            baseline_summary = {
                "out_len": int(out_ref.size),
                "out_sum": int(out_ref.sum()),
                "out_first": [int(x) for x in out_ref[: min(32, out_ref.size)]],
            }
        elif kernel == "group_norm_kernel" and "Y" in baseline:
            y = np.asarray(baseline["Y"], dtype=np.float32).reshape(-1)
            baseline_summary = {"Y_len": int(y.size), "Y_mean": float(y.mean()), "Y_std": float(y.std())}
        elif kernel == "_attn_fwd" and "Out" in baseline:
            o = np.asarray(baseline["Out"], dtype=np.float32).reshape(-1)
            baseline_summary = {"Out_len": int(o.size), "Out_mean": float(o.mean()), "Out_std": float(o.std())}
        elif kernel == "softmax_inner" and "output" in baseline:
            o = np.asarray(baseline["output"], dtype=np.float32).reshape(-1)
            baseline_summary = {"output_len": int(o.size), "output_mean": float(o.mean()), "output_std": float(o.std())}
        elif kernel == "layer_norm_persistent" and "out_ptr" in baseline:
            o = np.asarray(baseline["out_ptr"], dtype=np.float32).reshape(-1)
            baseline_summary = {"out_ptr_len": int(o.size), "out_ptr_mean": float(o.mean()), "out_ptr_std": float(o.std())}
    except Exception:
        baseline_summary = {}
    return {
        "frontend": str(frontend),
        "triton_provider": (str(triton_provider) if frontend == "triton" else None),
        "artifact_dir": str(artifact_dir),
        "require_mlir_artifacts": bool(require_mlir_artifacts),
        "contract_artifact_used": bool(contract_artifact_path),
        "contract_artifact_path": str(contract_artifact_path),
        "mlir_artifact_used": bool(mlir_artifact_path),
        "mlir_artifact_path": str(mlir_artifact_path),
        "backend": backend_used,
        "execution_mode": str(execution_mode),
        "execution_reason": str(execution_plan.get("reason") or ""),
        "execution_elf_machine": str(execution_plan.get("elf_machine") or ""),
        "execution_llvm_triple": str(execution_plan.get("llvm_triple") or ""),
        "omp_threads": int(omp_threads),
        "omp": chosen.get("omp"),
        "schedule": {
            "tile_m": chosen_schedule.tile_m,
            "tile_n": chosen_schedule.tile_n,
            "tile_k": chosen_schedule.tile_k,
            "vec_width": chosen_schedule.vec_width,
            "pipeline_depth": chosen_schedule.pipeline_depth,
            "axis_bindings": dict(chosen_schedule.axis_bindings or {}),
            "vec_axis": chosen_schedule.vec_axis,
            "parallel_axes": list(chosen_schedule.parallel_axes or []),
            "memory_hint": dict(chosen_schedule.memory_hint or {}),
        },
        "compile_rc": rc,
        "compile_mode": str(chosen.get("compile_mode") or execution_mode),
        "run_rc": run_rc,
        "lower_ms": 0.0,
        "compile_ms": compile_ms,
        "launch_ms": launch_ms,
        "total_ms": total_ms,
        "stdout": run_out,
        "stderr": run_err,
        "baseline_summary": baseline_summary,
        "tuning": tuning_info,
        "bench": bench,
        "profile_ops": prof,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="any_kernel_dim")
    ap.add_argument("--frontend", choices=["triton", "tilelang", "cuda"], default="triton")
    ap.add_argument(
        "--triton-provider",
        choices=["native", "flaggems"],
        default="native",
        help="Triton artifact provider (default: native)",
    )
    ap.add_argument("--artifact-dir", default=None, help="Override artifact report directory.")
    ap.add_argument(
        "--require-mlir-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require MLIR artifacts in reports and forbid fallback to legacy intent JSON.",
    )
    ap.add_argument(
        "--host",
        default=DEFAULT_RVV_HOST,
        help=f"RVV host (default: {DEFAULT_RVV_HOST}; env: INTENTIR_RVV_HOST)",
    )
    ap.add_argument(
        "--user",
        default=DEFAULT_RVV_USER,
        help=f"SSH user (default: {DEFAULT_RVV_USER}; env: INTENTIR_RVV_USER)",
    )
    ap.add_argument("--password", default=None, help="SSH password (prefer env INTENTIR_SSH_PASSWORD or prompt)")
    ap.add_argument("--use-key", action="store_true", help="use SSH key auth (no password prompt)")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--case-index", type=int, default=0, help="pick case from artifacts report (default 0)")
    ap.add_argument(
        "--shape",
        action="append",
        default=[],
        help="override a shape symbol binding (repeatable), e.g. --shape M=256",
    )
    ap.add_argument("--baseline-npz", default=None, help="override baseline npz path (default: from artifact report)")
    ap.add_argument("--prefer-live-baseline", action="store_true", help="re-launch frontend baseline even if npz exists")
    ap.add_argument("--no-tune", action="store_true", help="disable backend schedule selection (use IntentIR schedule as-is)")
    ap.add_argument("--tune-mode", choices=["auto", "guided", "locked"], default="auto")
    ap.add_argument("--tune-budget", type=int, default=1, help="if >1, benchmark top-K predicted schedules (requires --bench-iters>0)")
    ap.add_argument("--tune-debug", action="store_true", help="include structured tuning/cost-model debug in JSON output")
    ap.add_argument("--lock", action="append", default=[], help="repeatable; e.g. --lock tile_n=128")
    ap.add_argument("--constraint", action="append", default=[], help="repeatable; e.g. --constraint 'tile_n in (64,128)'")
    ap.add_argument("--profile", default=None, help="RVV profile name or JSON path (default: query remote host)")
    ap.add_argument("--bench-iters", type=int, default=0, help="if >0, run microbenchmark loop and print INTENTIR_BENCH JSON line")
    ap.add_argument("--bench-warmup", type=int, default=1, help="warmup iterations for benchmark loop")
    ap.add_argument("--bench-seed", type=int, default=0, help="deterministic seed for bench-only input fill (default: 0)")
    ap.add_argument("--omp-threads", type=int, default=1, help="OpenMP threads for RVV backend (default: 1)")
    ap.add_argument("--omp-proc-bind", default="auto", help="OpenMP proc bind policy (spread/close/false/auto)")
    ap.add_argument("--omp-places", default="cores", help="OpenMP places (e.g., cores, threads)")
    ap.add_argument("--gomp-cpu-affinity", default=None, help="Optional GOMP_CPU_AFFINITY override (advanced)")
    ap.add_argument("--profile-ops", action="store_true", help="emit per-op timing JSON line (INTENTIR_PROFILE) from the RVV program")
    ap.add_argument(
        "--bench-only",
        action="store_true",
        help="perf-only mode: do not upload inputs/refs; emit a bench-only C runner that allocates/fills inputs on target and skips output compare",
    )
    ap.add_argument("--json", action="store_true", help="print result as JSON (stable for tooling)")
    ap.add_argument("--quiet", action="store_true", help="disable progress logs")
    args = ap.parse_args()
    password: str | None = None
    if not bool(args.use_key):
        password = args.password or os.getenv("INTENTIR_SSH_PASSWORD")
        if password is None:
            password = getpass.getpass(f"SSH password for {args.user}@{args.host}: ")
    tune_req = None
    if not bool(args.no_tune):
        tune_req = TuningRequest(
            mode=str(args.tune_mode),
            budget=int(args.tune_budget),
            locks=parse_locks(args.lock or []),
            constraints=parse_constraints(args.constraint or []),
            debug=bool(args.tune_debug),
        )
    shape_overrides = {}
    for item in list(args.shape or []):
        if not item:
            continue
        if "=" not in str(item):
            raise ValueError(f"--shape expects key=value, got: {item!r}")
        k, v = str(item).split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"--shape expects key=value, got: {item!r}")
        try:
            shape_overrides[k] = int(v)
        except Exception:
            raise ValueError(f"--shape value must be int, got: {item!r}")

    def _log(msg: str) -> None:
        if bool(args.quiet):
            return
        print(str(msg), file=sys.stderr, flush=True)

    res = run_remote(
        args.kernel,
        args.frontend,
        args.host,
        args.user,
        password,
        port=args.port,
        case_index=args.case_index,
        shape_overrides=(shape_overrides if shape_overrides else None),
        baseline_npz=args.baseline_npz,
        prefer_live_baseline=bool(args.prefer_live_baseline),
        tune_request=tune_req,
        tune_profile=str(args.profile) if args.profile else None,
        bench_iters=int(args.bench_iters),
        bench_warmup=int(args.bench_warmup),
        bench_seed=int(args.bench_seed),
        omp_threads=int(args.omp_threads),
        omp_proc_bind=str(args.omp_proc_bind),
        omp_places=str(args.omp_places),
        gomp_cpu_affinity=(str(args.gomp_cpu_affinity) if args.gomp_cpu_affinity else None),
        profile_ops=bool(args.profile_ops),
        bench_only=bool(args.bench_only),
        log=_log,
        triton_provider=str(args.triton_provider),
        artifact_dir=(str(args.artifact_dir) if args.artifact_dir else None),
        require_mlir_artifacts=bool(args.require_mlir_artifacts),
    )
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print(res)


if __name__ == "__main__":
    main()
