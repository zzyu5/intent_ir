"""
Backend codegen smoke test (no LLM, no remote).

For each kernel artifact under `artifacts/<frontend>_full_pipeline/`, this script:
  - loads expanded IntentIR (or expands macros)
  - loads baseline inputs/outputs from `<kernel>.baseline.npz`
  - invokes Task6 backend codegen (C++ tool) to generate a standalone C program
  - compiles and runs the C program locally to compare against baseline outputs

This validates "IntentIR ops -> C" backend generation end-to-end, without
requiring an RVV host.
"""

from __future__ import annotations

import argparse
import re
import json
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.spmd_rvv.analysis.device_query import load_profile  # noqa: E402
from backends.spmd_rvv.analysis.tuning import TuningRequest, parse_constraints, parse_locks, select_schedule_from_intent_json  # noqa: E402
from backends.common.mlir_contract import MlirBackendContract  # noqa: E402
from intent_ir.ir import IntentFunction  # noqa: E402
from intent_ir.macros import expand_macros_json  # noqa: E402
from intent_ir.mlir import to_mlir  # noqa: E402
from intent_ir.mlir.convert_to_intent import to_intent  # noqa: E402
from intent_ir.mlir.passes.emit_rvv_contract import build_rvv_contract  # noqa: E402


DEFAULT_KERNELS = [
    "any_kernel_dim",
    "group_norm_kernel",
    "_attn_fwd",
    "softmax_inner",
    "layer_norm_persistent",
    "upsample_bicubic2d_aa",
]


def _intent_from_json_payload(intent_json: dict[str, Any]) -> IntentFunction:
    loader = getattr(IntentFunction, "from_json_dict")
    return loader(dict(intent_json))


def lower_intent_to_c_with_files(
    intent_or_json: Any,
    *,
    shape_bindings: dict[str, Any],
    atol: float = 1e-3,
    rtol: float = 1e-3,
    mode: str = "verify",
) -> str:
    if not isinstance(intent_or_json, dict):
        raise TypeError("backend_codegen_smoke requires mlir backend contract JSON payload")
    if not str(intent_or_json.get("schema_version") or "").startswith("intent_mlir_backend_contract_"):
        raise RuntimeError(
            "rvv strict hard-cut: backend_codegen_smoke lower_intent_to_c_with_files accepts only mlir backend contract JSON"
        )
    raise RuntimeError(
        "backend_codegen_smoke contract-to-C lowering path has been removed in strict hard-cut mode; "
        "use RVV remote contract execution for runtime validation"
    )

def _default_kernels_for(
    *,
    frontend: str,
    triton_provider: str,
    flaggems_opset: str,
    backend_target: str,
) -> list[str]:
    if str(frontend) == "triton" and str(triton_provider) == "flaggems":
        from pipeline.triton.providers.flaggems.specs import default_flaggems_kernel_specs  # noqa: PLC0415

        return [
            str(s.name)
            for s in default_flaggems_kernel_specs(
                flaggems_opset=str(flaggems_opset),
                backend_target=str(backend_target),
            )
        ]
    return list(DEFAULT_KERNELS)


def _artifact_dir_for_frontend(frontend: str, *, triton_provider: str = "native") -> str:
    if frontend == "triton":
        p = str(triton_provider)
        if p == "flaggems":
            return "flaggems_triton_full_pipeline"
        if p == "native":
            return "triton_full_pipeline"
        raise ValueError(f"unsupported triton provider: {triton_provider}")
    if frontend == "tilelang":
        return "tilelang_full_pipeline"
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


def _normalize_io_name(name: str) -> str:
    s = str(name).strip().strip("_").lower()
    if s.startswith("ptr_"):
        s = s[4:]
    if s.endswith("_ptr"):
        s = s[:-4]
    if s.endswith("ptr") and len(s) > 3:
        s = s[:-3]
    if s == "i":
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
    if s == "in":
        s = "input"
    if s == "out":
        s = "output"
    return s.replace("_", "")


def _with_io_aliases_for_names(wanted: list[str], io: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(io)
    norm_to_keys: dict[str, list[str]] = {}
    for k in io.keys():
        norm_to_keys.setdefault(_normalize_io_name(k), []).append(k)
    for name in list(wanted or []):
        n = str(name)
        if not n or n in out:
            continue
        keys = norm_to_keys.get(_normalize_io_name(n)) or []
        if len(keys) == 1:
            out[n] = io[keys[0]]
            continue
        if len(keys) > 1:
            lower_name = n.lower()
            picked = None
            for k in keys:
                if str(k).lower() == lower_name:
                    picked = k
                    break
            if picked is None:
                picked = keys[0]
            out[n] = io[picked]
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
            intent = to_intent(module_text)
            return intent.to_json_dict(), io_spec, payload_json
        except Exception:
            pass
    module_path = _resolve_contract_module_path(payload, contract_path=path)
    if module_path is not None:
        try:
            intent = to_intent(module_path.read_text(encoding="utf-8"))
            return intent.to_json_dict(), io_spec, payload_json
        except Exception:
            pass
    return _synthesize_intent_from_contract(payload), io_spec, payload_json


def _load_intent_and_contract(
    report: dict,
    *,
    artifact_root: Path,
    require_mlir_artifacts: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for contract_path in _contract_report_paths(report, artifact_root=artifact_root):
        parsed = _intent_from_contract_path(contract_path)
        if parsed is None:
            continue
        parsed_intent, io_spec, payload = parsed
        if isinstance(parsed_intent, dict):
            return parsed_intent, io_spec, payload
        # Contract payload is still valid for lowering even when module->intent recovery fails.
        break
    else:
        parsed_intent = None
        io_spec = {}
        payload = {}
    for mlir_path in _mlir_report_paths(report, artifact_root=artifact_root):
        try:
            intent = to_intent(mlir_path.read_text(encoding="utf-8"))
            mod = to_mlir(intent)
            contract = build_rvv_contract(mod, source_kind="mlir_path_fallback")
            artifacts = dict(contract.artifacts or {})
            artifacts["mlir_module_text"] = str(mod.module_text or "")
            contract.artifacts = artifacts
            intent_json = intent.to_json_dict()
            return intent_json, io_spec, payload if payload else contract.to_json_dict()
        except Exception:
            continue
    if bool(require_mlir_artifacts):
        raise RuntimeError("mlir_artifact_missing: no readable MLIR module path in report")
    intent_expanded_json = report.get("intent_expanded")
    if not isinstance(intent_expanded_json, dict):
        intent_expanded_json = expand_macros_json(dict(report["intent"]))
    intent = _intent_from_json_payload(dict(intent_expanded_json))
    mod = to_mlir(intent)
    contract = build_rvv_contract(mod, source_kind="intent_json_fallback")
    artifacts = dict(contract.artifacts or {})
    artifacts["mlir_module_text"] = str(mod.module_text or "")
    contract.artifacts = artifacts
    return dict(intent_expanded_json), {}, contract.to_json_dict()


def _external_inputs(intent_json: dict[str, Any]) -> tuple[list[str], list[str]]:
    ops = [x for x in list(intent_json.get("ops") or []) if isinstance(x, dict)]
    tensors = dict(intent_json.get("tensors") or {})
    outputs = [str(x) for x in list(intent_json.get("outputs") or []) if str(x).strip()]
    if not ops and tensors:
        external_inputs = sorted([str(n) for n in tensors.keys() if str(n) and str(n) not in set(outputs)])
        return external_inputs, outputs
    produced = {str(op.get("output") or "") for op in ops if str(op.get("output") or "").strip()}
    used: set[str] = set()
    for op in ops:
        for inp in list(op.get("inputs") or []):
            inp_s = str(inp).strip()
            if inp_s:
                used.add(inp_s)
    external_inputs = sorted([n for n in used if n in tensors and n not in produced])
    return external_inputs, outputs


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


def _resolve_tensor_shape(shape_spec: list[Any], bindings: dict[str, Any]) -> tuple[int, ...] | None:
    shape: list[int] = []
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
    if isinstance(shape, list):
        return list(shape)
    return []


def _tensor_dtype(tensor_spec: dict[str, Any]) -> str:
    return str(tensor_spec.get("dtype") or "f32")


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


def _outputs_from_io_spec(io_spec: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for x in list(io_spec.get("outputs") or []):
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    return out


def _produced_outputs(intent_json: dict[str, Any]) -> set[str]:
    produced: set[str] = set()
    for op in list(intent_json.get("ops") or []):
        if not isinstance(op, dict):
            continue
        out = str(op.get("output") or "").strip()
        if out:
            produced.add(out)
    return produced


def _augment_bindings_from_arrays(
    *,
    tensor_specs: dict[str, dict[str, Any]],
    bindings: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    out = dict(bindings)
    for name, arr in arrays.items():
        spec = tensor_specs.get(str(name))
        if not isinstance(spec, dict):
            continue
        arr_np = np.asarray(arr)
        spec_shape = _tensor_shape_spec(spec)
        if len(spec_shape) == 0 and arr_np.size == 1:
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
            elif isinstance(dim_spec, str):
                try:
                    int(dim_spec)
                except Exception:
                    key = str(dim_spec)
            if key and key not in out:
                out[key] = int(dim_val)
    return out


def _derive_optional_input_array(name: str, *, tensor_spec: dict[str, Any], bindings: dict[str, Any]) -> np.ndarray | None:
    if str(name) == "sm_scale":
        hd = bindings.get("HEAD_DIM")
        try:
            if hd is not None and int(hd) > 0:
                return np.array(1.0 / np.sqrt(float(hd)), dtype=_np_dtype(_tensor_dtype(tensor_spec)))
        except Exception:
            pass
    if str(name) == "attn_mask":
        shape = _resolve_tensor_shape(_tensor_shape_spec(tensor_spec), bindings)
        if shape is not None:
            return np.zeros(shape, dtype=_np_dtype(_tensor_dtype(tensor_spec)))
    return None


def _intent_outputs(intent_json: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for x in list(intent_json.get("outputs") or []):
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    return out


def _resolve_tensor_shape_compat(tensor: Any, bindings: dict) -> tuple[int, ...] | None:
    if isinstance(tensor, dict):
        return _resolve_tensor_shape(_tensor_shape_spec(tensor), bindings)
    shape: list[int] = []
    for d in list(getattr(tensor, "shape", []) or []):
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


def _write_bin(path: Path, arr: np.ndarray, dtype: str) -> None:
    arr_np = np.asarray(arr)
    dtype = str(dtype)
    # Respect declared tensor dtype first. Baseline arrays may carry bool views
    # for tensors that are declared as f32/i32 in intent tensors.
    if dtype in {"bool", "i1"}:
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
    elif dtype == "f16":
        raw = np.asarray(arr_np, dtype=np.float16).tobytes(order="C")
    elif dtype == "f64":
        raw = np.asarray(arr_np, dtype=np.float64).tobytes(order="C")
    else:
        raw = np.asarray(arr_np, dtype=np.float32).tobytes(order="C")
    path.write_bytes(raw)


_BUF_DESC_RE = re.compile(r'\{\s*"(?P<name>[^"]+)"\s*,.*?(?P<dtype>INTENTIR_DTYPE_[A-Z0-9_]+)')


def _dtype_from_buffer_token(tok: str) -> str | None:
    m = {
        "INTENTIR_DTYPE_F16": "f16",
        "INTENTIR_DTYPE_BF16": "bf16",
        "INTENTIR_DTYPE_F32": "f32",
        "INTENTIR_DTYPE_F64": "f64",
        "INTENTIR_DTYPE_I8": "i8",
        "INTENTIR_DTYPE_U8": "u8",
        "INTENTIR_DTYPE_I16": "i16",
        "INTENTIR_DTYPE_I32": "i32",
        "INTENTIR_DTYPE_I64": "i64",
        "INTENTIR_DTYPE_BOOL": "bool",
        "INTENTIR_DTYPE_I1": "i1",
    }
    return m.get(str(tok))


def _extract_buffer_declared_dtypes(c_src: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in str(c_src).splitlines():
        s = line.strip()
        if not s.startswith('{"') or "INTENTIR_DTYPE_" not in s:
            continue
        m = _BUF_DESC_RE.search(s)
        if not m:
            continue
        name = str(m.group("name"))
        dt = _dtype_from_buffer_token(str(m.group("dtype")))
        if dt:
            out[name] = dt
    return out


def run_one(
    kernel: str,
    *,
    frontend: str = "triton",
    triton_provider: str = "native",
    artifact_dir: str | None = None,
    require_mlir_artifacts: bool = False,
    keep_tmp: bool = False,
    tune_request: TuningRequest | None = None,
    tune_profile: str | None = None,
) -> dict:
    t_total = perf_counter()
    lower_ms = 0.0
    compile_ms = 0.0
    launch_ms = 0.0
    artifact_rel = _artifact_dir_for_frontend(frontend, triton_provider=str(triton_provider))
    artifact_root = (Path(artifact_dir) if artifact_dir else (ROOT / "artifacts" / artifact_rel)).resolve()
    report_path = artifact_root / f"{kernel}.json"
    baseline_npz_path = artifact_root / f"{kernel}.baseline.npz"
    if (
        artifact_dir is None
        and (not report_path.exists())
        and str(frontend) == "triton"
        and str(triton_provider) == "native"
        and str(artifact_rel) == "triton_full_pipeline"
    ):
        legacy_root = (ROOT / "artifacts" / "full_pipeline_verify").resolve()
        legacy_report = legacy_root / f"{kernel}.json"
        if legacy_report.exists():
            artifact_root = legacy_root
            report_path = legacy_report
            baseline_npz_path = legacy_root / f"{kernel}.baseline.npz"
    if not report_path.exists():
        raise FileNotFoundError(f"missing artifact report: {report_path}")
    if not baseline_npz_path.exists():
        raise FileNotFoundError(f"missing baseline npz: {baseline_npz_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    intent_json, io_spec, mlir_contract = _load_intent_and_contract(
        report,
        artifact_root=artifact_root,
        require_mlir_artifacts=bool(require_mlir_artifacts),
    )
    tensor_specs = _tensor_specs_from_io_spec(io_spec) or _tensor_specs_from_intent_json(intent_json)
    if not tensor_specs:
        raise RuntimeError("invalid intent payload: missing tensor specs")
    cert_v2 = report.get("certificate_v2") or {}
    tile_hints: list[int] = []
    try:
        sh = cert_v2.get("schedule_hints") or {}
        th = sh.get("tile_hints")
        if isinstance(th, list):
            tile_hints = [int(x) for x in th if isinstance(x, (int, float, str)) and int(x) > 0]
    except Exception:
        tile_hints = []

    baseline = dict(np.load(baseline_npz_path, allow_pickle=False))
    wanted_aliases = sorted(set(list(tensor_specs.keys()) + _outputs_from_io_spec(io_spec) + _intent_outputs(intent_json)))
    baseline = _with_io_aliases_for_names(wanted_aliases, baseline)
    external_inputs, outputs = _external_inputs(intent_json)
    produced = _produced_outputs(intent_json)
    io_outputs = _outputs_from_io_spec(io_spec)
    if io_outputs:
        outputs = io_outputs
    if baseline:
        # Verify only outputs available in the baseline bundle. Some kernels
        # expose auxiliary outputs (e.g., indices) that are intentionally not
        # included in baseline artifacts.
        outputs = [name for name in outputs if name in baseline]
    else:
        outputs = [name for name in outputs if name in produced]
    if not outputs:
        outputs = [name for name in _intent_outputs(intent_json) if (name in baseline)] if baseline else _intent_outputs(intent_json)
    if not outputs:
        # Keep declared outputs to avoid constructing an invalid empty-output intent.
        outputs = _intent_outputs(intent_json)
    intent_codegen_json = dict(intent_json)
    lower_payload: Any = dict(mlir_contract)
    if outputs != _intent_outputs(intent_json):
        intent_codegen_json["outputs"] = list(outputs)
        # Keep contract-first lowering: patch invocation outputs instead of
        # downgrading to non-contract intent JSON payload.
        payload = dict(lower_payload)
        executable = dict(payload.get("executable") or {})
        invocation = dict(executable.get("invocation") or {})
        io_spec = dict(invocation.get("io_spec") or {})
        if io_spec:
            io_spec["outputs"] = list(outputs)
            invocation["io_spec"] = io_spec
        invocation["output_names"] = list(outputs)
        executable["invocation"] = invocation
        payload["executable"] = executable
        lower_payload = payload

    bindings = ((report.get("baseline") or {}).get("shapes") or {}) if isinstance(report.get("baseline"), dict) else {}
    # Common axis aliases (match pipeline/runner conventions).
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
    if "N" in bindings and "group_size" in bindings and "G" not in bindings:
        try:
            n = int(bindings["N"])
            gs = int(bindings["group_size"])
            if gs > 0 and n % gs == 0:
                bindings["G"] = n // gs
        except Exception:
            pass
    bindings = _augment_bindings_from_arrays(tensor_specs=tensor_specs, bindings=bindings, arrays=baseline)
    if tune_request is not None:
        prof = load_profile(tune_profile or "generic_rvv_256")
        tuned = select_schedule_from_intent_json(
            intent_codegen_json,
            shape_bindings=bindings,
            profile=prof,
            request=tune_request,
            tile_hints=tile_hints,
            evidence=cert_v2,
        )
        intent_codegen_json["schedule"] = tuned.schedule.to_json_dict()
    tol = {
        "any_kernel_dim": (0.0, 0.0),
        "group_norm_kernel": (1e-3, 1e-3),
        "_attn_fwd": (1e-2, 1e-2),
        "softmax_inner": (1e-3, 1e-3),
        "layer_norm_persistent": (1e-3, 1e-3),
        "upsample_bicubic2d_aa": (1e-3, 1e-3),
    }
    atol, rtol = tol.get(kernel, (1e-3, 1e-3))

    tmp_ctx = tempfile.TemporaryDirectory(prefix=f"intentir_codegen_smoke_{kernel}_")
    td = Path(tmp_ctx.name)
    try:
        t_lower = perf_counter()
        c_src = lower_intent_to_c_with_files(
            lower_payload,
            shape_bindings=bindings,
            atol=float(atol),
            rtol=float(rtol),
        )
        lower_ms = (perf_counter() - t_lower) * 1000.0
        declared_dtypes = _extract_buffer_declared_dtypes(c_src)

        # Write inputs / reference outputs after lowering, so file dtypes follow
        # emitted buffer descriptors instead of potentially stale intent tensor dtypes.
        for name in external_inputs:
            if name not in baseline:
                ts = tensor_specs.get(str(name))
                if isinstance(ts, dict):
                    derived = _derive_optional_input_array(name, tensor_spec=ts, bindings=bindings)
                    if derived is not None:
                        baseline[name] = derived
            if name not in baseline:
                raise RuntimeError(f"baseline missing input {name} for {kernel}")
            ts = tensor_specs.get(str(name)) or {}
            dtype = declared_dtypes.get(name, _tensor_dtype(ts))
            _write_bin(td / f"{name}.bin", np.asarray(baseline[name]), dtype)
        for name in outputs:
            if name not in baseline:
                raise RuntimeError(f"baseline missing output {name} for {kernel}")
            ts = tensor_specs.get(str(name)) or {}
            dtype = declared_dtypes.get(name, _tensor_dtype(ts))
            _write_bin(td / f"{name}_ref.bin", np.asarray(baseline[name]), dtype)

        (td / "main.c").write_text(c_src, encoding="utf-8")

        runtime_dir = ROOT / "backends" / "spmd_rvv" / "runtime"
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

        compile_cmd = [
            "gcc",
            "-O2",
            "-std=c11",
            "-D_POSIX_C_SOURCE=200809L",
            "-I.",
            "-o",
            str(td / "run"),
            str(td / "main.c"),
            str(td / "intentir_runtime.c"),
            str(td / "intentir_driver.c"),
            str(td / "intentir_ops.c"),
            "-lm",
            "-lrt",
        ]
        t_compile = perf_counter()
        cp = subprocess.run(compile_cmd, cwd=td, capture_output=True, text=True)
        compile_ms = (perf_counter() - t_compile) * 1000.0
        if cp.returncode != 0:
            return {
                "kernel": kernel,
                "ok": False,
                "rc": int(cp.returncode),
                "reason_code": "compile_fail",
                "stdout": (cp.stdout or "").strip(),
                "stderr": (cp.stderr or "").strip(),
                "tmpdir": str(td) if keep_tmp else None,
                "lower_ms": float(lower_ms),
                "compile_ms": float(compile_ms),
                "launch_ms": float(launch_ms),
                "total_ms": float((perf_counter() - t_total) * 1000.0),
            }

        t_launch = perf_counter()
        rp = subprocess.run([str(td / "run")], cwd=td, capture_output=True, text=True)
        launch_ms = (perf_counter() - t_launch) * 1000.0
        return {
            "kernel": kernel,
            "ok": rp.returncode == 0,
            "rc": rp.returncode,
            "reason_code": ("ok" if rp.returncode == 0 else "runtime_fail"),
            "stdout": (rp.stdout or "").strip(),
            "stderr": (rp.stderr or "").strip(),
            "tmpdir": str(td) if keep_tmp else None,
            "lower_ms": float(lower_ms),
            "compile_ms": float(compile_ms),
            "launch_ms": float(launch_ms),
            "total_ms": float((perf_counter() - t_total) * 1000.0),
        }
    finally:
        if keep_tmp:
            # Keep temporary artifacts for postmortem debugging.
            # TemporaryDirectory schedules deletion via an internal finalizer,
            # so overriding cleanup() is not sufficient.
            try:
                tmp_ctx._finalizer.detach()  # type: ignore[attr-defined]
            except Exception:
                tmp_ctx.cleanup = lambda: None  # type: ignore[attr-defined]
        else:
            tmp_ctx.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang"], default="triton")
    ap.add_argument(
        "--triton-provider",
        choices=["native", "flaggems"],
        default="native",
        help="Triton artifact provider (default: native)",
    )
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; default runs 6 kernels")
    ap.add_argument(
        "--flaggems-opset",
        choices=["deterministic_forward"],
        default="deterministic_forward",
        help="FlagGems semantic-op set used to resolve default kernels.",
    )
    ap.add_argument(
        "--backend-target",
        choices=["rvv", "cuda_h100", "cuda_5090d"],
        default="rvv",
        help="Capability target passed to FlagGems spec registry when selecting defaults.",
    )
    ap.add_argument("--artifact-dir", default=None, help="Override artifact report directory.")
    ap.add_argument(
        "--require-mlir-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require MLIR artifacts in reports and forbid fallback to legacy intent JSON.",
    )
    ap.add_argument("--keep-tmp", action="store_true", help="keep generated C + binaries in a temp dir")
    ap.add_argument("--tune-mode", choices=["auto", "guided", "locked"], default=None)
    ap.add_argument("--lock", action="append", default=[], help="repeatable; e.g. --lock tile_n=128")
    ap.add_argument("--constraint", action="append", default=[], help="repeatable; e.g. --constraint 'tile_n in (64,128)'")
    ap.add_argument("--profile", default=None, help="RVV profile name or JSON path (default: generic_rvv_256 when tuning)")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-kernel START/DONE progress lines even in --json mode.",
    )
    ap.add_argument("--json", action="store_true", help="print machine-readable summary JSON")
    ap.add_argument("--out", default=None, help="write summary JSON to this path")
    args = ap.parse_args()

    if args.kernel:
        kernels = list(args.kernel)
    else:
        kernels = _default_kernels_for(
            frontend=str(args.frontend),
            triton_provider=str(args.triton_provider),
            flaggems_opset=str(args.flaggems_opset),
            backend_target=str(args.backend_target),
        )
    tune_req = None
    if args.tune_mode:
        tune_req = TuningRequest(
            mode=str(args.tune_mode),
            budget=0,
            locks=parse_locks(args.lock or []),
            constraints=parse_constraints(args.constraint or []),
        )
    results: list[dict[str, Any]] = []
    ok_all = True
    total_kernels = len(kernels)
    for idx, k in enumerate(kernels, start=1):
        if bool(args.progress):
            print(f"[rvv][{idx}/{total_kernels}] START kernel={k}", flush=True)
        try:
            r = run_one(
                k,
                frontend=str(args.frontend),
                triton_provider=str(args.triton_provider),
                artifact_dir=(str(args.artifact_dir) if args.artifact_dir else None),
                require_mlir_artifacts=bool(args.require_mlir_artifacts),
                keep_tmp=bool(args.keep_tmp),
                tune_request=tune_req,
                tune_profile=str(args.profile) if args.profile else None,
            )
        except Exception as e:
            reason = "runtime_fail"
            msg = str(e).lower()
            if "mlir_artifact_missing" in msg:
                reason = "artifact_missing"
            r = {
                "kernel": str(k),
                "ok": False,
                "rc": 1,
                "reason_code": str(reason),
                "stdout": "",
                "stderr": f"{type(e).__name__}: {e}",
                "error": {"type": type(e).__name__, "message": str(e)},
                "lower_ms": 0.0,
                "compile_ms": 0.0,
                "launch_ms": 0.0,
                "total_ms": 0.0,
            }
        results.append(r)
        ok_all = ok_all and bool(r["ok"])
        if bool(args.progress):
            print(
                f"[rvv][{idx}/{total_kernels}] DONE kernel={k} ok={bool(r.get('ok'))} "
                f"reason={str(r.get('reason_code') or '')} "
                f"lower_ms={float(r.get('lower_ms', 0.0)):.3f} "
                f"compile_ms={float(r.get('compile_ms', 0.0)):.3f} "
                f"launch_ms={float(r.get('launch_ms', 0.0)):.3f} "
                f"total_ms={float(r.get('total_ms', 0.0)):.3f}",
                flush=True,
            )
        if not bool(args.json):
            status = "OK" if r["ok"] else "FAIL"
            print(f"[{k}] {status} rc={r.get('rc', 1)}")
            if r.get("stdout"):
                print(r["stdout"])
            if r.get("stderr"):
                print(r["stderr"])
            if args.keep_tmp and r.get("tmpdir"):
                print(f"  tmpdir={r['tmpdir']}")

    summary = {
        "frontend": str(args.frontend),
        "triton_provider": (str(args.triton_provider) if str(args.frontend) == "triton" else None),
        "flaggems_opset": str(args.flaggems_opset),
        "backend_target": str(args.backend_target),
        "artifact_dir": (str(args.artifact_dir) if args.artifact_dir else None),
        "require_mlir_artifacts": bool(args.require_mlir_artifacts),
        "kernels": list(kernels),
        "results": results,
        "ok": bool(ok_all),
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if bool(args.json):
        print(json.dumps(summary, ensure_ascii=False))
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
