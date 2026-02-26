"""
CUDA backend smoke from existing frontend artifacts (no LLM, no remote).

For each kernel artifact under `artifacts/<frontend>_full_pipeline/`, this script:
  - loads MLIR backend contract (contract-first; fallback to intent JSON only when needed)
  - loads baseline inputs/outputs from `<kernel>.baseline.npz`
  - lowers intent JSON to CUDA
  - runs the CUDA kernel via runtime wrapper
  - compares outputs against baseline reference
"""

from __future__ import annotations

import atexit
import argparse
import json
import math
import multiprocessing as mp
import os
from functools import lru_cache
from queue import Empty
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.cuda.pipeline.driver import lower_cuda_contract_to_kernel  # noqa: E402
from backends.cuda.runtime import (  # noqa: E402
    CudaLaunch,
    CudaRuntimeError,
    compile_cuda_extension,
    cuda_extension_cache_info,
    load_cuda_ptx_module,
    run_cuda_kernel,
    run_cuda_kernel_ptx,
)
from backends.common.mlir_contract import MlirBackendContract  # noqa: E402
from intent_ir.ir import IntentFunction  # noqa: E402
from intent_ir.macros import expand_macros_json  # noqa: E402
from intent_ir.mlir import to_mlir  # noqa: E402
from intent_ir.mlir.convert_to_intent import to_intent  # noqa: E402
from intent_ir.mlir.passes.emit_cuda_contract import build_cuda_contract  # noqa: E402


DEFAULT_KERNELS = [
    "any_kernel_dim",
    "group_norm_kernel",
    "_attn_fwd",
    "softmax_inner",
    "layer_norm_persistent",
    "upsample_bicubic2d_aa",
]


_PERSISTENT_WORKERS: dict[str, dict[str, Any]] = {}
_PERSISTENT_TASK_SEQ = 0


def _intent_from_json_payload(intent_json: dict[str, Any]) -> IntentFunction:
    loader = getattr(IntentFunction, "from_json_dict")
    return loader(dict(intent_json))


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
            return "full_pipeline_verify"
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


@lru_cache(maxsize=4096)
def _discover_kernel_artifact_path_cached(artifact_root_raw: str, kernel: str, suffix: str) -> str:
    root = Path(str(artifact_root_raw or "")).resolve()
    k = str(kernel).strip()
    sfx = str(suffix).strip()
    if not k or not sfx:
        return ""
    direct = root / f"{k}{sfx}"
    if direct.is_file():
        return str(direct)
    try:
        matches = [p for p in root.rglob(f"{k}{sfx}") if p.is_file()]
    except Exception:
        matches = []
    if not matches:
        return ""
    preferred: list[Path] = []
    fallback: list[Path] = []
    for p in matches:
        if "pipeline_reports" in p.parts:
            preferred.append(p)
        else:
            fallback.append(p)
    pool = preferred or fallback
    pool.sort(key=lambda p: float(p.stat().st_mtime), reverse=True)
    return str(pool[0]) if pool else ""


def _discover_kernel_artifact_path(*, artifact_root: Path, kernel: str, suffix: str) -> Path | None:
    raw = _discover_kernel_artifact_path_cached(str(artifact_root), str(kernel), str(suffix))
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_file() else None


def _mlir_report_paths(report: dict, *, artifact_root: Path) -> list[Path]:
    mlir = report.get("mlir")
    if not isinstance(mlir, dict):
        return []
    out: list[Path] = []
    # Prefer downstream contracts first, then midend/module.
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
        # CUDA path must prefer CUDA-specific contracts first.
        "downstream_cuda_llvm_contract_path",
        "downstream_cuda_contract_path",
        "downstream_llvm_contract_path",
        "midend_cuda_contract_path",
        # Generic downstream contract is allowed only as a late fallback.
        "downstream_contract_path",
        "downstream_rvv_contract_path",
        "midend_rvv_contract_path",
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


def _resolve_contract_module_path(
    contract: MlirBackendContract,
    *,
    contract_path: Path,
) -> Path | None:
    artifacts = dict(contract.artifacts or {})
    module_path_raw = str(artifacts.get("mlir_module_path") or "").strip()
    if not module_path_raw and str(contract.executable.format or "").endswith("mlir_module"):
        module_path_raw = str(contract.executable.path or "").strip()
    if not module_path_raw:
        return None
    p = Path(module_path_raw)
    if p.is_absolute():
        return p if p.is_file() else None
    candidates = [
        (contract_path.parent / p),
        (ROOT / p),
    ]
    for cand in candidates:
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


def _intent_from_contract_path(
    path: Path,
    *,
    expected_backend: str = "cuda",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
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
    backend = str(payload.backend or "").strip().lower()
    if backend and backend != str(expected_backend).strip().lower():
        return None
    module_text = str((payload.artifacts or {}).get("mlir_module_text") or "").strip()
    io_spec = _runtime_io_spec_from_contract(payload)
    if module_text:
        try:
            intent = to_intent(module_text)
            return intent.to_json_dict(), io_spec, payload.to_json_dict()
        except Exception:
            pass
    module_path = _resolve_contract_module_path(payload, contract_path=path)
    if module_path is not None:
        try:
            intent = to_intent(module_path.read_text(encoding="utf-8"))
            return intent.to_json_dict(), io_spec, payload.to_json_dict()
        except Exception:
            pass
    intent_json = _synthesize_intent_from_contract(payload)
    return intent_json, io_spec, payload.to_json_dict()


def _load_intent_and_contract(
    report: dict,
    *,
    artifact_root: Path,
    require_mlir_artifacts: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for contract_path in _contract_report_paths(report, artifact_root=artifact_root):
        parsed = _intent_from_contract_path(contract_path, expected_backend="cuda")
        if parsed is not None:
            return parsed
    for mlir_path in _mlir_report_paths(report, artifact_root=artifact_root):
        try:
            parsed = to_intent(mlir_path.read_text(encoding="utf-8"))
            mlir_mod = to_mlir(parsed)
            contract = build_cuda_contract(mlir_mod, source_kind="mlir_path_fallback")
            artifacts = dict(contract.artifacts or {})
            artifacts["mlir_module_text"] = str(mlir_mod.module_text or "")
            contract.artifacts = artifacts
            return parsed.to_json_dict(), {}, contract.to_json_dict()
        except Exception:
            continue
    if bool(require_mlir_artifacts):
        raise RuntimeError("mlir_artifact_missing: no readable MLIR module path in report")
    intent_expanded_json = report.get("intent_expanded")
    if not isinstance(intent_expanded_json, dict):
        intent_raw = report.get("intent")
        if not isinstance(intent_raw, dict):
            raise RuntimeError("intent_payload_missing: report has no contract/mlir/intent payload")
        intent_expanded_json = expand_macros_json(dict(intent_raw))
    intent = _intent_from_json_payload(dict(intent_expanded_json))
    mlir_mod = to_mlir(intent)
    contract = build_cuda_contract(mlir_mod, source_kind="intent_json_fallback")
    artifacts = dict(contract.artifacts or {})
    artifacts["mlir_module_text"] = str(mlir_mod.module_text or "")
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


def _coerce_bindings(bindings: dict) -> dict:
    out = dict(bindings)
    if "batch" in out and "Z" not in out:
        out["Z"] = out["batch"]
    if "Z" in out and "batch" not in out:
        out["batch"] = out["Z"]
    if "group" in out and "num_groups" not in out:
        out["num_groups"] = out["group"]
    if "num_groups" in out and "C" in out and "group_size" not in out:
        try:
            g = int(out["num_groups"])
            c = int(out["C"])
            if g > 0:
                out["group_size"] = c // g if (c % g == 0) else (c + g - 1) // g
        except Exception:
            pass
    if "group_size" in out and "HW" in out and "num_elements" not in out:
        try:
            out["num_elements"] = int(out["group_size"]) * int(out["HW"])
        except Exception:
            pass
    if "N" in out and "group_size" in out and "G" not in out:
        try:
            n = int(out["N"])
            gs = int(out["group_size"])
            if gs > 0 and n % gs == 0:
                out["G"] = n // gs
        except Exception:
            pass
    if "HEAD_DIM" in out:
        try:
            hd = int(out["HEAD_DIM"])
            if hd > 0:
                out.setdefault("HEAD_DIM_DIV2", hd // 2)
        except Exception:
            pass
    return out


def _derive_scalar_input(name: str, *, dtype: str, bindings: dict) -> np.ndarray | None:
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


def _augment_bindings_from_arrays(
    *,
    tensor_specs: dict[str, Any],
    bindings: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    out = dict(bindings)
    for name, arr in arrays.items():
        spec_raw = tensor_specs.get(str(name))
        if not isinstance(spec_raw, dict):
            continue
        arr_np = np.asarray(arr)
        spec_shape = _tensor_shape_spec(spec_raw)
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


def _derive_optional_tensor_input(name: str, *, tensor_spec: dict[str, Any], bindings: dict[str, Any]) -> np.ndarray | None:
    if str(name) != "attn_mask":
        return None
    shape = _resolve_tensor_shape(_tensor_shape_spec(tensor_spec), bindings)
    if shape is None:
        return None
    return np.zeros(shape, dtype=_np_dtype(_tensor_dtype(tensor_spec)))


def _outputs_from_io_spec(io_spec: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for x in list(io_spec.get("outputs") or []):
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    return out


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
    out: list[str] = []
    for x in list(intent_json.get("outputs") or []):
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


def _resolve_tensor_shape_compat(tensor: Any, bindings: dict) -> tuple[int, ...] | None:
    # Kept for backward compatibility in worker state reload paths.
    if isinstance(tensor, dict):
        return _resolve_tensor_shape(_tensor_shape_spec(tensor), bindings)
    shape = []
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


def _compare_output(name: str, got: np.ndarray, ref: np.ndarray, *, atol: float, rtol: float) -> tuple[bool, str]:
    g = np.asarray(got)
    r = np.asarray(ref)
    if g.shape != r.shape:
        return False, f"shape mismatch {name}: {g.shape} vs {r.shape}"
    if g.dtype == np.bool_ or r.dtype == np.bool_:
        ok = bool(np.array_equal(g.astype(np.bool_), r.astype(np.bool_)))
        return ok, ("ok" if ok else f"bool mismatch in {name}")
    ok = bool(np.allclose(g, r, atol=float(atol), rtol=float(rtol), equal_nan=True))
    if ok:
        return True, "ok"
    abs_err = np.max(np.abs(g.astype(np.float64) - r.astype(np.float64))) if g.size else 0.0
    return False, f"mismatch in {name} (max_abs={float(abs_err):.6g}, atol={atol}, rtol={rtol})"


def _prepare_kernel_context(
    kernel: str,
    *,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    require_mlir_artifacts: bool = False,
) -> dict[str, Any]:
    artifact_rel = _artifact_dir_for_frontend(frontend, triton_provider=str(triton_provider))
    artifact_root = (Path(artifact_dir) if artifact_dir else (ROOT / "artifacts" / artifact_rel)).resolve()
    report_path = artifact_root / f"{kernel}.json"
    baseline_npz_path = artifact_root / f"{kernel}.baseline.npz"
    if not report_path.is_file():
        discovered = _discover_kernel_artifact_path(artifact_root=artifact_root, kernel=str(kernel), suffix=".json")
        if discovered is not None:
            report_path = discovered
    if not baseline_npz_path.is_file():
        discovered = _discover_kernel_artifact_path(
            artifact_root=artifact_root,
            kernel=str(kernel),
            suffix=".baseline.npz",
        )
        if discovered is not None:
            baseline_npz_path = discovered
        elif report_path.is_file():
            sibling = report_path.with_name(f"{kernel}.baseline.npz")
            if sibling.is_file():
                baseline_npz_path = sibling
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
    baseline = dict(np.load(baseline_npz_path, allow_pickle=False))
    wanted_aliases = sorted(set(list(tensor_specs.keys()) + _outputs_from_io_spec(io_spec) + _intent_outputs(intent_json)))
    baseline = _with_io_aliases_for_names(wanted_aliases, baseline)
    external_inputs, outputs = _external_inputs(intent_json)
    produced = _produced_outputs(intent_json)
    io_outputs = _outputs_from_io_spec(io_spec)
    if io_outputs:
        outputs = io_outputs
    outputs = [name for name in outputs if (name in baseline or name in produced)]
    if not outputs:
        outputs = _intent_outputs(intent_json)
    raw_bindings = ((report.get("baseline") or {}).get("shapes") or {}) if isinstance(report.get("baseline"), dict) else {}
    bindings = _coerce_bindings(raw_bindings)
    bindings = _augment_bindings_from_arrays(tensor_specs=tensor_specs, bindings=bindings, arrays=baseline)
    tol = (report.get("tolerances") or {}) if isinstance(report.get("tolerances"), dict) else {}
    return {
        "kernel": str(kernel),
        "intent_json": dict(intent_json),
        "mlir_contract": dict(mlir_contract),
        "io_spec": dict(io_spec),
        "tensor_specs": dict(tensor_specs),
        "baseline": baseline,
        "external_inputs": external_inputs,
        "outputs": outputs,
        "bindings": bindings,
        "atol": float(tol.get("atol", 1e-3)),
        "rtol": float(tol.get("rtol", 1e-3)),
    }


def _build_inputs_np(
    *,
    kernel: str,
    tensor_specs: dict[str, dict[str, Any]],
    baseline: dict[str, np.ndarray],
    external_inputs: list[str],
    bindings: dict[str, Any],
) -> dict[str, np.ndarray]:
    inputs_np: dict[str, np.ndarray] = {}
    for name in external_inputs:
        if name in baseline:
            inputs_np[name] = np.asarray(baseline[name])
            continue
        spec = tensor_specs.get(str(name))
        if not isinstance(spec, dict):
            raise RuntimeError(f"baseline missing input {name} for {kernel}")
        shape_spec = _tensor_shape_spec(spec)
        if shape_spec:
            derived_t = _derive_optional_tensor_input(name, tensor_spec=spec, bindings=bindings)
            if derived_t is None:
                raise RuntimeError(f"baseline missing input {name} for {kernel}")
            inputs_np[name] = derived_t
            continue
        derived_s = _derive_scalar_input(name, dtype=_tensor_dtype(spec), bindings=bindings)
        if derived_s is None:
            raise RuntimeError(f"baseline missing input {name} for {kernel}")
        inputs_np[name] = derived_s
    return inputs_np


def _set_codegen_mode_env() -> None:
    # Pure-compiler mode: keep environment neutral and avoid deprecated knobs.
    os.environ.pop("INTENTIR_CUDA_CODEGEN", None)
    os.environ.pop("INTENTIR_CUDA_CODEGEN_STRICT", None)


def _set_runtime_backend_env(runtime_backend: str) -> None:
    mode = str(runtime_backend).strip().lower()
    if mode == "nvrtc":
        os.environ["INTENTIR_CUDA_FORCE_NVRTC"] = "1"
        os.environ.setdefault("INTENTIR_CUDA_NVRTC_FALLBACK", "1")
        try:
            try:
                from cuda import nvrtc as _nvrtc  # type: ignore[attr-defined]  # noqa: PLC0415
            except Exception:
                from cuda.bindings import nvrtc as _nvrtc  # type: ignore[assignment]  # noqa: PLC0415
        except Exception as e:
            raise RuntimeError(f"nvrtc_unavailable: {type(e).__name__}: {e}") from e
    else:
        os.environ.pop("INTENTIR_CUDA_FORCE_NVRTC", None)


def _parse_launch_dict(launch_raw: Any) -> CudaLaunch:
    launch = dict(launch_raw or {}) if isinstance(launch_raw, dict) else {}
    grid = launch.get("grid")
    block = launch.get("block")
    if not (isinstance(grid, list) and len(grid) == 3 and isinstance(block, list) and len(block) == 3):
        raise RuntimeError("invalid_cuda_launch: missing grid/block in lowered payload")
    return CudaLaunch(
        grid=(int(grid[0]), int(grid[1]), int(grid[2])),
        block=(int(block[0]), int(block[1]), int(block[2])),
        shared_mem=int(launch.get("shared_mem", 0)),
    )


def _run_compile_stage(
    kernel: str,
    *,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    require_mlir_artifacts: bool = False,
) -> dict[str, Any]:
    t_all = time.perf_counter()
    ctx = _prepare_kernel_context(
        kernel,
        frontend=frontend,
        triton_provider=triton_provider,
        artifact_dir=artifact_dir,
        require_mlir_artifacts=bool(require_mlir_artifacts),
    )
    t_lower = time.perf_counter()
    lowered = lower_cuda_contract_to_kernel(dict(ctx["mlir_contract"]), shape_bindings=ctx["bindings"])
    lower_ms = (time.perf_counter() - t_lower) * 1000.0
    kernel_name = str(lowered.get("kernel_name") or "")
    io_spec = dict(lowered.get("io_spec") or {})
    is_ptx = lowered.get("cuda_ptx") is not None
    cache_info = (
        {
            "artifact_exists": False,
            "module_name": f"intentir_cuda_ptx_{kernel_name}",
            "build_dir": "",
        }
        if is_ptx
        else cuda_extension_cache_info(
            kernel_name=kernel_name,
            cuda_src=str(lowered.get("cuda_src") or ""),
        )
    )

    t_compile = time.perf_counter()
    if is_ptx:
        _ = load_cuda_ptx_module(kernel_name=kernel_name, ptx=lowered.get("cuda_ptx"), io_spec=io_spec)
    else:
        _ = compile_cuda_extension(
            kernel_name=kernel_name,
            cuda_src=str(lowered.get("cuda_src") or ""),
            io_spec=io_spec,
        )
    compile_ms = (time.perf_counter() - t_compile) * 1000.0

    return {
        "kernel": str(kernel),
        "ok": True,
        "reason_code": "ok",
        "lower_ms": float(lower_ms),
        "compile_ms": float(compile_ms),
        "launch_ms": 0.0,
        "total_ms": float((time.perf_counter() - t_all) * 1000.0),
        "bindings": dict(ctx["bindings"]),
        "compile_cache_hit": bool(cache_info.get("artifact_exists")),
        "compile_module_name": str(cache_info.get("module_name") or ""),
        "compile_build_dir": str(cache_info.get("build_dir") or ""),
    }


def _run_launch_stage(
    kernel: str,
    *,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    require_mlir_artifacts: bool = False,
) -> dict[str, Any]:
    t_all = time.perf_counter()
    ctx = _prepare_kernel_context(
        kernel,
        frontend=frontend,
        triton_provider=triton_provider,
        artifact_dir=artifact_dir,
        require_mlir_artifacts=bool(require_mlir_artifacts),
    )

    t_lower = time.perf_counter()
    lowered = lower_cuda_contract_to_kernel(dict(ctx["mlir_contract"]), shape_bindings=ctx["bindings"])
    lower_ms = (time.perf_counter() - t_lower) * 1000.0
    kernel_name = str(lowered.get("kernel_name") or "")
    io_spec = dict(lowered.get("io_spec") or {})
    is_ptx = lowered.get("cuda_ptx") is not None
    cache_info = (
        {
            "artifact_exists": False,
            "module_name": f"intentir_cuda_ptx_{kernel_name}",
            "build_dir": "",
        }
        if is_ptx
        else cuda_extension_cache_info(
            kernel_name=kernel_name,
            cuda_src=str(lowered.get("cuda_src") or ""),
        )
    )

    t_compile = time.perf_counter()
    compiled_mod = (
        load_cuda_ptx_module(kernel_name=kernel_name, ptx=lowered.get("cuda_ptx"), io_spec=io_spec)
        if is_ptx
        else compile_cuda_extension(
            kernel_name=kernel_name,
            cuda_src=str(lowered.get("cuda_src") or ""),
            io_spec=io_spec,
        )
    )
    compile_ms = (time.perf_counter() - t_compile) * 1000.0

    inputs_np = _build_inputs_np(
        kernel=str(kernel),
        tensor_specs=ctx["tensor_specs"],
        baseline=ctx["baseline"],
        external_inputs=ctx["external_inputs"],
        bindings=ctx["bindings"],
    )

    t_launch = time.perf_counter()
    if is_ptx:
        out = run_cuda_kernel_ptx(
            kernel_name=kernel_name,
            ptx=lowered.get("cuda_ptx"),
            io_spec=io_spec,
            launch=_parse_launch_dict(lowered.get("launch")),
            bindings=dict(lowered.get("bindings") or {}),
            inputs_np=inputs_np,
            output_names=list(lowered.get("output_names") or ctx["outputs"]),
            compiled_module=compiled_mod,
        )
    else:
        out = run_cuda_kernel(
            kernel_name=kernel_name,
            cuda_src=str(lowered.get("cuda_src") or ""),
            io_spec=io_spec,
            launch=_parse_launch_dict(lowered.get("launch")),
            bindings=dict(lowered.get("bindings") or {}),
            inputs_np=inputs_np,
            output_names=list(lowered.get("output_names") or ctx["outputs"]),
            compiled_module=compiled_mod,
        )
    out = _with_io_aliases_for_names(
        sorted(set(list(ctx["outputs"]) + list(dict(ctx["tensor_specs"]).keys()))),
        out,
    )
    launch_ms = (time.perf_counter() - t_launch) * 1000.0

    checks: list[dict[str, Any]] = []
    ok_all = True
    for name in ctx["outputs"]:
        if name not in ctx["baseline"]:
            checks.append({"name": str(name), "ok": False, "summary": f"baseline missing output {name}"})
            ok_all = False
            continue
        if name not in out:
            checks.append({"name": str(name), "ok": False, "summary": f"cuda output missing {name}"})
            ok_all = False
            continue
        ok, summary = _compare_output(name, out[name], ctx["baseline"][name], atol=ctx["atol"], rtol=ctx["rtol"])
        checks.append({"name": str(name), "ok": bool(ok), "summary": str(summary)})
        ok_all = ok_all and bool(ok)

    return {
        "kernel": str(kernel),
        "ok": bool(ok_all),
        "reason_code": ("ok" if ok_all else "diff_fail"),
        "atol": float(ctx["atol"]),
        "rtol": float(ctx["rtol"]),
        "checks": checks,
        "bindings": dict(ctx["bindings"]),
        "lower_ms": float(lower_ms),
        "compile_ms": float(compile_ms),
        "launch_ms": float(launch_ms),
        "total_ms": float((time.perf_counter() - t_all) * 1000.0),
        "compile_cache_hit": bool(cache_info.get("artifact_exists")),
        "compile_module_name": str(cache_info.get("module_name") or ""),
        "compile_build_dir": str(cache_info.get("build_dir") or ""),
    }


def run_one(
    kernel: str,
    *,
    frontend: str = "triton",
    triton_provider: str = "native",
    artifact_dir: str | None = None,
    require_mlir_artifacts: bool = False,
) -> dict:
    # Compatibility helper for ad-hoc invocations.
    _ = _run_compile_stage(
        kernel,
        frontend=frontend,
        triton_provider=triton_provider,
        artifact_dir=artifact_dir,
        require_mlir_artifacts=bool(require_mlir_artifacts),
    )
    return _run_launch_stage(
        kernel,
        frontend=frontend,
        triton_provider=triton_provider,
        artifact_dir=artifact_dir,
        require_mlir_artifacts=bool(require_mlir_artifacts),
    )


def _reason_code_for_exception(e: Exception) -> str:
    msg = str(e).lower()
    if "unsupported" in msg or "missing op" in msg:
        return "lowering_missing_op"
    if isinstance(e, CudaRuntimeError):
        if "timeout" in msg:
            return "launch_timeout"
        return "runtime_fail"
    if isinstance(e, FileNotFoundError):
        return "artifact_missing"
    if isinstance(e, RuntimeError):
        if "mlir_artifact_missing" in str(e).lower():
            return "artifact_missing"
        if "nvrtc_unavailable" in str(e).lower():
            return "env_unavailable"
        return "runtime_fail"
    if isinstance(e, ValueError):
        return "runtime_fail"
    return "runtime_fail"


def _cuda_compile_worker(
    queue: mp.Queue,
    kernel: str,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    require_mlir_artifacts: bool,
    runtime_backend: str,
) -> None:
    try:
        _set_codegen_mode_env()
        _set_runtime_backend_env(runtime_backend)
        res = _run_compile_stage(
            kernel,
            frontend=frontend,
            triton_provider=triton_provider,
            artifact_dir=artifact_dir,
            require_mlir_artifacts=bool(require_mlir_artifacts),
        )
        queue.put({"ok": True, "result": res})
    except Exception as ex:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": {
                    "type": type(ex).__name__,
                    "message": str(ex),
                    "reason_code": _reason_code_for_exception(ex),
                },
            }
        )


def _cuda_launch_worker(
    queue: mp.Queue,
    kernel: str,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    require_mlir_artifacts: bool,
    runtime_backend: str,
) -> None:
    try:
        _set_codegen_mode_env()
        _set_runtime_backend_env(runtime_backend)
        res = _run_launch_stage(
            kernel,
            frontend=frontend,
            triton_provider=triton_provider,
            artifact_dir=artifact_dir,
            require_mlir_artifacts=bool(require_mlir_artifacts),
        )
        queue.put({"ok": True, "result": res})
    except Exception as ex:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": {
                    "type": type(ex).__name__,
                    "message": str(ex),
                    "reason_code": _reason_code_for_exception(ex),
                },
            }
        )


def _resolve_start_method_for_stage(stage: str) -> str:
    stage_key = str(stage).strip().lower()
    explicit_global = os.getenv("INTENTIR_CUDA_SMOKE_MP_START_METHOD")
    explicit_stage = os.getenv(f"INTENTIR_CUDA_SMOKE_{stage_key.upper()}_MP_START_METHOD")
    if explicit_stage is not None:
        preferred = str(explicit_stage).strip().lower()
    elif explicit_global is not None:
        preferred = str(explicit_global).strip().lower()
    else:
        preferred = "spawn"
    available = list(mp.get_all_start_methods())
    if preferred in available:
        return preferred
    if "spawn" in available:
        return "spawn"
    if available:
        return str(available[0])
    return "spawn"


def _persistent_workers_enabled() -> bool:
    raw = str(os.getenv("INTENTIR_CUDA_SMOKE_PERSISTENT_WORKER", "1")).strip().lower()
    return raw in {"1", "true", "yes", "y"}


def _shutdown_persistent_workers() -> None:
    for stage, handle in list(_PERSISTENT_WORKERS.items()):
        proc = handle.get("proc")
        in_q = handle.get("in_q")
        try:
            if in_q is not None:
                in_q.put_nowait(None)
        except Exception:
            pass
        try:
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        except Exception:
            pass
        _PERSISTENT_WORKERS.pop(stage, None)


def _reset_persistent_worker(stage: str) -> None:
    stage_key = str(stage).strip().lower()
    handle = _PERSISTENT_WORKERS.get(stage_key)
    if handle is None:
        return
    proc = handle.get("proc")
    in_q = handle.get("in_q")
    try:
        if in_q is not None:
            in_q.put_nowait(None)
    except Exception:
        pass
    try:
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
    except Exception:
        pass
    _PERSISTENT_WORKERS.pop(stage_key, None)


atexit.register(_shutdown_persistent_workers)


def _cuda_stage_worker_loop(
    in_q: Any,
    out_q: Any,
    *,
    stage: str,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    require_mlir_artifacts: bool,
    runtime_backend: str,
) -> None:
    while True:
        item = in_q.get()
        if item is None:
            break
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        kernel = str(item.get("kernel") or "")
        if not kernel:
            out_q.put(
                {
                    "task_id": task_id,
                    "ok": False,
                    "error": {
                        "type": "ValueError",
                        "message": "missing kernel in stage worker request",
                        "reason_code": "runtime_fail",
                    },
                }
            )
            continue
        try:
            _set_codegen_mode_env()
            _set_runtime_backend_env(runtime_backend)
            if stage == "compile":
                res = _run_compile_stage(
                    kernel,
                    frontend=frontend,
                    triton_provider=triton_provider,
                    artifact_dir=artifact_dir,
                    require_mlir_artifacts=bool(require_mlir_artifacts),
                )
            elif stage == "launch":
                res = _run_launch_stage(
                    kernel,
                    frontend=frontend,
                    triton_provider=triton_provider,
                    artifact_dir=artifact_dir,
                    require_mlir_artifacts=bool(require_mlir_artifacts),
                )
            else:
                raise ValueError(f"unsupported worker stage: {stage}")
            out_q.put({"task_id": task_id, "ok": True, "result": res})
        except Exception as ex:  # noqa: BLE001
            out_q.put(
                {
                    "task_id": task_id,
                    "ok": False,
                    "error": {
                        "type": type(ex).__name__,
                        "message": str(ex),
                        "reason_code": _reason_code_for_exception(ex),
                    },
                }
            )


def _get_or_start_persistent_worker(
    *,
    stage: str,
    start_method: str,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    require_mlir_artifacts: bool,
    runtime_backend: str,
) -> dict[str, Any]:
    cfg = {
        "start_method": str(start_method),
        "frontend": str(frontend),
        "triton_provider": str(triton_provider),
        "artifact_dir": (str(artifact_dir) if artifact_dir else ""),
        "require_mlir_artifacts": bool(require_mlir_artifacts),
        "runtime_backend": str(runtime_backend),
    }
    existing = _PERSISTENT_WORKERS.get(stage)
    if existing is not None:
        same_cfg = existing.get("cfg") == cfg
        proc = existing.get("proc")
        if same_cfg and proc is not None and proc.is_alive():
            return existing
        try:
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        except Exception:
            pass
        _PERSISTENT_WORKERS.pop(stage, None)

    ctx = mp.get_context(start_method)
    in_q = ctx.Queue()
    out_q = ctx.Queue()
    proc = ctx.Process(
        target=_cuda_stage_worker_loop,
        kwargs={
            "in_q": in_q,
            "out_q": out_q,
            "stage": str(stage),
            "frontend": str(frontend),
            "triton_provider": str(triton_provider),
            "artifact_dir": (str(artifact_dir) if artifact_dir else None),
            "require_mlir_artifacts": bool(require_mlir_artifacts),
            "runtime_backend": str(runtime_backend),
        },
    )
    proc.start()
    handle = {"cfg": cfg, "ctx": ctx, "in_q": in_q, "out_q": out_q, "proc": proc}
    _PERSISTENT_WORKERS[stage] = handle
    return handle


def _run_worker_with_timeout(
    worker: Any,
    *,
    kernel: str,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    require_mlir_artifacts: bool,
    runtime_backend: str,
    timeout_sec: int,
    stage: str,
) -> dict[str, Any]:
    stage_key = str(stage).strip().lower()
    start_method = _resolve_start_method_for_stage(stage_key)
    if _persistent_workers_enabled() and stage_key in {"compile", "launch"}:
        global _PERSISTENT_TASK_SEQ
        handle = _get_or_start_persistent_worker(
            stage=stage_key,
            start_method=start_method,
            frontend=frontend,
            triton_provider=triton_provider,
            artifact_dir=artifact_dir,
            require_mlir_artifacts=bool(require_mlir_artifacts),
            runtime_backend=runtime_backend,
        )
        _PERSISTENT_TASK_SEQ += 1
        task_id = int(_PERSISTENT_TASK_SEQ)
        in_q = handle["in_q"]
        out_q = handle["out_q"]
        proc = handle["proc"]
        in_q.put({"task_id": task_id, "kernel": str(kernel)})
        timeout = None if int(timeout_sec) <= 0 else float(timeout_sec)
        deadline = None if timeout is None else (time.time() + timeout)
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.time())
            if deadline is not None and remaining <= 0.0:
                try:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=2.0)
                except Exception:
                    pass
                _PERSISTENT_WORKERS.pop(stage_key, None)
                return {"ok": False, "timed_out": True}
            try:
                payload = out_q.get(timeout=remaining)
            except Empty:
                try:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=2.0)
                except Exception:
                    pass
                _PERSISTENT_WORKERS.pop(stage_key, None)
                return {"ok": False, "timed_out": True}
            if not isinstance(payload, dict):
                continue
            if int(payload.get("task_id", -1)) != task_id:
                continue
            payload["timed_out"] = False
            return payload

    # one-shot worker fallback
    ctx = mp.get_context(start_method)
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=worker,
        args=(
            q,
            str(kernel),
            str(frontend),
            str(triton_provider),
            artifact_dir,
            bool(require_mlir_artifacts),
            str(runtime_backend),
        ),
    )
    proc.start()
    timeout = None if int(timeout_sec) <= 0 else float(timeout_sec)
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        return {"ok": False, "timed_out": True}
    try:
        payload = q.get_nowait()
    except Empty:
        payload = None
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "timed_out": False,
            "error": {
                "type": "RuntimeError",
                "message": "no result returned from worker process",
                "reason_code": "runtime_fail",
            },
        }
    payload["timed_out"] = False
    return payload


def _run_one_with_stage_timeouts(
    kernel: str,
    *,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    require_mlir_artifacts: bool,
    compile_timeout_sec: int,
    launch_timeout_sec: int,
    runtime_backend: str,
) -> dict[str, Any]:
    compile_res = _run_worker_with_timeout(
        _cuda_compile_worker,
        kernel=kernel,
        frontend=frontend,
        triton_provider=triton_provider,
        artifact_dir=artifact_dir,
        require_mlir_artifacts=bool(require_mlir_artifacts),
        runtime_backend=runtime_backend,
        timeout_sec=int(compile_timeout_sec),
        stage="compile",
    )
    if bool(compile_res.get("timed_out")):
        return {
            "kernel": str(kernel),
            "ok": False,
            "reason_code": "compile_timeout",
            "error": {"type": "TimeoutError", "message": f"compile stage exceeded timeout_sec={int(compile_timeout_sec)}"},
        }
    if not bool(compile_res.get("ok")):
        err = compile_res.get("error") if isinstance(compile_res.get("error"), dict) else {}
        return {
            "kernel": str(kernel),
            "ok": False,
            "reason_code": str(err.get("reason_code") or "runtime_fail"),
            "error": {"type": str(err.get("type") or "RuntimeError"), "message": str(err.get("message") or "compile stage failed")},
        }
    compile_payload = dict(compile_res.get("result") or {})

    launch_res = _run_worker_with_timeout(
        _cuda_launch_worker,
        kernel=kernel,
        frontend=frontend,
        triton_provider=triton_provider,
        artifact_dir=artifact_dir,
        require_mlir_artifacts=bool(require_mlir_artifacts),
        runtime_backend=runtime_backend,
        timeout_sec=int(launch_timeout_sec),
        stage="launch",
    )
    if bool(launch_res.get("timed_out")):
        return {
            "kernel": str(kernel),
            "ok": False,
            "reason_code": "launch_timeout",
            "error": {"type": "TimeoutError", "message": f"launch stage exceeded timeout_sec={int(launch_timeout_sec)}"},
            "lower_ms": float(compile_payload.get("lower_ms", 0.0)),
            "compile_ms": float(compile_payload.get("compile_ms", 0.0)),
            "launch_ms": 0.0,
            "total_ms": float(compile_payload.get("total_ms", 0.0)),
            "compile_cache_hit": bool(compile_payload.get("compile_cache_hit", False)),
            "compile_module_name": str(compile_payload.get("compile_module_name") or ""),
            "compile_build_dir": str(compile_payload.get("compile_build_dir") or ""),
        }
    if not bool(launch_res.get("ok")):
        err = launch_res.get("error") if isinstance(launch_res.get("error"), dict) else {}
        msg = str(err.get("message") or "")
        msg_l = msg.lower()
        _reset_persistent_worker("launch")
        if any(
            token in msg_l
            for token in (
                "illegal memory access",
                "cumoduleloaddata failed",
                "cuda error",
                "device-side assert",
            )
        ):
            _reset_persistent_worker("compile")
        return {
            "kernel": str(kernel),
            "ok": False,
            "reason_code": str(err.get("reason_code") or "runtime_fail"),
            "error": {"type": str(err.get("type") or "RuntimeError"), "message": str(err.get("message") or "launch stage failed")},
            "lower_ms": float(compile_payload.get("lower_ms", 0.0)),
            "compile_ms": float(compile_payload.get("compile_ms", 0.0)),
            "launch_ms": 0.0,
            "total_ms": float(compile_payload.get("total_ms", 0.0)),
            "compile_cache_hit": bool(compile_payload.get("compile_cache_hit", False)),
            "compile_module_name": str(compile_payload.get("compile_module_name") or ""),
            "compile_build_dir": str(compile_payload.get("compile_build_dir") or ""),
        }

    launch_payload = dict(launch_res.get("result") or {})
    launch_ms = float(launch_payload.get("launch_ms", 0.0))
    lower_ms = float(compile_payload.get("lower_ms", launch_payload.get("lower_ms", 0.0)))
    compile_ms = float(compile_payload.get("compile_ms", launch_payload.get("compile_ms", 0.0)))
    launch_payload["lower_ms"] = lower_ms
    launch_payload["compile_ms"] = compile_ms
    launch_payload["launch_ms"] = launch_ms
    launch_payload["total_ms"] = float(lower_ms + compile_ms + launch_ms)
    launch_payload["compile_cache_hit"] = bool(compile_payload.get("compile_cache_hit", False))
    launch_payload["compile_module_name"] = str(compile_payload.get("compile_module_name") or "")
    launch_payload["compile_build_dir"] = str(compile_payload.get("compile_build_dir") or "")
    return launch_payload


def _probe_timeout_runtime_detail(
    *,
    kernel: str,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    compile_timeout_sec: int,
    launch_timeout_sec: int,
    runtime_backend: str,
) -> dict[str, Any]:
    probe = _run_one_with_stage_timeouts(
        kernel=kernel,
        frontend=frontend,
        triton_provider=triton_provider,
        artifact_dir=artifact_dir,
        require_mlir_artifacts=True,
        compile_timeout_sec=max(1, min(int(compile_timeout_sec), 30)),
        launch_timeout_sec=max(1, min(int(launch_timeout_sec), 30)),
        runtime_backend=runtime_backend,
    )
    detail = {
        "ok": bool(probe.get("ok")),
        "reason_code": str(probe.get("reason_code") or "runtime_fail"),
    }
    if isinstance(probe.get("error"), dict):
        detail["error"] = dict(probe.get("error") or {})
    return detail


def _cuda_env_ready() -> tuple[bool, str]:
    if shutil.which("nvcc") is None:
        return False, "nvcc_not_found"
    try:
        import torch  # noqa: PLC0415

        if not bool(torch.cuda.is_available()):
            return False, "torch_cuda_unavailable"
    except Exception as e:
        return False, f"torch_import_error:{type(e).__name__}"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang"], default="triton")
    ap.add_argument(
        "--triton-provider",
        choices=["native", "flaggems"],
        default="native",
        help="Triton artifact provider (default: native)",
    )
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; default runs kernel suite")
    ap.add_argument(
        "--flaggems-opset",
        choices=["deterministic_forward"],
        default="deterministic_forward",
        help="FlagGems semantic-op set used to resolve default kernels.",
    )
    ap.add_argument(
        "--backend-target",
        choices=["rvv", "cuda_h100", "cuda_5090d"],
        default="cuda_h100",
        help="Capability target passed to FlagGems spec registry when selecting defaults.",
    )
    ap.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Compatibility default timeout in seconds used when stage-specific timeouts are not set.",
    )
    ap.add_argument(
        "--compile-timeout-sec",
        type=int,
        default=None,
        help="Compile-stage timeout in seconds (defaults to --timeout-sec).",
    )
    ap.add_argument(
        "--launch-timeout-sec",
        type=int,
        default=None,
        help="Launch-stage timeout in seconds (defaults to --timeout-sec).",
    )
    ap.add_argument(
        "--runtime-backend",
        choices=["auto", "nvcc", "nvrtc"],
        default="auto",
        help="CUDA runtime compile backend selector. nvrtc forces NVRTC path, nvcc forces extension path.",
    )
    ap.add_argument(
        "--refine-timeout-reason",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When timeout happens under auto mode, run a short py-codegen probe and attach runtime_detail.",
    )
    ap.add_argument("--artifact-dir", default=None, help="Override artifact report directory.")
    ap.add_argument(
        "--require-mlir-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require MLIR artifacts in reports and forbid fallback to legacy intent JSON.",
    )
    ap.add_argument("--allow-skip", action="store_true", help="exit 0 with ok=false when CUDA environment is unavailable")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-kernel START/DONE progress lines even in --json mode.",
    )
    ap.add_argument("--json", action="store_true", help="print machine-readable summary JSON")
    ap.add_argument("--out", default=None, help="write summary JSON to this path")
    args = ap.parse_args()

    compile_timeout_sec = int(args.compile_timeout_sec) if args.compile_timeout_sec is not None else int(args.timeout_sec)
    launch_timeout_sec = int(args.launch_timeout_sec) if args.launch_timeout_sec is not None else int(args.timeout_sec)
    runtime_backend = str(args.runtime_backend)
    if runtime_backend == "auto":
        runtime_backend = "nvcc"
    if args.kernel:
        kernels = list(args.kernel)
    else:
        kernels = _default_kernels_for(
            frontend=str(args.frontend),
            triton_provider=str(args.triton_provider),
            flaggems_opset=str(args.flaggems_opset),
            backend_target=str(args.backend_target),
        )

    env_ok, env_reason = _cuda_env_ready()
    if not env_ok:
        summary = {
            "frontend": str(args.frontend),
            "triton_provider": (str(args.triton_provider) if str(args.frontend) == "triton" else None),
            "flaggems_opset": str(args.flaggems_opset),
            "backend_target": str(args.backend_target),
            "artifact_dir": (str(args.artifact_dir) if args.artifact_dir else None),
            "require_mlir_artifacts": bool(args.require_mlir_artifacts),
            "runtime_backend": str(runtime_backend),
            "timeout_sec": int(args.timeout_sec),
            "compile_timeout_sec": int(compile_timeout_sec),
            "launch_timeout_sec": int(launch_timeout_sec),
            "kernels": list(kernels),
            "results": [],
            "ok": False,
            "skipped": True,
            "skip_reason": str(env_reason),
        }
        if args.out:
            Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.json:
            print(json.dumps(summary, ensure_ascii=False))
        if args.allow_skip:
            _shutdown_persistent_workers()
            raise SystemExit(0)
        _shutdown_persistent_workers()
        raise SystemExit(1)

    results: list[dict[str, Any]] = []
    ok_all = True
    total_kernels = len(kernels)
    for idx, k in enumerate(kernels, start=1):
        if bool(args.progress):
            print(f"[cuda][{idx}/{total_kernels}] START kernel={k}", flush=True)
        try:
            r = _run_one_with_stage_timeouts(
                str(k),
                frontend=str(args.frontend),
                triton_provider=str(args.triton_provider),
                artifact_dir=(str(args.artifact_dir) if args.artifact_dir else None),
                require_mlir_artifacts=bool(args.require_mlir_artifacts),
                compile_timeout_sec=int(compile_timeout_sec),
                launch_timeout_sec=int(launch_timeout_sec),
                runtime_backend=runtime_backend,
            )
        except (CudaRuntimeError, FileNotFoundError, RuntimeError, ValueError) as e:
            r = {
                "kernel": str(k),
                "ok": False,
                "reason_code": _reason_code_for_exception(e),
                "error": {"type": type(e).__name__, "message": str(e)},
            }
        if str(r.get("reason_code") or "") not in {"ok", "compile_timeout", "launch_timeout"}:
            if "lower_ms" not in r:
                r["lower_ms"] = 0.0
            if "compile_ms" not in r:
                r["compile_ms"] = 0.0
            if "launch_ms" not in r:
                r["launch_ms"] = 0.0
            if "total_ms" not in r:
                r["total_ms"] = float(r["lower_ms"]) + float(r["compile_ms"]) + float(r["launch_ms"])

        if (
            (not bool(r.get("ok")))
            and str(r.get("reason_code") or "") in {"compile_timeout", "launch_timeout"}
            and bool(args.refine_timeout_reason)
        ):
            detail = _probe_timeout_runtime_detail(
                kernel=str(k),
                frontend=str(args.frontend),
                triton_provider=str(args.triton_provider),
                artifact_dir=(str(args.artifact_dir) if args.artifact_dir else None),
                compile_timeout_sec=int(compile_timeout_sec),
                launch_timeout_sec=int(launch_timeout_sec),
                runtime_backend=runtime_backend,
            )
            runtime_detail = dict(r.get("runtime_detail") or {})
            runtime_detail["timeout_probe"] = detail
            r["runtime_detail"] = runtime_detail
        if "reason_code" not in r:
            r["reason_code"] = ("ok" if bool(r.get("ok")) else "runtime_fail")
        if "lower_ms" not in r:
            r["lower_ms"] = 0.0
        if "compile_ms" not in r:
            r["compile_ms"] = 0.0
        if "launch_ms" not in r:
            r["launch_ms"] = 0.0
        if "total_ms" not in r:
            r["total_ms"] = float(r["lower_ms"]) + float(r["compile_ms"]) + float(r["launch_ms"])
        results.append(r)
        ok_all = ok_all and bool(r.get("ok"))
        if bool(args.progress):
            print(
                f"[cuda][{idx}/{total_kernels}] DONE kernel={k} ok={bool(r.get('ok'))} "
                f"reason={str(r.get('reason_code') or '')} "
                f"lower_ms={float(r.get('lower_ms', 0.0)):.3f} "
                f"compile_ms={float(r.get('compile_ms', 0.0)):.3f} "
                f"launch_ms={float(r.get('launch_ms', 0.0)):.3f} "
                f"total_ms={float(r.get('total_ms', 0.0)):.3f} "
                f"cache_hit={bool(r.get('compile_cache_hit', False))}",
                flush=True,
            )
        if not args.json:
            print(f"[{k}] {'OK' if r.get('ok') else 'FAIL'}")
            if r.get("error"):
                print(f"  {r['error']['type']}: {r['error']['message']}")

    summary = {
        "frontend": str(args.frontend),
        "triton_provider": (str(args.triton_provider) if str(args.frontend) == "triton" else None),
        "flaggems_opset": str(args.flaggems_opset),
        "backend_target": str(args.backend_target),
        "artifact_dir": (str(args.artifact_dir) if args.artifact_dir else None),
        "require_mlir_artifacts": bool(args.require_mlir_artifacts),
        "runtime_backend": str(runtime_backend),
        "timeout_sec": int(args.timeout_sec),
        "compile_timeout_sec": int(compile_timeout_sec),
        "launch_timeout_sec": int(launch_timeout_sec),
        "kernels": list(kernels),
        "results": results,
        "ok": bool(ok_all),
        "skipped": False,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    _shutdown_persistent_workers()
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
