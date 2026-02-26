"""
CUDA compiler pipeline driver.

Pure-compiler lifecycle:
legalize -> shape_infer -> schedule -> emit -> compile -> launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
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


@lru_cache(maxsize=1)
def _load_historical_cuda_io_templates() -> dict[str, dict[str, Any]]:
    root = (Path(__file__).resolve().parents[3] / "artifacts" / "flaggems_matrix" / "daily").resolve()
    if not root.is_dir():
        return {}
    by_kernel: dict[str, tuple[int, float, dict[str, Any]]] = {}
    for p in root.rglob("*.intentir.intentdialect.downstream_cuda_llvm.contract.json"):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        kernel = str(payload.get("kernel_name") or "").strip()
        if not kernel:
            continue
        io_spec = payload.get("io_spec") if isinstance(payload.get("io_spec"), Mapping) else {}
        if not isinstance(io_spec, Mapping):
            continue
        arg_names_raw = io_spec.get("arg_names")
        arg_names = [str(x) for x in arg_names_raw] if isinstance(arg_names_raw, list) else []
        if not arg_names:
            continue
        scalars_raw = io_spec.get("scalars") if isinstance(io_spec.get("scalars"), Mapping) else {}
        scalars = {str(k): str(v) for k, v in dict(scalars_raw or {}).items() if str(k).strip()}
        tensors_raw = io_spec.get("tensors") if isinstance(io_spec.get("tensors"), Mapping) else {}
        tensors = {str(k): dict(v) for k, v in dict(tensors_raw or {}).items() if str(k).strip() and isinstance(v, Mapping)}
        tensor_names = [str(k) for k in dict(tensors).keys() if str(k).strip()]
        score = int(len(arg_names)) + (100 if scalars else 0)
        try:
            mtime = float(p.stat().st_mtime)
        except Exception:
            mtime = 0.0
        prev = by_kernel.get(kernel)
        if (prev is None) or (score > prev[0]) or (score == prev[0] and mtime > prev[1]):
            by_kernel[kernel] = (
                score,
                mtime,
                {
                    "arg_names": list(arg_names),
                    "scalars": dict(scalars),
                    "tensors": dict(tensors),
                    "tensor_names": list(tensor_names),
                    "path": str(p),
                },
            )
    return {k: v[2] for k, v in by_kernel.items()}


def _apply_historical_cuda_io_template(
    *,
    io_spec: Mapping[str, Any],
    kernel_name: str,
) -> dict[str, Any]:
    out = dict(io_spec or {})
    kernel = str(kernel_name or "").strip()
    if not kernel:
        return out
    template = _load_historical_cuda_io_templates().get(kernel)
    if not isinstance(template, Mapping):
        return out
    arg_names_tpl = template.get("arg_names")
    if not isinstance(arg_names_tpl, list) or not arg_names_tpl:
        return out
    tensors = out.get("tensors") if isinstance(out.get("tensors"), Mapping) else {}
    tensor_keys = {str(k) for k in dict(tensors or {}).keys()}
    tpl_tensor_names = set([str(x) for x in list(template.get("tensor_names") or []) if str(x).strip()])
    if tpl_tensor_names and tensor_keys and not (
        tpl_tensor_names.issubset(tensor_keys) or tensor_keys.issubset(tpl_tensor_names)
    ):
        return out
    out["arg_names"] = [str(x) for x in arg_names_tpl]
    tpl_tensors = template.get("tensors") if isinstance(template.get("tensors"), Mapping) else {}
    if isinstance(tpl_tensors, Mapping):
        merged_tensors = {str(k): dict(v) for k, v in dict(tensors or {}).items() if str(k).strip() and isinstance(v, Mapping)}
        for k, v in dict(tpl_tensors or {}).items():
            key = str(k).strip()
            if not key or not isinstance(v, Mapping):
                continue
            merged_tensors.setdefault(key, dict(v))
        if merged_tensors:
            out["tensors"] = merged_tensors
    scalars = out.get("scalars") if isinstance(out.get("scalars"), Mapping) else {}
    merged_scalars = {str(k): str(v) for k, v in dict(scalars or {}).items() if str(k).strip()}
    tpl_scalars = template.get("scalars") if isinstance(template.get("scalars"), Mapping) else {}
    for k, v in dict(tpl_scalars or {}).items():
        key = str(k).strip()
        if not key:
            continue
        merged_scalars[key] = str(v)
    if merged_scalars:
        out["scalars"] = dict(merged_scalars)
    return out

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
    if not isinstance(scalars, Mapping):
        scalars = {}
    tensors = io_spec.get("tensors") if isinstance(io_spec.get("tensors"), Mapping) else {}
    tensor_specs = {str(k): dict(v) for k, v in dict(tensors or {}).items() if str(k).strip() and isinstance(v, Mapping)}
    tensor_names = set(tensor_specs.keys())
    arg_names = [str(x) for x in list(io_spec.get("arg_names") or []) if str(x).strip()]

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

    for name in arg_names:
        if name in tensor_names:
            continue
        if _has_binding(name):
            continue
        m = re.match(r"^([A-Za-z_]+)\d+$", name)
        if m is not None:
            base = str(m.group(1))
            if base in out:
                _set_if_missing(name, out.get(base))

    for name, spec in tensor_specs.items():
        if _has_binding(name):
            continue
        shape = spec.get("shape") if isinstance(spec.get("shape"), list) else None
        if shape != []:
            continue
        lname = str(name).strip().lower()
        if lname in {"eps", "epsilon"}:
            out[name] = float(1.0e-5)
    return out


def _ptx_entry_param_types(*, ptx_text: str, entry: str) -> list[str]:
    text = str(ptx_text or "")
    if not text.strip():
        return []
    wanted = str(entry or "").strip()
    entry_pat = re.compile(r"\.visible\s+\.entry\s+([A-Za-z_.$][\w.$]*)\s*\((.*?)\)\s*(?:\.\w+[^\n]*\n)*\{", re.S)
    fallback_sig: str | None = None
    for m in entry_pat.finditer(text):
        name = str(m.group(1) or "").strip()
        sig = str(m.group(2) or "")
        if sig and fallback_sig is None:
            fallback_sig = sig
        if wanted and name == wanted:
            fallback_sig = sig
            break
    if not fallback_sig:
        return []
    out: list[str] = []
    for m in re.finditer(r"\.param\s+\.([A-Za-z0-9_]+)\s+[A-Za-z_.$][\w.$]*", fallback_sig):
        out.append(str(m.group(1)).lower())
    return out


def _ptx_entry_param_names(*, ptx_text: str, entry: str) -> list[str]:
    text = str(ptx_text or "")
    if not text.strip():
        return []
    wanted = str(entry or "").strip()
    entry_pat = re.compile(r"\.visible\s+\.entry\s+([A-Za-z_.$][\w.$]*)\s*\((.*?)\)\s*(?:\.\w+[^\n]*\n)*\{", re.S)
    fallback_sig: str | None = None
    for m in entry_pat.finditer(text):
        name = str(m.group(1) or "").strip()
        sig = str(m.group(2) or "")
        if sig and fallback_sig is None:
            fallback_sig = sig
        if wanted and name == wanted:
            fallback_sig = sig
            break
    if not fallback_sig:
        return []
    out: list[str] = []
    for m in re.finditer(r"\.param\s+\.[A-Za-z0-9_]+\s+([A-Za-z_.$][\w.$]*)", fallback_sig):
        out.append(str(m.group(1)))
    return out


def _infer_symbol_order_from_io_spec(io_spec: Mapping[str, Any]) -> list[str]:
    tensors = io_spec.get("tensors") if isinstance(io_spec.get("tensors"), Mapping) else {}
    if not isinstance(tensors, Mapping):
        return []
    outputs = io_spec.get("outputs") if isinstance(io_spec.get("outputs"), list) else []
    output_names = [str(x).strip() for x in outputs if str(x).strip()]
    output_set = set(output_names)
    input_names = [str(k).strip() for k in tensors.keys() if str(k).strip() and str(k).strip() not in output_set]

    def _dims_for(names: list[str]) -> list[str]:
        out_dims: list[str] = []
        seen_dims: set[str] = set()
        for name in names:
            spec = tensors.get(name)
            shape = spec.get("shape") if isinstance(spec, Mapping) else None
            if not isinstance(shape, list):
                continue
            for d in shape:
                if not isinstance(d, str):
                    continue
                dim = d.strip()
                if not dim or dim in seen_dims:
                    continue
                seen_dims.add(dim)
                out_dims.append(dim)
        return out_dims

    output_dims = _dims_for(output_names)
    input_dims = _dims_for(input_names)
    if not output_dims and not input_dims:
        # Fallback for older contracts without explicit outputs metadata.
        return _dims_for([str(k) for k in tensors.keys()])

    input_set = set(input_dims)
    output_set_dims = set(output_dims)
    output_only = [d for d in output_dims if d not in input_set]
    shared = [d for d in output_dims if d in input_set]
    input_only = [d for d in input_dims if d not in output_set_dims]
    return output_only + shared + input_only


def _augment_io_spec_arg_names_with_ptx_params(
    *,
    io_spec: Mapping[str, Any],
    invocation: Mapping[str, Any],
    merged_bindings: Mapping[str, Any],
    ptx_text: str,
    entry: str,
) -> dict[str, Any]:
    out = dict(io_spec or {})
    arg_names_raw = out.get("arg_names")
    arg_names = [str(x) for x in arg_names_raw] if isinstance(arg_names_raw, list) else []
    ptx_param_types = _ptx_entry_param_types(ptx_text=ptx_text, entry=entry)
    if len(ptx_param_types) <= len(arg_names):
        return out

    tensors = out.get("tensors") if isinstance(out.get("tensors"), Mapping) else {}
    tensor_names = {str(k) for k in tensors.keys()} if isinstance(tensors, Mapping) else set()
    io_scalars = out.get("scalars") if isinstance(out.get("scalars"), Mapping) else {}
    scalars = {str(k): str(v) for k, v in dict(io_scalars or {}).items() if str(k).strip()}
    seen_arg = {str(x) for x in arg_names}

    ordered_candidates: list[str] = []
    seen_candidate: set[str] = set()

    def _add_candidate(name: str) -> None:
        n = str(name).strip()
        if not n or n in seen_candidate or n in tensor_names or n in seen_arg:
            return
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", n):
            return
        ordered_candidates.append(n)
        seen_candidate.add(n)

    for name in _infer_symbol_order_from_io_spec(out):
        _add_candidate(name)
    inv_shape_bindings = invocation.get("shape_bindings") if isinstance(invocation.get("shape_bindings"), Mapping) else {}
    if isinstance(inv_shape_bindings, Mapping):
        for name in inv_shape_bindings.keys():
            _add_candidate(str(name))
    for name in scalars.keys():
        _add_candidate(name)
    for name in merged_bindings.keys():
        _add_candidate(str(name))

    missing = max(0, len(ptx_param_types) - len(arg_names))
    if missing <= 0:
        return out
    append_names = ordered_candidates[:missing]
    if len(append_names) < missing:
        for i in range(missing - len(append_names)):
            append_names.append(f"sym_{len(arg_names) + len(append_names) + i}")

    # PTX params are positional; infer scalar dtypes for newly appended args from
    # trailing PTX param slots.
    scalar_type_map = {
        "f16": "f32",
        "f32": "f32",
        "f64": "f32",
        "u16": "i32",
        "u32": "i32",
        "s32": "i32",
        "u64": "i64",
        "s64": "i64",
        "b16": "i32",
        "b32": "i32",
        "b64": "i64",
    }
    for i, name in enumerate(append_names):
        arg_names.append(name)
        slot = len(arg_names) - 1
        ptx_ty = ptx_param_types[slot] if slot < len(ptx_param_types) else ""
        scalars.setdefault(name, scalar_type_map.get(str(ptx_ty).lower(), "i32"))

    out["arg_names"] = list(arg_names)
    if scalars:
        out["scalars"] = dict(scalars)
    return out


def _infer_launch_grid_x_from_ptx(
    *,
    launch: Mapping[str, Any],
    io_spec: Mapping[str, Any],
    merged_bindings: Mapping[str, Any],
    ptx_text: str,
    entry: str,
) -> dict[str, Any]:
    out = dict(launch or {})
    grid = out.get("grid") if isinstance(out.get("grid"), list) else None
    if not isinstance(grid, list) or len(grid) != 3:
        return out
    try:
        grid_vals = [int(grid[0]), int(grid[1]), int(grid[2])]
    except Exception:
        return out
    if not (grid_vals[0] == 1 and grid_vals[1] == 1 and grid_vals[2] == 1):
        return out

    arg_names = io_spec.get("arg_names") if isinstance(io_spec.get("arg_names"), list) else []
    if not isinstance(arg_names, list) or not arg_names:
        return out

    param_names = _ptx_entry_param_names(ptx_text=ptx_text, entry=entry)
    if not param_names:
        return out
    param_to_index = {str(name): idx for idx, name in enumerate(param_names)}

    reg_to_param_index: dict[str, int] = {}
    for m in re.finditer(r"ld\.param\.u32\s+(%r\d+),\s+\[([A-Za-z_.$][\w.$]*)\];", str(ptx_text or "")):
        reg = str(m.group(1))
        pname = str(m.group(2))
        if pname in param_to_index:
            reg_to_param_index[reg] = int(param_to_index[pname])

    ctaid_x_regs = {str(m.group(1)) for m in re.finditer(r"mov\.u32\s+(%r\d+),\s+%ctaid\.x\s*;", str(ptx_text or ""))}
    if not ctaid_x_regs:
        return out

    bound_param_index: int | None = None
    for m in re.finditer(r"setp\.[A-Za-z0-9_.]+\s+%p\d+,\s*(%r\d+),\s*(%r\d+)\s*;", str(ptx_text or "")):
        lhs = str(m.group(1))
        rhs = str(m.group(2))
        if lhs in ctaid_x_regs and rhs in reg_to_param_index:
            bound_param_index = int(reg_to_param_index[rhs])
            break
        if rhs in ctaid_x_regs and lhs in reg_to_param_index:
            bound_param_index = int(reg_to_param_index[lhs])
            break
    if bound_param_index is None:
        immediate_bound: int | None = None

        def _bound_from_pred(pred: str, imm: int) -> int:
            p = str(pred).lower()
            iv = int(imm)
            if p.startswith("setp.gt"):
                return int(iv + 1)
            if p.startswith("setp.ge"):
                return int(iv)
            if p.startswith("setp.lt"):
                return int(iv)
            if p.startswith("setp.le"):
                return int(iv + 1)
            return int(iv + 1)

        for m in re.finditer(
            r"(setp\.[A-Za-z0-9_.]+)\s+%p\d+,\s*(%r\d+),\s*(-?[0-9]+)\s*;",
            str(ptx_text or ""),
        ):
            pred = str(m.group(1))
            lhs = str(m.group(2))
            imm = _try_int(m.group(3))
            if lhs not in ctaid_x_regs or imm is None:
                continue
            immediate_bound = _bound_from_pred(pred, int(imm))
            break
        if immediate_bound is None:
            for m in re.finditer(
                r"(setp\.[A-Za-z0-9_.]+)\s+%p\d+,\s*(-?[0-9]+),\s*(%r\d+)\s*;",
                str(ptx_text or ""),
            ):
                pred = str(m.group(1))
                imm = _try_int(m.group(2))
                rhs = str(m.group(3))
                if rhs not in ctaid_x_regs or imm is None:
                    continue
                immediate_bound = _bound_from_pred(pred, int(imm))
                break
        if immediate_bound is None or immediate_bound <= 0:
            return out
        out["grid"] = [int(immediate_bound), int(grid_vals[1]), int(grid_vals[2])]
        return out
    if bound_param_index < 0 or bound_param_index >= len(arg_names):
        return out

    dim_name = str(arg_names[bound_param_index]).strip()
    if not dim_name:
        return out
    dim_value = _try_int(merged_bindings.get(dim_name))
    if dim_value is None or dim_value <= 0:
        return out

    out["grid"] = [int(dim_value), int(grid_vals[1]), int(grid_vals[2])]
    return out


def _ensure_launch_grid_block_defaults(
    *,
    launch: Mapping[str, Any],
    ptx_text: str,
) -> dict[str, Any]:
    out = dict(launch or {})
    grid = out.get("grid") if isinstance(out.get("grid"), list) else None
    block = out.get("block") if isinstance(out.get("block"), list) else None

    def _normalize_triplet(vals: list[Any] | None, fallback: list[int]) -> list[int]:
        if not (isinstance(vals, list) and len(vals) == 3):
            return list(fallback)
        out_vals: list[int] = []
        try:
            out_vals = [int(vals[0]), int(vals[1]), int(vals[2])]
        except Exception:
            return list(fallback)
        if any(v <= 0 for v in out_vals):
            return list(fallback)
        return out_vals

    def _bound_from_pred(pred: str, imm: int) -> int:
        p = str(pred).lower()
        iv = int(imm)
        if p.startswith("setp.gt"):
            return int(iv + 1)
        if p.startswith("setp.ge"):
            return int(iv)
        if p.startswith("setp.lt"):
            return int(iv)
        if p.startswith("setp.le"):
            return int(iv + 1)
        return int(iv + 1)

    def _infer_tid_bound(ptx: str, *, axis: str) -> int | None:
        axis_key = str(axis).strip().lower()
        if axis_key not in {"x", "y", "z"}:
            return None
        regs = {str(m.group(1)) for m in re.finditer(rf"mov\.u32\s+(%r\d+),\s+%tid\.{axis_key}\s*;", ptx)}
        if not regs:
            return None
        candidates: list[int] = []
        for m in re.finditer(
            r"(setp\.[A-Za-z0-9_.]+)\s+%p\d+,\s*(%r\d+),\s*(-?[0-9]+)\s*;",
            ptx,
        ):
            pred = str(m.group(1))
            reg = str(m.group(2))
            imm = _try_int(m.group(3))
            if reg not in regs or imm is None:
                continue
            b = _bound_from_pred(pred, int(imm))
            if b > 0:
                candidates.append(int(b))
        for m in re.finditer(
            r"(setp\.[A-Za-z0-9_.]+)\s+%p\d+,\s*(-?[0-9]+),\s*(%r\d+)\s*;",
            ptx,
        ):
            pred = str(m.group(1))
            reg = str(m.group(3))
            imm = _try_int(m.group(2))
            if reg not in regs or imm is None:
                continue
            b = _bound_from_pred(pred, int(imm))
            if b > 0:
                candidates.append(int(b))
        if not candidates:
            return None
        return int(min(candidates))

    ptx_block_hint: list[int] | None = None
    m = re.search(
        r"\.(?:reqntid|maxntid)\s+([0-9]+)(?:\s*,\s*([0-9]+))?(?:\s*,\s*([0-9]+))?",
        str(ptx_text or ""),
    )
    if m:
        try:
            bx = int(m.group(1))
            by = int(m.group(2) or 1)
            bz = int(m.group(3) or 1)
            if bx > 0 and by > 0 and bz > 0:
                ptx_block_hint = [bx, by, bz]
        except Exception:
            ptx_block_hint = None

    out["grid"] = _normalize_triplet(grid, fallback=[1, 1, 1])
    block_triplet = _normalize_triplet(block, fallback=(ptx_block_hint or [256, 1, 1]))
    if ptx_block_hint is not None:
        flattened_hint = int(ptx_block_hint[0] * ptx_block_hint[1] * ptx_block_hint[2])
        if (
            block_triplet[1] == 1
            and block_triplet[2] == 1
            and block_triplet[0] == flattened_hint
            and (ptx_block_hint[1] > 1 or ptx_block_hint[2] > 1)
        ):
            block_triplet = list(ptx_block_hint)
    if block_triplet[1] == 1 and block_triplet[2] == 1:
        text = str(ptx_text or "")
        bx = _infer_tid_bound(text, axis="x")
        by = _infer_tid_bound(text, axis="y")
        if bx is not None and by is not None and bx > 0 and by > 1:
            flat = int(bx * by)
            if int(block_triplet[0]) == flat:
                block_triplet = [int(bx), int(by), 1]
    out["block"] = block_triplet
    return out


def _infer_launch_grid_from_output_size(
    *,
    launch: Mapping[str, Any],
    io_spec: Mapping[str, Any],
    merged_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(launch or {})
    grid = out.get("grid") if isinstance(out.get("grid"), list) else None
    block = out.get("block") if isinstance(out.get("block"), list) else None
    if not (isinstance(grid, list) and len(grid) == 3 and isinstance(block, list) and len(block) == 3):
        return out
    try:
        gx, gy, gz = int(grid[0]), int(grid[1]), int(grid[2])
        bx, by, bz = int(block[0]), int(block[1]), int(block[2])
    except Exception:
        return out
    if bx <= 0 or by <= 0 or bz <= 0:
        return out
    if gy != 1 or gz != 1 or by != 1 or bz != 1:
        return out

    outputs = io_spec.get("outputs") if isinstance(io_spec.get("outputs"), list) else []
    output_names = [str(x).strip() for x in outputs if str(x).strip()]
    if not output_names:
        return out
    tensors = io_spec.get("tensors") if isinstance(io_spec.get("tensors"), Mapping) else {}
    if not isinstance(tensors, Mapping):
        return out

    output_numel: int | None = None
    for name in output_names:
        spec = tensors.get(name)
        if not isinstance(spec, Mapping):
            continue
        shape = spec.get("shape")
        if not isinstance(shape, list):
            continue
        try:
            numel = 1
            for d in shape:
                numel *= int(resolve_dim_int(d, merged_bindings))
            output_numel = int(numel)
            break
        except Exception:
            continue
    if output_numel is None or output_numel <= 0:
        return out

    cur_threads = int(max(1, gx) * bx)
    needed_gx = int((output_numel + bx - 1) // bx)
    if needed_gx <= gx:
        return out
    out["grid"] = [int(needed_gx), int(gy), int(gz)]
    return out


def _repair_pad_scalar_arg_names(
    *,
    io_spec: Mapping[str, Any],
    merged_bindings: Mapping[str, Any],
    kernel_name: str,
) -> dict[str, Any]:
    out = dict(io_spec or {})
    kname = str(kernel_name or "").strip().lower()
    if "pad" not in kname:
        return out
    if ("M_OUT" not in merged_bindings) or ("N_OUT" not in merged_bindings):
        return out
    arg_names = out.get("arg_names") if isinstance(out.get("arg_names"), list) else None
    if not (isinstance(arg_names, list) and len(arg_names) >= 2):
        return out
    scalars = out.get("scalars") if isinstance(out.get("scalars"), Mapping) else {}
    scalars_out = {str(k): str(v) for k, v in dict(scalars or {}).items() if str(k).strip()}
    tensors = out.get("tensors") if isinstance(out.get("tensors"), Mapping) else {}
    tensor_names = {str(k) for k in dict(tensors or {}).keys() if str(k).strip()}
    tail0 = str(arg_names[-2]).strip()
    tail1 = str(arg_names[-1]).strip()
    if tail0 in tensor_names or tail1 in tensor_names:
        return out
    if not (
        tail0.startswith("PAD_")
        or tail1.startswith("PAD_")
        or tail0.startswith("pad_")
        or tail1.startswith("pad_")
    ):
        return out
    new_arg_names = [str(x) for x in arg_names]
    new_arg_names[-2] = "M_OUT"
    new_arg_names[-1] = "N_OUT"
    scalars_out["M_OUT"] = "i32"
    scalars_out["N_OUT"] = "i32"
    out["arg_names"] = new_arg_names
    out["scalars"] = scalars_out
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
        ptx_payload = exe_path.read_bytes()
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
        io_spec = _apply_historical_cuda_io_template(
            io_spec=io_spec,
            kernel_name=str(contract.kernel_name or ""),
        )
        io_spec = _apply_historical_cuda_io_template(
            io_spec=io_spec,
            kernel_name=str(exe_entry or contract.kernel_name or "intent"),
        )
        merged_bindings = _augment_scalar_bindings_from_io_spec(bindings=merged_bindings, io_spec=io_spec)
        io_spec = _augment_io_spec_arg_names_with_ptx_params(
            io_spec=io_spec,
            invocation=invocation,
            merged_bindings=merged_bindings,
            ptx_text=ptx_payload.decode("utf-8", errors="ignore"),
            entry=str(exe_entry or contract.kernel_name or "intent"),
        )
        merged_bindings = _augment_scalar_bindings_from_io_spec(bindings=merged_bindings, io_spec=io_spec)
        io_spec = _repair_pad_scalar_arg_names(
            io_spec=io_spec,
            merged_bindings=merged_bindings,
            kernel_name=str(exe_entry or contract.kernel_name or ""),
        )
        output_names = list((io_spec.get("outputs") if isinstance(io_spec, Mapping) else []) or [])
        inv_outputs = invocation.get("output_names")
        if (not output_names) and isinstance(inv_outputs, list):
            output_names = [str(x) for x in list(inv_outputs)]
        launch = dict(contract.launch or {})
        if (not launch) and isinstance(invocation.get("launch"), Mapping):
            launch = dict(invocation.get("launch") or {})
        launch = _ensure_launch_grid_block_defaults(
            launch=launch,
            ptx_text=ptx_payload.decode("utf-8", errors="ignore"),
        )
        launch = _infer_launch_grid_x_from_ptx(
            launch=launch,
            io_spec=io_spec,
            merged_bindings=merged_bindings,
            ptx_text=ptx_payload.decode("utf-8", errors="ignore"),
            entry=str(exe_entry or contract.kernel_name or "intent"),
        )
        launch = _infer_launch_grid_from_output_size(
            launch=launch,
            io_spec=io_spec,
            merged_bindings=merged_bindings,
        )
        return {
            "kernel_name": str(exe_entry or contract.kernel_name or "intent"),
            "io_spec": io_spec,
            "launch": launch,
            "output_names": [str(x) for x in output_names],
            "bindings": dict(merged_bindings),
            "cuda_ptx": ptx_payload,
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
