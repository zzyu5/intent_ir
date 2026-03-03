from __future__ import annotations

import base64
import json
import re
from typing import Any, Mapping

from intent_ir.ir import IntentFunction
from intent_ir.ir.repair import materialize_missing_op_output_tensors
from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.mlir.module import IntentMLIRModule


def _resolve_dim_int(dim: Any, bindings: Mapping[str, Any]) -> int | None:
    if dim is None:
        return None
    kind = getattr(dim, "kind", None)
    raw = getattr(dim, "value", dim) if kind in {"sym", "const"} else dim
    if isinstance(raw, int):
        return int(raw)
    key = str(raw).strip()
    if not key:
        return None
    if key in bindings:
        try:
            return int(bindings[key])
        except Exception:
            return None
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\+\s*(\d+)$", key)
    if m:
        base = str(m.group(1))
        delta = int(m.group(2))
        if base in bindings:
            try:
                return int(bindings[base]) + int(delta)
            except Exception:
                return None
    try:
        return int(key)
    except Exception:
        return None


def _mlir_ident(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return "v"
    out = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s)
    if out and out[0].isdigit():
        out = f"v_{out}"
    return out or "v"


def _b64_json(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _dtype(dt: str) -> str:
    s = str(dt or "").strip().lower()
    if s in {"f32", "f16"}:
        return s
    if s in {"bool", "i1"}:
        # Baseline bundles stage bool tensors as bytes; use i8 in the ABI.
        return "i8"
    if s in {"u8", "i8"}:
        return "i8"
    if s in {"i16"}:
        return "i16"
    if s in {"i32"}:
        return "i32"
    if s in {"i64"}:
        return "i64"
    raise RuntimeError(f"rvv cpu-loops v1 supports only f16/f32/bool/u8/i8/i16/i32/i64, got dtype={dt!r}")


def _shape(name: str, *, intent: IntentFunction, bindings: Mapping[str, Any]) -> list[int]:
    tt = (intent.tensors or {}).get(str(name))
    if tt is None:
        return []
    dims: list[int] = []
    for d in list(getattr(tt, "shape", []) or []):
        v = _resolve_dim_int(d, bindings)
        if v is None:
            raise RuntimeError(f"unbound dim for tensor={name} dim={d}")
        dims.append(int(v))
    return dims


def _io_arg_order(intent: IntentFunction) -> list[str]:
    tensors = dict(intent.tensors or {})
    ops = [op for op in list(intent.ops or []) if op is not None]
    outputs = [str(x).strip() for x in list(intent.outputs or []) if str(x).strip()]
    produced = {str(getattr(op, "output", "")).strip() for op in ops if str(getattr(op, "output", "")).strip()}
    used = {str(x).strip() for op in ops for x in list(getattr(op, "inputs", []) or []) if str(x).strip()}
    external_inputs = sorted([n for n in used if n in tensors and n not in produced])
    ext_set = set(external_inputs)
    return list(external_inputs) + [n for n in outputs if n in tensors and n not in ext_set]


def _const_scalar_value(value: Any, *, dtype: str, bindings: Mapping[str, Any]) -> float | int | None:
    dt = str(dtype or "").strip().lower()
    if dt == "f32":
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip()
        if s in bindings:
            try:
                return float(bindings[s])
            except Exception:
                return None
        try:
            return float(s)
        except Exception:
            return None
    if dt == "i32":
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and float(value).is_integer():
            return int(value)
        s = str(value).strip()
        if s in bindings:
            try:
                return int(bindings[s])
            except Exception:
                return None
        try:
            return int(s)
        except Exception:
            return None
    return None


def _f32_lit(value: float) -> str:
    s = repr(float(value))
    if s.lower() in {"nan", "inf", "-inf"}:
        raise RuntimeError(f"unsupported f32 literal: {s}")
    if "e" in s or "E" in s:
        head, exp = (s.split("e", 1) if "e" in s else s.split("E", 1))
        if "." not in head:
            head = f"{head}.0"
        return f"{head}e{exp}"
    if "." not in s:
        return f"{s}.0"
    return s


def _find_const_scalar(
    *,
    intent: IntentFunction,
    output: str,
    dtype: str,
    bindings: Mapping[str, Any],
) -> float | int | None:
    out_name = str(output).strip()
    if not out_name:
        return None
    for op in list(intent.ops or []):
        if op is None:
            continue
        op_name = str(getattr(op, "op", "")).strip()
        if op_name != "const":
            continue
        out = str(getattr(op, "output", "")).strip()
        if out != out_name:
            continue
        attrs = dict(getattr(op, "attrs", {}) or {})
        dt = str(attrs.get("dtype") or dtype or "").strip().lower()
        return _const_scalar_value(attrs.get("value"), dtype=dt, bindings=bindings)
    return None


def _emit_rms_norm2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x).strip() for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 rms_norm2d expects two outputs, got outputs={outputs}")
    out0, out1 = outputs
    out0_shape = _shape(out0, intent=intent, bindings=bindings)
    out1_shape = _shape(out1, intent=intent, bindings=bindings)
    if len(out0_shape) == 2 and len(out1_shape) == 1:
        out_2d, out_inv = out0, out1
        out_shape, inv_shape = out0_shape, out1_shape
    elif len(out1_shape) == 2 and len(out0_shape) == 1:
        out_2d, out_inv = out1, out0
        out_shape, inv_shape = out1_shape, out0_shape
    else:
        raise RuntimeError(f"rvv cpu-loops v1 rms_norm2d expects [M,N] and [M] outputs, got {out0_shape} and {out1_shape}")
    if len(out_shape) != 2 or len(inv_shape) != 1:
        raise RuntimeError("rvv cpu-loops v1 rms_norm2d expects rank-2 output and rank-1 INV_RMS")
    m_dim, n_dim = int(out_shape[0]), int(out_shape[1])
    if m_dim <= 0 or n_dim <= 0 or int(inv_shape[0]) != int(m_dim):
        raise RuntimeError(f"rvv cpu-loops v1 rms_norm2d shape mismatch: out={out_shape} inv={inv_shape}")

    io_names = _io_arg_order(intent)
    if not io_names:
        raise RuntimeError("rvv cpu-loops v1 rms_norm2d missing io tensors")
    ext_inputs = [n for n in io_names if n not in set(outputs)]

    def _find_by_shape(target: list[int]) -> str | None:
        for name in ext_inputs:
            if list(_shape(name, intent=intent, bindings=bindings)) == list(target):
                return str(name)
        return None

    inp_name = _find_by_shape([m_dim, n_dim])
    w_name = _find_by_shape([n_dim])
    if inp_name is None or w_name is None:
        raise RuntimeError("rvv cpu-loops v1 rms_norm2d expects external input[M,N] and weight[N] tensors")

    eps_val = _find_const_scalar(intent=intent, output="eps", dtype="f32", bindings=bindings)
    if eps_val is None:
        eps_val = _find_const_scalar(intent=intent, output="epsilon", dtype="f32", bindings=bindings)
    if eps_val is None:
        # Conservative default; should not happen for the standard rms_norm2d seed.
        eps_val = 1e-5

    n_scalar_val = _find_const_scalar(intent=intent, output="N_scalar", dtype="f32", bindings=bindings)
    if n_scalar_val is None:
        n_scalar_val = float(n_dim)

    arg_decls: list[str] = []
    arg_ssa: dict[str, str] = {}
    memref_ty_by_name: dict[str, str] = {}
    for name in io_names:
        tt = (intent.tensors or {}).get(name)
        if tt is None:
            raise RuntimeError(f"missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "f32"))
        if elem_ty != "f32":
            raise RuntimeError(f"rvv cpu-loops v1 rms_norm2d supports only f32 tensors, got {name} dtype={elem_ty!r}")
        sh = _shape(name, intent=intent, bindings=bindings)
        numel = 1
        for d in sh:
            numel *= int(d)
        memref_ty = f"memref<{int(numel)}x{elem_ty}>"
        ssa = f"%{_mlir_ident(name)}"
        arg_ssa[name] = ssa
        memref_ty_by_name[name] = memref_ty
        arg_decls.append(f"    {ssa}: {memref_ty}")

    inp_memref = memref_ty_by_name[inp_name]
    w_memref = memref_ty_by_name[w_name]
    out_memref = memref_ty_by_name[out_2d]
    inv_memref = memref_ty_by_name[out_inv]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}(")
    lines.append(",\n".join(arg_decls))
    lines.append("  ) {")
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %c1 = arith.constant 1 : index")
    lines.append(f"  %cM = arith.constant {m_dim} : index")
    lines.append(f"  %cN = arith.constant {n_dim} : index")
    lines.append("  %c0f = arith.constant 0.0 : f32")
    lines.append("  %c1f = arith.constant 1.0 : f32")
    lines.append(f"  %eps = arith.constant {_f32_lit(float(eps_val))} : f32")
    lines.append(f"  %n_scalar = arith.constant {_f32_lit(float(n_scalar_val))} : f32")
    lines.append("  scf.for %m = %c0 to %cM step %c1 {")
    lines.append("    %base = arith.muli %m, %cN : index")
    lines.append("    %sum_sq = scf.for %n = %c0 to %cN step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("      %i = arith.addi %base, %n : index")
    lines.append(f"      %x = memref.load {arg_ssa[inp_name]}[%i] : {inp_memref}")
    lines.append("      %x2 = arith.mulf %x, %x : f32")
    lines.append("      %acc2 = arith.addf %acc, %x2 : f32")
    lines.append("      scf.yield %acc2 : f32")
    lines.append("    }")
    lines.append("    %mean_sq = arith.divf %sum_sq, %n_scalar : f32")
    lines.append("    %var_eps = arith.addf %mean_sq, %eps : f32")
    lines.append("    %inv = math.rsqrt %var_eps : f32")
    lines.append(f"    memref.store %inv, {arg_ssa[out_inv]}[%m] : {inv_memref}")
    lines.append("    scf.for %n2 = %c0 to %cN step %c1 {")
    lines.append("      %i2 = arith.addi %base, %n2 : index")
    lines.append(f"      %xv = memref.load {arg_ssa[inp_name]}[%i2] : {inp_memref}")
    lines.append(f"      %w = memref.load {arg_ssa[w_name]}[%n2] : {w_memref}")
    lines.append("      %x_norm = arith.mulf %xv, %inv : f32")
    lines.append("      %y = arith.mulf %x_norm, %w : f32")
    lines.append(f"      memref.store %y, {arg_ssa[out_2d]}[%i2] : {out_memref}")
    lines.append("    }")
    lines.append("  }")
    lines.append("  return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_rms_norm_residual2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x).strip() for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 rms_norm_residual2d expects two outputs, got outputs={outputs}")
    out0, out1 = outputs
    out0_shape = _shape(out0, intent=intent, bindings=bindings)
    out1_shape = _shape(out1, intent=intent, bindings=bindings)
    if len(out0_shape) == 2 and len(out1_shape) == 1:
        out_2d, out_rstd = out0, out1
        out_shape, rstd_shape = out0_shape, out1_shape
    elif len(out1_shape) == 2 and len(out0_shape) == 1:
        out_2d, out_rstd = out1, out0
        out_shape, rstd_shape = out1_shape, out0_shape
    else:
        raise RuntimeError(
            f"rvv cpu-loops v1 rms_norm_residual2d expects [M,N] and [M] outputs, got {out0_shape} and {out1_shape}"
        )
    m_dim, n_dim = int(out_shape[0]), int(out_shape[1])
    if m_dim <= 0 or n_dim <= 0 or int(rstd_shape[0]) != int(m_dim):
        raise RuntimeError(f"rvv cpu-loops v1 rms_norm_residual2d shape mismatch: out={out_shape} rstd={rstd_shape}")

    io_names = _io_arg_order(intent)
    ext_inputs = [n for n in io_names if n not in set(outputs)]

    def _name_key(raw: str) -> str:
        return str(raw or "").strip().lower().replace("-", "_")

    rank2_mn = [n for n in ext_inputs if list(_shape(n, intent=intent, bindings=bindings)) == [m_dim, n_dim]]
    rank1_n = [n for n in ext_inputs if list(_shape(n, intent=intent, bindings=bindings)) == [n_dim]]

    inp_name = next((n for n in rank2_mn if _name_key(n) in {"inp", "input", "x"} or "inp" in _name_key(n)), "")
    residual_name = next((n for n in rank2_mn if "resid" in _name_key(n)), "")
    if not inp_name:
        for n in list(rank2_mn):
            if n != residual_name:
                inp_name = str(n)
                break
    if not residual_name:
        for n in list(rank2_mn):
            if n != inp_name:
                residual_name = str(n)
                break
    if not (inp_name and residual_name):
        raise RuntimeError("rvv cpu-loops v1 rms_norm_residual2d expects inp[M,N] and residual[M,N] inputs")

    weight_name = next((n for n in rank1_n if _name_key(n) in {"weight", "w", "weight_ptr", "gamma"}), "")
    bias_name = next((n for n in rank1_n if _name_key(n) in {"bias", "b", "bias_ptr", "beta"}), "")
    if (not weight_name) or (not bias_name):
        # Deterministic fallback: prefer bias first (matches many frontends).
        ordered = sorted([str(x) for x in list(rank1_n)])
        if len(ordered) >= 2:
            bias_name = bias_name or ordered[0]
            weight_name = weight_name or ordered[1]
    if not (weight_name and bias_name):
        raise RuntimeError("rvv cpu-loops v1 rms_norm_residual2d expects weight[N] and bias[N] inputs")

    eps_val = _find_const_scalar(intent=intent, output="eps", dtype="f32", bindings=bindings)
    if eps_val is None:
        eps_val = _find_const_scalar(intent=intent, output="epsilon", dtype="f32", bindings=bindings)
    if eps_val is None:
        eps_val = 1.0e-5
    n_scalar_val = _find_const_scalar(intent=intent, output="N_scalar", dtype="f32", bindings=bindings)
    if n_scalar_val is None:
        n_scalar_val = float(n_dim)

    arg_decls: list[str] = []
    arg_ssa: dict[str, str] = {}
    memref_ty_by_name: dict[str, str] = {}
    for name in io_names:
        tt = (intent.tensors or {}).get(name)
        if tt is None:
            raise RuntimeError(f"missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "f32"))
        if elem_ty != "f32":
            raise RuntimeError(
                f"rvv cpu-loops v1 rms_norm_residual2d supports only f32 tensors, got {name} dtype={elem_ty!r}"
            )
        sh = _shape(name, intent=intent, bindings=bindings)
        numel = 1
        for d in sh:
            numel *= int(d)
        memref_ty = f"memref<{int(numel)}x{elem_ty}>"
        ssa = f"%{_mlir_ident(name)}"
        arg_ssa[name] = ssa
        memref_ty_by_name[name] = memref_ty
        arg_decls.append(f"    {ssa}: {memref_ty}")

    inp_memref = memref_ty_by_name[inp_name]
    res_memref = memref_ty_by_name[residual_name]
    w_memref = memref_ty_by_name[weight_name]
    b_memref = memref_ty_by_name[bias_name]
    out_memref = memref_ty_by_name[out_2d]
    rstd_memref = memref_ty_by_name[out_rstd]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}(")
    lines.append(",\n".join(arg_decls))
    lines.append("  ) {")
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %c1 = arith.constant 1 : index")
    lines.append(f"  %cM = arith.constant {m_dim} : index")
    lines.append(f"  %cN = arith.constant {n_dim} : index")
    lines.append("  %c0f = arith.constant 0.0 : f32")
    lines.append(f"  %cNf = arith.constant {_f32_lit(float(n_scalar_val))} : f32")
    lines.append(f"  %eps = arith.constant {_f32_lit(float(eps_val))} : f32")
    lines.append("  scf.for %m = %c0 to %cM step %c1 {")
    lines.append("    %base = arith.muli %m, %cN : index")
    lines.append("    %sum_sq = scf.for %n = %c0 to %cN step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("      %i = arith.addi %base, %n : index")
    lines.append(f"      %x = memref.load {arg_ssa[inp_name]}[%i] : {inp_memref}")
    lines.append(f"      %r = memref.load {arg_ssa[residual_name]}[%i] : {res_memref}")
    lines.append(f"      %b = memref.load {arg_ssa[bias_name]}[%n] : {b_memref}")
    lines.append("      %z0 = arith.addf %x, %r : f32")
    lines.append("      %z = arith.addf %z0, %b : f32")
    lines.append("      %z2 = arith.mulf %z, %z : f32")
    lines.append("      %acc2 = arith.addf %acc, %z2 : f32")
    lines.append("      scf.yield %acc2 : f32")
    lines.append("    }")
    lines.append("    %mean_sq = arith.divf %sum_sq, %cNf : f32")
    lines.append("    %var_eps = arith.addf %mean_sq, %eps : f32")
    lines.append("    %rstd_val = math.rsqrt %var_eps : f32")
    lines.append(f"    memref.store %rstd_val, {arg_ssa[out_rstd]}[%m] : {rstd_memref}")
    lines.append("    scf.for %n2 = %c0 to %cN step %c1 {")
    lines.append("      %i2 = arith.addi %base, %n2 : index")
    lines.append(f"      %xv = memref.load {arg_ssa[inp_name]}[%i2] : {inp_memref}")
    lines.append(f"      %rv = memref.load {arg_ssa[residual_name]}[%i2] : {res_memref}")
    lines.append(f"      %bv = memref.load {arg_ssa[bias_name]}[%n2] : {b_memref}")
    lines.append(f"      %w = memref.load {arg_ssa[weight_name]}[%n2] : {w_memref}")
    lines.append("      %zv0 = arith.addf %xv, %rv : f32")
    lines.append("      %zv = arith.addf %zv0, %bv : f32")
    lines.append("      %zn = arith.mulf %zv, %rstd_val : f32")
    lines.append("      %y = arith.mulf %zn, %w : f32")
    lines.append(f"      memref.store %y, {arg_ssa[out_2d]}[%i2] : {out_memref}")
    lines.append("    }")
    lines.append("  }")
    lines.append("  return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_layer_norm_persistent_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x).strip() for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 3:
        raise RuntimeError(f"rvv cpu-loops v1 layer_norm_persistent expects three outputs, got outputs={outputs}")

    shapes = {n: _shape(n, intent=intent, bindings=bindings) for n in outputs}
    out_2d = next((n for n, sh in shapes.items() if len(sh) == 2), "")
    if not out_2d:
        raise RuntimeError(f"rvv cpu-loops v1 layer_norm_persistent missing rank-2 output in outputs={outputs}")
    m_dim, n_dim = map(int, shapes[out_2d])
    out_mean = next((n for n in outputs if n != out_2d and len(shapes.get(n) or []) == 1), "")
    out_rstd = next((n for n in outputs if n not in {out_2d, out_mean} and len(shapes.get(n) or []) == 1), "")
    if not out_mean or not out_rstd:
        raise RuntimeError("rvv cpu-loops v1 layer_norm_persistent expects rank-1 mean/rstd outputs")
    if list(shapes.get(out_mean) or []) != [m_dim] or list(shapes.get(out_rstd) or []) != [m_dim]:
        raise RuntimeError(
            f"rvv cpu-loops v1 layer_norm_persistent expects mean/rstd shape [M], got mean={shapes.get(out_mean)} rstd={shapes.get(out_rstd)}"
        )

    io_names = _io_arg_order(intent)
    ext_inputs = [n for n in io_names if n not in set(outputs)]

    def _find_by_shape(target: list[int]) -> str | None:
        for name in ext_inputs:
            if list(_shape(name, intent=intent, bindings=bindings)) == list(target):
                return str(name)
        return None

    inp_name = _find_by_shape([m_dim, n_dim])
    if inp_name is None:
        raise RuntimeError("rvv cpu-loops v1 layer_norm_persistent expects external input[M,N] tensor")
    rank1_names = [n for n in ext_inputs if list(_shape(n, intent=intent, bindings=bindings)) == [n_dim]]

    def _name_key(raw: str) -> str:
        return str(raw or "").strip().lower().replace("-", "_")

    weight_name = next(
        (n for n in rank1_names if _name_key(n) in {"weight", "w", "weight_ptr", "gamma"}), None
    )
    bias_name = next((n for n in rank1_names if _name_key(n) in {"bias", "b", "bias_ptr", "beta"}), None)
    if weight_name is None or bias_name is None:
        if len(rank1_names) >= 2:
            # Heuristic: bias tends to sort before weight (e.g. bias_ptr < weight_ptr).
            bias_name = str(rank1_names[0])
            weight_name = str(rank1_names[1])
    if weight_name is None or bias_name is None:
        raise RuntimeError("rvv cpu-loops v1 layer_norm_persistent expects external weight[N] and bias[N] tensors")

    eps_const = _find_const_scalar(intent=intent, output="eps", dtype="f32", bindings=bindings)
    if eps_const is None:
        eps_const = 1.0e-5
    n_f = float(n_dim)

    arg_decls: list[str] = []
    arg_ssa: dict[str, str] = {}
    memref_ty_by_name: dict[str, str] = {}
    for name in io_names:
        tt = (intent.tensors or {}).get(name)
        if tt is None:
            raise RuntimeError(f"missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "f32"))
        if elem_ty != "f32":
            raise RuntimeError(
                f"rvv cpu-loops v1 layer_norm_persistent supports only f32 tensors, got {name} dtype={elem_ty!r}"
            )
        sh = _shape(name, intent=intent, bindings=bindings)
        numel = 1
        for d in sh:
            numel *= int(d)
        memref_ty = f"memref<{int(numel)}x{elem_ty}>"
        ssa = f"%{_mlir_ident(name)}"
        arg_ssa[name] = ssa
        memref_ty_by_name[name] = memref_ty
        arg_decls.append(f"    {ssa}: {memref_ty}")

    inp_memref = memref_ty_by_name[inp_name]
    w_memref = memref_ty_by_name[weight_name]
    b_memref = memref_ty_by_name[bias_name]
    out_memref = memref_ty_by_name[out_2d]
    mean_memref = memref_ty_by_name[out_mean]
    rstd_memref = memref_ty_by_name[out_rstd]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}(")
    lines.append(",\n".join(arg_decls))
    lines.append("  ) {")
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %c1 = arith.constant 1 : index")
    lines.append(f"  %cM = arith.constant {m_dim} : index")
    lines.append(f"  %cN = arith.constant {n_dim} : index")
    lines.append("  %c0f = arith.constant 0.0 : f32")
    lines.append(f"  %cNf = arith.constant {_f32_lit(n_f)} : f32")
    lines.append(f"  %eps = arith.constant {_f32_lit(float(eps_const))} : f32")
    lines.append("  scf.for %m = %c0 to %cM step %c1 {")
    lines.append("    %base = arith.muli %m, %cN : index")
    lines.append("    %sum = scf.for %n = %c0 to %cN step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("      %i = arith.addi %base, %n : index")
    lines.append(f"      %x = memref.load {arg_ssa[inp_name]}[%i] : {inp_memref}")
    lines.append("      %acc2 = arith.addf %acc, %x : f32")
    lines.append("      scf.yield %acc2 : f32")
    lines.append("    }")
    lines.append("    %mean_val = arith.divf %sum, %cNf : f32")
    lines.append(f"    memref.store %mean_val, {arg_ssa[out_mean]}[%m] : {mean_memref}")
    lines.append("    %sum_sq = scf.for %n2 = %c0 to %cN step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("      %i2 = arith.addi %base, %n2 : index")
    lines.append(f"      %x2 = memref.load {arg_ssa[inp_name]}[%i2] : {inp_memref}")
    lines.append("      %d = arith.subf %x2, %mean_val : f32")
    lines.append("      %d2 = arith.mulf %d, %d : f32")
    lines.append("      %acc3 = arith.addf %acc, %d2 : f32")
    lines.append("      scf.yield %acc3 : f32")
    lines.append("    }")
    lines.append("    %var = arith.divf %sum_sq, %cNf : f32")
    lines.append("    %var_eps = arith.addf %var, %eps : f32")
    lines.append("    %rstd = math.rsqrt %var_eps : f32")
    lines.append(f"    memref.store %rstd, {arg_ssa[out_rstd]}[%m] : {rstd_memref}")
    lines.append("    scf.for %n3 = %c0 to %cN step %c1 {")
    lines.append("      %i3 = arith.addi %base, %n3 : index")
    lines.append(f"      %x3 = memref.load {arg_ssa[inp_name]}[%i3] : {inp_memref}")
    lines.append("      %d3 = arith.subf %x3, %mean_val : f32")
    lines.append("      %norm = arith.mulf %d3, %rstd : f32")
    lines.append(f"      %w = memref.load {arg_ssa[weight_name]}[%n3] : {w_memref}")
    lines.append(f"      %b = memref.load {arg_ssa[bias_name]}[%n3] : {b_memref}")
    lines.append("      %scaled = arith.mulf %norm, %w : f32")
    lines.append("      %y = arith.addf %scaled, %b : f32")
    lines.append(f"      memref.store %y, {arg_ssa[out_2d]}[%i3] : {out_memref}")
    lines.append("    }")
    lines.append("  }")
    lines.append("  return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_layer_norm_residual2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x).strip() for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 3:
        raise RuntimeError(f"rvv cpu-loops v1 layer_norm_residual2d expects three outputs, got outputs={outputs}")

    shapes = {n: _shape(n, intent=intent, bindings=bindings) for n in outputs}
    out_2d = next((n for n, sh in shapes.items() if len(sh) == 2), "")
    if not out_2d:
        raise RuntimeError(f"rvv cpu-loops v1 layer_norm_residual2d missing rank-2 output in outputs={outputs}")
    m_dim, n_dim = map(int, shapes[out_2d])
    out_mean = next((n for n in outputs if n != out_2d and len(shapes.get(n) or []) == 1), "")
    out_rstd = next((n for n in outputs if n not in {out_2d, out_mean} and len(shapes.get(n) or []) == 1), "")
    if not out_mean or not out_rstd:
        raise RuntimeError("rvv cpu-loops v1 layer_norm_residual2d expects rank-1 mean/rstd outputs")
    if list(shapes.get(out_mean) or []) != [m_dim] or list(shapes.get(out_rstd) or []) != [m_dim]:
        raise RuntimeError("rvv cpu-loops v1 layer_norm_residual2d expects mean/rstd shape [M]")

    io_names = _io_arg_order(intent)
    ext_inputs = [n for n in io_names if n not in set(outputs)]

    def _name_key(raw: str) -> str:
        return str(raw or "").strip().lower().replace("-", "_")

    rank2_mn = [n for n in ext_inputs if list(_shape(n, intent=intent, bindings=bindings)) == [m_dim, n_dim]]
    rank1_n = [n for n in ext_inputs if list(_shape(n, intent=intent, bindings=bindings)) == [n_dim]]
    if len(rank2_mn) < 2:
        raise RuntimeError("rvv cpu-loops v1 layer_norm_residual2d expects inp[M,N] and residual[M,N] inputs")

    inp_name = next((n for n in rank2_mn if _name_key(n) in {"inp", "input", "x"} or "inp" in _name_key(n)), "")
    residual_name = next((n for n in rank2_mn if "resid" in _name_key(n)), "")
    if not inp_name:
        for n in list(rank2_mn):
            if n != residual_name:
                inp_name = str(n)
                break
    if not residual_name:
        for n in list(rank2_mn):
            if n != inp_name:
                residual_name = str(n)
                break

    weight_name = next((n for n in rank1_n if _name_key(n) in {"weight", "w", "weight_ptr", "gamma"}), "")
    bias_name = next((n for n in rank1_n if _name_key(n) in {"bias", "b", "bias_ptr", "beta"}), "")
    if (not weight_name) or (not bias_name):
        ordered = sorted([str(x) for x in list(rank1_n)])
        if len(ordered) >= 2:
            bias_name = bias_name or ordered[0]
            weight_name = weight_name or ordered[1]
    if not (inp_name and residual_name and weight_name and bias_name):
        raise RuntimeError("rvv cpu-loops v1 layer_norm_residual2d failed to infer inputs")

    eps_val = _find_const_scalar(intent=intent, output="eps", dtype="f32", bindings=bindings)
    if eps_val is None:
        eps_val = _find_const_scalar(intent=intent, output="epsilon", dtype="f32", bindings=bindings)
    if eps_val is None:
        eps_val = 1.0e-5
    n_scalar_val = _find_const_scalar(intent=intent, output="N_scalar", dtype="f32", bindings=bindings)
    if n_scalar_val is None:
        n_scalar_val = float(n_dim)

    arg_decls: list[str] = []
    arg_ssa: dict[str, str] = {}
    memref_ty_by_name: dict[str, str] = {}
    for name in io_names:
        tt = (intent.tensors or {}).get(name)
        if tt is None:
            raise RuntimeError(f"missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "f32"))
        if elem_ty != "f32":
            raise RuntimeError(
                f"rvv cpu-loops v1 layer_norm_residual2d supports only f32 tensors, got {name} dtype={elem_ty!r}"
            )
        sh = _shape(name, intent=intent, bindings=bindings)
        numel = 1
        for d in sh:
            numel *= int(d)
        memref_ty = f"memref<{int(numel)}x{elem_ty}>"
        ssa = f"%{_mlir_ident(name)}"
        arg_ssa[name] = ssa
        memref_ty_by_name[name] = memref_ty
        arg_decls.append(f"    {ssa}: {memref_ty}")

    inp_memref = memref_ty_by_name[inp_name]
    res_memref = memref_ty_by_name[residual_name]
    w_memref = memref_ty_by_name[weight_name]
    b_memref = memref_ty_by_name[bias_name]
    out_memref = memref_ty_by_name[out_2d]
    mean_memref = memref_ty_by_name[out_mean]
    rstd_memref = memref_ty_by_name[out_rstd]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}(")
    lines.append(",\n".join(arg_decls))
    lines.append("  ) {")
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %c1 = arith.constant 1 : index")
    lines.append(f"  %cM = arith.constant {m_dim} : index")
    lines.append(f"  %cN = arith.constant {n_dim} : index")
    lines.append("  %c0f = arith.constant 0.0 : f32")
    lines.append(f"  %cNf = arith.constant {_f32_lit(float(n_scalar_val))} : f32")
    lines.append(f"  %eps = arith.constant {_f32_lit(float(eps_val))} : f32")
    lines.append("  scf.for %m = %c0 to %cM step %c1 {")
    lines.append("    %base = arith.muli %m, %cN : index")
    lines.append("    %sum = scf.for %n = %c0 to %cN step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("      %i = arith.addi %base, %n : index")
    lines.append(f"      %x = memref.load {arg_ssa[inp_name]}[%i] : {inp_memref}")
    lines.append(f"      %r = memref.load {arg_ssa[residual_name]}[%i] : {res_memref}")
    lines.append("      %z = arith.addf %x, %r : f32")
    lines.append("      %acc2 = arith.addf %acc, %z : f32")
    lines.append("      scf.yield %acc2 : f32")
    lines.append("    }")
    lines.append("    %mean_val = arith.divf %sum, %cNf : f32")
    lines.append(f"    memref.store %mean_val, {arg_ssa[out_mean]}[%m] : {mean_memref}")
    lines.append("    %sum_sq = scf.for %n2 = %c0 to %cN step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("      %i2 = arith.addi %base, %n2 : index")
    lines.append(f"      %x2 = memref.load {arg_ssa[inp_name]}[%i2] : {inp_memref}")
    lines.append(f"      %r2 = memref.load {arg_ssa[residual_name]}[%i2] : {res_memref}")
    lines.append("      %z2 = arith.addf %x2, %r2 : f32")
    lines.append("      %d = arith.subf %z2, %mean_val : f32")
    lines.append("      %d2 = arith.mulf %d, %d : f32")
    lines.append("      %acc3 = arith.addf %acc, %d2 : f32")
    lines.append("      scf.yield %acc3 : f32")
    lines.append("    }")
    lines.append("    %var = arith.divf %sum_sq, %cNf : f32")
    lines.append("    %var_eps = arith.addf %var, %eps : f32")
    lines.append("    %rstd_val = math.rsqrt %var_eps : f32")
    lines.append(f"    memref.store %rstd_val, {arg_ssa[out_rstd]}[%m] : {rstd_memref}")
    lines.append("    scf.for %n3 = %c0 to %cN step %c1 {")
    lines.append("      %i3 = arith.addi %base, %n3 : index")
    lines.append(f"      %x3 = memref.load {arg_ssa[inp_name]}[%i3] : {inp_memref}")
    lines.append(f"      %r3 = memref.load {arg_ssa[residual_name]}[%i3] : {res_memref}")
    lines.append("      %z3 = arith.addf %x3, %r3 : f32")
    lines.append("      %d3 = arith.subf %z3, %mean_val : f32")
    lines.append("      %norm = arith.mulf %d3, %rstd_val : f32")
    lines.append(f"      %w = memref.load {arg_ssa[weight_name]}[%n3] : {w_memref}")
    lines.append(f"      %b = memref.load {arg_ssa[bias_name]}[%n3] : {b_memref}")
    lines.append("      %scaled = arith.mulf %norm, %w : f32")
    lines.append("      %y = arith.addf %scaled, %b : f32")
    lines.append(f"      memref.store %y, {arg_ssa[out_2d]}[%i3] : {out_memref}")
    lines.append("    }")
    lines.append("  }")
    lines.append("  return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_mlp2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x).strip() for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 mlp2d expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 mlp2d expects rank-2 output, got {out_shape}")
    m_dim, n_dim = int(out_shape[0]), int(out_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    mm_ops = [op for op in ops if str(getattr(op, "op", "")).strip() == "matmul"]
    if len(mm_ops) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 mlp2d expects exactly 2 matmul ops, got {len(mm_ops)}")
    mm0, mm1 = mm_ops
    mm0_ins = [str(x).strip() for x in list(getattr(mm0, "inputs", []) or []) if str(x).strip()]
    mm1_ins = [str(x).strip() for x in list(getattr(mm1, "inputs", []) or []) if str(x).strip()]
    if len(mm0_ins) != 2 or len(mm1_ins) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 mlp2d expects matmul inputs of length 2, got {mm0_ins} and {mm1_ins}")
    a_name, w1_name = mm0_ins
    hidden_name, w2_name = mm1_ins

    a_shape = _shape(a_name, intent=intent, bindings=bindings)
    w1_shape = _shape(w1_name, intent=intent, bindings=bindings)
    w2_shape = _shape(w2_name, intent=intent, bindings=bindings)
    if len(a_shape) != 2 or len(w1_shape) != 2 or len(w2_shape) != 2:
        raise RuntimeError("rvv cpu-loops v1 mlp2d expects rank-2 A/W1/W2 tensors")
    if int(a_shape[0]) != int(m_dim):
        raise RuntimeError(f"rvv cpu-loops v1 mlp2d expects A[M,K], got {a_name} shape={a_shape} for M={m_dim}")
    if int(w2_shape[1]) != int(n_dim):
        raise RuntimeError(f"rvv cpu-loops v1 mlp2d expects W2[H,N], got {w2_name} shape={w2_shape} for N={n_dim}")
    k_dim = int(a_shape[1])
    h_dim = int(w2_shape[0])
    if k_dim <= 0 or h_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 mlp2d invalid K/H inferred: K={k_dim} H={h_dim}")
    if [int(x) for x in w1_shape] != [int(k_dim), int(h_dim)]:
        raise RuntimeError(
            f"rvv cpu-loops v1 mlp2d expects W1[K,H], got {w1_name} shape={w1_shape} (K={k_dim},H={h_dim})"
        )

    b1_name = ""
    b2_name = ""

    # Prefer graph-based identification over shape heuristics (H can equal N, making shapes ambiguous).
    tensors = dict(intent.tensors or {})
    produced = {str(getattr(op, "output", "")).strip() for op in ops if str(getattr(op, "output", "")).strip()}
    op_by_output: dict[str, Any] = {str(getattr(op, "output", "")).strip(): op for op in ops if str(getattr(op, "output", "")).strip()}

    def _op_name(op: Any) -> str:
        return str(getattr(op, "op", "")).strip()

    def _op_inputs(op: Any) -> list[str]:
        return [str(x).strip() for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]

    def _is_external(name: str) -> bool:
        return str(name) in tensors and str(name) not in produced

    def _resolve_bias_input(name: str, *, expected_len: int) -> str | None:
        nm = str(name).strip()
        if not nm:
            return None
        op = op_by_output.get(nm)
        if op is not None and _op_name(op) == "broadcast_in_dim":
            b_ins = _op_inputs(op)
            if len(b_ins) == 1:
                return str(b_ins[0])
            return None
        if _is_external(nm) and _shape(nm, intent=intent, bindings=bindings) == [int(expected_len)]:
            return nm
        return None

    # b1: resolve from the pre-activation that feeds into `hidden` (max/relu output).
    op_hidden = op_by_output.get(str(hidden_name))
    pre_act = ""
    if op_hidden is not None:
        hn = _op_name(op_hidden)
        if hn == "relu":
            ins = _op_inputs(op_hidden)
            if len(ins) == 1:
                pre_act = str(ins[0])
        elif hn == "max":
            ins = _op_inputs(op_hidden)
            pre_act = next((n for n in ins if n in op_by_output and _op_name(op_by_output.get(n)) == "add"), "")
    if pre_act:
        op_add = op_by_output.get(str(pre_act))
        if op_add is not None and _op_name(op_add) == "add":
            for n in _op_inputs(op_add):
                cand = _resolve_bias_input(n, expected_len=h_dim)
                if cand:
                    b1_name = str(cand)
                    break

    # b2: resolve from the final add that produces the output.
    op_out = op_by_output.get(str(out_name))
    if op_out is not None and _op_name(op_out) == "add":
        for n in _op_inputs(op_out):
            cand = _resolve_bias_input(n, expected_len=n_dim)
            if cand:
                b2_name = str(cand)
                break

    if not (b1_name and b2_name):
        rank1_f32 = [
            str(nm)
            for nm, tt in tensors.items()
            if str(nm).strip()
            and str(nm) != out_name
            and _dtype(getattr(tt, "dtype", "f32")) == "f32"
            and len(_shape(str(nm), intent=intent, bindings=bindings)) == 1
        ]

        def _name_key(raw: str) -> str:
            return str(raw or "").strip().lower().replace("-", "_")

        if not b1_name:
            b1_name = next((nm for nm in rank1_f32 if _name_key(nm) in {"b1", "bias1", "bias_1"}), "")
        if not b2_name:
            b2_name = next((nm for nm in rank1_f32 if _name_key(nm) in {"b2", "bias2", "bias_2"}), "")

        # Last resort: pick by shape only if unambiguous.
        if not b1_name:
            cands = [nm for nm in rank1_f32 if _shape(nm, intent=intent, bindings=bindings) == [int(h_dim)]]
            if len(cands) == 1:
                b1_name = cands[0]
        if not b2_name:
            cands = [nm for nm in rank1_f32 if _shape(nm, intent=intent, bindings=bindings) == [int(n_dim)]]
            if len(cands) == 1:
                b2_name = cands[0]
    if not (b1_name and b2_name):
        raise RuntimeError(f"rvv cpu-loops v1 mlp2d missing b1[H]/b2[N] inputs (b1={b1_name!r} b2={b2_name!r})")

    io_names = _io_arg_order(intent)
    total_a = int(m_dim * k_dim)
    total_w1 = int(k_dim * h_dim)
    total_w2 = int(h_dim * n_dim)
    total_b1 = int(h_dim)
    total_b2 = int(n_dim)
    total_out = int(m_dim * n_dim)

    arg_types: dict[str, str] = {
        a_name: f"memref<{total_a}xf32>",
        w1_name: f"memref<{total_w1}xf32>",
        w2_name: f"memref<{total_w2}xf32>",
        b1_name: f"memref<{total_b1}xf32>",
        b2_name: f"memref<{total_b2}xf32>",
        out_name: f"memref<{total_out}xf32>",
    }
    for n in list(io_names):
        if n not in arg_types:
            raise RuntimeError(
                f"rvv cpu-loops v1 mlp2d: unexpected IO tensor {n!r} (known={sorted(arg_types)}) for kernel={kernel_name!r}"
            )
    arg_decls = [f"%{_mlir_ident(n)}: {arg_types[n]}" for n in io_names]

    a_memref_ty = arg_types[a_name]
    w1_memref_ty = arg_types[w1_name]
    w2_memref_ty = arg_types[w2_name]
    b1_memref_ty = arg_types[b1_name]
    b2_memref_ty = arg_types[b2_name]
    out_memref_ty = arg_types[out_name]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cM = arith.constant {m_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append(f"    %cK = arith.constant {k_dim} : index")
    lines.append(f"    %cH = arith.constant {h_dim} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    hidden_total = int(m_dim * h_dim)
    hidden_memref_ty = f"memref<{hidden_total}xf32>"
    lines.append(f"    %hidden = memref.alloca() : {hidden_memref_ty}")
    lines.append("    scf.for %m = %c0 to %cM step %c1 {")
    lines.append("      %a_base = arith.muli %m, %cK : index")
    lines.append("      %h_base = arith.muli %m, %cH : index")
    lines.append("      scf.for %h = %c0 to %cH step %c1 {")
    lines.append("        %sum = scf.for %k = %c0 to %cK step %c1 iter_args(%a = %c0f) -> (f32) {")
    lines.append("          %a_i = arith.addi %a_base, %k : index")
    lines.append("          %w1_row = arith.muli %k, %cH : index")
    lines.append("          %w1_i = arith.addi %w1_row, %h : index")
    lines.append(f"          %av = memref.load %{_mlir_ident(a_name)}[%a_i] : {a_memref_ty}")
    lines.append(f"          %w1v = memref.load %{_mlir_ident(w1_name)}[%w1_i] : {w1_memref_ty}")
    lines.append("          %p = arith.mulf %av, %w1v : f32")
    lines.append("          %a2 = arith.addf %a, %p : f32")
    lines.append("          scf.yield %a2 : f32")
    lines.append("        }")
    lines.append(f"        %b1v = memref.load %{_mlir_ident(b1_name)}[%h] : {b1_memref_ty}")
    lines.append("        %pre = arith.addf %sum, %b1v : f32")
    lines.append("        %hid = arith.maximumf %pre, %c0f : f32")
    lines.append("        %hi = arith.addi %h_base, %h : index")
    lines.append(f"        memref.store %hid, %hidden[%hi] : {hidden_memref_ty}")
    lines.append("      }")
    lines.append("      %out_row = arith.muli %m, %cN : index")
    lines.append("      scf.for %n = %c0 to %cN step %c1 {")
    lines.append("        %acc_out = scf.for %h2 = %c0 to %cH step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("          %hi2 = arith.addi %h_base, %h2 : index")
    lines.append(f"          %hv = memref.load %hidden[%hi2] : {hidden_memref_ty}")
    lines.append("          %w2_row = arith.muli %h2, %cN : index")
    lines.append("          %w2_i = arith.addi %w2_row, %n : index")
    lines.append(f"          %w2v = memref.load %{_mlir_ident(w2_name)}[%w2_i] : {w2_memref_ty}")
    lines.append("          %q = arith.mulf %hv, %w2v : f32")
    lines.append("          %acc2 = arith.addf %acc, %q : f32")
    lines.append("          scf.yield %acc2 : f32")
    lines.append("        }")
    lines.append(f"        %b2v = memref.load %{_mlir_ident(b2_name)}[%n] : {b2_memref_ty}")
    lines.append("        %y = arith.addf %acc_out, %b2v : f32")
    lines.append("        %out_i = arith.addi %out_row, %n : index")
    lines.append(f"        memref.store %y, %{_mlir_ident(out_name)}[%out_i] : {out_memref_ty}")
    lines.append("      }")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_group_norm_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x).strip() for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 3:
        raise RuntimeError(f"rvv cpu-loops v1 group_norm_kernel expects three outputs, got outputs={outputs}")
    y_name = next((n for n in outputs if len(_shape(n, intent=intent, bindings=bindings)) == 3), "")
    if not y_name:
        raise RuntimeError(f"rvv cpu-loops v1 group_norm_kernel missing rank-3 Y output in outputs={outputs}")
    mean_name = next((n for n in outputs if n != y_name and len(_shape(n, intent=intent, bindings=bindings)) == 2), "")
    rstd_name = next(
        (n for n in outputs if n not in {y_name, mean_name} and len(_shape(n, intent=intent, bindings=bindings)) == 2), ""
    )
    if not mean_name or not rstd_name:
        raise RuntimeError("rvv cpu-loops v1 group_norm_kernel expects rank-2 Mean/Rstd outputs")

    y_shape = _shape(y_name, intent=intent, bindings=bindings)
    mean_shape = _shape(mean_name, intent=intent, bindings=bindings)
    rstd_shape = _shape(rstd_name, intent=intent, bindings=bindings)
    if y_shape != _shape("X", intent=intent, bindings=bindings) and (intent.tensors or {}).get("X") is not None:
        # Keep this soft: `X` can be named differently across frontends.
        pass
    if mean_shape != rstd_shape:
        raise RuntimeError(f"rvv cpu-loops v1 group_norm_kernel expects Mean/Rstd shapes match, got {mean_shape} vs {rstd_shape}")
    if len(y_shape) != 3 or len(mean_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 group_norm_kernel expects Y[N,C,HW] and Mean[N,G], got Y={y_shape} Mean={mean_shape}")
    n_dim, c_dim, hw_dim = map(int, y_shape)
    if n_dim <= 0 or c_dim <= 0 or hw_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 group_norm_kernel invalid Y shape: {y_shape}")
    if int(mean_shape[0]) != int(n_dim):
        raise RuntimeError("rvv cpu-loops v1 group_norm_kernel expects Mean first dim == N")
    num_groups = int(mean_shape[1])
    if num_groups <= 0:
        raise RuntimeError("rvv cpu-loops v1 group_norm_kernel expects num_groups>0")
    group_size = int(bindings.get("group_size") or 0)
    if group_size <= 0:
        if c_dim % num_groups != 0:
            raise RuntimeError(f"rvv cpu-loops v1 group_norm_kernel expects C divisible by num_groups, got C={c_dim} G={num_groups}")
        group_size = int(c_dim // num_groups)
    if int(group_size) * int(num_groups) != int(c_dim):
        raise RuntimeError(f"rvv cpu-loops v1 group_norm_kernel expects C==group_size*num_groups, got C={c_dim} group_size={group_size} G={num_groups}")
    if list(rstd_shape) != [n_dim, num_groups]:
        raise RuntimeError(f"rvv cpu-loops v1 group_norm_kernel expects Rstd shape [N,G], got {rstd_shape}")

    eps_const = _find_const_scalar(intent=intent, output="eps", dtype="f32", bindings=bindings)
    if eps_const is None:
        eps_const = 1.0e-5
    num_elements = int(group_size) * int(hw_dim)
    num_elements_f = float(num_elements)

    io_names = _io_arg_order(intent)
    ext_inputs = [n for n in io_names if n not in set(outputs)]

    def _find_rank(rank: int) -> str | None:
        for name in ext_inputs:
            if len(_shape(name, intent=intent, bindings=bindings)) == rank:
                return str(name)
        return None

    x_name = _find_rank(3)
    if x_name is None:
        raise RuntimeError("rvv cpu-loops v1 group_norm_kernel expects external X[N,C,HW] tensor")
    rank1 = [n for n in ext_inputs if len(_shape(n, intent=intent, bindings=bindings)) == 1]
    if len(rank1) < 2:
        raise RuntimeError("rvv cpu-loops v1 group_norm_kernel expects external W[C] and B[C] tensors")

    def _name_key(raw: str) -> str:
        return str(raw or "").strip().lower().replace("-", "_")

    w_name = next((n for n in rank1 if _name_key(n) in {"w", "weight", "weight_ptr", "gamma"}), None)
    b_name = next((n for n in rank1 if _name_key(n) in {"b", "bias", "bias_ptr", "beta"}), None)
    if w_name is None or b_name is None:
        if len(rank1) >= 2:
            # Heuristic: bias sorts before weight (e.g. B < W).
            b_name = str(rank1[0])
            w_name = str(rank1[1])
    if w_name is None or b_name is None:
        raise RuntimeError("rvv cpu-loops v1 group_norm_kernel expects external W[C] and B[C] tensors")

    def _numel(name: str) -> int:
        sh = _shape(name, intent=intent, bindings=bindings)
        out = 1
        for d in sh:
            out *= int(d)
        return int(out)

    arg_decls: list[str] = []
    arg_ssa: dict[str, str] = {}
    memref_ty_by_name: dict[str, str] = {}
    for name in io_names:
        tt = (intent.tensors or {}).get(name)
        if tt is None:
            raise RuntimeError(f"missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "f32"))
        if elem_ty != "f32":
            raise RuntimeError(f"rvv cpu-loops v1 group_norm_kernel supports only f32 tensors, got {name} dtype={elem_ty!r}")
        memref_ty = f"memref<{_numel(name)}x{elem_ty}>"
        ssa = f"%{_mlir_ident(name)}"
        arg_ssa[name] = ssa
        memref_ty_by_name[name] = memref_ty
        arg_decls.append(f"    {ssa}: {memref_ty}")

    x_memref = memref_ty_by_name[x_name]
    w_memref = memref_ty_by_name[w_name]
    b_memref = memref_ty_by_name[b_name]
    y_memref = memref_ty_by_name[y_name]
    mean_memref = memref_ty_by_name[mean_name]
    rstd_memref = memref_ty_by_name[rstd_name]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}(")
    lines.append(",\n".join(arg_decls))
    lines.append("  ) {")
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %c1 = arith.constant 1 : index")
    lines.append(f"  %cN = arith.constant {n_dim} : index")
    lines.append(f"  %cC = arith.constant {c_dim} : index")
    lines.append(f"  %cHW = arith.constant {hw_dim} : index")
    lines.append(f"  %cG = arith.constant {num_groups} : index")
    lines.append(f"  %cGS = arith.constant {group_size} : index")
    lines.append(f"  %cT = arith.constant {num_elements} : index")
    lines.append("  %c0f = arith.constant 0.0 : f32")
    lines.append(f"  %cTf = arith.constant {_f32_lit(num_elements_f)} : f32")
    lines.append(f"  %eps = arith.constant {_f32_lit(float(eps_const))} : f32")
    lines.append("  scf.for %n = %c0 to %cN step %c1 {")
    lines.append("    %nC = arith.muli %n, %cC : index")
    lines.append("    %nG = arith.muli %n, %cG : index")
    lines.append("    scf.for %g = %c0 to %cG step %c1 {")
    lines.append("      %gGS = arith.muli %g, %cGS : index")
    lines.append("      %mean_idx = arith.addi %nG, %g : index")
    lines.append("      %sum = scf.for %t = %c0 to %cT step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("        %ci = arith.divui %t, %cHW : index")
    lines.append("        %hw = arith.remui %t, %cHW : index")
    lines.append("        %c = arith.addi %gGS, %ci : index")
    lines.append("        %nc = arith.addi %nC, %c : index")
    lines.append("        %mul = arith.muli %nc, %cHW : index")
    lines.append("        %idx = arith.addi %mul, %hw : index")
    lines.append(f"        %x = memref.load {arg_ssa[x_name]}[%idx] : {x_memref}")
    lines.append("        %acc2 = arith.addf %acc, %x : f32")
    lines.append("        scf.yield %acc2 : f32")
    lines.append("      }")
    lines.append("      %mean = arith.divf %sum, %cTf : f32")
    lines.append(f"      memref.store %mean, {arg_ssa[mean_name]}[%mean_idx] : {mean_memref}")
    lines.append("      %sum_sq = scf.for %t2 = %c0 to %cT step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("        %ci2 = arith.divui %t2, %cHW : index")
    lines.append("        %hw2 = arith.remui %t2, %cHW : index")
    lines.append("        %c2 = arith.addi %gGS, %ci2 : index")
    lines.append("        %nc2 = arith.addi %nC, %c2 : index")
    lines.append("        %mul2 = arith.muli %nc2, %cHW : index")
    lines.append("        %idx2 = arith.addi %mul2, %hw2 : index")
    lines.append(f"        %x2 = memref.load {arg_ssa[x_name]}[%idx2] : {x_memref}")
    lines.append("        %d = arith.subf %x2, %mean : f32")
    lines.append("        %d2 = arith.mulf %d, %d : f32")
    lines.append("        %acc3 = arith.addf %acc, %d2 : f32")
    lines.append("        scf.yield %acc3 : f32")
    lines.append("      }")
    lines.append("      %var = arith.divf %sum_sq, %cTf : f32")
    lines.append("      %var_eps = arith.addf %var, %eps : f32")
    lines.append("      %rstd = math.rsqrt %var_eps : f32")
    lines.append(f"      memref.store %rstd, {arg_ssa[rstd_name]}[%mean_idx] : {rstd_memref}")
    lines.append("      scf.for %t3 = %c0 to %cT step %c1 {")
    lines.append("        %ci3 = arith.divui %t3, %cHW : index")
    lines.append("        %hw3 = arith.remui %t3, %cHW : index")
    lines.append("        %c3 = arith.addi %gGS, %ci3 : index")
    lines.append("        %nc3 = arith.addi %nC, %c3 : index")
    lines.append("        %mul3 = arith.muli %nc3, %cHW : index")
    lines.append("        %idx3 = arith.addi %mul3, %hw3 : index")
    lines.append(f"        %x3 = memref.load {arg_ssa[x_name]}[%idx3] : {x_memref}")
    lines.append("        %d3 = arith.subf %x3, %mean : f32")
    lines.append("        %xh = arith.mulf %d3, %rstd : f32")
    lines.append(f"        %w = memref.load {arg_ssa[w_name]}[%c3] : {w_memref}")
    lines.append(f"        %b = memref.load {arg_ssa[b_name]}[%c3] : {b_memref}")
    lines.append("        %xs = arith.mulf %xh, %w : f32")
    lines.append("        %y = arith.addf %xs, %b : f32")
    lines.append(f"        memref.store %y, {arg_ssa[y_name]}[%idx3] : {y_memref}")
    lines.append("      }")
    lines.append("    }")
    lines.append("  }")
    lines.append("  return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_elementwise_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    out_elem_ty = _dtype(getattr(out_tt, "dtype", "f32"))
    if out_elem_ty not in {"f32", "i8"}:
        raise RuntimeError(
            f"rvv cpu-loops v1 elementwise supports only f32/bool outputs, got out={out_name} dtype={getattr(out_tt,'dtype','')}"
        )
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 elementwise expects rank-2 output, got {out_shape}")
    m_dim, n_dim = int(out_shape[0]), int(out_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")
    total = int(m_dim * n_dim)

    ops = [op for op in list(intent.ops or []) if op is not None]
    if not ops:
        raise RuntimeError("rvv cpu-loops v1 requires non-empty ops")

    produced = {str(getattr(op, "output", "")).strip() for op in ops if str(getattr(op, "output", "")).strip()}
    used = {str(x).strip() for op in ops for x in list(getattr(op, "inputs", []) or []) if str(x).strip()}
    external_inputs = sorted([n for n in used if n and n in (intent.tensors or {}) and n not in produced and n != out_name])
    arg_order = [*external_inputs, out_name]

    scalar_loads: list[str] = []
    arg_decls: list[str] = []
    arg_ssa: dict[str, str] = {}
    scalar_ssa: dict[str, str] = {}
    memref_ty_by_name: dict[str, str] = {}
    shape_by_name: dict[str, list[int]] = {}
    for name in arg_order:
        tt = (intent.tensors or {}).get(name)
        if tt is None:
            raise RuntimeError(f"missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "f32"))
        if name == out_name and elem_ty not in {"f32", "i8"}:
            raise RuntimeError(f"rvv cpu-loops v1 elementwise supports only f32/bool outputs, got output dtype={elem_ty!r}")
        sh = _shape(name, intent=intent, bindings=bindings)
        shape_by_name[name] = list(sh)
        if len(sh) == 2:
            if name != out_name and list(sh) != list(out_shape):
                raise RuntimeError(
                    f"rvv cpu-loops v1 elementwise expects rank-2 inputs to match output shape={out_shape}, got {name} shape={sh}"
                )
            numel = int(sh[0] * sh[1])
        elif len(sh) == 1:
            dim0 = int(sh[0])
            if name == out_name:
                raise RuntimeError(f"rvv cpu-loops v1 output must be rank-2, got {out_name} shape={sh}")
            if dim0 not in {m_dim, n_dim}:
                raise RuntimeError(
                    f"rvv cpu-loops v1 supports only rank-1 broadcast along M/N (len==M or len==N), got {name} shape={sh} output_shape={out_shape}"
                )
            numel = dim0
        elif len(sh) == 0:
            numel = 1
        else:
            raise RuntimeError(
                f"rvv cpu-loops v1 elementwise supports only rank-2/rank-1/scalar tensors, got {name} shape={sh}"
            )
        memref_ty = f"memref<{numel}x{elem_ty}>"
        ssa = f"%{_mlir_ident(name)}"
        arg_ssa[name] = ssa
        memref_ty_by_name[name] = memref_ty
        arg_decls.append(f"    {ssa}: {memref_ty}")
        if len(sh) == 0 and name != out_name:
            vssa = f"%{_mlir_ident(name)}_s"
            scalar_ssa[name] = vssa
            scalar_loads.append(f"  {vssa} = memref.load {ssa}[%c0] : {memref_ty}")

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append("  func.func private @erff(%arg0: f32) -> f32")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}(")
    lines.append(",\n".join(arg_decls))
    lines.append("  ) {")
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %c1 = arith.constant 1 : index")
    lines.append(f"  %cM = arith.constant {m_dim} : index")
    lines.append(f"  %cN = arith.constant {n_dim} : index")
    lines.append("  %c0f = arith.constant 0.0 : f32")
    lines.append("  %c1f = arith.constant 1.0 : f32")
    lines.append("  %c0i8 = arith.constant 0 : i8")
    lines.append("  %c1i8 = arith.constant 1 : i8")
    lines.append("  %c0i32 = arith.constant 0 : i32")
    lines.append("  %c1i32 = arith.constant 1 : i32")
    lines.append("  %true = arith.constant true")
    lines.extend(scalar_loads)
    lines.append("  scf.for %m = %c0 to %cM step %c1 {")
    lines.append("    %mN = arith.muli %m, %cN : index")
    lines.append("    scf.for %n = %c0 to %cN step %c1 {")
    lines.append("      %i = arith.addi %mN, %n : index")

    tmp_id = 0

    def _fresh(prefix: str) -> str:
        nonlocal tmp_id
        tmp_id += 1
        return f"%{prefix}{tmp_id}"

    load_cache: dict[tuple[str, str], str] = {}

    def _load(name: str) -> str:
        tt = (intent.tensors or {}).get(name)
        if tt is None:
            raise RuntimeError(f"unknown tensor referenced: {name}")
        sh = list(shape_by_name.get(name) or [])
        if len(sh) == 0:
            return scalar_ssa.get(name) or f"%{_mlir_ident(name)}_s"
        if len(sh) == 1:
            dim0 = int(sh[0])
            if dim0 == n_dim:
                idx = "%n"
            elif dim0 == m_dim:
                idx = "%m"
            else:
                raise RuntimeError(
                    f"rvv cpu-loops v1 rank-1 broadcast requires len==M or len==N, got {name} shape={sh} output_shape={out_shape}"
                )
            numel = dim0
        elif len(sh) == 2:
            idx = "%i"
            numel = int(sh[0] * sh[1])
        else:
            raise RuntimeError(f"rvv cpu-loops v1 unsupported tensor rank for {name}: shape={sh}")
        memref_ty = memref_ty_by_name.get(name)
        if not memref_ty:
            elem_ty = _dtype(getattr(tt, "dtype", "f32"))
            memref_ty = f"memref<{int(numel)}x{elem_ty}>"
        cache_key = (str(name), str(idx))
        cached = load_cache.get(cache_key)
        if cached:
            return cached
        ssa = _fresh(f"{_mlir_ident(name)}_v")
        lines.append(f"    {ssa} = memref.load {arg_ssa[name]}[{idx}] : {memref_ty}")
        load_cache[cache_key] = ssa
        return ssa

    computed: dict[str, str] = {}

    for op in ops:
        name = str(getattr(op, "op", "")).strip()
        out = str(getattr(op, "output", "")).strip()
        ins = [str(x).strip() for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
        attrs = dict(getattr(op, "attrs", {}) or {})

        def _in(idx: int) -> str:
            nm = ins[idx]
            return computed.get(nm) or _load(nm)

        if name == "const":
            if not out:
                raise RuntimeError("invalid const op: missing output")
            out_tt = (intent.tensors or {}).get(out)
            if out_tt is None:
                raise RuntimeError(f"rvv cpu-loops v1 const missing tensor spec: {out}")
            if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
                raise RuntimeError(f"rvv cpu-loops v1 supports only f32 const, got out={out} dtype={getattr(out_tt,'dtype','')}")
            if len(_shape(out, intent=intent, bindings=bindings)) != 0:
                raise RuntimeError(f"rvv cpu-loops v1 const supports only scalar outputs, got out={out}")
            try:
                value = float(attrs.get("value"))
            except Exception:
                raise RuntimeError(f"rvv cpu-loops v1 const requires numeric attrs.value, got {attrs.get('value')!r}")
            v = f"%{_mlir_ident(out)}_c"
            lines.append(f"    {v} = arith.constant {value!r} : f32")
            computed[out] = v
            continue

        if name == "cast":
            if len(ins) != 1 or not out:
                raise RuntimeError(f"invalid cast op: inputs={ins} output={out!r}")
            in_name = str(ins[0]).strip()
            in_tt = (intent.tensors or {}).get(in_name)
            out_tt = (intent.tensors or {}).get(out)
            if in_tt is None or out_tt is None:
                raise RuntimeError(f"rvv cpu-loops v1 cast missing tensor specs: in={in_name!r} out={out!r}")
            in_ty = _dtype(getattr(in_tt, "dtype", "f32"))
            out_ty = _dtype(getattr(out_tt, "dtype", "f32"))
            to_raw = str(attrs.get("to") or attrs.get("dtype") or out_ty).strip().lower()
            try:
                to_norm = _dtype(to_raw) if to_raw else out_ty
            except Exception:
                to_norm = to_raw
            if to_norm != out_ty:
                raise RuntimeError(
                    f"rvv cpu-loops v1 cast dtype mismatch: attrs.to={to_raw!r} normalized={to_norm!r} tensor.dtype={out_ty!r}"
                )
            a = _in(0)
            if in_ty == out_ty:
                computed[out] = a
            elif in_ty == "f16" and out_ty == "f32":
                v = f"%{_mlir_ident(out)}_t"
                lines.append(f"    {v} = arith.extf {a} : f16 to f32")
                computed[out] = v
            elif in_ty == "i8" and out_ty == "f32":
                p = f"%{_mlir_ident(out)}_p"
                lines.append(f"    {p} = arith.cmpi ne, {a}, %c0i8 : i8")
                v = f"%{_mlir_ident(out)}_t"
                lines.append(f"    {v} = arith.select {p}, %c1f, %c0f : f32")
                computed[out] = v
            elif in_ty == "i32" and out_ty == "f32":
                v = f"%{_mlir_ident(out)}_t"
                lines.append(f"    {v} = arith.sitofp {a} : i32 to f32")
                computed[out] = v
            elif in_ty == "f32" and out_ty == "i8":
                p = f"%{_mlir_ident(out)}_p"
                lines.append(f"    {p} = arith.cmpf one, {a}, %c0f : f32")
                v = f"%{_mlir_ident(out)}_t"
                lines.append(f"    {v} = arith.select {p}, %c1i8, %c0i8 : i8")
                computed[out] = v
            elif in_ty == "i8" and out_ty == "i32":
                p = f"%{_mlir_ident(out)}_p"
                lines.append(f"    {p} = arith.cmpi ne, {a}, %c0i8 : i8")
                v = f"%{_mlir_ident(out)}_t"
                lines.append(f"    {v} = arith.select {p}, %c1i32, %c0i32 : i32")
                computed[out] = v
            elif in_ty == "i32" and out_ty == "i8":
                p = f"%{_mlir_ident(out)}_p"
                lines.append(f"    {p} = arith.cmpi ne, {a}, %c0i32 : i32")
                v = f"%{_mlir_ident(out)}_t"
                lines.append(f"    {v} = arith.select {p}, %c1i8, %c0i8 : i8")
                computed[out] = v
            else:
                raise RuntimeError(f"rvv cpu-loops v1 unsupported cast: {in_ty} -> {out_ty}")
            continue

        if name == "broadcast_in_dim":
            if len(ins) != 1 or not out:
                raise RuntimeError(f"invalid broadcast_in_dim op: inputs={ins} output={out!r}")
            in_name = str(ins[0]).strip()
            in_tt = (intent.tensors or {}).get(in_name)
            out_tt = (intent.tensors or {}).get(out)
            if in_tt is None or out_tt is None:
                raise RuntimeError(f"rvv cpu-loops v1 broadcast_in_dim missing tensor specs: in={in_name!r} out={out!r}")
            in_ty = _dtype(getattr(in_tt, "dtype", "f32"))
            out_ty = _dtype(getattr(out_tt, "dtype", "f32"))
            if in_ty != out_ty:
                raise RuntimeError(f"rvv cpu-loops v1 broadcast_in_dim requires same dtype, got {in_ty}->{out_ty}")
            in_sh = _shape(in_name, intent=intent, bindings=bindings)
            out_sh = _shape(out, intent=intent, bindings=bindings)
            if list(out_sh) != [m_dim, n_dim]:
                raise RuntimeError(
                    f"rvv cpu-loops v1 broadcast_in_dim currently requires out_shape=[M,N], got out={out} shape={out_sh} kernel_out_shape={out_shape}"
                )
            b_dims = list(attrs.get("broadcast_dims") or []) if isinstance(attrs.get("broadcast_dims"), list) else []
            if len(in_sh) == 0:
                if b_dims:
                    raise RuntimeError(f"rvv cpu-loops v1 broadcast_in_dim scalar expects broadcast_dims=[], got {b_dims}")
                computed[out] = _in(0)
            elif len(in_sh) == 1:
                if b_dims not in ([0], [1]):
                    raise RuntimeError(f"rvv cpu-loops v1 broadcast_in_dim rank1 expects broadcast_dims [0] or [1], got {b_dims}")
                dim0 = int(in_sh[0])
                expected = int(m_dim if b_dims == [0] else n_dim)
                if dim0 != expected:
                    raise RuntimeError(
                        f"rvv cpu-loops v1 broadcast_in_dim rank1 dim mismatch: in_shape={in_sh} broadcast_dims={b_dims} expected={expected}"
                    )
                computed[out] = _in(0)
            else:
                raise RuntimeError(f"rvv cpu-loops v1 broadcast_in_dim supports only scalar or rank1 inputs, got in_shape={in_sh}")
            continue

        if name == "where":
            if len(ins) != 3 or not out:
                raise RuntimeError(f"invalid where op: inputs={ins} output={out!r}")
            cond, a0, a1 = [str(x).strip() for x in ins]
            cond_tt = (intent.tensors or {}).get(cond)
            out_tt = (intent.tensors or {}).get(out)
            a0_tt = (intent.tensors or {}).get(a0)
            a1_tt = (intent.tensors or {}).get(a1)
            if cond_tt is None or out_tt is None or a0_tt is None or a1_tt is None:
                raise RuntimeError(f"rvv cpu-loops v1 where missing tensor specs: inputs={ins} output={out!r}")
            cond_ty = _dtype(getattr(cond_tt, "dtype", "bool"))
            if cond_ty not in {"i8", "i32"}:
                raise RuntimeError(
                    f"rvv cpu-loops v1 where expects bool/u8/i32 mask, got tensor={cond} dtype={getattr(cond_tt, 'dtype', '')}"
                )
            for nm in [a0, a1, out]:
                tt = (intent.tensors or {}).get(str(nm))
                if tt is None:
                    raise RuntimeError(f"rvv cpu-loops v1 missing tensor spec: {nm}")
                if _dtype(getattr(tt, "dtype", "f32")) != "f32":
                    raise RuntimeError(
                        f"rvv cpu-loops v1 supports only f32 for where values, got tensor={nm} dtype={getattr(tt, 'dtype', '')}"
                    )
            m = _in(0)
            x0 = _in(1)
            x1 = _in(2)
            p = f"%{_mlir_ident(out)}_p"
            if cond_ty == "i8":
                lines.append(f"    {p} = arith.cmpi ne, {m}, %c0i8 : i8")
            else:
                lines.append(f"    {p} = arith.cmpi ne, {m}, %c0i32 : i32")
            v = f"%{_mlir_ident(out)}_t"
            lines.append(f"    {v} = arith.select {p}, {x0}, {x1} : f32")
            computed[out] = v
            continue

        if name == "identity":
            if len(ins) != 1 or not out:
                raise RuntimeError(f"invalid identity op: inputs={ins} output={out!r}")
            for nm in [*ins, out]:
                tt = (intent.tensors or {}).get(str(nm))
                if tt is None:
                    raise RuntimeError(f"rvv cpu-loops v1 missing tensor spec: {nm}")
                if _dtype(getattr(tt, "dtype", "f32")) != "f32":
                    raise RuntimeError(
                        f"rvv cpu-loops v1 supports only f32 for op {name}, got tensor={nm} dtype={getattr(tt, 'dtype', '')}"
                    )
            a = _in(0)
            computed[out] = a
            continue

        if name == "iota":
            if ins or not out:
                raise RuntimeError(f"invalid iota op: inputs={ins} output={out!r}")
            out_tt = (intent.tensors or {}).get(out)
            if out_tt is None:
                raise RuntimeError(f"rvv cpu-loops v1 iota missing tensor spec: out={out!r}")
            out_ty = _dtype(getattr(out_tt, "dtype", "i32"))
            if out_ty != "i32":
                raise RuntimeError(f"rvv cpu-loops v1 iota supports only i32 outputs, got out={out} dtype={getattr(out_tt,'dtype','')}")
            out_sh = _shape(out, intent=intent, bindings=bindings)
            if list(out_sh) != [m_dim, n_dim]:
                raise RuntimeError(f"rvv cpu-loops v1 iota requires out shape [M,N], got out={out} shape={out_sh} expected={[m_dim,n_dim]}")
            axis = attrs.get("axis")
            if not isinstance(axis, int):
                try:
                    axis = int(axis)
                except Exception:
                    axis = None
            if axis not in {0, 1}:
                raise RuntimeError(f"rvv cpu-loops v1 iota supports only axis 0/1, got axis={attrs.get('axis')!r}")
            v = f"%{_mlir_ident(out)}_t"
            if axis == 0:
                lines.append(f"    {v} = arith.index_cast %m : index to i32")
            else:
                lines.append(f"    {v} = arith.index_cast %n : index to i32")
            computed[out] = v
            continue

        if name in {"eq", "ne", "lt", "le", "gt", "ge"}:
            if len(ins) != 2 or not out:
                raise RuntimeError(f"invalid cmp op: {name} inputs={ins} output={out!r}")
            lhs_name, rhs_name = str(ins[0]), str(ins[1])
            lhs_tt = (intent.tensors or {}).get(lhs_name)
            rhs_tt = (intent.tensors or {}).get(rhs_name)
            out_tt = (intent.tensors or {}).get(out)
            if lhs_tt is None or rhs_tt is None or out_tt is None:
                raise RuntimeError(f"rvv cpu-loops v1 cmp missing tensor specs: op={name} inputs={ins} out={out!r}")
            lhs_ty = _dtype(getattr(lhs_tt, "dtype", "f32"))
            rhs_ty = _dtype(getattr(rhs_tt, "dtype", "f32"))
            out_ty = _dtype(getattr(out_tt, "dtype", "bool"))
            if lhs_ty != rhs_ty:
                raise RuntimeError(f"rvv cpu-loops v1 cmp requires same input dtypes, got {lhs_name}={lhs_ty} {rhs_name}={rhs_ty}")
            if out_ty not in {"i8", "i32"}:
                raise RuntimeError(f"rvv cpu-loops v1 cmp supports only bool/i32 outputs, got out={out} dtype={getattr(out_tt,'dtype','')}")
            a = _in(0)
            b = _in(1)
            p = f"%{_mlir_ident(out)}_p"
            if lhs_ty == "f32":
                pred = {
                    "eq": "oeq",
                    "ne": "une",
                    "lt": "olt",
                    "le": "ole",
                    "gt": "ogt",
                    "ge": "oge",
                }[name]
                lines.append(f"    {p} = arith.cmpf {pred}, {a}, {b} : f32")
            elif lhs_ty in {"i8", "i16", "i32", "i64"}:
                pred = {
                    "eq": "eq",
                    "ne": "ne",
                    "lt": "slt",
                    "le": "sle",
                    "gt": "sgt",
                    "ge": "sge",
                }[name]
                lines.append(f"    {p} = arith.cmpi {pred}, {a}, {b} : {lhs_ty}")
            else:
                raise RuntimeError(f"rvv cpu-loops v1 cmp supports only f32/i* inputs, got {lhs_ty}")
            v = f"%{_mlir_ident(out)}_t"
            if out_ty == "i32":
                lines.append(f"    {v} = arith.select {p}, %c1i32, %c0i32 : i32")
            else:
                lines.append(f"    {v} = arith.select {p}, %c1i8, %c0i8 : i8")
            computed[out] = v
            continue

        if name in {"and", "or", "xor"}:
            if len(ins) != 2 or not out:
                raise RuntimeError(f"invalid logical op: {name} inputs={ins} output={out!r}")
            a0_name, a1_name = str(ins[0]), str(ins[1])
            a0_tt = (intent.tensors or {}).get(a0_name)
            a1_tt = (intent.tensors or {}).get(a1_name)
            out_tt = (intent.tensors or {}).get(out)
            if a0_tt is None or a1_tt is None or out_tt is None:
                raise RuntimeError(f"rvv cpu-loops v1 logical op missing tensor specs: op={name} inputs={ins} out={out!r}")
            in_ty0 = _dtype(getattr(a0_tt, "dtype", "bool"))
            in_ty1 = _dtype(getattr(a1_tt, "dtype", "bool"))
            out_ty = _dtype(getattr(out_tt, "dtype", "bool"))
            if in_ty0 != in_ty1 or out_ty != in_ty0:
                raise RuntimeError(f"rvv cpu-loops v1 logical op requires same dtypes, got {a0_name}={in_ty0} {a1_name}={in_ty1} out={out_ty}")
            if out_ty not in {"i8", "i32"}:
                raise RuntimeError(f"rvv cpu-loops v1 logical op supports only bool/i32 tensors, got out={out} dtype={getattr(out_tt,'dtype','')}")
            a = _in(0)
            b = _in(1)
            pa = f"%{_mlir_ident(out)}_pa"
            pb = f"%{_mlir_ident(out)}_pb"
            if out_ty == "i32":
                lines.append(f"    {pa} = arith.cmpi ne, {a}, %c0i32 : i32")
                lines.append(f"    {pb} = arith.cmpi ne, {b}, %c0i32 : i32")
            else:
                lines.append(f"    {pa} = arith.cmpi ne, {a}, %c0i8 : i8")
                lines.append(f"    {pb} = arith.cmpi ne, {b}, %c0i8 : i8")
            p = f"%{_mlir_ident(out)}_p"
            lop = {"and": "andi", "or": "ori", "xor": "xori"}[name]
            lines.append(f"    {p} = arith.{lop} {pa}, {pb} : i1")
            v = f"%{_mlir_ident(out)}_t"
            if out_ty == "i32":
                lines.append(f"    {v} = arith.select {p}, %c1i32, %c0i32 : i32")
            else:
                lines.append(f"    {v} = arith.select {p}, %c1i8, %c0i8 : i8")
            computed[out] = v
            continue

        if name == "not":
            if len(ins) != 1 or not out:
                raise RuntimeError(f"invalid logical not op: inputs={ins} output={out!r}")
            in_name = str(ins[0])
            in_tt = (intent.tensors or {}).get(in_name)
            out_tt = (intent.tensors or {}).get(out)
            if in_tt is None or out_tt is None:
                raise RuntimeError(f"rvv cpu-loops v1 not missing tensor specs: in={in_name!r} out={out!r}")
            in_ty = _dtype(getattr(in_tt, "dtype", "bool"))
            out_ty = _dtype(getattr(out_tt, "dtype", "bool"))
            if out_ty != in_ty or out_ty not in {"i8", "i32"}:
                raise RuntimeError(f"rvv cpu-loops v1 not supports only bool/i32 tensors, got in={in_ty} out={out_ty}")
            a = _in(0)
            pa = f"%{_mlir_ident(out)}_pa"
            if out_ty == "i32":
                lines.append(f"    {pa} = arith.cmpi ne, {a}, %c0i32 : i32")
            else:
                lines.append(f"    {pa} = arith.cmpi ne, {a}, %c0i8 : i8")
            p = f"%{_mlir_ident(out)}_p"
            lines.append(f"    {p} = arith.xori {pa}, %true : i1")
            v = f"%{_mlir_ident(out)}_t"
            if out_ty == "i32":
                lines.append(f"    {v} = arith.select {p}, %c1i32, %c0i32 : i32")
            else:
                lines.append(f"    {v} = arith.select {p}, %c1i8, %c0i8 : i8")
            computed[out] = v
            continue

        if name in {"add", "sub", "mul", "div", "max", "min", "remainder"}:
            if len(ins) != 2 or not out:
                raise RuntimeError(f"invalid binary op: {name} inputs={ins} output={out!r}")
            in0_tt = (intent.tensors or {}).get(str(ins[0]))
            in1_tt = (intent.tensors or {}).get(str(ins[1]))
            out_tt = (intent.tensors or {}).get(str(out))
            if in0_tt is None or in1_tt is None or out_tt is None:
                raise RuntimeError(f"rvv cpu-loops v1 missing tensor spec: op={name} inputs={ins} out={out!r}")
            in0_ty = _dtype(getattr(in0_tt, "dtype", "f32"))
            in1_ty = _dtype(getattr(in1_tt, "dtype", "f32"))
            out_ty = _dtype(getattr(out_tt, "dtype", "f32"))
            if in0_ty != in1_ty or out_ty != in0_ty:
                raise RuntimeError(f"rvv cpu-loops v1 binary op requires same dtypes, got {ins[0]}={in0_ty} {ins[1]}={in1_ty} out={out_ty}")
            a = _in(0)
            b = _in(1)
            v = f"%{_mlir_ident(out)}_t"
            if in0_ty == "f32":
                if name == "add":
                    lines.append(f"    {v} = arith.addf {a}, {b} : f32")
                elif name == "sub":
                    lines.append(f"    {v} = arith.subf {a}, {b} : f32")
                elif name == "mul":
                    lines.append(f"    {v} = arith.mulf {a}, {b} : f32")
                elif name == "div":
                    lines.append(f"    {v} = arith.divf {a}, {b} : f32")
                elif name == "max":
                    lines.append(f"    {v} = arith.maximumf {a}, {b} : f32")
                elif name == "min":
                    lines.append(f"    {v} = arith.minimumf {a}, {b} : f32")
                else:
                    # Match torch.remainder semantics (Python-style modulo):
                    #   r = a - floor(a / b) * b
                    q = _fresh(f"{_mlir_ident(out)}_q")
                    qf = _fresh(f"{_mlir_ident(out)}_qf")
                    qb = _fresh(f"{_mlir_ident(out)}_qb")
                    lines.append(f"    {q} = arith.divf {a}, {b} : f32")
                    lines.append(f"    {qf} = math.floor {q} : f32")
                    lines.append(f"    {qb} = arith.mulf {qf}, {b} : f32")
                    lines.append(f"    {v} = arith.subf {a}, {qb} : f32")
            elif in0_ty in {"i8", "i32"} and name in {"max", "min"}:
                iop = "maxui" if name == "max" else "minui"
                lines.append(f"    {v} = arith.{iop} {a}, {b} : {in0_ty}")
            elif in0_ty in {"i8", "i32"} and name in {"add", "sub", "mul", "div"}:
                iop = {"add": "addi", "sub": "subi", "mul": "muli", "div": "divsi"}[name]
                lines.append(f"    {v} = arith.{iop} {a}, {b} : {in0_ty}")
            else:
                raise RuntimeError(f"rvv cpu-loops v1 unsupported binary op: {name} dtype={in0_ty}")
            computed[out] = v
            continue

        if name in {
            "relu",
            "abs",
            "sqrt",
            "rsqrt",
            "exp",
            "exp2",
            "neg",
            "floor",
            "ceil",
            "log",
            "sin",
            "cos",
            "tan",
            "atan",
            "acos",
            "erf",
        }:
            if len(ins) != 1 or not out:
                raise RuntimeError(f"invalid unary op: {name} inputs={ins} output={out!r}")
            for nm in [*ins, out]:
                tt = (intent.tensors or {}).get(str(nm))
                if tt is None:
                    raise RuntimeError(f"rvv cpu-loops v1 missing tensor spec: {nm}")
                if _dtype(getattr(tt, "dtype", "f32")) != "f32":
                    raise RuntimeError(
                        f"rvv cpu-loops v1 supports only f32 for op {name}, got tensor={nm} dtype={getattr(tt, 'dtype', '')}"
                    )
            a = _in(0)
            v = f"%{_mlir_ident(out)}_t"
            if name == "relu":
                lines.append(f"    {v} = arith.maximumf {a}, %c0f : f32")
            elif name == "abs":
                lines.append(f"    {v} = math.absf {a} : f32")
            elif name == "sqrt":
                lines.append(f"    {v} = math.sqrt {a} : f32")
            elif name == "rsqrt":
                lines.append(f"    {v} = math.rsqrt {a} : f32")
            elif name == "neg":
                lines.append(f"    {v} = arith.negf {a} : f32")
            elif name == "exp2":
                lines.append(f"    {v} = math.exp2 {a} : f32")
            elif name == "floor":
                lines.append(f"    {v} = math.floor {a} : f32")
            elif name == "ceil":
                lines.append(f"    {v} = math.ceil {a} : f32")
            elif name == "log":
                lines.append(f"    {v} = math.log {a} : f32")
            elif name == "erf":
                lines.append(f"    {v} = func.call @erff({a}) : (f32) -> f32")
            elif name in {"sin", "cos", "tan", "atan", "acos"}:
                mop = {
                    "sin": "math.sin",
                    "cos": "math.cos",
                    "tan": "math.tan",
                    "atan": "math.atan",
                    "acos": "math.acos",
                }[name]
                lines.append(f"    {v} = {mop} {a} : f32")
            else:
                base = attrs.get("base")
                if base in (2, 2.0):
                    lines.append(f"    {v} = math.exp2 {a} : f32")
                else:
                    lines.append(f"    {v} = math.exp {a} : f32")
            computed[out] = v
            continue

        raise RuntimeError(f"rvv cpu-loops v1 unsupported op: {name}")

    final = computed.get(out_name) or _load(out_name)
    out_memref_ty = memref_ty_by_name[out_name]
    lines.append(f"    memref.store {final}, {arg_ssa[out_name]}[%i] : {out_memref_ty}")
    lines.append("    }")
    lines.append("  }")
    lines.append("  return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_cmp2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 cmp2d expects single output, got outputs={outputs}")
    out_name = str(outputs[0]).strip()
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"rvv cpu-loops v1 cmp2d missing output tensor spec: {out_name}")
    out_ty = _dtype(getattr(out_tt, "dtype", "bool"))
    if out_ty not in {"i8", "i32"}:
        raise RuntimeError(f"rvv cpu-loops v1 cmp2d supports only bool/i32 outputs, got out dtype={getattr(out_tt,'dtype','')}")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 cmp2d expects rank-2 output, got {out_shape}")
    m_dim, n_dim = int(out_shape[0]), int(out_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"invalid cmp2d output shape: {out_shape}")
    total = int(m_dim * n_dim)

    ops = [op for op in list(intent.ops or []) if op is not None]
    if not ops:
        raise RuntimeError("rvv cpu-loops v1 cmp2d requires non-empty ops")
    cmp_ops = [op for op in ops if str(getattr(op, "op", "")).strip() in {"eq", "ne", "lt", "le", "gt", "ge"}]
    if len(cmp_ops) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 cmp2d expects exactly 1 cmp op, got {[getattr(o,'op','') for o in cmp_ops]}")
    cmp_op = cmp_ops[0]
    cmp_name = str(getattr(cmp_op, "op", "")).strip()
    ins = [str(x).strip() for x in list(getattr(cmp_op, "inputs", []) or []) if str(x).strip()]
    if len(ins) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 cmp2d {cmp_name} expects 2 inputs, got inputs={ins}")
    lhs_name, rhs_name = str(ins[0]), str(ins[1])
    lhs_tt = (intent.tensors or {}).get(lhs_name)
    rhs_tt = (intent.tensors or {}).get(rhs_name)
    if lhs_tt is None or rhs_tt is None:
        raise RuntimeError(f"rvv cpu-loops v1 cmp2d missing input tensor specs: lhs={lhs_name!r} rhs={rhs_name!r}")
    if _dtype(getattr(lhs_tt, "dtype", "f32")) != "f32" or _dtype(getattr(rhs_tt, "dtype", "f32")) != "f32":
        raise RuntimeError(f"rvv cpu-loops v1 cmp2d supports only f32 inputs, got {lhs_name}={getattr(lhs_tt,'dtype','')}, {rhs_name}={getattr(rhs_tt,'dtype','')}")
    lhs_shape = _shape(lhs_name, intent=intent, bindings=bindings)
    rhs_shape = _shape(rhs_name, intent=intent, bindings=bindings)
    if list(lhs_shape) != [m_dim, n_dim] or list(rhs_shape) != [m_dim, n_dim]:
        raise RuntimeError(f"rvv cpu-loops v1 cmp2d expects inputs shape [M,N]=={out_shape}, got lhs={lhs_shape} rhs={rhs_shape}")

    io_names = _io_arg_order(intent)
    if not io_names:
        raise RuntimeError("rvv cpu-loops v1 cmp2d missing io tensors")

    arg_decls: list[str] = []
    arg_ssa: dict[str, str] = {}
    memref_ty_by_name: dict[str, str] = {}
    for name in io_names:
        tt = (intent.tensors or {}).get(name)
        if tt is None:
            raise RuntimeError(f"rvv cpu-loops v1 cmp2d missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "f32"))
        sh = _shape(name, intent=intent, bindings=bindings)
        if len(sh) == 2:
            if list(sh) != [m_dim, n_dim]:
                raise RuntimeError(f"rvv cpu-loops v1 cmp2d expects rank-2 io tensors to match [M,N], got {name} shape={sh}")
            numel = int(m_dim * n_dim)
        elif len(sh) == 0:
            numel = 1
        else:
            raise RuntimeError(f"rvv cpu-loops v1 cmp2d supports only rank-2 or scalar io tensors, got {name} shape={sh}")
        memref_ty = f"memref<{numel}x{elem_ty}>"
        ssa = f"%{_mlir_ident(name)}"
        arg_ssa[name] = ssa
        memref_ty_by_name[name] = memref_ty
        arg_decls.append(f"    {ssa}: {memref_ty}")

    lhs_memref = memref_ty_by_name[lhs_name]
    rhs_memref = memref_ty_by_name[rhs_name]
    out_memref = memref_ty_by_name[out_name]

    pred_kind = {
        "eq": "oeq",
        "ne": "one",
        "lt": "olt",
        "le": "ole",
        "gt": "ogt",
        "ge": "oge",
    }[cmp_name]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}(")
    lines.append(",\n".join(arg_decls))
    lines.append("  ) {")
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %c1 = arith.constant 1 : index")
    lines.append(f"  %cM = arith.constant {m_dim} : index")
    lines.append(f"  %cN = arith.constant {n_dim} : index")
    if out_ty == "i8":
        lines.append("  %c0o = arith.constant 0 : i8")
        lines.append("  %c1o = arith.constant 1 : i8")
    else:
        lines.append("  %c0o = arith.constant 0 : i32")
        lines.append("  %c1o = arith.constant 1 : i32")
    lines.append("  scf.for %m = %c0 to %cM step %c1 {")
    lines.append("    %mN = arith.muli %m, %cN : index")
    lines.append("    scf.for %n = %c0 to %cN step %c1 {")
    lines.append("      %i = arith.addi %mN, %n : index")
    lines.append(f"      %a = memref.load {arg_ssa[lhs_name]}[%i] : {lhs_memref}")
    lines.append(f"      %b = memref.load {arg_ssa[rhs_name]}[%i] : {rhs_memref}")
    lines.append(f"      %p = arith.cmpf {pred_kind}, %a, %b : f32")
    lines.append(f"      %v = arith.select %p, %c1o, %c0o : {out_ty}")
    lines.append(f"      memref.store %v, {arg_ssa[out_name]}[%i] : {out_memref}")
    lines.append("    }")
    lines.append("  }")
    lines.append("  return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_bitwise2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 bitwise2d expects single output, got outputs={outputs}")
    out_name = str(outputs[0]).strip()
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"rvv cpu-loops v1 bitwise2d missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "i32")) != "i32":
        raise RuntimeError(f"rvv cpu-loops v1 bitwise2d supports only i32 output, got dtype={getattr(out_tt,'dtype','')}")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 bitwise2d expects rank-2 output, got {out_shape}")
    m_dim, n_dim = int(out_shape[0]), int(out_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"invalid bitwise2d output shape: {out_shape}")
    total = int(m_dim * n_dim)

    ops = [op for op in list(intent.ops or []) if op is not None]
    if len(ops) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 bitwise2d expects exactly 1 op, got {len(ops)}")
    op0 = ops[0]
    op_name = str(getattr(op0, "op", "")).strip()
    ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
    out = str(getattr(op0, "output", "")).strip()
    if out != out_name:
        raise RuntimeError(f"rvv cpu-loops v1 bitwise2d output mismatch: op.output={out!r} expected={out_name!r}")
    if op_name not in {
        "bitwise_and",
        "bitwise_or",
        "bitwise_not",
        "bitwise_left_shift",
        "bitwise_right_shift",
    }:
        raise RuntimeError(f"rvv cpu-loops v1 bitwise2d unsupported op: {op_name}")

    if op_name == "bitwise_not":
        if len(ins) != 1:
            raise RuntimeError(f"rvv cpu-loops v1 bitwise_not expects 1 input, got inputs={ins}")
    else:
        if len(ins) != 2:
            raise RuntimeError(f"rvv cpu-loops v1 {op_name} expects 2 inputs, got inputs={ins}")

    io_names = _io_arg_order(intent)
    if not io_names:
        raise RuntimeError("rvv cpu-loops v1 bitwise2d missing io tensors")

    arg_decls: list[str] = []
    arg_ssa: dict[str, str] = {}
    memref_ty_by_name: dict[str, str] = {}
    for name in io_names:
        tt = (intent.tensors or {}).get(name)
        if tt is None:
            raise RuntimeError(f"rvv cpu-loops v1 bitwise2d missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "i32"))
        if elem_ty != "i32":
            raise RuntimeError(f"rvv cpu-loops v1 bitwise2d supports only i32 tensors, got {name} dtype={getattr(tt,'dtype','')}")
        sh = _shape(name, intent=intent, bindings=bindings)
        if len(sh) != 2 or list(sh) != [m_dim, n_dim]:
            raise RuntimeError(f"rvv cpu-loops v1 bitwise2d expects [M,N] tensors, got {name} shape={sh}")
        memref_ty = f"memref<{total}xi32>"
        ssa = f"%{_mlir_ident(name)}"
        arg_ssa[name] = ssa
        memref_ty_by_name[name] = memref_ty
        arg_decls.append(f"    {ssa}: {memref_ty}")

    a_name = str(ins[0])
    b_name = str(ins[1]) if len(ins) > 1 else ""
    if a_name not in arg_ssa or (b_name and b_name not in arg_ssa):
        raise RuntimeError(f"rvv cpu-loops v1 bitwise2d expects inputs in io tensors, got inputs={ins} io={io_names}")

    out_memref = memref_ty_by_name[out_name]
    a_memref = memref_ty_by_name[a_name]
    b_memref = memref_ty_by_name[b_name] if b_name else ""

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}(")
    lines.append(",\n".join(arg_decls))
    lines.append("  ) {")
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %c1 = arith.constant 1 : index")
    lines.append(f"  %cM = arith.constant {m_dim} : index")
    lines.append(f"  %cN = arith.constant {n_dim} : index")
    lines.append("  %all_ones = arith.constant -1 : i32")
    lines.append("  scf.for %m = %c0 to %cM step %c1 {")
    lines.append("    %mN = arith.muli %m, %cN : index")
    lines.append("    scf.for %n = %c0 to %cN step %c1 {")
    lines.append("      %i = arith.addi %mN, %n : index")
    lines.append(f"      %a = memref.load {arg_ssa[a_name]}[%i] : {a_memref}")
    if b_name:
        lines.append(f"      %b = memref.load {arg_ssa[b_name]}[%i] : {b_memref}")
    if op_name == "bitwise_and":
        lines.append("      %v = arith.andi %a, %b : i32")
    elif op_name == "bitwise_or":
        lines.append("      %v = arith.ori %a, %b : i32")
    elif op_name == "bitwise_left_shift":
        lines.append("      %v = arith.shli %a, %b : i32")
    elif op_name == "bitwise_right_shift":
        lines.append("      %v = arith.shrsi %a, %b : i32")
    else:
        lines.append("      %v = arith.xori %a, %all_ones : i32")
    lines.append(f"      memref.store %v, {arg_ssa[out_name]}[%i] : {out_memref}")
    lines.append("    }")
    lines.append("  }")
    lines.append("  return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_arange1d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 arange1d expects single output, got outputs={outputs}")
    out_name = str(outputs[0]).strip()
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"rvv cpu-loops v1 arange1d missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "i64")) != "i64":
        raise RuntimeError(f"rvv cpu-loops v1 arange1d supports only i64 output, got dtype={getattr(out_tt,'dtype','')}")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 arange1d expects rank-1 output, got {out_shape}")
    n_dim = int(out_shape[0])
    if n_dim <= 0:
        raise RuntimeError(f"invalid arange1d output shape: {out_shape}")

    io_names = _io_arg_order(intent)
    if not io_names:
        raise RuntimeError("rvv cpu-loops v1 arange1d missing io tensors")

    arg_decls: list[str] = []
    arg_ssa: dict[str, str] = {}
    memref_ty_by_name: dict[str, str] = {}
    for name in io_names:
        tt = (intent.tensors or {}).get(name)
        if tt is None:
            raise RuntimeError(f"rvv cpu-loops v1 arange1d missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "i64"))
        sh = _shape(name, intent=intent, bindings=bindings)
        if name == out_name:
            if elem_ty != "i64":
                raise RuntimeError(f"rvv cpu-loops v1 arange1d output must be i64, got {name} dtype={getattr(tt,'dtype','')}")
            if len(sh) != 1 or int(sh[0]) != int(n_dim):
                raise RuntimeError(f"rvv cpu-loops v1 arange1d output shape mismatch: out_shape={out_shape} got {name} shape={sh}")
            numel = int(n_dim)
        else:
            if elem_ty != "i64":
                raise RuntimeError(f"rvv cpu-loops v1 arange1d expects scalar i64 inputs, got {name} dtype={getattr(tt,'dtype','')}")
            if len(sh) != 0:
                raise RuntimeError(f"rvv cpu-loops v1 arange1d expects scalar inputs, got {name} shape={sh}")
            numel = 1
        memref_ty = f"memref<{numel}x{elem_ty}>"
        ssa = f"%{_mlir_ident(name)}"
        arg_ssa[name] = ssa
        memref_ty_by_name[name] = memref_ty
        arg_decls.append(f"    {ssa}: {memref_ty}")

    # Expect the standard FlagGems arange seed: inputs are scalar start/step.
    scalar_inputs = [n for n in io_names if n != out_name]
    if len(scalar_inputs) < 2:
        raise RuntimeError(f"rvv cpu-loops v1 arange1d expects at least 2 scalar inputs (start, step), got io={io_names}")
    start_name = None
    step_name = None
    for n in scalar_inputs:
        if str(n).lower() == "start":
            start_name = n
        if str(n).lower() == "step":
            step_name = n
    if start_name is None or step_name is None:
        # Fall back to stable order: [start, step, out]
        start_name, step_name = scalar_inputs[0], scalar_inputs[1]

    out_memref = memref_ty_by_name[out_name]
    start_memref = memref_ty_by_name[start_name]
    step_memref = memref_ty_by_name[step_name]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}(")
    lines.append(",\n".join(arg_decls))
    lines.append("  ) {")
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %c1 = arith.constant 1 : index")
    lines.append(f"  %cN = arith.constant {n_dim} : index")
    lines.append(f"  %start_val = memref.load {arg_ssa[start_name]}[%c0] : {start_memref}")
    lines.append(f"  %step_val = memref.load {arg_ssa[step_name]}[%c0] : {step_memref}")
    lines.append("  scf.for %i = %c0 to %cN step %c1 {")
    lines.append("    %ii64 = arith.index_cast %i : index to i64")
    lines.append("    %mul = arith.muli %ii64, %step_val : i64")
    lines.append("    %v = arith.addi %mul, %start_val : i64")
    lines.append(f"    memref.store %v, {arg_ssa[out_name]}[%i] : {out_memref}")
    lines.append("  }")
    lines.append("  return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_transpose2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 supports only f32 outputs")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 transpose2d expects rank-2 output, got {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    if len(ops) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 transpose2d expects exactly 1 op, got {len(ops)}")
    op0 = ops[0]
    op_name = str(getattr(op0, "op", "")).strip()
    ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
    attrs = dict(getattr(op0, "attrs", {}) or {})
    perm = list(attrs.get("perm") or []) if isinstance(attrs.get("perm"), list) else []
    if op_name != "transpose" or len(ins) != 1 or perm != [1, 0]:
        raise RuntimeError(f"rvv cpu-loops v1 unsupported transpose op={op_name} inputs={ins} perm={perm}")
    in_name = str(ins[0])
    in_tt = (intent.tensors or {}).get(in_name)
    if in_tt is None:
        raise RuntimeError(f"missing input tensor spec: {in_name}")
    if _dtype(getattr(in_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 supports only f32 inputs for transpose2d")
    in_shape = _shape(in_name, intent=intent, bindings=bindings)
    if len(in_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 transpose2d expects rank-2 input, got {in_shape}")
    m_dim, n_dim = int(in_shape[0]), int(in_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"invalid transpose2d input shape: {in_shape}")
    if list(out_shape) != [n_dim, m_dim]:
        raise RuntimeError(f"rvv cpu-loops v1 transpose2d expects output shape [N,M], got out_shape={out_shape} in_shape={in_shape}")
    total = int(m_dim * n_dim)

    arg_order = [in_name, out_name]
    arg_decls: list[str] = []
    arg_ssa: dict[str, str] = {}
    memref_ty_by_name: dict[str, str] = {}
    for name in arg_order:
        tt = (intent.tensors or {}).get(name)
        if tt is None:
            raise RuntimeError(f"missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "f32"))
        if elem_ty != "f32":
            raise RuntimeError("rvv cpu-loops v1 transpose2d supports only f32 tensors")
        memref_ty = f"memref<{total}x{elem_ty}>"
        ssa = f"%{_mlir_ident(name)}"
        arg_ssa[name] = ssa
        memref_ty_by_name[name] = memref_ty
        arg_decls.append(f"    {ssa}: {memref_ty}")

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}(")
    lines.append(",\n".join(arg_decls))
    lines.append("  ) {")
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %c1 = arith.constant 1 : index")
    lines.append(f"  %cM = arith.constant {m_dim} : index")
    lines.append(f"  %cN = arith.constant {n_dim} : index")
    lines.append("  scf.for %m = %c0 to %cM step %c1 {")
    lines.append("    %mN = arith.muli %m, %cN : index")
    lines.append("    scf.for %n = %c0 to %cN step %c1 {")
    lines.append("      %in_idx = arith.addi %mN, %n : index")
    lines.append(f"      %v = memref.load {arg_ssa[in_name]}[%in_idx] : {memref_ty_by_name[in_name]}")
    lines.append("      %nM = arith.muli %n, %cM : index")
    lines.append("      %out_idx = arith.addi %nM, %m : index")
    lines.append(f"      memref.store %v, {arg_ssa[out_name]}[%out_idx] : {memref_ty_by_name[out_name]}")
    lines.append("    }")
    lines.append("  }")
    lines.append("  return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_row_reduce_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 supports only f32 outputs")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 row-reduce expects rank-1 output, got {out_shape}")
    m_dim = int(out_shape[0])
    if m_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    if len(ops) != 1:
        raise RuntimeError("rvv cpu-loops v1 row-reduce expects exactly 1 op")
    op0 = ops[0]
    op_name = str(getattr(op0, "op", "")).strip()
    ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
    attrs = dict(getattr(op0, "attrs", {}) or {})
    dims = attrs.get("dims")
    dims_list = list(dims) if isinstance(dims, list) else []
    if op_name not in {"reduce_sum", "reduce_max"} or dims_list != [1] or len(ins) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 unsupported row-reduce op={op_name} dims={dims_list} inputs={ins}")
    in_name = str(ins[0])
    in_tt = (intent.tensors or {}).get(in_name)
    if in_tt is None:
        raise RuntimeError(f"missing input tensor spec: {in_name}")
    if _dtype(getattr(in_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 supports only f32 inputs")
    in_shape = _shape(in_name, intent=intent, bindings=bindings)
    if len(in_shape) != 2 or int(in_shape[0]) != int(m_dim):
        raise RuntimeError(f"rvv cpu-loops v1 row-reduce expects [M,N] input, got {in_name} shape={in_shape}")
    n_dim = int(in_shape[1])
    if n_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 row-reduce expects N>0, got N={n_dim}")
    total = int(m_dim * n_dim)

    in_memref_ty = f"memref<{total}xf32>"
    out_memref_ty = f"memref<{m_dim}xf32>"

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}(%{_mlir_ident(in_name)}: {in_memref_ty}, %{_mlir_ident(out_name)}: {out_memref_ty}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cM = arith.constant {m_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    if op_name == "reduce_sum":
        lines.append("    %init = arith.constant 0.0 : f32")
    else:
        lines.append("    %init = arith.constant 0xFF800000 : f32")
    lines.append("    scf.for %m = %c0 to %cM step %c1 {")
    lines.append("      %base = arith.muli %m, %cN : index")
    lines.append("      %acc = scf.for %n = %c0 to %cN step %c1 iter_args(%x = %init) -> (f32) {")
    lines.append("        %idx = arith.addi %base, %n : index")
    lines.append(f"        %v = memref.load %{_mlir_ident(in_name)}[%idx] : {in_memref_ty}")
    if op_name == "reduce_sum":
        lines.append("        %y = arith.addf %x, %v : f32")
    else:
        lines.append("        %y = arith.maximumf %x, %v : f32")
    lines.append("        scf.yield %y : f32")
    lines.append("      }")
    lines.append(f"      memref.store %acc, %{_mlir_ident(out_name)}[%m] : {out_memref_ty}")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_row_mean_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    """
    Implements `row_mean`: out[m] = sum(inp[m, :]) / N.
    """

    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 row_mean expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 row_mean supports only f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 row_mean expects rank-1 output, got {out_shape}")
    m_dim = int(out_shape[0])
    if m_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    io_names = _io_arg_order(intent)
    ext_inputs = [n for n in io_names if n not in set(outputs)]
    inp_name = ""
    n_dim = 0
    for name in ext_inputs:
        tt = (intent.tensors or {}).get(str(name))
        if tt is None:
            continue
        if _dtype(getattr(tt, "dtype", "f32")) != "f32":
            continue
        shp = _shape(str(name), intent=intent, bindings=bindings)
        if len(shp) == 2 and int(shp[0]) == int(m_dim) and int(shp[1]) > 0:
            inp_name = str(name)
            n_dim = int(shp[1])
            break
    if not inp_name:
        raise RuntimeError("rvv cpu-loops v1 row_mean expects external f32 input [M,N]")
    if n_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 row_mean expects N>0, got N={n_dim}")

    total = int(m_dim * n_dim)
    in_memref_ty = f"memref<{total}xf32>"
    out_memref_ty = f"memref<{m_dim}xf32>"

    n_lit = _f32_lit(float(n_dim))

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(
        f"  func.func @{_mlir_ident(kernel_name)}(%{_mlir_ident(inp_name)}: {in_memref_ty}, %{_mlir_ident(out_name)}: {out_memref_ty}) {{"
    )
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cM = arith.constant {m_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append("    %init = arith.constant 0.0 : f32")
    lines.append(f"    %nf = arith.constant {n_lit} : f32")
    lines.append("    scf.for %m = %c0 to %cM step %c1 {")
    lines.append("      %base = arith.muli %m, %cN : index")
    lines.append("      %acc = scf.for %n = %c0 to %cN step %c1 iter_args(%x = %init) -> (f32) {")
    lines.append("        %idx = arith.addi %base, %n : index")
    lines.append(f"        %v = memref.load %{_mlir_ident(inp_name)}[%idx] : {in_memref_ty}")
    lines.append("        %y = arith.addf %x, %v : f32")
    lines.append("        scf.yield %y : f32")
    lines.append("      }")
    lines.append("      %mean = arith.divf %acc, %nf : f32")
    lines.append(f"      memref.store %mean, %{_mlir_ident(out_name)}[%m] : {out_memref_ty}")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_row_reduce_any_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    """
    Implements `any_kernel_dim`: out[m] = any(inp[m, :] != 0).

    ABI notes:
      - bool/i1 tensors are represented as i8 in the exported memref ABI.
    """

    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "bool")) != "i8":
        raise RuntimeError("rvv cpu-loops v1 reduce_any expects bool output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 reduce_any expects rank-1 output, got {out_shape}")
    m_dim = int(out_shape[0])
    if m_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    # Canonical intent: const(0) -> ne -> reduce_any(axis=1,dims=[1])
    if not ops:
        raise RuntimeError("rvv cpu-loops v1 reduce_any requires non-empty ops")

    io_names = _io_arg_order(intent)
    ext_inputs = [n for n in io_names if n not in set(outputs)]
    inp_name = ""
    n_dim = 0
    for name in ext_inputs:
        tt = (intent.tensors or {}).get(str(name))
        if tt is None:
            continue
        if _dtype(getattr(tt, "dtype", "f32")) != "f32":
            continue
        shp = _shape(str(name), intent=intent, bindings=bindings)
        if len(shp) == 2 and int(shp[0]) == int(m_dim) and int(shp[1]) > 0:
            inp_name = str(name)
            n_dim = int(shp[1])
            break
    if not inp_name:
        raise RuntimeError("rvv cpu-loops v1 reduce_any expects external input[M,N] f32 tensor")

    total = int(m_dim * n_dim)
    in_memref_ty = f"memref<{total}xf32>"
    out_memref_ty = f"memref<{m_dim}xi8>"

    arg_types: dict[str, str] = {inp_name: in_memref_ty, out_name: out_memref_ty}
    for n in list(io_names):
        if n not in arg_types:
            raise RuntimeError(
                f"rvv cpu-loops v1 reduce_any: unexpected IO tensor {n!r} (known={sorted(arg_types)}) for kernel={kernel_name!r}"
            )
    arg_decls = [f"%{_mlir_ident(n)}: {arg_types[n]}" for n in io_names]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cM = arith.constant {m_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    lines.append("    %false = arith.constant false")
    lines.append("    scf.for %m = %c0 to %cM step %c1 {")
    lines.append("      %base = arith.muli %m, %cN : index")
    lines.append("      %acc = scf.for %n = %c0 to %cN step %c1 iter_args(%a = %false) -> (i1) {")
    lines.append("        %i = arith.addi %base, %n : index")
    lines.append(f"        %x = memref.load %{_mlir_ident(inp_name)}[%i] : {in_memref_ty}")
    lines.append("        %nz = arith.cmpf une, %x, %c0f : f32")
    lines.append("        %a2 = arith.ori %a, %nz : i1")
    lines.append("        scf.yield %a2 : i1")
    lines.append("      }")
    lines.append("      %out_i8 = arith.extui %acc : i1 to i8")
    lines.append(f"      memref.store %out_i8, %{_mlir_ident(out_name)}[%m] : {out_memref_ty}")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_row_argminmax_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
    mode: str,
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "i32")) != "i32":
        raise RuntimeError("rvv cpu-loops v1 argmin/argmax expects i32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 argmin/argmax expects rank-1 output, got {out_shape}")
    m_dim = int(out_shape[0])
    if m_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    if len(ops) != 1:
        raise RuntimeError("rvv cpu-loops v1 argmin/argmax expects exactly 1 op")
    op0 = ops[0]
    op_name = str(getattr(op0, "op", "")).strip()
    ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
    attrs = dict(getattr(op0, "attrs", {}) or {})
    axis = attrs.get("axis")
    try:
        axis_i = int(axis)
    except Exception:
        axis_i = None
    expected_op = "argmax" if mode == "max" else "argmin" if mode == "min" else None
    if expected_op is None:
        raise RuntimeError(f"invalid mode for argmin/argmax: {mode!r}")
    if op_name != expected_op or axis_i != 1 or len(ins) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 unsupported {expected_op}: axis={axis!r} inputs={ins}")
    in_name = str(ins[0])
    in_tt = (intent.tensors or {}).get(in_name)
    if in_tt is None:
        raise RuntimeError(f"missing input tensor spec: {in_name}")
    if _dtype(getattr(in_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 argmin/argmax supports only f32 inputs")
    in_shape = _shape(in_name, intent=intent, bindings=bindings)
    if len(in_shape) != 2 or int(in_shape[0]) != int(m_dim):
        raise RuntimeError(f"rvv cpu-loops v1 argmin/argmax expects [M,N] input, got {in_name} shape={in_shape}")
    n_dim = int(in_shape[1])
    if n_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 argmin/argmax expects N>0, got N={n_dim}")
    total = int(m_dim * n_dim)

    in_memref_ty = f"memref<{total}xf32>"
    out_memref_ty = f"memref<{m_dim}xi32>"

    cmp_op = "ogt" if mode == "max" else "olt"
    init_hex = "0xFF800000" if mode == "max" else "0x7F800000"

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(
        f"  func.func @{_mlir_ident(kernel_name)}(%{_mlir_ident(in_name)}: {in_memref_ty}, %{_mlir_ident(out_name)}: {out_memref_ty}) {{"
    )
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cM = arith.constant {m_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append("    %c0i32 = arith.constant 0 : i32")
    lines.append(f"    %init = arith.constant {init_hex} : f32")
    lines.append("    scf.for %m = %c0 to %cM step %c1 {")
    lines.append("      %base = arith.muli %m, %cN : index")
    lines.append(
        "      %best, %best_idx = scf.for %n = %c0 to %cN step %c1 iter_args(%v = %init, %idx = %c0i32) -> (f32, i32) {"
    )
    lines.append("        %i = arith.addi %base, %n : index")
    lines.append(f"        %x = memref.load %{_mlir_ident(in_name)}[%i] : {in_memref_ty}")
    lines.append(f"        %cmp = arith.cmpf {cmp_op}, %x, %v : f32")
    lines.append("        %eq = arith.cmpf oeq, %x, %v : f32")
    lines.append("        %n_i32 = arith.index_cast %n : index to i32")
    lines.append("        %lt = arith.cmpi slt, %n_i32, %idx : i32")
    lines.append("        %tie = arith.andi %eq, %lt : i1")
    lines.append("        %take = arith.ori %cmp, %tie : i1")
    lines.append("        %v2 = arith.select %take, %x, %v : f32")
    lines.append("        %idx2 = arith.select %take, %n_i32, %idx : i32")
    lines.append("        scf.yield %v2, %idx2 : f32, i32")
    lines.append("      }")
    lines.append(f"      memref.store %best_idx, %{_mlir_ident(out_name)}[%m] : {out_memref_ty}")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_matmul2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
    relu: bool,
    require_bias: bool,
    require_row_col_masks: bool,
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 matmul expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 matmul expects rank-2 output, got {out_shape}")
    m_dim, n_dim = int(out_shape[0]), int(out_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    tensors = dict(intent.tensors or {})
    rank2_f32: list[str] = []
    rank1_f32: list[str] = []
    rank1_i8: list[str] = []
    for name, tt in tensors.items():
        nm = str(name).strip()
        if not nm or nm == out_name:
            continue
        dt = _dtype(getattr(tt, "dtype", "f32"))
        shp = _shape(nm, intent=intent, bindings=bindings)
        if dt == "f32" and len(shp) == 2:
            rank2_f32.append(nm)
        elif dt == "f32" and len(shp) == 1:
            rank1_f32.append(nm)
        elif dt == "i8" and len(shp) == 1:
            rank1_i8.append(nm)

    a_name = ""
    b_name = ""
    k_dim = 0
    for a in list(rank2_f32):
        a_shape = _shape(a, intent=intent, bindings=bindings)
        if len(a_shape) != 2 or int(a_shape[0]) != int(m_dim):
            continue
        for b in list(rank2_f32):
            if b == a:
                continue
            b_shape = _shape(b, intent=intent, bindings=bindings)
            if len(b_shape) != 2 or int(b_shape[1]) != int(n_dim):
                continue
            if int(a_shape[1]) <= 0 or int(b_shape[0]) <= 0:
                continue
            if int(a_shape[1]) != int(b_shape[0]):
                continue
            a_name, b_name = str(a), str(b)
            k_dim = int(a_shape[1])
            break
        if a_name and b_name:
            break
    if not (a_name and b_name and k_dim > 0):
        raise RuntimeError(
            f"rvv cpu-loops v1 matmul: failed to infer A/B for kernel={kernel_name!r} output={out_name!r}"
        )

    bias_name = ""
    if bool(require_bias):
        for b in list(rank1_f32):
            b_shape = _shape(b, intent=intent, bindings=bindings)
            if b_shape == [int(n_dim)]:
                bias_name = str(b)
                break
        if not bias_name:
            raise RuntimeError(f"rvv cpu-loops v1 matmul: missing bias [N] for kernel={kernel_name!r}")

    row_mask_name = ""
    col_mask_name = ""
    if bool(require_row_col_masks):
        for nm in list(rank1_i8):
            shp = _shape(nm, intent=intent, bindings=bindings)
            if shp == [int(m_dim)] and not row_mask_name:
                row_mask_name = str(nm)
            elif shp == [int(n_dim)] and not col_mask_name:
                col_mask_name = str(nm)
        if not (row_mask_name and col_mask_name):
            raise RuntimeError(f"rvv cpu-loops v1 matmul: missing row/col masks for kernel={kernel_name!r}")

    total_a = int(m_dim * k_dim)
    total_b = int(k_dim * n_dim)
    total_out = int(m_dim * n_dim)
    a_memref_ty = f"memref<{total_a}xf32>"
    b_memref_ty = f"memref<{total_b}xf32>"
    out_memref_ty = f"memref<{total_out}xf32>"
    bias_memref_ty = f"memref<{int(n_dim)}xf32>" if bias_name else ""
    rm_memref_ty = f"memref<{int(m_dim)}xi8>" if row_mask_name else ""
    cm_memref_ty = f"memref<{int(n_dim)}xi8>" if col_mask_name else ""

    arg_types: dict[str, str] = {a_name: a_memref_ty, b_name: b_memref_ty, out_name: out_memref_ty}
    if bias_name:
        arg_types[str(bias_name)] = bias_memref_ty
    if row_mask_name:
        arg_types[str(row_mask_name)] = rm_memref_ty
    if col_mask_name:
        arg_types[str(col_mask_name)] = cm_memref_ty

    arg_names = _io_arg_order(intent)
    for n in list(arg_names):
        if n not in arg_types:
            raise RuntimeError(
                f"rvv cpu-loops v1 matmul: unexpected IO tensor {n!r} (known={sorted(arg_types)}) for kernel={kernel_name!r}"
            )
    arg_decls = [f"%{_mlir_ident(n)}: {arg_types[n]}" for n in arg_names]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cM = arith.constant {m_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append(f"    %cK = arith.constant {k_dim} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    if relu:
        lines.append("    %c0f_relu = arith.constant 0.0 : f32")
    if row_mask_name or col_mask_name:
        lines.append("    %c0i8 = arith.constant 0 : i8")

    # Row-major matmul with optional epilogue.
    lines.append("    scf.for %m = %c0 to %cM step %c1 {")
    lines.append("      %a_row = arith.muli %m, %cK : index")
    lines.append("      %out_row = arith.muli %m, %cN : index")
    if row_mask_name:
        rm = f"%{_mlir_ident(row_mask_name)}"
        lines.append(f"      %rmv = memref.load {rm}[%m] : {rm_memref_ty}")
        lines.append("      %rmnz = arith.cmpi ne, %rmv, %c0i8 : i8")
    lines.append("      scf.for %n = %c0 to %cN step %c1 {")
    lines.append("        %acc = scf.for %k = %c0 to %cK step %c1 iter_args(%x = %c0f) -> (f32) {")
    lines.append("          %a_idx = arith.addi %a_row, %k : index")
    lines.append(f"          %av = memref.load %{_mlir_ident(a_name)}[%a_idx] : {a_memref_ty}")
    lines.append("          %b_row = arith.muli %k, %cN : index")
    lines.append("          %b_idx = arith.addi %b_row, %n : index")
    lines.append(f"          %bv = memref.load %{_mlir_ident(b_name)}[%b_idx] : {b_memref_ty}")
    lines.append("          %prod = arith.mulf %av, %bv : f32")
    lines.append("          %x2 = arith.addf %x, %prod : f32")
    lines.append("          scf.yield %x2 : f32")
    lines.append("        }")
    val_cur = "%acc"
    if bias_name:
        lines.append(f"        %bias = memref.load %{_mlir_ident(bias_name)}[%n] : {bias_memref_ty}")
        lines.append(f"        %accb = arith.addf {val_cur}, %bias : f32")
        val_cur = "%accb"
    if relu:
        lines.append(f"        %relu = arith.maximumf {val_cur}, %c0f_relu : f32")
        val_cur = "%relu"
    if col_mask_name:
        cm = f"%{_mlir_ident(col_mask_name)}"
        lines.append(f"        %cmv = memref.load {cm}[%n] : {cm_memref_ty}")
        lines.append("        %cmnz = arith.cmpi ne, %cmv, %c0i8 : i8")
        if row_mask_name:
            lines.append("        %cond = arith.andi %rmnz, %cmnz : i1")
            cond = "%cond"
        else:
            cond = "%cmnz"
        lines.append(f"        %masked = arith.select {cond}, {val_cur}, %c0f : f32")
        val_cur = "%masked"

    lines.append("        %out_idx = arith.addi %out_row, %n : index")
    lines.append(f"        memref.store {val_cur}, %{_mlir_ident(out_name)}[%out_idx] : {out_memref_ty}")
    lines.append("      }")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_row_softmax2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
    require_mask: bool,
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 softmax expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 softmax expects rank-2 output, got {out_shape}")
    m_dim, n_dim = int(out_shape[0]), int(out_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    tensors = dict(intent.tensors or {})
    arg_names = _io_arg_order(intent)

    in_name = ""
    # Prefer IO tensors over intermediate nodes.
    for nm in list(arg_names):
        if nm == out_name:
            continue
        tt = tensors.get(str(nm))
        if tt is None:
            continue
        if _dtype(getattr(tt, "dtype", "f32")) != "f32":
            continue
        shp = _shape(str(nm), intent=intent, bindings=bindings)
        if shp == [int(m_dim), int(n_dim)]:
            in_name = str(nm)
            break
    if not in_name:
        for name, tt in tensors.items():
            nm = str(name).strip()
            if not nm or nm == out_name:
                continue
            if _dtype(getattr(tt, "dtype", "f32")) != "f32":
                continue
            shp = _shape(nm, intent=intent, bindings=bindings)
            if shp == [int(m_dim), int(n_dim)]:
                in_name = nm
                break
    if not in_name:
        raise RuntimeError(f"rvv cpu-loops v1 softmax: failed to infer input for kernel={kernel_name!r}")

    mask_name = ""
    mask_dt = ""
    if bool(require_mask):
        # Prefer the public IO mask tensor, not internal casts (e.g. mask_bool).
        for nm in list(arg_names):
            nm = str(nm).strip()
            if not nm or nm in {out_name, in_name}:
                continue
            tt = tensors.get(nm)
            if tt is None:
                continue
            dt = _dtype(getattr(tt, "dtype", "bool"))
            if dt not in {"i8", "i32"}:
                continue
            shp = _shape(nm, intent=intent, bindings=bindings)
            if shp == [int(n_dim)]:
                mask_name = nm
                mask_dt = dt
                break
        if not mask_name:
            raise RuntimeError(f"rvv cpu-loops v1 softmax: missing mask [N] for kernel={kernel_name!r}")

    total = int(m_dim * n_dim)
    in_memref_ty = f"memref<{total}xf32>"
    out_memref_ty = f"memref<{total}xf32>"
    mask_elem_ty = "i8" if (not mask_dt) else str(mask_dt)
    mask_memref_ty = f"memref<{int(n_dim)}x{mask_elem_ty}>" if mask_name else ""

    arg_types: dict[str, str] = {in_name: in_memref_ty, out_name: out_memref_ty}
    if mask_name:
        arg_types[mask_name] = mask_memref_ty
    for n in list(arg_names):
        if n not in arg_types:
            raise RuntimeError(
                f"rvv cpu-loops v1 softmax: unexpected IO tensor {n!r} (known={sorted(arg_types)}) for kernel={kernel_name!r}"
            )
    arg_decls = [f"%{_mlir_ident(n)}: {arg_types[n]}" for n in arg_names]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cM = arith.constant {m_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    lines.append("    %neg_inf = arith.constant 0xFF800000 : f32")
    if mask_name:
        if mask_elem_ty == "i32":
            lines.append("    %c0mask = arith.constant 0 : i32")
        else:
            lines.append("    %c0mask = arith.constant 0 : i8")
        lines.append("    %masked_neg = arith.constant -1.0e9 : f32")

    lines.append("    scf.for %m = %c0 to %cM step %c1 {")
    lines.append("      %row = arith.muli %m, %cN : index")

    # Pass 1: max
    lines.append("      %mx = scf.for %n = %c0 to %cN step %c1 iter_args(%x = %neg_inf) -> (f32) {")
    lines.append("        %idx = arith.addi %row, %n : index")
    lines.append(f"        %v0 = memref.load %{_mlir_ident(in_name)}[%idx] : {in_memref_ty}")
    if mask_name:
        lines.append(f"        %m0 = memref.load %{_mlir_ident(mask_name)}[%n] : {mask_memref_ty}")
        lines.append(f"        %men = arith.cmpi ne, %m0, %c0mask : {mask_elem_ty}")
        lines.append("        %v = arith.select %men, %v0, %masked_neg : f32")
    else:
        lines.append("        %v = arith.addf %v0, %c0f : f32")
    lines.append("        %x2 = arith.maximumf %x, %v : f32")
    lines.append("        scf.yield %x2 : f32")
    lines.append("      }")

    # Pass 2: sum exp
    lines.append("      %sm = scf.for %n2 = %c0 to %cN step %c1 iter_args(%s = %c0f) -> (f32) {")
    lines.append("        %idx2 = arith.addi %row, %n2 : index")
    lines.append(f"        %u0 = memref.load %{_mlir_ident(in_name)}[%idx2] : {in_memref_ty}")
    if mask_name:
        lines.append(f"        %m1 = memref.load %{_mlir_ident(mask_name)}[%n2] : {mask_memref_ty}")
        lines.append(f"        %men1 = arith.cmpi ne, %m1, %c0mask : {mask_elem_ty}")
        lines.append("        %u = arith.select %men1, %u0, %masked_neg : f32")
    else:
        lines.append("        %u = arith.addf %u0, %c0f : f32")
    lines.append("        %shift = arith.subf %u, %mx : f32")
    lines.append("        %ex = math.exp %shift : f32")
    lines.append("        %s2 = arith.addf %s, %ex : f32")
    lines.append("        scf.yield %s2 : f32")
    lines.append("      }")

    # Pass 3: write
    lines.append("      scf.for %n3 = %c0 to %cN step %c1 {")
    lines.append("        %idx3 = arith.addi %row, %n3 : index")
    lines.append(f"        %w0 = memref.load %{_mlir_ident(in_name)}[%idx3] : {in_memref_ty}")
    if mask_name:
        lines.append(f"        %m2 = memref.load %{_mlir_ident(mask_name)}[%n3] : {mask_memref_ty}")
        lines.append(f"        %men2 = arith.cmpi ne, %m2, %c0mask : {mask_elem_ty}")
        lines.append("        %w = arith.select %men2, %w0, %masked_neg : f32")
    else:
        lines.append("        %w = arith.addf %w0, %c0f : f32")
    lines.append("        %shift2 = arith.subf %w, %mx : f32")
    lines.append("        %ex2 = math.exp %shift2 : f32")
    lines.append("        %y = arith.divf %ex2, %sm : f32")
    lines.append(f"        memref.store %y, %{_mlir_ident(out_name)}[%idx3] : {out_memref_ty}")
    lines.append("      }")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_gather2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 gather2d expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 gather2d expects rank-1 output, got {out_shape}")
    l_dim = int(out_shape[0])
    if l_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    tensors = dict(intent.tensors or {})
    inp_name = ""
    row_idx_name = ""
    col_idx_name = ""
    m_dim = 0
    n_dim = 0
    for name, tt in tensors.items():
        nm = str(name).strip()
        if not nm or nm == out_name:
            continue
        dt = _dtype(getattr(tt, "dtype", "f32"))
        shp = _shape(nm, intent=intent, bindings=bindings)
        if dt == "f32" and len(shp) == 2 and not inp_name:
            inp_name = nm
            m_dim, n_dim = int(shp[0]), int(shp[1])
        if dt == "i32" and shp == [int(l_dim)]:
            if not row_idx_name:
                row_idx_name = nm
            elif not col_idx_name and nm != row_idx_name:
                col_idx_name = nm
    if not (inp_name and row_idx_name and col_idx_name and m_dim > 0 and n_dim > 0):
        raise RuntimeError(f"rvv cpu-loops v1 gather2d: failed to infer inputs for kernel={kernel_name!r}")

    total_inp = int(m_dim * n_dim)
    inp_memref_ty = f"memref<{total_inp}xf32>"
    idx_memref_ty = f"memref<{l_dim}xi32>"
    out_memref_ty = f"memref<{l_dim}xf32>"

    arg_types: dict[str, str] = {
        inp_name: inp_memref_ty,
        row_idx_name: idx_memref_ty,
        col_idx_name: idx_memref_ty,
        out_name: out_memref_ty,
    }
    arg_names = _io_arg_order(intent)
    for n in list(arg_names):
        if n not in arg_types:
            raise RuntimeError(
                f"rvv cpu-loops v1 gather2d: unexpected IO tensor {n!r} (known={sorted(arg_types)}) for kernel={kernel_name!r}"
            )
    arg_decls = [f"%{_mlir_ident(n)}: {arg_types[n]}" for n in arg_names]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cL = arith.constant {l_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append("    scf.for %i = %c0 to %cL step %c1 {")
    lines.append(f"      %r32 = memref.load %{_mlir_ident(row_idx_name)}[%i] : {idx_memref_ty}")
    lines.append(f"      %c32 = memref.load %{_mlir_ident(col_idx_name)}[%i] : {idx_memref_ty}")
    lines.append("      %r = arith.index_cast %r32 : i32 to index")
    lines.append("      %c = arith.index_cast %c32 : i32 to index")
    lines.append("      %mul = arith.muli %r, %cN : index")
    lines.append("      %idx = arith.addi %mul, %c : index")
    lines.append(f"      %x = memref.load %{_mlir_ident(inp_name)}[%idx] : {inp_memref_ty}")
    lines.append(f"      memref.store %x, %{_mlir_ident(out_name)}[%i] : {out_memref_ty}")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_grouped_row_sum2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 grouped_row_sum2d expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 grouped_row_sum2d expects rank-2 output, got {out_shape}")
    m_dim, g_dim = int(out_shape[0]), int(out_shape[1])
    if m_dim <= 0 or g_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    tensors = dict(intent.tensors or {})
    in_name = ""
    n_dim = 0
    for name, tt in tensors.items():
        nm = str(name).strip()
        if not nm or nm == out_name:
            continue
        if _dtype(getattr(tt, "dtype", "f32")) != "f32":
            continue
        shp = _shape(nm, intent=intent, bindings=bindings)
        if len(shp) == 2 and int(shp[0]) == int(m_dim):
            in_name = nm
            n_dim = int(shp[1])
            break
    if not (in_name and n_dim > 0):
        raise RuntimeError(f"rvv cpu-loops v1 grouped_row_sum2d: failed to infer input for kernel={kernel_name!r}")

    group_size = int(bindings.get("group_size") or bindings.get("GROUP_SIZE") or 0)
    if group_size <= 0:
        if int(g_dim) > 0 and (int(n_dim) % int(g_dim)) == 0:
            group_size = int(n_dim) // int(g_dim)
    if group_size <= 0 or (int(n_dim) % int(group_size)) != 0 or (int(n_dim) // int(group_size)) != int(g_dim):
        raise RuntimeError(
            f"rvv cpu-loops v1 grouped_row_sum2d expects N divisible by group_size and G=N/group_size; "
            f"got M={m_dim} N={n_dim} G={g_dim} group_size={group_size}"
        )

    total_in = int(m_dim * n_dim)
    total_out = int(m_dim * g_dim)
    in_memref_ty = f"memref<{total_in}xf32>"
    out_memref_ty = f"memref<{total_out}xf32>"

    arg_types: dict[str, str] = {in_name: in_memref_ty, out_name: out_memref_ty}
    arg_names = _io_arg_order(intent)
    for n in list(arg_names):
        if n not in arg_types:
            raise RuntimeError(
                f"rvv cpu-loops v1 grouped_row_sum2d: unexpected IO tensor {n!r} (known={sorted(arg_types)}) for kernel={kernel_name!r}"
            )
    arg_decls = [f"%{_mlir_ident(n)}: {arg_types[n]}" for n in arg_names]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cM = arith.constant {m_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append(f"    %cG = arith.constant {g_dim} : index")
    lines.append(f"    %cGS = arith.constant {group_size} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    lines.append("    scf.for %m = %c0 to %cM step %c1 {")
    lines.append("      %row = arith.muli %m, %cN : index")
    lines.append("      %out_row = arith.muli %m, %cG : index")
    lines.append("      scf.for %g = %c0 to %cG step %c1 {")
    lines.append("        %g0 = arith.muli %g, %cGS : index")
    lines.append("        %base = arith.addi %row, %g0 : index")
    lines.append("        %acc = scf.for %i = %c0 to %cGS step %c1 iter_args(%x = %c0f) -> (f32) {")
    lines.append("          %idx = arith.addi %base, %i : index")
    lines.append(f"          %v = memref.load %{_mlir_ident(in_name)}[%idx] : {in_memref_ty}")
    lines.append("          %x2 = arith.addf %x, %v : f32")
    lines.append("          scf.yield %x2 : f32")
    lines.append("        }")
    lines.append("        %o = arith.addi %out_row, %g : index")
    lines.append(f"        memref.store %acc, %{_mlir_ident(out_name)}[%o] : {out_memref_ty}")
    lines.append("      }")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"

def _emit_row_reduce_prod_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 reduce_prod expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 reduce_prod expects rank-1 output, got {out_shape}")
    m_dim = int(out_shape[0])
    if m_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    if len(ops) != 1:
        raise RuntimeError("rvv cpu-loops v1 reduce_prod expects exactly 1 op")
    op0 = ops[0]
    op_name = str(getattr(op0, "op", "")).strip()
    ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
    attrs = dict(getattr(op0, "attrs", {}) or {})
    dims = attrs.get("dims")
    dims_list = list(dims) if isinstance(dims, list) else []
    if op_name != "reduce_prod" or dims_list != [1] or len(ins) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 unsupported reduce_prod dims={dims_list} inputs={ins}")
    in_name = str(ins[0])
    in_tt = (intent.tensors or {}).get(in_name)
    if in_tt is None:
        raise RuntimeError(f"missing input tensor spec: {in_name}")
    if _dtype(getattr(in_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 reduce_prod supports only f32 inputs")
    in_shape = _shape(in_name, intent=intent, bindings=bindings)
    if len(in_shape) != 2 or int(in_shape[0]) != int(m_dim):
        raise RuntimeError(f"rvv cpu-loops v1 reduce_prod expects [M,N] input, got {in_name} shape={in_shape}")
    n_dim = int(in_shape[1])
    if n_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 reduce_prod expects N>0, got N={n_dim}")
    total = int(m_dim * n_dim)

    in_memref_ty = f"memref<{total}xf32>"
    out_memref_ty = f"memref<{m_dim}xf32>"

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(
        f"  func.func @{_mlir_ident(kernel_name)}(%{_mlir_ident(in_name)}: {in_memref_ty}, %{_mlir_ident(out_name)}: {out_memref_ty}) {{"
    )
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cM = arith.constant {m_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append("    %init = arith.constant 1.0 : f32")
    lines.append("    scf.for %m = %c0 to %cM step %c1 {")
    lines.append("      %base = arith.muli %m, %cN : index")
    lines.append("      %acc = scf.for %n = %c0 to %cN step %c1 iter_args(%x = %init) -> (f32) {")
    lines.append("        %idx = arith.addi %base, %n : index")
    lines.append(f"        %v = memref.load %{_mlir_ident(in_name)}[%idx] : {in_memref_ty}")
    lines.append("        %y = arith.mulf %x, %v : f32")
    lines.append("        scf.yield %y : f32")
    lines.append("      }")
    lines.append(f"      memref.store %acc, %{_mlir_ident(out_name)}[%m] : {out_memref_ty}")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_row_reduce_min_argmin_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 min_dim expects two outputs, got outputs={outputs}")
    out_value, out_index = outputs
    out_value_tt = (intent.tensors or {}).get(out_value)
    out_index_tt = (intent.tensors or {}).get(out_index)
    if out_value_tt is None or out_index_tt is None:
        raise RuntimeError(f"missing output tensor specs: {outputs}")
    if _dtype(getattr(out_value_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 min_dim expects f32 out_value")
    if _dtype(getattr(out_index_tt, "dtype", "i32")) != "i32":
        raise RuntimeError("rvv cpu-loops v1 min_dim expects i32 indices")
    out_shape = _shape(out_value, intent=intent, bindings=bindings)
    idx_shape = _shape(out_index, intent=intent, bindings=bindings)
    if len(out_shape) != 1 or list(out_shape) != list(idx_shape):
        raise RuntimeError(
            f"rvv cpu-loops v1 min_dim expects matching rank-1 outputs, got value={out_shape} idx={idx_shape}"
        )
    m_dim = int(out_shape[0])
    if m_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_value}: {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    if len(ops) != 2:
        raise RuntimeError("rvv cpu-loops v1 min_dim expects exactly 2 ops")
    reduce_op = None
    argmin_op = None
    for op in ops:
        nm = str(getattr(op, "op", "")).strip()
        if nm == "reduce_min" and reduce_op is None:
            reduce_op = op
        elif nm == "argmin" and argmin_op is None:
            argmin_op = op
    if reduce_op is None or argmin_op is None:
        raise RuntimeError(
            f"rvv cpu-loops v1 min_dim expects reduce_min+argmin ops, got {[str(getattr(o, 'op', '')) for o in ops]}"
        )
    red_attrs = dict(getattr(reduce_op, "attrs", {}) or {})
    dims = red_attrs.get("dims")
    dims_list = list(dims) if isinstance(dims, list) else []
    if dims_list != [1]:
        raise RuntimeError(f"rvv cpu-loops v1 min_dim expects reduce_min dims=[1], got {dims_list}")
    argmin_attrs = dict(getattr(argmin_op, "attrs", {}) or {})
    axis = argmin_attrs.get("axis")
    try:
        axis_i = int(axis)
    except Exception:
        axis_i = None
    if axis_i != 1:
        raise RuntimeError(f"rvv cpu-loops v1 min_dim expects argmin axis=1, got {axis!r}")

    red_ins = [str(x).strip() for x in list(getattr(reduce_op, "inputs", []) or []) if str(x).strip()]
    arg_ins = [str(x).strip() for x in list(getattr(argmin_op, "inputs", []) or []) if str(x).strip()]
    if len(red_ins) != 1 or len(arg_ins) != 1 or str(red_ins[0]) != str(arg_ins[0]):
        raise RuntimeError(
            f"rvv cpu-loops v1 min_dim expects shared single input, got reduce={red_ins} argmin={arg_ins}"
        )
    in_name = str(red_ins[0])
    in_tt = (intent.tensors or {}).get(in_name)
    if in_tt is None:
        raise RuntimeError(f"missing input tensor spec: {in_name}")
    if _dtype(getattr(in_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 min_dim supports only f32 inputs")
    in_shape = _shape(in_name, intent=intent, bindings=bindings)
    if len(in_shape) != 2 or int(in_shape[0]) != int(m_dim):
        raise RuntimeError(f"rvv cpu-loops v1 min_dim expects [M,N] input, got {in_name} shape={in_shape}")
    n_dim = int(in_shape[1])
    if n_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 min_dim expects N>0, got N={n_dim}")
    total = int(m_dim * n_dim)

    in_memref_ty = f"memref<{total}xf32>"
    out_value_ty = f"memref<{m_dim}xf32>"
    out_index_ty = f"memref<{m_dim}xi32>"

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(
        f"  func.func @{_mlir_ident(kernel_name)}(%{_mlir_ident(in_name)}: {in_memref_ty}, %{_mlir_ident(out_value)}: {out_value_ty}, %{_mlir_ident(out_index)}: {out_index_ty}) {{"
    )
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cM = arith.constant {m_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append("    %c0i32 = arith.constant 0 : i32")
    lines.append("    %init = arith.constant 0x7F800000 : f32")
    lines.append("    scf.for %m = %c0 to %cM step %c1 {")
    lines.append("      %base = arith.muli %m, %cN : index")
    lines.append(
        "      %best, %best_idx = scf.for %n = %c0 to %cN step %c1 iter_args(%v = %init, %idx = %c0i32) -> (f32, i32) {"
    )
    lines.append("        %i = arith.addi %base, %n : index")
    lines.append(f"        %x = memref.load %{_mlir_ident(in_name)}[%i] : {in_memref_ty}")
    lines.append("        %cmp = arith.cmpf olt, %x, %v : f32")
    lines.append("        %eq = arith.cmpf oeq, %x, %v : f32")
    lines.append("        %n_i32 = arith.index_cast %n : index to i32")
    lines.append("        %lt = arith.cmpi slt, %n_i32, %idx : i32")
    lines.append("        %tie = arith.andi %eq, %lt : i1")
    lines.append("        %take = arith.ori %cmp, %tie : i1")
    lines.append("        %v2 = arith.select %take, %x, %v : f32")
    lines.append("        %idx2 = arith.select %take, %n_i32, %idx : i32")
    lines.append("        scf.yield %v2, %idx2 : f32, i32")
    lines.append("      }")
    lines.append(f"      memref.store %best, %{_mlir_ident(out_value)}[%m] : {out_value_ty}")
    lines.append(f"      memref.store %best_idx, %{_mlir_ident(out_index)}[%m] : {out_index_ty}")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_reduce_min_all_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 reduce_min_all expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 0:
        raise RuntimeError(f"rvv cpu-loops v1 reduce_min_all expects scalar output, got {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    if len(ops) != 1:
        raise RuntimeError("rvv cpu-loops v1 reduce_min_all expects exactly 1 op")
    op0 = ops[0]
    op_name = str(getattr(op0, "op", "")).strip()
    ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
    attrs = dict(getattr(op0, "attrs", {}) or {})
    dims = attrs.get("dims")
    dims_list = list(dims) if isinstance(dims, list) else []
    if op_name != "reduce_min" or dims_list != [0, 1] or len(ins) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 unsupported reduce_min dims={dims_list} inputs={ins}")
    in_name = str(ins[0])
    in_tt = (intent.tensors or {}).get(in_name)
    if in_tt is None:
        raise RuntimeError(f"missing input tensor spec: {in_name}")
    if _dtype(getattr(in_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 reduce_min_all supports only f32 inputs")
    in_shape = _shape(in_name, intent=intent, bindings=bindings)
    if len(in_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 reduce_min_all expects [M,N] input, got {in_name} shape={in_shape}")
    m_dim, n_dim = int(in_shape[0]), int(in_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 reduce_min_all expects M,N>0, got {in_shape}")
    total = int(m_dim * n_dim)

    in_memref_ty = f"memref<{total}xf32>"
    out_memref_ty = "memref<1xf32>"

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(
        f"  func.func @{_mlir_ident(kernel_name)}(%{_mlir_ident(in_name)}: {in_memref_ty}, %{_mlir_ident(out_name)}: {out_memref_ty}) {{"
    )
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cT = arith.constant {total} : index")
    lines.append("    %init = arith.constant 0x7F800000 : f32")
    lines.append("    %acc = scf.for %i = %c0 to %cT step %c1 iter_args(%x = %init) -> (f32) {")
    lines.append(f"      %v = memref.load %{_mlir_ident(in_name)}[%i] : {in_memref_ty}")
    lines.append("      %y = arith.minimumf %x, %v : f32")
    lines.append("      scf.yield %y : f32")
    lines.append("    }")
    lines.append(f"    memref.store %acc, %{_mlir_ident(out_name)}[%c0] : {out_memref_ty}")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_count_nonzero2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "i64")) != "i64":
        raise RuntimeError("rvv cpu-loops v1 count_nonzero2d expects i64 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 0:
        raise RuntimeError(f"rvv cpu-loops v1 count_nonzero2d expects scalar output, got {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    if not ops:
        raise RuntimeError("rvv cpu-loops v1 count_nonzero2d expects non-empty ops")
    input_name = None
    for op in ops:
        if str(getattr(op, "op", "")).strip() != "ne":
            continue
        ins = [str(x).strip() for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
        if len(ins) == 2:
            input_name = str(ins[0])
            break
    if not input_name:
        raise RuntimeError("rvv cpu-loops v1 count_nonzero2d: failed to locate input via ne op")
    in_tt = (intent.tensors or {}).get(str(input_name))
    if in_tt is None:
        raise RuntimeError(f"missing input tensor spec: {input_name}")
    if _dtype(getattr(in_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 count_nonzero2d supports only f32 inputs")
    in_shape = _shape(str(input_name), intent=intent, bindings=bindings)
    if len(in_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 count_nonzero2d expects [M,N] input, got {input_name} shape={in_shape}")
    m_dim, n_dim = int(in_shape[0]), int(in_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 count_nonzero2d expects M,N>0, got {in_shape}")
    total = int(m_dim * n_dim)

    in_memref_ty = f"memref<{total}xf32>"
    out_memref_ty = "memref<1xi64>"

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(
        f"  func.func @{_mlir_ident(kernel_name)}(%{_mlir_ident(input_name)}: {in_memref_ty}, %{_mlir_ident(out_name)}: {out_memref_ty}) {{"
    )
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cT = arith.constant {total} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    lines.append("    %c0i64 = arith.constant 0 : i64")
    lines.append("    %c1i64 = arith.constant 1 : i64")
    lines.append("    %sum = scf.for %i = %c0 to %cT step %c1 iter_args(%acc = %c0i64) -> (i64) {")
    lines.append(f"      %v = memref.load %{_mlir_ident(input_name)}[%i] : {in_memref_ty}")
    lines.append("      %p = arith.cmpf une, %v, %c0f : f32")
    lines.append("      %inc = arith.select %p, %c1i64, %c0i64 : i64")
    lines.append("      %acc2 = arith.addi %acc, %inc : i64")
    lines.append("      scf.yield %acc2 : i64")
    lines.append("    }")
    lines.append(f"    memref.store %sum, %{_mlir_ident(out_name)}[%c0] : {out_memref_ty}")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_trace2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 trace2d expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 0:
        raise RuntimeError(f"rvv cpu-loops v1 trace2d expects scalar output, got {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    if not ops:
        raise RuntimeError("rvv cpu-loops v1 trace2d expects non-empty ops")
    input_name = None
    for op in ops:
        if str(getattr(op, "op", "")).strip() != "where":
            continue
        ins = [str(x).strip() for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
        if len(ins) == 3:
            input_name = str(ins[1])
            break
    if not input_name:
        raise RuntimeError("rvv cpu-loops v1 trace2d: failed to locate input via where op")
    in_tt = (intent.tensors or {}).get(str(input_name))
    if in_tt is None:
        raise RuntimeError(f"missing input tensor spec: {input_name}")
    if _dtype(getattr(in_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 trace2d supports only f32 inputs")
    in_shape = _shape(str(input_name), intent=intent, bindings=bindings)
    if len(in_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 trace2d expects [M,N] input, got {input_name} shape={in_shape}")
    m_dim, n_dim = int(in_shape[0]), int(in_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 trace2d expects M,N>0, got {in_shape}")
    total = int(m_dim * n_dim)
    l_dim = int(min(m_dim, n_dim))

    in_memref_ty = f"memref<{total}xf32>"
    out_memref_ty = "memref<1xf32>"

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(
        f"  func.func @{_mlir_ident(kernel_name)}(%{_mlir_ident(input_name)}: {in_memref_ty}, %{_mlir_ident(out_name)}: {out_memref_ty}) {{"
    )
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cL = arith.constant {l_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    lines.append("    %sum = scf.for %i = %c0 to %cL step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("      %iN = arith.muli %i, %cN : index")
    lines.append("      %idx = arith.addi %iN, %i : index")
    lines.append(f"      %v = memref.load %{_mlir_ident(input_name)}[%idx] : {in_memref_ty}")
    lines.append("      %acc2 = arith.addf %acc, %v : f32")
    lines.append("      scf.yield %acc2 : f32")
    lines.append("    }")
    lines.append(f"    memref.store %sum, %{_mlir_ident(out_name)}[%c0] : {out_memref_ty}")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_allclose2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "bool")) != "i8":
        raise RuntimeError("rvv cpu-loops v1 allclose2d expects bool/i8 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 0:
        raise RuntimeError(f"rvv cpu-loops v1 allclose2d expects scalar output, got {out_shape}")

    def _tensor_is_scalar_f32(name: str) -> bool:
        tt = (intent.tensors or {}).get(str(name))
        if tt is None:
            return False
        if _dtype(getattr(tt, "dtype", "f32")) != "f32":
            return False
        return len(_shape(str(name), intent=intent, bindings=bindings)) == 0

    def _tensor_is_rank2_f32(name: str) -> bool:
        tt = (intent.tensors or {}).get(str(name))
        if tt is None:
            return False
        if _dtype(getattr(tt, "dtype", "f32")) != "f32":
            return False
        return len(_shape(str(name), intent=intent, bindings=bindings)) == 2

    ops = [op for op in list(intent.ops or []) if op is not None]
    if not ops:
        raise RuntimeError("rvv cpu-loops v1 allclose2d expects non-empty ops")

    a_name = None
    b_name = None
    for op in ops:
        if str(getattr(op, "op", "")).strip() != "sub":
            continue
        ins = [str(x).strip() for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
        if len(ins) == 2 and _tensor_is_rank2_f32(ins[0]) and _tensor_is_rank2_f32(ins[1]):
            a_name, b_name = str(ins[0]), str(ins[1])
            break
    if not (a_name and b_name):
        raise RuntimeError("rvv cpu-loops v1 allclose2d: failed to infer A/B via sub op")

    rtol_name = None
    for op in ops:
        if str(getattr(op, "op", "")).strip() != "mul":
            continue
        ins = [str(x).strip() for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
        if len(ins) != 2:
            continue
        if _tensor_is_scalar_f32(ins[0]) and _tensor_is_rank2_f32(ins[1]):
            rtol_name = str(ins[0])
            break
        if _tensor_is_rank2_f32(ins[0]) and _tensor_is_scalar_f32(ins[1]):
            rtol_name = str(ins[1])
            break
    if not rtol_name:
        raise RuntimeError("rvv cpu-loops v1 allclose2d: failed to infer rtol via mul op")

    atol_name = None
    for op in ops:
        if str(getattr(op, "op", "")).strip() != "add":
            continue
        ins = [str(x).strip() for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
        if len(ins) != 2:
            continue
        if _tensor_is_scalar_f32(ins[0]) and _tensor_is_rank2_f32(ins[1]):
            atol_name = str(ins[0])
            break
        if _tensor_is_rank2_f32(ins[0]) and _tensor_is_scalar_f32(ins[1]):
            atol_name = str(ins[1])
            break
    if not atol_name:
        raise RuntimeError("rvv cpu-loops v1 allclose2d: failed to infer atol via add op")

    a_shape = _shape(str(a_name), intent=intent, bindings=bindings)
    b_shape = _shape(str(b_name), intent=intent, bindings=bindings)
    if list(a_shape) != list(b_shape):
        raise RuntimeError(f"rvv cpu-loops v1 allclose2d expects A/B same shape, got A={a_shape} B={b_shape}")
    m_dim, n_dim = int(a_shape[0]), int(a_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 allclose2d expects M,N>0, got {a_shape}")
    total = int(m_dim * n_dim)

    a_memref_ty = f"memref<{total}xf32>"
    b_memref_ty = f"memref<{total}xf32>"
    rtol_memref_ty = "memref<1xf32>"
    atol_memref_ty = "memref<1xf32>"
    out_memref_ty = "memref<1xi8>"

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    arg_types = {
        str(a_name): a_memref_ty,
        str(b_name): b_memref_ty,
        str(atol_name): atol_memref_ty,
        str(rtol_name): rtol_memref_ty,
        str(out_name): out_memref_ty,
    }
    arg_order = sorted({str(a_name), str(b_name), str(atol_name), str(rtol_name)}) + [str(out_name)]
    arg_decls = [f"%{_mlir_ident(nm)}: {arg_types[nm]}" for nm in arg_order]
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cT = arith.constant {total} : index")
    lines.append("    %true = arith.constant 1 : i1")
    lines.append("    %false = arith.constant 0 : i1")
    lines.append(f"    %rtol_s = memref.load %{_mlir_ident(rtol_name)}[%c0] : {rtol_memref_ty}")
    lines.append(f"    %atol_s = memref.load %{_mlir_ident(atol_name)}[%c0] : {atol_memref_ty}")
    lines.append("    %any = scf.for %i = %c0 to %cT step %c1 iter_args(%acc = %false) -> (i1) {")
    lines.append(f"      %a = memref.load %{_mlir_ident(a_name)}[%i] : {a_memref_ty}")
    lines.append(f"      %b = memref.load %{_mlir_ident(b_name)}[%i] : {b_memref_ty}")
    lines.append("      %diff = arith.subf %a, %b : f32")
    lines.append("      %abs_diff = math.absf %diff : f32")
    lines.append("      %abs_b = math.absf %b : f32")
    lines.append("      %rtol_term = arith.mulf %rtol_s, %abs_b : f32")
    lines.append("      %tol = arith.addf %atol_s, %rtol_term : f32")
    lines.append("      %close = arith.cmpf ole, %abs_diff, %tol : f32")
    lines.append("      %not_close = arith.xori %close, %true : i1")
    lines.append("      %acc2 = arith.ori %acc, %not_close : i1")
    lines.append("      scf.yield %acc2 : i1")
    lines.append("    }")
    lines.append("    %allclose = arith.xori %any, %true : i1")
    lines.append("    %out_i8 = arith.extui %allclose : i1 to i8")
    lines.append(f"    memref.store %out_i8, %{_mlir_ident(out_name)}[%c0] : {out_memref_ty}")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_softmax_inner_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 softmax expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 softmax supports only f32 outputs")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 softmax expects rank-2 output, got {out_shape}")
    m_dim, n_dim = int(out_shape[0]), int(out_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")
    total = int(m_dim * n_dim)

    ops = [op for op in list(intent.ops or []) if op is not None]
    if not ops:
        raise RuntimeError("rvv cpu-loops v1 softmax_inner requires non-empty ops")

    in_name = ""
    softmax_ops = [op for op in ops if str(getattr(op, "op", "")).strip() == "softmax"]
    if softmax_ops:
        op0 = softmax_ops[0]
        ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
        axis = dict(getattr(op0, "attrs", {}) or {}).get("axis")
        try:
            axis_i = int(axis)
        except Exception:
            axis_i = None
        if axis_i != 1 or len(ins) != 1:
            raise RuntimeError(f"rvv cpu-loops v1 softmax_inner unsupported softmax: axis={axis!r} inputs={ins}")
        in_name = str(ins[0])
    else:
        reduce_max_ops = [op for op in ops if str(getattr(op, "op", "")).strip() == "reduce_max"]
        if not reduce_max_ops:
            raise RuntimeError(
                f"rvv cpu-loops v1 softmax_inner expects reduce_max or softmax op, got {[str(getattr(o,'op','')) for o in ops]}"
            )
        op0 = reduce_max_ops[0]
        ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
        dims = dict(getattr(op0, "attrs", {}) or {}).get("dims")
        dims_list = list(dims) if isinstance(dims, list) else []
        if dims_list != [1] or len(ins) != 1:
            raise RuntimeError(f"rvv cpu-loops v1 softmax_inner unsupported reduce_max: dims={dims_list} inputs={ins}")
        in_name = str(ins[0])
        allowed = {"reduce_max", "reduce_sum", "sub", "exp", "div", "broadcast_in_dim"}
        op_names = [str(getattr(op, "op", "")).strip() for op in ops]
        unexpected = sorted({n for n in op_names if n and n not in allowed})
        if unexpected:
            raise RuntimeError(f"rvv cpu-loops v1 softmax_inner unexpected ops: {unexpected}")
        required = {"reduce_max", "reduce_sum", "sub", "exp", "div"}
        missing = sorted([n for n in required if n not in set(op_names)])
        if missing:
            raise RuntimeError(f"rvv cpu-loops v1 softmax_inner missing ops: {missing}")
        for op in ops:
            op_name = str(getattr(op, "op", "")).strip()
            if op_name not in {"reduce_max", "reduce_sum"}:
                continue
            attrs = dict(getattr(op, "attrs", {}) or {})
            dims = attrs.get("dims")
            dims_list = list(dims) if isinstance(dims, list) else []
            if dims_list != [1]:
                raise RuntimeError(f"rvv cpu-loops v1 softmax_inner expects reduce dims=[1], got dims={dims_list}")
    in_tt = (intent.tensors or {}).get(in_name)
    if in_tt is None:
        raise RuntimeError(f"missing input tensor spec: {in_name}")
    if _dtype(getattr(in_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 softmax supports only f32 inputs")
    in_shape = _shape(in_name, intent=intent, bindings=bindings)
    if list(in_shape) != [m_dim, n_dim]:
        raise RuntimeError(
            f"rvv cpu-loops v1 softmax expects input shape [M,N] to match output; got {in_name} shape={in_shape} out_shape={out_shape}"
        )

    in_memref_ty = f"memref<{total}xf32>"
    out_memref_ty = f"memref<{total}xf32>"

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(
        f"  func.func @{_mlir_ident(kernel_name)}(%{_mlir_ident(in_name)}: {in_memref_ty}, %{_mlir_ident(out_name)}: {out_memref_ty}) {{"
    )
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cM = arith.constant {m_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    lines.append("    %init_max = arith.constant 0xFF800000 : f32")
    lines.append("    scf.for %m = %c0 to %cM step %c1 {")
    lines.append("      %base = arith.muli %m, %cN : index")
    lines.append("      %row_max = scf.for %n = %c0 to %cN step %c1 iter_args(%mx = %init_max) -> (f32) {")
    lines.append("        %idx = arith.addi %base, %n : index")
    lines.append(f"        %x = memref.load %{_mlir_ident(in_name)}[%idx] : {in_memref_ty}")
    lines.append("        %mx2 = arith.maximumf %mx, %x : f32")
    lines.append("        scf.yield %mx2 : f32")
    lines.append("      }")
    lines.append("      %sum = scf.for %n = %c0 to %cN step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("        %idx = arith.addi %base, %n : index")
    lines.append(f"        %x = memref.load %{_mlir_ident(in_name)}[%idx] : {in_memref_ty}")
    lines.append("        %centered = arith.subf %x, %row_max : f32")
    lines.append("        %e = math.exp %centered : f32")
    lines.append("        %acc2 = arith.addf %acc, %e : f32")
    lines.append("        scf.yield %acc2 : f32")
    lines.append("      }")
    lines.append("      scf.for %n = %c0 to %cN step %c1 {")
    lines.append("        %idx = arith.addi %base, %n : index")
    lines.append(f"        %x = memref.load %{_mlir_ident(in_name)}[%idx] : {in_memref_ty}")
    lines.append("        %centered = arith.subf %x, %row_max : f32")
    lines.append("        %e = math.exp %centered : f32")
    lines.append("        %y = arith.divf %e, %sum : f32")
    lines.append(f"        memref.store %y, %{_mlir_ident(out_name)}[%idx] : {out_memref_ty}")
    lines.append("      }")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_log_softmax2d_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 log_softmax expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 log_softmax supports only f32 outputs")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 log_softmax expects rank-2 output, got {out_shape}")
    m_dim, n_dim = int(out_shape[0]), int(out_shape[1])
    if m_dim <= 0 or n_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")
    total = int(m_dim * n_dim)

    ops = [op for op in list(intent.ops or []) if op is not None]
    if not ops:
        raise RuntimeError("rvv cpu-loops v1 log_softmax2d requires non-empty ops")

    in_name = ""
    op_names = [str(getattr(op, "op", "")).strip() for op in ops]
    if len(ops) >= 2 and op_names[0] == "softmax":
        op0, op1 = ops[0], ops[1]
        op0_ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
        op0_attrs = dict(getattr(op0, "attrs", {}) or {})
        axis = op0_attrs.get("axis")
        try:
            axis_i = int(axis)
        except Exception:
            axis_i = None
        if axis_i != 1 or len(op0_ins) != 1:
            raise RuntimeError(f"rvv cpu-loops v1 log_softmax2d unsupported softmax: axis={axis!r} inputs={op0_ins}")
        in_name = str(op0_ins[0])
        op1_name = str(getattr(op1, "op", "")).strip()
        op1_ins = [str(x).strip() for x in list(getattr(op1, "inputs", []) or []) if str(x).strip()]
        if op1_name != "log" or len(op1_ins) != 1:
            raise RuntimeError(f"rvv cpu-loops v1 log_softmax2d unsupported log op: inputs={op1_ins}")
    else:
        reduce_max_ops = [op for op in ops if str(getattr(op, "op", "")).strip() == "reduce_max"]
        if not reduce_max_ops:
            raise RuntimeError(
                f"rvv cpu-loops v1 log_softmax2d expects reduce_max or softmax op, got {op_names}"
            )
        op0 = reduce_max_ops[0]
        op0_ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
        op0_attrs = dict(getattr(op0, "attrs", {}) or {})
        dims = op0_attrs.get("dims")
        dims_list = list(dims) if isinstance(dims, list) else []
        if dims_list != [1] or len(op0_ins) != 1:
            raise RuntimeError(f"rvv cpu-loops v1 log_softmax2d unsupported reduce_max: dims={dims_list} inputs={op0_ins}")
        in_name = str(op0_ins[0])
        allowed = {"reduce_max", "reduce_sum", "sub", "exp", "div", "log", "broadcast_in_dim"}
        unexpected = sorted({n for n in op_names if n and n not in allowed})
        if unexpected:
            raise RuntimeError(f"rvv cpu-loops v1 log_softmax2d unexpected ops: {unexpected}")
        required = {"reduce_max", "reduce_sum", "sub", "exp", "log"}
        missing = sorted([n for n in required if n not in set(op_names)])
        if missing:
            raise RuntimeError(f"rvv cpu-loops v1 log_softmax2d missing ops: {missing}")
        for op in ops:
            op_name = str(getattr(op, "op", "")).strip()
            if op_name not in {"reduce_max", "reduce_sum"}:
                continue
            attrs = dict(getattr(op, "attrs", {}) or {})
            dims = attrs.get("dims")
            dims_list = list(dims) if isinstance(dims, list) else []
            if dims_list != [1]:
                raise RuntimeError(f"rvv cpu-loops v1 log_softmax2d expects reduce dims=[1], got dims={dims_list}")
    in_tt = (intent.tensors or {}).get(in_name)
    if in_tt is None:
        raise RuntimeError(f"missing input tensor spec: {in_name}")
    if _dtype(getattr(in_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 log_softmax supports only f32 inputs")
    in_shape = _shape(in_name, intent=intent, bindings=bindings)
    if list(in_shape) != [m_dim, n_dim]:
        raise RuntimeError(
            f"rvv cpu-loops v1 log_softmax expects input shape [M,N] to match output; got {in_name} shape={in_shape} out_shape={out_shape}"
        )

    in_memref_ty = f"memref<{total}xf32>"
    out_memref_ty = f"memref<{total}xf32>"

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(
        f"  func.func @{_mlir_ident(kernel_name)}(%{_mlir_ident(in_name)}: {in_memref_ty}, %{_mlir_ident(out_name)}: {out_memref_ty}) {{"
    )
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cM = arith.constant {m_dim} : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    lines.append("    %init_max = arith.constant 0xFF800000 : f32")
    lines.append("    scf.for %m = %c0 to %cM step %c1 {")
    lines.append("      %base = arith.muli %m, %cN : index")
    lines.append("      %row_max = scf.for %n = %c0 to %cN step %c1 iter_args(%mx = %init_max) -> (f32) {")
    lines.append("        %idx = arith.addi %base, %n : index")
    lines.append(f"        %x = memref.load %{_mlir_ident(in_name)}[%idx] : {in_memref_ty}")
    lines.append("        %mx2 = arith.maximumf %mx, %x : f32")
    lines.append("        scf.yield %mx2 : f32")
    lines.append("      }")
    lines.append("      %sum = scf.for %n = %c0 to %cN step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("        %idx = arith.addi %base, %n : index")
    lines.append(f"        %x = memref.load %{_mlir_ident(in_name)}[%idx] : {in_memref_ty}")
    lines.append("        %centered = arith.subf %x, %row_max : f32")
    lines.append("        %e = math.exp %centered : f32")
    lines.append("        %acc2 = arith.addf %acc, %e : f32")
    lines.append("        scf.yield %acc2 : f32")
    lines.append("      }")
    lines.append("      %log_sum = math.log %sum : f32")
    lines.append("      scf.for %n = %c0 to %cN step %c1 {")
    lines.append("        %idx = arith.addi %base, %n : index")
    lines.append(f"        %x = memref.load %{_mlir_ident(in_name)}[%idx] : {in_memref_ty}")
    lines.append("        %centered = arith.subf %x, %row_max : f32")
    lines.append("        %y = arith.subf %centered, %log_sum : f32")
    lines.append(f"        memref.store %y, %{_mlir_ident(out_name)}[%idx] : {out_memref_ty}")
    lines.append("      }")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_ai_bench_rope_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_rope expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_rope expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 4:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_rope expects rank-4 output, got {out_shape}")
    seq_len, batch, head_num, head_dim = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]), int(out_shape[3]))
    if seq_len <= 0 or batch <= 0 or head_num <= 0 or head_dim <= 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")
    if (head_dim % 2) != 0:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_rope expects even HEAD_DIM, got HEAD_DIM={head_dim}")
    half = head_dim // 2

    ops = [op for op in list(intent.ops or []) if op is not None]
    rope_ops = [op for op in ops if str(getattr(op, "op", "")).strip() == "rope"]
    if len(rope_ops) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_rope expects exactly 1 rope op, got {len(rope_ops)}")
    op0 = rope_ops[0]
    op0_out = str(getattr(op0, "output", "")).strip()
    op0_ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
    if op0_out != str(out_name) or len(op0_ins) != 3:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_rope invalid rope op: inputs={op0_ins} output={op0_out!r}")
    x_name, cos_name, sin_name = op0_ins

    x_tt = (intent.tensors or {}).get(x_name)
    cos_tt = (intent.tensors or {}).get(cos_name)
    sin_tt = (intent.tensors or {}).get(sin_name)
    if x_tt is None or cos_tt is None or sin_tt is None:
        raise RuntimeError("rvv cpu-loops v1 ai_bench_rope missing tensor specs for inputs")
    for nm, tt in [(x_name, x_tt), (cos_name, cos_tt), (sin_name, sin_tt)]:
        if _dtype(getattr(tt, "dtype", "f32")) != "f32":
            raise RuntimeError(f"rvv cpu-loops v1 ai_bench_rope expects f32 tensors, got {nm} dtype={getattr(tt, 'dtype', '')}")

    x_shape = _shape(x_name, intent=intent, bindings=bindings)
    cos_shape = _shape(cos_name, intent=intent, bindings=bindings)
    sin_shape = _shape(sin_name, intent=intent, bindings=bindings)
    if list(x_shape) != list(out_shape):
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_rope expects input shape==output shape, got x={x_shape} out={out_shape}")
    if cos_shape != [seq_len, half] or sin_shape != [seq_len, half]:
        raise RuntimeError(
            f"rvv cpu-loops v1 ai_bench_rope expects cos/sin shape [SEQ,HEAD_DIM/2]=[{seq_len},{half}], "
            f"got cos={cos_shape} sin={sin_shape}"
        )

    io_names = _io_arg_order(intent)
    total_x = int(seq_len * batch * head_num * head_dim)
    total_cs = int(seq_len * half)
    total_out = total_x

    arg_types: dict[str, str] = {
        x_name: f"memref<{total_x}xf32>",
        cos_name: f"memref<{total_cs}xf32>",
        sin_name: f"memref<{total_cs}xf32>",
        out_name: f"memref<{total_out}xf32>",
    }
    for n in list(io_names):
        if n not in arg_types:
            raise RuntimeError(
                f"rvv cpu-loops v1 ai_bench_rope: unexpected IO tensor {n!r} (known={sorted(arg_types)}) for kernel={kernel_name!r}"
            )
    arg_decls = [f"%{_mlir_ident(n)}: {arg_types[n]}" for n in io_names]

    x_memref = arg_types[x_name]
    cs_memref = arg_types[cos_name]
    out_memref = arg_types[out_name]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cSEQ = arith.constant {seq_len} : index")
    lines.append(f"    %cB = arith.constant {batch} : index")
    lines.append(f"    %cH = arith.constant {head_num} : index")
    lines.append(f"    %cD = arith.constant {head_dim} : index")
    lines.append(f"    %cHalf = arith.constant {half} : index")
    lines.append("    scf.for %s = %c0 to %cSEQ step %c1 {")
    lines.append("      %sB = arith.muli %s, %cB : index")
    lines.append("      %cos_base = arith.muli %s, %cHalf : index")
    lines.append("      scf.for %b = %c0 to %cB step %c1 {")
    lines.append("        %sb = arith.addi %sB, %b : index")
    lines.append("        %sbH = arith.muli %sb, %cH : index")
    lines.append("        scf.for %h = %c0 to %cH step %c1 {")
    lines.append("          %sbh = arith.addi %sbH, %h : index")
    lines.append("          %base = arith.muli %sbh, %cD : index")
    lines.append("          scf.for %d = %c0 to %cHalf step %c1 {")
    lines.append("            %cs_i = arith.addi %cos_base, %d : index")
    lines.append(f"            %c = memref.load %{_mlir_ident(cos_name)}[%cs_i] : {cs_memref}")
    lines.append(f"            %sn = memref.load %{_mlir_ident(sin_name)}[%cs_i] : {cs_memref}")
    lines.append("            %x1_i = arith.addi %base, %d : index")
    lines.append("            %d2 = arith.addi %d, %cHalf : index")
    lines.append("            %x2_i = arith.addi %base, %d2 : index")
    lines.append(f"            %x1 = memref.load %{_mlir_ident(x_name)}[%x1_i] : {x_memref}")
    lines.append(f"            %x2 = memref.load %{_mlir_ident(x_name)}[%x2_i] : {x_memref}")
    lines.append("            %x1c = arith.mulf %x1, %c : f32")
    lines.append("            %x2s = arith.mulf %x2, %sn : f32")
    lines.append("            %y1 = arith.subf %x1c, %x2s : f32")
    lines.append("            %x1s = arith.mulf %x1, %sn : f32")
    lines.append("            %x2c = arith.mulf %x2, %c : f32")
    lines.append("            %y2 = arith.addf %x1s, %x2c : f32")
    lines.append(f"            memref.store %y1, %{_mlir_ident(out_name)}[%x1_i] : {out_memref}")
    lines.append(f"            memref.store %y2, %{_mlir_ident(out_name)}[%x2_i] : {out_memref}")
    lines.append("          }")
    lines.append("        }")
    lines.append("      }")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_ai_bench_dropout_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_dropout expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_dropout expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_dropout expects rank-1 output, got {out_shape}")
    n = int(out_shape[0])
    if n < 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    if len(ops) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_dropout expects exactly 1 op, got {len(ops)}")
    op0 = ops[0]
    op0_name = str(getattr(op0, "op", "")).strip()
    op0_out = str(getattr(op0, "output", "")).strip()
    op0_ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
    if op0_name != "dropout" or op0_out != str(out_name) or len(op0_ins) != 3:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_dropout invalid dropout op: inputs={op0_ins} output={op0_out!r}")
    x_name, p_name, seed_name = op0_ins

    x_tt = (intent.tensors or {}).get(x_name)
    p_tt = (intent.tensors or {}).get(p_name)
    seed_tt = (intent.tensors or {}).get(seed_name)
    if x_tt is None or p_tt is None or seed_tt is None:
        raise RuntimeError("rvv cpu-loops v1 ai_bench_dropout missing tensor specs for inputs")
    if _dtype(getattr(x_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_dropout expects f32 x")
    if _dtype(getattr(p_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_dropout expects f32 p")
    if _dtype(getattr(seed_tt, "dtype", "i32")) != "i32":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_dropout expects i32 seed")
    x_shape = _shape(x_name, intent=intent, bindings=bindings)
    p_shape = _shape(p_name, intent=intent, bindings=bindings)
    seed_shape = _shape(seed_name, intent=intent, bindings=bindings)
    if x_shape != [n]:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_dropout expects x shape [n_elements]=[{n}], got {x_shape}")
    if p_shape != [] or seed_shape != []:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_dropout expects scalar p/seed, got p={p_shape} seed={seed_shape}")

    io_names = _io_arg_order(intent)
    arg_types: dict[str, str] = {
        x_name: f"memref<{n}xf32>",
        out_name: f"memref<{n}xf32>",
        p_name: "memref<1xf32>",
        seed_name: "memref<1xi32>",
    }
    for nm in list(io_names):
        if nm not in arg_types:
            raise RuntimeError(
                f"rvv cpu-loops v1 ai_bench_dropout: unexpected IO tensor {nm!r} (known={sorted(arg_types)}) for kernel={kernel_name!r}"
            )
    arg_decls = [f"%{_mlir_ident(nm)}: {arg_types[nm]}" for nm in io_names]

    x_memref = arg_types[x_name]
    out_memref = arg_types[out_name]

    # Match Triton's tl.rand(seed, offsets) semantics (Philox + uint_to_uniform_float).
    # Constants are specified in signed-i32 form (same two's-complement bits).
    KEY_A = -1640531527  # 0x9E3779B9 as signed i32
    KEY_B = -1150833019  # 0xBB67AE85 as signed i32
    ROUND_A_U64 = 3528531795  # 0xD2511F53
    ROUND_B_U64 = 3449720151  # 0xCD9E8D57
    SCALE = 4.6566127342e-10
    n_rounds = 10

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append("  func.func private @__intentir_philox_u32(%seed: i32, %offset: i32) -> i32 {")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cRounds = arith.constant {n_rounds} : index")
    lines.append("    %c0i32 = arith.constant 0 : i32")
    lines.append(f"    %keyA = arith.constant {KEY_A} : i32")
    lines.append(f"    %keyB = arith.constant {KEY_B} : i32")
    lines.append(f"    %roundA = arith.constant {ROUND_A_U64} : i64")
    lines.append(f"    %roundB = arith.constant {ROUND_B_U64} : i64")
    lines.append("    %c32i64 = arith.constant 32 : i64")
    lines.append(
        "    %r_c0, %r_c1, %r_c2, %r_c3, %r_k0, %r_k1 = scf.for %r = %c0 to %cRounds step %c1 "
        "iter_args(%pc0 = %offset, %pc1 = %c0i32, %pc2 = %c0i32, %pc3 = %c0i32, %pk0 = %seed, %pk1 = %c0i32) "
        "-> (i32, i32, i32, i32, i32, i32) {"
    )
    lines.append("      %pc2_u64 = arith.extui %pc2 : i32 to i64")
    lines.append("      %pc0_u64 = arith.extui %pc0 : i32 to i64")
    lines.append("      %prodB = arith.muli %pc2_u64, %roundB : i64")
    lines.append("      %prodA = arith.muli %pc0_u64, %roundA : i64")
    lines.append("      %hiB64 = arith.shrui %prodB, %c32i64 : i64")
    lines.append("      %hiA64 = arith.shrui %prodA, %c32i64 : i64")
    lines.append("      %hiB = arith.trunci %hiB64 : i64 to i32")
    lines.append("      %hiA = arith.trunci %hiA64 : i64 to i32")
    lines.append("      %tmp0 = arith.xori %hiB, %pc1 : i32")
    lines.append("      %nc0 = arith.xori %tmp0, %pk0 : i32")
    lines.append("      %tmp2 = arith.xori %hiA, %pc3 : i32")
    lines.append("      %nc2 = arith.xori %tmp2, %pk1 : i32")
    lines.append("      %nc1 = arith.trunci %prodB : i64 to i32")
    lines.append("      %nc3 = arith.trunci %prodA : i64 to i32")
    lines.append("      %nk0 = arith.addi %pk0, %keyA : i32")
    lines.append("      %nk1 = arith.addi %pk1, %keyB : i32")
    lines.append("      scf.yield %nc0, %nc1, %nc2, %nc3, %nk0, %nk1 : i32, i32, i32, i32, i32, i32")
    lines.append("    }")
    lines.append("    return %r_c0 : i32")
    lines.append("  }")
    lines.append("  func.func private @__intentir_uint_to_uniform_float_u32(%x: i32) -> f32 {")
    lines.append("    %c0i32 = arith.constant 0 : i32")
    lines.append("    %cMinus1i32 = arith.constant -1 : i32")
    lines.append("    %neg = arith.cmpi slt, %x, %c0i32 : i32")
    lines.append("    %x_not = arith.xori %x, %cMinus1i32 : i32")
    lines.append("    %u = arith.select %neg, %x_not, %x : i32")
    lines.append("    %u_f = arith.sitofp %u : i32 to f32")
    lines.append(f"    %scale = arith.constant {_f32_lit(float(SCALE))} : f32")
    lines.append("    %rnd = arith.mulf %u_f, %scale : f32")
    lines.append("    return %rnd : f32")
    lines.append("  }")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cN = arith.constant {n} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    lines.append("    %c1f = arith.constant 1.0 : f32")
    lines.append(f"    %p0 = memref.load %{_mlir_ident(p_name)}[%c0] : memref<1xf32>")
    lines.append(f"    %seed0 = memref.load %{_mlir_ident(seed_name)}[%c0] : memref<1xi32>")
    lines.append("    %keep_prob = arith.subf %c1f, %p0 : f32")
    lines.append("    scf.for %i = %c0 to %cN step %c1 {")
    lines.append(f"      %xv = memref.load %{_mlir_ident(x_name)}[%i] : {x_memref}")
    lines.append("      %off = arith.index_cast %i : index to i32")
    lines.append("      %u32 = func.call @__intentir_philox_u32(%seed0, %off) : (i32, i32) -> i32")
    lines.append("      %rnd = func.call @__intentir_uint_to_uniform_float_u32(%u32) : (i32) -> f32")
    lines.append("      %keep = arith.cmpf ogt, %rnd, %p0 : f32")
    lines.append("      %scaled = arith.divf %xv, %keep_prob : f32")
    lines.append("      %y = arith.select %keep, %scaled, %c0f : f32")
    lines.append(f"      memref.store %y, %{_mlir_ident(out_name)}[%i] : {out_memref}")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_ai_bench_correlation_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_correlation expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "i8")) != "i8":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_correlation expects i8 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 3:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_correlation expects rank-3 output, got {out_shape}")
    out_c, h_dim, w_dim = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if out_c < 0 or h_dim < 0 or w_dim < 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    if len(ops) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_correlation expects exactly 1 op, got {len(ops)}")
    op0 = ops[0]
    op0_name = str(getattr(op0, "op", "")).strip()
    op0_out = str(getattr(op0, "output", "")).strip()
    op0_ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
    if op0_name != "correlation" or op0_out != str(out_name) or len(op0_ins) != 3:
        raise RuntimeError(
            f"rvv cpu-loops v1 ai_bench_correlation invalid op: op={op0_name!r} inputs={op0_ins} output={op0_out!r}"
        )
    src0_name, src1_name, out_shift_name = op0_ins

    src0_tt = (intent.tensors or {}).get(src0_name)
    src1_tt = (intent.tensors or {}).get(src1_name)
    out_shift_tt = (intent.tensors or {}).get(out_shift_name)
    if src0_tt is None or src1_tt is None or out_shift_tt is None:
        raise RuntimeError("rvv cpu-loops v1 ai_bench_correlation missing tensor specs for inputs")
    if _dtype(getattr(src0_tt, "dtype", "i8")) != "i8" or _dtype(getattr(src1_tt, "dtype", "i8")) != "i8":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_correlation expects i8 src0/src1")
    if _dtype(getattr(out_shift_tt, "dtype", "i32")) != "i32":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_correlation expects i32 out_shift")

    src0_shape = _shape(src0_name, intent=intent, bindings=bindings)
    src1_shape = _shape(src1_name, intent=intent, bindings=bindings)
    if len(src0_shape) != 3 or src0_shape != src1_shape:
        raise RuntimeError(
            f"rvv cpu-loops v1 ai_bench_correlation expects src0/src1 rank-3 same shape, got {src0_shape} and {src1_shape}"
        )
    in_c, h0, w0 = (int(src0_shape[0]), int(src0_shape[1]), int(src0_shape[2]))
    if (h0, w0) != (h_dim, w_dim):
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_correlation spatial mismatch: src={(h0, w0)} out={(h_dim, w_dim)}")
    if out_shift_name and _shape(out_shift_name, intent=intent, bindings=bindings) != []:
        raise RuntimeError("rvv cpu-loops v1 ai_bench_correlation expects scalar out_shift")

    hw = int(h_dim * w_dim)
    total_in = int(in_c * hw)
    total_out = int(out_c * hw)
    io_names = _io_arg_order(intent)
    arg_types: dict[str, str] = {
        src0_name: f"memref<{total_in}xi8>",
        src1_name: f"memref<{total_in}xi8>",
        out_shift_name: "memref<1xi32>",
        out_name: f"memref<{total_out}xi8>",
    }
    for nm in list(io_names):
        if nm not in arg_types:
            raise RuntimeError(
                f"rvv cpu-loops v1 ai_bench_correlation: unexpected IO tensor {nm!r} (known={sorted(arg_types)}) for kernel={kernel_name!r}"
            )
    arg_decls = [f"%{_mlir_ident(nm)}: {arg_types[nm]}" for nm in io_names]

    src_memref = arg_types[src0_name]
    out_memref = arg_types[out_name]

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cOC = arith.constant {out_c} : index")
    lines.append(f"    %cIC = arith.constant {in_c} : index")
    lines.append(f"    %cH = arith.constant {h_dim} : index")
    lines.append(f"    %cW = arith.constant {w_dim} : index")
    lines.append(f"    %cHW = arith.constant {hw} : index")
    lines.append("    %c0i32 = arith.constant 0 : i32")
    lines.append("    %c0i8 = arith.constant 0 : i8")
    lines.append(f"    %out_shift0 = memref.load %{_mlir_ident(out_shift_name)}[%c0] : memref<1xi32>")
    lines.append("    scf.for %oc = %c0 to %cOC step %c1 {")
    lines.append("      %oc_base = arith.muli %oc, %cHW : index")
    lines.append("      %oc_nonzero = arith.cmpi ne, %oc, %c0 : index")
    lines.append("      scf.for %h = %c0 to %cH step %c1 {")
    lines.append("        %hW = arith.muli %h, %cW : index")
    lines.append("        scf.for %w = %c0 to %cW step %c1 {")
    lines.append("          %w_lt_oc = arith.cmpi ult, %w, %oc : index")
    lines.append("          %skip = arith.andi %oc_nonzero, %w_lt_oc : i1")
    lines.append("          %out_hw = arith.addi %oc_base, %hW : index")
    lines.append("          %out_idx = arith.addi %out_hw, %w : index")
    lines.append("          scf.if %skip {")
    lines.append(f"            memref.store %c0i8, %{_mlir_ident(out_name)}[%out_idx] : {out_memref}")
    lines.append("          } else {")
    lines.append("            %w2 = arith.subi %w, %oc : index")
    lines.append("            %acc = scf.for %k = %c0 to %cIC step %c1 iter_args(%a = %c0i32) -> (i32) {")
    lines.append("              %k_base = arith.muli %k, %cHW : index")
    lines.append("              %in_hw = arith.addi %k_base, %hW : index")
    lines.append("              %idx0 = arith.addi %in_hw, %w : index")
    lines.append("              %idx1 = arith.addi %in_hw, %w2 : index")
    lines.append(f"              %a8 = memref.load %{_mlir_ident(src0_name)}[%idx0] : {src_memref}")
    lines.append(f"              %b8 = memref.load %{_mlir_ident(src1_name)}[%idx1] : {src_memref}")
    lines.append("              %a16 = arith.extsi %a8 : i8 to i16")
    lines.append("              %b16 = arith.extsi %b8 : i8 to i16")
    lines.append("              %p16 = arith.muli %a16, %b16 : i16")
    lines.append("              %p32 = arith.extsi %p16 : i16 to i32")
    lines.append("              %a2 = arith.addi %a, %p32 : i32")
    lines.append("              scf.yield %a2 : i32")
    lines.append("            }")
    lines.append("            %sh = arith.shrsi %acc, %out_shift0 : i32")
    lines.append("            %o8 = arith.trunci %sh : i32 to i8")
    lines.append(f"            memref.store %o8, %{_mlir_ident(out_name)}[%out_idx] : {out_memref}")
    lines.append("          }")
    lines.append("        }")
    lines.append("      }")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_ai_bench_resize_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_resize expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "i8")) != "i8":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_resize expects i8 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 3:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_resize expects rank-3 output, got {out_shape}")
    c_dim, oh_dim, ow_dim = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if c_dim < 0 or oh_dim < 0 or ow_dim < 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    if len(ops) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_resize expects exactly 1 op, got {len(ops)}")
    op0 = ops[0]
    op0_name = str(getattr(op0, "op", "")).strip()
    op0_out = str(getattr(op0, "output", "")).strip()
    op0_ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
    attrs = dict(getattr(op0, "attrs", {}) or {})
    if op0_name != "resize" or op0_out != str(out_name) or len(op0_ins) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_resize invalid op: op={op0_name!r} inputs={op0_ins} output={op0_out!r}")
    src_name = str(op0_ins[0])
    src_tt = (intent.tensors or {}).get(src_name)
    if src_tt is None:
        raise RuntimeError(f"missing input tensor spec: {src_name}")
    if _dtype(getattr(src_tt, "dtype", "i8")) != "i8":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_resize expects i8 src")
    src_shape = _shape(src_name, intent=intent, bindings=bindings)
    if len(src_shape) != 3:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_resize expects rank-3 src, got {src_shape}")
    c0, h_dim, w_dim = (int(src_shape[0]), int(src_shape[1]), int(src_shape[2]))
    if c0 != int(c_dim):
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_resize channel mismatch: src C={c0} out C={c_dim}")

    scale_factor = int(attrs.get("scale_factor", 2))
    mode = str(attrs.get("mode", "")).strip().lower()
    if scale_factor != 2 or mode != "bilinear":
        raise RuntimeError(
            f"rvv cpu-loops v1 ai_bench_resize expects scale_factor=2 bilinear, got scale_factor={scale_factor} mode={mode!r}"
        )
    hw_fl = int(attrs.get("hw_fl", 7))
    if hw_fl <= 0:
        hw_fl = 7
    if oh_dim != 2 * h_dim or ow_dim != 2 * w_dim:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_resize expects 2x upsample: src={(h_dim, w_dim)} out={(oh_dim, ow_dim)}")

    factor = int(1 << int(hw_fl))
    shift1 = int(hw_fl - 1)
    if shift1 < 0:
        shift1 = 0

    total_in = int(c_dim * h_dim * w_dim)
    total_out = int(c_dim * oh_dim * ow_dim)
    io_names = _io_arg_order(intent)
    arg_types: dict[str, str] = {
        src_name: f"memref<{total_in}xi8>",
        out_name: f"memref<{total_out}xi8>",
    }
    for nm in list(io_names):
        if nm not in arg_types:
            raise RuntimeError(
                f"rvv cpu-loops v1 ai_bench_resize: unexpected IO tensor {nm!r} (known={sorted(arg_types)}) for kernel={kernel_name!r}"
            )
    arg_decls = [f"%{_mlir_ident(nm)}: {arg_types[nm]}" for nm in io_names]

    in_memref = arg_types[src_name]
    out_memref = arg_types[out_name]

    hw = int(h_dim * w_dim)
    ohow = int(oh_dim * ow_dim)

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cC = arith.constant {c_dim} : index")
    lines.append(f"    %cH = arith.constant {h_dim} : index")
    lines.append(f"    %cW = arith.constant {w_dim} : index")
    lines.append(f"    %cOH = arith.constant {oh_dim} : index")
    lines.append(f"    %cOW = arith.constant {ow_dim} : index")
    lines.append(f"    %cHW = arith.constant {hw} : index")
    lines.append(f"    %cOHOW = arith.constant {ohow} : index")
    lines.append("    %c0i32 = arith.constant 0 : i32")
    lines.append("    %c1i32 = arith.constant 1 : i32")
    lines.append(f"    %cHWFl = arith.constant {hw_fl} : i32")
    lines.append(f"    %cShift1 = arith.constant {shift1} : i32")
    lines.append(f"    %factor = arith.constant {factor} : i32")
    lines.append(f"    %cWm1 = arith.constant {w_dim - 1} : i32")
    lines.append(f"    %cHm1 = arith.constant {h_dim - 1} : i32")
    lines.append("    scf.for %c = %c0 to %cC step %c1 {")
    lines.append("      %c_in_base = arith.muli %c, %cHW : index")
    lines.append("      %c_out_base = arith.muli %c, %cOHOW : index")
    lines.append("      scf.for %oh = %c0 to %cOH step %c1 {")
    lines.append("        %oh_i32 = arith.index_cast %oh : index to i32")
    lines.append("        %input_y = arith.shli %oh_i32, %cShift1 : i32")
    lines.append("        %y0 = arith.shrsi %input_y, %cHWFl : i32")
    lines.append("        %y0_sh = arith.shli %y0, %cHWFl : i32")
    lines.append("        %h1 = arith.subi %input_y, %y0_sh : i32")
    lines.append("        %h0 = arith.subi %factor, %h1 : i32")
    lines.append("        %y1_tmp = arith.addi %y0, %c1i32 : i32")
    lines.append("        %y1_gt = arith.cmpi sgt, %y1_tmp, %cHm1 : i32")
    lines.append("        %y1 = arith.select %y1_gt, %cHm1, %y1_tmp : i32")
    lines.append("        %y0_idx = arith.index_cast %y0 : i32 to index")
    lines.append("        %y1_idx = arith.index_cast %y1 : i32 to index")
    lines.append("        %y0W = arith.muli %y0_idx, %cW : index")
    lines.append("        %y1W = arith.muli %y1_idx, %cW : index")
    lines.append("        %y0_base = arith.addi %c_in_base, %y0W : index")
    lines.append("        %y1_base = arith.addi %c_in_base, %y1W : index")
    lines.append("        %out_oh = arith.muli %oh, %cOW : index")
    lines.append("        scf.for %ow = %c0 to %cOW step %c1 {")
    lines.append("          %ow_i32 = arith.index_cast %ow : index to i32")
    lines.append("          %input_x = arith.shli %ow_i32, %cShift1 : i32")
    lines.append("          %x0 = arith.shrsi %input_x, %cHWFl : i32")
    lines.append("          %x0_sh = arith.shli %x0, %cHWFl : i32")
    lines.append("          %w1 = arith.subi %input_x, %x0_sh : i32")
    lines.append("          %w0 = arith.subi %factor, %w1 : i32")
    lines.append("          %x1_tmp = arith.addi %x0, %c1i32 : i32")
    lines.append("          %x1_gt = arith.cmpi sgt, %x1_tmp, %cWm1 : i32")
    lines.append("          %x1 = arith.select %x1_gt, %cWm1, %x1_tmp : i32")
    lines.append("          %x0_idx = arith.index_cast %x0 : i32 to index")
    lines.append("          %x1_idx = arith.index_cast %x1 : i32 to index")
    lines.append("          %idx00 = arith.addi %y0_base, %x0_idx : index")
    lines.append("          %idx01 = arith.addi %y0_base, %x1_idx : index")
    lines.append("          %idx10 = arith.addi %y1_base, %x0_idx : index")
    lines.append("          %idx11 = arith.addi %y1_base, %x1_idx : index")
    lines.append(f"          %v00 = memref.load %{_mlir_ident(src_name)}[%idx00] : {in_memref}")
    lines.append(f"          %v01 = memref.load %{_mlir_ident(src_name)}[%idx01] : {in_memref}")
    lines.append(f"          %v10 = memref.load %{_mlir_ident(src_name)}[%idx10] : {in_memref}")
    lines.append(f"          %v11 = memref.load %{_mlir_ident(src_name)}[%idx11] : {in_memref}")
    lines.append("          %v00_32 = arith.extsi %v00 : i8 to i32")
    lines.append("          %v01_32 = arith.extsi %v01 : i8 to i32")
    lines.append("          %v10_32 = arith.extsi %v10 : i8 to i32")
    lines.append("          %v11_32 = arith.extsi %v11 : i8 to i32")
    lines.append("          %m00 = arith.muli %v00_32, %w0 : i32")
    lines.append("          %m01 = arith.muli %v01_32, %w1 : i32")
    lines.append("          %m10 = arith.muli %v10_32, %w0 : i32")
    lines.append("          %m11 = arith.muli %v11_32, %w1 : i32")
    lines.append("          %s1 = arith.addi %m00, %m01 : i32")
    lines.append("          %s2 = arith.addi %m10, %m11 : i32")
    lines.append("          %sum1 = arith.shrsi %s1, %cHWFl : i32")
    lines.append("          %sum2 = arith.shrsi %s2, %cHWFl : i32")
    lines.append("          %t0 = arith.muli %sum1, %h0 : i32")
    lines.append("          %t1 = arith.muli %sum2, %h1 : i32")
    lines.append("          %t = arith.addi %t0, %t1 : i32")
    lines.append("          %val = arith.shrsi %t, %cHWFl : i32")
    lines.append("          %o8 = arith.trunci %val : i32 to i8")
    lines.append("          %out_off = arith.addi %out_oh, %ow : index")
    lines.append("          %out_idx = arith.addi %c_out_base, %out_off : index")
    lines.append(f"          memref.store %o8, %{_mlir_ident(out_name)}[%out_idx] : {out_memref}")
    lines.append("        }")
    lines.append("      }")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_ai_bench_warp_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_warp expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "i8")) != "i8":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_warp expects i8 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 3:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_warp expects rank-3 output, got {out_shape}")
    c_dim, h_dim, w_dim = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if c_dim < 0 or h_dim < 0 or w_dim < 0:
        raise RuntimeError(f"invalid output shape for {out_name}: {out_shape}")

    ops = [op for op in list(intent.ops or []) if op is not None]
    if len(ops) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_warp expects exactly 1 op, got {len(ops)}")
    op0 = ops[0]
    op0_name = str(getattr(op0, "op", "")).strip()
    op0_out = str(getattr(op0, "output", "")).strip()
    op0_ins = [str(x).strip() for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
    if op0_name != "warp" or op0_out != str(out_name) or len(op0_ins) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_warp invalid op: op={op0_name!r} inputs={op0_ins} output={op0_out!r}")
    src_name, offset_name = op0_ins

    src_tt = (intent.tensors or {}).get(src_name)
    offset_tt = (intent.tensors or {}).get(offset_name)
    if src_tt is None or offset_tt is None:
        raise RuntimeError("rvv cpu-loops v1 ai_bench_warp missing tensor specs for inputs")
    if _dtype(getattr(src_tt, "dtype", "i8")) != "i8":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_warp expects i8 src")
    if _dtype(getattr(offset_tt, "dtype", "i16")) != "i16":
        raise RuntimeError("rvv cpu-loops v1 ai_bench_warp expects i16 offset")
    src_shape = _shape(src_name, intent=intent, bindings=bindings)
    off_shape = _shape(offset_name, intent=intent, bindings=bindings)
    if src_shape != [c_dim, h_dim, w_dim]:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_warp expects src shape {out_shape}, got {src_shape}")
    if off_shape != [h_dim, w_dim]:
        raise RuntimeError(f"rvv cpu-loops v1 ai_bench_warp expects offset shape [{h_dim},{w_dim}], got {off_shape}")

    total_src = int(c_dim * h_dim * w_dim)
    total_off = int(h_dim * w_dim)
    total_out = total_src
    io_names = _io_arg_order(intent)
    arg_types: dict[str, str] = {
        src_name: f"memref<{total_src}xi8>",
        offset_name: f"memref<{total_off}xi16>",
        out_name: f"memref<{total_out}xi8>",
    }
    for nm in list(io_names):
        if nm not in arg_types:
            raise RuntimeError(
                f"rvv cpu-loops v1 ai_bench_warp: unexpected IO tensor {nm!r} (known={sorted(arg_types)}) for kernel={kernel_name!r}"
            )
    arg_decls = [f"%{_mlir_ident(nm)}: {arg_types[nm]}" for nm in io_names]

    src_memref = arg_types[src_name]
    off_memref = arg_types[offset_name]
    out_memref = arg_types[out_name]

    hw = int(h_dim * w_dim)
    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cC = arith.constant {c_dim} : index")
    lines.append(f"    %cH = arith.constant {h_dim} : index")
    lines.append(f"    %cW = arith.constant {w_dim} : index")
    lines.append(f"    %cHW = arith.constant {hw} : index")
    lines.append("    %c0i8 = arith.constant 0 : i8")
    lines.append("    %c0i16 = arith.constant 0 : i16")
    lines.append("    %c1i8 = arith.constant 1 : i8")
    lines.append("    %c8i16 = arith.constant 8 : i16")
    lines.append("    %c0i32 = arith.constant 0 : i32")
    lines.append(f"    %cWm1i32 = arith.constant {w_dim - 1} : i32")
    lines.append("    scf.for %h = %c0 to %cH step %c1 {")
    lines.append("      %hW = arith.muli %h, %cW : index")
    lines.append("      scf.for %w = %c0 to %cW step %c1 {")
    lines.append("        %off_i = arith.addi %hW, %w : index")
    lines.append(f"        %off = memref.load %{_mlir_ident(offset_name)}[%off_i] : {off_memref}")
    lines.append("        %offset_int = arith.shrsi %off, %c8i16 : i16")
    lines.append("        %offset_frac_i8 = arith.trunci %off : i16 to i8")
    lines.append("        %offset_frac = arith.extsi %offset_frac_i8 : i8 to i16")
    lines.append("        %w_i32 = arith.index_cast %w : index to i32")
    lines.append("        %w_i8 = arith.trunci %w_i32 : i32 to i8")
    lines.append("        %offset_int_i8 = arith.trunci %offset_int : i16 to i8")
    lines.append("        %right_i8 = arith.subi %w_i8, %offset_int_i8 : i8")
    lines.append("        %left_i8 = arith.subi %right_i8, %c1i8 : i8")
    lines.append("        %right = arith.extsi %right_i8 : i8 to i16")
    lines.append("        %left = arith.extsi %left_i8 : i8 to i16")
    lines.append("        %right_ok = arith.cmpi sge, %right, %c0i16 : i16")
    lines.append("        %left_ok = arith.cmpi sge, %left, %c0i16 : i16")
    lines.append("        %right_i32 = arith.extsi %right : i16 to i32")
    lines.append("        %left_i32 = arith.extsi %left : i16 to i32")
    lines.append("        %r_lt0 = arith.cmpi slt, %right_i32, %c0i32 : i32")
    lines.append("        %l_lt0 = arith.cmpi slt, %left_i32, %c0i32 : i32")
    lines.append("        %r0 = arith.select %r_lt0, %c0i32, %right_i32 : i32")
    lines.append("        %l0 = arith.select %l_lt0, %c0i32, %left_i32 : i32")
    lines.append("        %r_gt = arith.cmpi sgt, %r0, %cWm1i32 : i32")
    lines.append("        %l_gt = arith.cmpi sgt, %l0, %cWm1i32 : i32")
    lines.append("        %r_cl = arith.select %r_gt, %cWm1i32, %r0 : i32")
    lines.append("        %l_cl = arith.select %l_gt, %cWm1i32, %l0 : i32")
    lines.append("        %r_idx = arith.index_cast %r_cl : i32 to index")
    lines.append("        %l_idx = arith.index_cast %l_cl : i32 to index")
    lines.append("        scf.for %c = %c0 to %cC step %c1 {")
    lines.append("          %c_base = arith.muli %c, %cHW : index")
    lines.append("          %row_base = arith.addi %c_base, %hW : index")
    lines.append("          %idx_r = arith.addi %row_base, %r_idx : index")
    lines.append("          %idx_l = arith.addi %row_base, %l_idx : index")
    lines.append(f"          %rv0 = memref.load %{_mlir_ident(src_name)}[%idx_r] : {src_memref}")
    lines.append(f"          %lv0 = memref.load %{_mlir_ident(src_name)}[%idx_l] : {src_memref}")
    lines.append("          %rv = arith.select %right_ok, %rv0, %c0i8 : i8")
    lines.append("          %lv = arith.select %left_ok, %lv0, %c0i8 : i8")
    lines.append("          %rv16 = arith.extsi %rv : i8 to i16")
    lines.append("          %lv16 = arith.extsi %lv : i8 to i16")
    lines.append("          %rv8 = arith.shli %rv16, %c8i16 : i16")
    lines.append("          %diff = arith.subi %lv16, %rv16 : i16")
    lines.append("          %mul = arith.muli %diff, %offset_frac : i16")
    lines.append("          %acc = arith.addi %rv8, %mul : i16")
    lines.append("          %out16 = arith.shrsi %acc, %c8i16 : i16")
    lines.append("          %out8 = arith.trunci %out16 : i16 to i8")
    lines.append("          %out_idx = arith.addi %row_base, %w : index")
    lines.append(f"          memref.store %out8, %{_mlir_ident(out_name)}[%out_idx] : {out_memref}")
    lines.append("        }")
    lines.append("      }")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_attention2d_causal_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
    block_kv: int,
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 attention2d expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 attention2d expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 attention2d expects rank-2 output, got out_shape={out_shape}")
    q_ctx, head_dim = (int(out_shape[0]), int(out_shape[1]))
    if q_ctx <= 0 or head_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 attention2d invalid out_shape={out_shape}")

    tensors = dict(intent.tensors or {})
    io_names = _io_arg_order(intent)
    ext_inputs = [n for n in io_names if n not in set(outputs)]

    rank2_f32: list[tuple[str, list[int]]] = []
    scalar_f32: list[str] = []
    for nm in list(ext_inputs):
        tt = tensors.get(str(nm))
        if tt is None:
            continue
        dt = _dtype(getattr(tt, "dtype", "f32"))
        shp = _shape(str(nm), intent=intent, bindings=bindings)
        if dt == "f32" and len(shp) == 0:
            scalar_f32.append(str(nm))
        if dt == "f32" and len(shp) == 2 and int(shp[1]) == int(head_dim):
            rank2_f32.append((str(nm), list(shp)))

    q_candidates = [nm for nm, shp in rank2_f32 if list(shp) == list(out_shape)]
    q_name = ""
    if q_candidates:
        for nm in q_candidates:
            n = str(nm).strip().lower()
            if n in {"q", "query"} or n.startswith("q_") or n.startswith("qptr") or "query" in n:
                q_name = str(nm)
                break
        if not q_name:
            for nm in q_candidates:
                n = str(nm).strip().lower()
                if (n == "q") or ("q" in n and "kv" not in n):
                    q_name = str(nm)
                    break
    if not q_name and len(q_candidates) == 1:
        q_name = str(q_candidates[0])
    if not q_name:
        raise RuntimeError(
            f"rvv cpu-loops v1 attention2d failed to infer Q tensor for kernel={kernel_name!r} (candidates={q_candidates})"
        )
    kv_candidates = [(nm, shp) for nm, shp in rank2_f32 if nm != q_name]
    groups: dict[tuple[int, int], list[str]] = {}
    for nm, shp in kv_candidates:
        if len(shp) != 2:
            continue
        groups.setdefault((int(shp[0]), int(shp[1])), []).append(str(nm))
    kv_shape = None
    kv_names: list[str] = []
    for sh, names in groups.items():
        if len(names) >= 2:
            kv_shape = list(sh)
            kv_names = list(names[:2])
            break
    if kv_shape is None or len(kv_names) != 2:
        raise RuntimeError(f"rvv cpu-loops v1 attention2d failed to infer K/V tensors for kernel={kernel_name!r}")
    kv_ctx = int(kv_shape[0])
    if kv_ctx <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 attention2d invalid KV_CTX={kv_ctx}")

    # Assign K/V by name heuristic.
    k_name = next((n for n in kv_names if "k" in str(n).lower()), kv_names[0])
    v_name = next((n for n in kv_names if ("v" in str(n).lower()) and str(n) != str(k_name)), "")
    if not v_name:
        v_name = kv_names[1] if kv_names[0] == k_name else kv_names[0]

    sm_scale_name = ""
    for nm in scalar_f32:
        if "sm_scale" in str(nm).lower() or str(nm).lower() in {"scale", "sm_scale", "smscale"}:
            sm_scale_name = str(nm)
            break
    if not sm_scale_name and scalar_f32:
        sm_scale_name = str(scalar_f32[0])

    arg_decls: list[str] = []
    memref_ty_by_name: dict[str, str] = {}
    for name in list(io_names):
        tt = tensors.get(str(name))
        if tt is None:
            raise RuntimeError(f"missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "f32"))
        sh = _shape(str(name), intent=intent, bindings=bindings)
        numel = 1
        for d in sh:
            numel *= int(d)
        memref_ty = f"memref<{int(numel)}x{elem_ty}>"
        memref_ty_by_name[str(name)] = memref_ty
        arg_decls.append(f"%{_mlir_ident(name)}: {memref_ty}")

    q_memref_ty = memref_ty_by_name[q_name]
    k_memref_ty = memref_ty_by_name[k_name]
    v_memref_ty = memref_ty_by_name[v_name]
    out_memref_ty = memref_ty_by_name[out_name]
    sm_scale_memref_ty = memref_ty_by_name.get(str(sm_scale_name), "")

    acc_memref_ty = f"memref<{int(head_dim)}xf32>"

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cQ = arith.constant {q_ctx} : index")
    lines.append(f"    %cKV = arith.constant {kv_ctx} : index")
    lines.append(f"    %cHD = arith.constant {head_dim} : index")
    lines.append(f"    %cBlockKV = arith.constant {int(block_kv)} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    lines.append("    %neg_inf = arith.constant 0xFF800000 : f32")
    if sm_scale_name:
        lines.append(f"    %sm = memref.load %{_mlir_ident(sm_scale_name)}[%c0] : {sm_scale_memref_ty}")
    else:
        import math as _math  # noqa: PLC0415

        sm = 1.0 / _math.sqrt(float(head_dim))
        lines.append(f"    %sm = arith.constant {_f32_lit(float(sm))} : f32")
    lines.append(f"    %acc = memref.alloca() : {acc_memref_ty}")
    lines.append("    scf.for %q = %c0 to %cQ step %c1 {")
    lines.append("      %q_base = arith.muli %q, %cHD : index")
    lines.append("      scf.for %d0 = %c0 to %cHD step %c1 {")
    lines.append(f"        memref.store %c0f, %acc[%d0] : {acc_memref_ty}")
    lines.append("      }")
    # kv_end = min(KV_CTX, q+1) for causal attention.
    lines.append("      %q1 = arith.addi %q, %c1 : index")
    lines.append("      %p_kvend = arith.cmpi ule, %q1, %cKV : index")
    lines.append("      %kv_end = arith.select %p_kvend, %q1, %cKV : index")
    lines.append(
        "      %m_out, %l_out = scf.for %tile = %c0 to %kv_end step %cBlockKV "
        "iter_args(%m_i = %neg_inf, %l_i = %c0f) -> (f32, f32) {"
    )
    lines.append("        %tile_end0 = arith.addi %tile, %cBlockKV : index")
    lines.append("        %p_tile_end = arith.cmpi ule, %tile_end0, %kv_end : index")
    lines.append("        %tile_end = arith.select %p_tile_end, %tile_end0, %kv_end : index")
    # Pass1: max score in tile.
    lines.append("        %m_tile = scf.for %kv = %tile to %tile_end step %c1 iter_args(%mx = %neg_inf) -> (f32) {")
    lines.append("          %k_base1 = arith.muli %kv, %cHD : index")
    lines.append("          %dot1 = scf.for %d1 = %c0 to %cHD step %c1 iter_args(%s1 = %c0f) -> (f32) {")
    lines.append("            %q_idx1 = arith.addi %q_base, %d1 : index")
    lines.append("            %k_idx1 = arith.addi %k_base1, %d1 : index")
    lines.append(f"            %qv1 = memref.load %{_mlir_ident(q_name)}[%q_idx1] : {q_memref_ty}")
    lines.append(f"            %kvv1 = memref.load %{_mlir_ident(k_name)}[%k_idx1] : {k_memref_ty}")
    lines.append("            %p0 = arith.mulf %qv1, %kvv1 : f32")
    lines.append("            %s2 = arith.addf %s1, %p0 : f32")
    lines.append("            scf.yield %s2 : f32")
    lines.append("          }")
    lines.append("          %score1 = arith.mulf %dot1, %sm : f32")
    lines.append("          %mx2 = arith.maximumf %mx, %score1 : f32")
    lines.append("          scf.yield %mx2 : f32")
    lines.append("        }")
    # m_new, alpha, scale l/acc
    lines.append("        %m_new = arith.maximumf %m_i, %m_tile : f32")
    lines.append("        %delta = arith.subf %m_i, %m_new : f32")
    lines.append("        %alpha = math.exp %delta : f32")
    lines.append("        %l_scaled = arith.mulf %l_i, %alpha : f32")
    lines.append("        scf.for %d2 = %c0 to %cHD step %c1 {")
    lines.append(f"          %old = memref.load %acc[%d2] : {acc_memref_ty}")
    lines.append("          %scaled = arith.mulf %old, %alpha : f32")
    lines.append(f"          memref.store %scaled, %acc[%d2] : {acc_memref_ty}")
    lines.append("        }")
    # Pass2: sum p and accumulate acc.
    lines.append("        %sum_p = scf.for %kv2 = %tile to %tile_end step %c1 iter_args(%sp = %c0f) -> (f32) {")
    lines.append("          %k_base2 = arith.muli %kv2, %cHD : index")
    lines.append("          %dot2 = scf.for %d3 = %c0 to %cHD step %c1 iter_args(%s3 = %c0f) -> (f32) {")
    lines.append("            %q_idx2 = arith.addi %q_base, %d3 : index")
    lines.append("            %k_idx2 = arith.addi %k_base2, %d3 : index")
    lines.append(f"            %qv2 = memref.load %{_mlir_ident(q_name)}[%q_idx2] : {q_memref_ty}")
    lines.append(f"            %kvv2 = memref.load %{_mlir_ident(k_name)}[%k_idx2] : {k_memref_ty}")
    lines.append("            %p1 = arith.mulf %qv2, %kvv2 : f32")
    lines.append("            %s4 = arith.addf %s3, %p1 : f32")
    lines.append("            scf.yield %s4 : f32")
    lines.append("          }")
    lines.append("          %score2 = arith.mulf %dot2, %sm : f32")
    lines.append("          %shift = arith.subf %score2, %m_new : f32")
    lines.append("          %p2 = math.exp %shift : f32")
    lines.append("          %sp2 = arith.addf %sp, %p2 : f32")
    lines.append("          %v_base2 = arith.muli %kv2, %cHD : index")
    lines.append("          scf.for %d4 = %c0 to %cHD step %c1 {")
    lines.append("            %v_idx = arith.addi %v_base2, %d4 : index")
    lines.append(f"            %vv = memref.load %{_mlir_ident(v_name)}[%v_idx] : {v_memref_ty}")
    lines.append(f"            %cur = memref.load %acc[%d4] : {acc_memref_ty}")
    lines.append("            %pv = arith.mulf %p2, %vv : f32")
    lines.append("            %nxt = arith.addf %cur, %pv : f32")
    lines.append(f"            memref.store %nxt, %acc[%d4] : {acc_memref_ty}")
    lines.append("          }")
    lines.append("          scf.yield %sp2 : f32")
    lines.append("        }")
    lines.append("        %l_next = arith.addf %l_scaled, %sum_p : f32")
    lines.append("        scf.yield %m_new, %l_next : f32, f32")
    lines.append("      }")
    # Write output: acc / l_out.
    lines.append("      scf.for %d5 = %c0 to %cHD step %c1 {")
    lines.append(f"        %av = memref.load %acc[%d5] : {acc_memref_ty}")
    lines.append("        %outv = arith.divf %av, %l_out : f32")
    lines.append("        %o_idx = arith.addi %q_base, %d5 : index")
    lines.append(f"        memref.store %outv, %{_mlir_ident(out_name)}[%o_idx] : {out_memref_ty}")
    lines.append("      }")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_attn_fwd_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 _attn_fwd expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 _attn_fwd expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 4:
        raise RuntimeError(f"rvv cpu-loops v1 _attn_fwd expects rank-4 output, got out_shape={out_shape}")
    z_dim, q_heads, q_ctx, head_dim = map(int, out_shape)
    if z_dim != 1 or q_heads != 1:
        raise RuntimeError(f"rvv cpu-loops v1 _attn_fwd supports only Z=q_numhead=1, got Z={z_dim} heads={q_heads}")
    kv_ctx = int(bindings.get("KV_CTX") or 0)
    kv_heads = int(bindings.get("kv_numhead") or 1)
    if kv_heads != 1:
        raise RuntimeError(f"rvv cpu-loops v1 _attn_fwd supports only kv_numhead=1, got kv_numhead={kv_heads}")

    tensors = dict(intent.tensors or {})
    io_names = _io_arg_order(intent)
    ext_inputs = [n for n in io_names if n not in set(outputs)]

    # Infer Q/K/V/scalars by shape (ignore attn_mask and other metadata).
    q_name = ""
    k_name = ""
    v_name = ""
    sm_scale_name = ""
    q_candidates: list[str] = []
    for nm in ext_inputs:
        tt = tensors.get(str(nm))
        if tt is None:
            continue
        if _dtype(getattr(tt, "dtype", "f32")) != "f32":
            continue
        shp = _shape(str(nm), intent=intent, bindings=bindings)
        if list(shp) == list(out_shape):
            q_candidates.append(str(nm))
    if q_candidates:
        for nm in q_candidates:
            n = str(nm).strip().lower()
            if n in {"q", "query"} or n.startswith("q_") or n.startswith("qptr") or "query" in n:
                q_name = str(nm)
                break
        if not q_name:
            for nm in q_candidates:
                n = str(nm).strip().lower()
                if (n == "q") or ("q" in n and "kv" not in n):
                    q_name = str(nm)
                    break
    if not q_name and len(q_candidates) == 1:
        q_name = str(q_candidates[0])

    # Find scalar sm_scale.
    for nm in ext_inputs:
        tt = tensors.get(str(nm))
        if tt is None:
            continue
        if _dtype(getattr(tt, "dtype", "f32")) != "f32":
            continue
        shp = _shape(str(nm), intent=intent, bindings=bindings)
        if len(shp) == 0 and ("sm_scale" in str(nm).lower() or str(nm).lower() in {"scale", "sm_scale", "smscale"}):
            sm_scale_name = str(nm)
            break

    # Find K/V: rank-4 f32 with last dim == head_dim and third dim == KV_CTX (if known).
    rank4_f32: list[tuple[str, list[int]]] = []
    for nm in ext_inputs:
        tt = tensors.get(str(nm))
        if tt is None:
            continue
        if _dtype(getattr(tt, "dtype", "f32")) != "f32":
            continue
        if str(nm) == q_name:
            continue
        shp = _shape(str(nm), intent=intent, bindings=bindings)
        if len(shp) == 4 and int(shp[-1]) == int(head_dim):
            rank4_f32.append((str(nm), list(shp)))
    groups: dict[tuple[int, int, int, int], list[str]] = {}
    for nm, shp in rank4_f32:
        groups.setdefault(tuple(int(x) for x in shp), []).append(nm)
    for shp, names in groups.items():
        if len(names) < 2:
            continue
        if kv_ctx > 0 and int(shp[2]) != int(kv_ctx):
            continue
        k_name = next((n for n in names if "k" in str(n).lower()), names[0])
        v_name = next((n for n in names if ("v" in str(n).lower()) and str(n) != str(k_name)), "")
        if not v_name:
            v_name = names[1] if names[0] == k_name else names[0]
        if kv_ctx <= 0:
            kv_ctx = int(shp[2])
        break
    if not q_name or not k_name or not v_name:
        raise RuntimeError(f"rvv cpu-loops v1 _attn_fwd failed to infer Q/K/V tensors for kernel={kernel_name!r}")
    if kv_ctx <= 0:
        raise RuntimeError("rvv cpu-loops v1 _attn_fwd missing KV_CTX")

    if not sm_scale_name:
        for nm in ext_inputs:
            tt = tensors.get(str(nm))
            if tt is None:
                continue
            if _dtype(getattr(tt, "dtype", "f32")) != "f32":
                continue
            shp = _shape(str(nm), intent=intent, bindings=bindings)
            if len(shp) == 0:
                sm_scale_name = str(nm)
                break

    arg_decls: list[str] = []
    memref_ty_by_name: dict[str, str] = {}
    for name in list(io_names):
        tt = tensors.get(str(name))
        if tt is None:
            raise RuntimeError(f"missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "f32"))
        sh = _shape(str(name), intent=intent, bindings=bindings)
        numel = 1
        for d in sh:
            numel *= int(d)
        memref_ty = f"memref<{int(numel)}x{elem_ty}>"
        memref_ty_by_name[str(name)] = memref_ty
        arg_decls.append(f"%{_mlir_ident(name)}: {memref_ty}")

    q_memref_ty = memref_ty_by_name[q_name]
    k_memref_ty = memref_ty_by_name[k_name]
    v_memref_ty = memref_ty_by_name[v_name]
    out_memref_ty = memref_ty_by_name[out_name]
    sm_scale_memref_ty = memref_ty_by_name.get(str(sm_scale_name), "")

    acc_memref_ty = f"memref<{int(head_dim)}xf32>"

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %cQ = arith.constant {int(q_ctx)} : index")
    lines.append(f"    %cKV = arith.constant {int(kv_ctx)} : index")
    lines.append(f"    %cHD = arith.constant {int(head_dim)} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    lines.append("    %neg_inf = arith.constant 0xFF800000 : f32")
    if sm_scale_name:
        lines.append(f"    %sm = memref.load %{_mlir_ident(sm_scale_name)}[%c0] : {sm_scale_memref_ty}")
    else:
        import math as _math  # noqa: PLC0415

        sm = 1.0 / _math.sqrt(float(head_dim))
        lines.append(f"    %sm = arith.constant {_f32_lit(float(sm))} : f32")
    lines.append(f"    %acc = memref.alloca() : {acc_memref_ty}")
    # Only supports single batch/head; flatten indices are row-major.
    lines.append("    scf.for %q = %c0 to %cQ step %c1 {")
    lines.append("      %q_base = arith.muli %q, %cHD : index")
    lines.append("      scf.for %d0 = %c0 to %cHD step %c1 {")
    lines.append(f"        memref.store %c0f, %acc[%d0] : {acc_memref_ty}")
    lines.append("      }")
    # Pass1: max over kv.
    lines.append("      %mx = scf.for %kv = %c0 to %cKV step %c1 iter_args(%x = %neg_inf) -> (f32) {")
    lines.append("        %k_base1 = arith.muli %kv, %cHD : index")
    lines.append("        %dot1 = scf.for %d1 = %c0 to %cHD step %c1 iter_args(%s1 = %c0f) -> (f32) {")
    lines.append("          %q_idx1 = arith.addi %q_base, %d1 : index")
    lines.append("          %k_idx1 = arith.addi %k_base1, %d1 : index")
    lines.append(f"          %qv1 = memref.load %{_mlir_ident(q_name)}[%q_idx1] : {q_memref_ty}")
    lines.append(f"          %kvv1 = memref.load %{_mlir_ident(k_name)}[%k_idx1] : {k_memref_ty}")
    lines.append("          %p0 = arith.mulf %qv1, %kvv1 : f32")
    lines.append("          %s2 = arith.addf %s1, %p0 : f32")
    lines.append("          scf.yield %s2 : f32")
    lines.append("        }")
    lines.append("        %score1 = arith.mulf %dot1, %sm : f32")
    lines.append("        %x2 = arith.maximumf %x, %score1 : f32")
    lines.append("        scf.yield %x2 : f32")
    lines.append("      }")
    # Pass2: sum exp and accumulate.
    lines.append("      %sum = scf.for %kv2 = %c0 to %cKV step %c1 iter_args(%sp = %c0f) -> (f32) {")
    lines.append("        %k_base2 = arith.muli %kv2, %cHD : index")
    lines.append("        %dot2 = scf.for %d2 = %c0 to %cHD step %c1 iter_args(%s3 = %c0f) -> (f32) {")
    lines.append("          %q_idx2 = arith.addi %q_base, %d2 : index")
    lines.append("          %k_idx2 = arith.addi %k_base2, %d2 : index")
    lines.append(f"          %qv2 = memref.load %{_mlir_ident(q_name)}[%q_idx2] : {q_memref_ty}")
    lines.append(f"          %kvv2 = memref.load %{_mlir_ident(k_name)}[%k_idx2] : {k_memref_ty}")
    lines.append("          %p1 = arith.mulf %qv2, %kvv2 : f32")
    lines.append("          %s4 = arith.addf %s3, %p1 : f32")
    lines.append("          scf.yield %s4 : f32")
    lines.append("        }")
    lines.append("        %score2 = arith.mulf %dot2, %sm : f32")
    lines.append("        %shift = arith.subf %score2, %mx : f32")
    lines.append("        %p2 = math.exp %shift : f32")
    lines.append("        %sp2 = arith.addf %sp, %p2 : f32")
    lines.append("        %v_base2 = arith.muli %kv2, %cHD : index")
    lines.append("        scf.for %d3 = %c0 to %cHD step %c1 {")
    lines.append("          %v_idx = arith.addi %v_base2, %d3 : index")
    lines.append(f"          %vv = memref.load %{_mlir_ident(v_name)}[%v_idx] : {v_memref_ty}")
    lines.append(f"          %cur = memref.load %acc[%d3] : {acc_memref_ty}")
    lines.append("          %pv = arith.mulf %p2, %vv : f32")
    lines.append("          %nxt = arith.addf %cur, %pv : f32")
    lines.append(f"          memref.store %nxt, %acc[%d3] : {acc_memref_ty}")
    lines.append("        }")
    lines.append("        scf.yield %sp2 : f32")
    lines.append("      }")
    lines.append("      scf.for %d4 = %c0 to %cHD step %c1 {")
    lines.append(f"        %av = memref.load %acc[%d4] : {acc_memref_ty}")
    lines.append("        %outv = arith.divf %av, %sum : f32")
    lines.append("        %o_idx = arith.addi %q_base, %d4 : index")
    lines.append(f"        memref.store %outv, %{_mlir_ident(out_name)}[%o_idx] : {out_memref_ty}")
    lines.append("      }")
    lines.append("    }")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_upsample_bicubic2d_aa_kernel(
    *,
    kernel_name: str,
    intent: IntentFunction,
    bindings: Mapping[str, Any],
) -> str:
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"rvv cpu-loops v1 upsample_bicubic2d_aa expects single output, got outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 upsample_bicubic2d_aa expects f32 output")
    out_shape = _shape(out_name, intent=intent, bindings=bindings)
    if len(out_shape) != 4:
        raise RuntimeError(f"rvv cpu-loops v1 upsample_bicubic2d_aa expects rank-4 output, got out_shape={out_shape}")
    n_dim, c_dim, oh_dim, ow_dim = map(int, out_shape)
    if n_dim <= 0 or c_dim <= 0 or oh_dim <= 0 or ow_dim <= 0:
        raise RuntimeError(f"rvv cpu-loops v1 upsample_bicubic2d_aa invalid output shape: {out_shape}")

    tensors = dict(intent.tensors or {})
    io_names = _io_arg_order(intent)
    ext_inputs = [n for n in io_names if n not in set(outputs)]

    # Infer input tensor by shape (N,C,IH,IW) and dtype.
    in_name = ""
    ih_dim = 0
    iw_dim = 0
    for nm in ext_inputs:
        tt = tensors.get(str(nm))
        if tt is None:
            continue
        if _dtype(getattr(tt, "dtype", "f32")) != "f32":
            continue
        shp = _shape(str(nm), intent=intent, bindings=bindings)
        if len(shp) == 4 and int(shp[0]) == int(n_dim) and int(shp[1]) == int(c_dim):
            in_name = str(nm)
            ih_dim = int(shp[2])
            iw_dim = int(shp[3])
            break
    if not in_name or ih_dim <= 0 or iw_dim <= 0:
        raise RuntimeError("rvv cpu-loops v1 upsample_bicubic2d_aa failed to infer input tensor")

    arg_decls: list[str] = []
    memref_ty_by_name: dict[str, str] = {}
    for name in list(io_names):
        tt = tensors.get(str(name))
        if tt is None:
            raise RuntimeError(f"missing tensor spec: {name}")
        elem_ty = _dtype(getattr(tt, "dtype", "f32"))
        sh = _shape(str(name), intent=intent, bindings=bindings)
        numel = 1
        for d in sh:
            numel *= int(d)
        memref_ty = f"memref<{int(numel)}x{elem_ty}>"
        memref_ty_by_name[str(name)] = memref_ty
        arg_decls.append(f"%{_mlir_ident(name)}: {memref_ty}")

    in_memref_ty = memref_ty_by_name[in_name]
    out_memref_ty = memref_ty_by_name[out_name]

    # reciprocal_scale_h/w = IH/OH, IW/OW (runner semantics).
    rec_h = float(ih_dim) / float(oh_dim) if oh_dim > 0 else 1.0
    rec_w = float(iw_dim) / float(ow_dim) if ow_dim > 0 else 1.0

    lines: list[str] = []
    lines.append("module attributes {")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{_b64_json(intent.to_json_dict())}"')
    lines.append("} {")
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}({', '.join(arg_decls)}) {{")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append("    %c2 = arith.constant 2 : index")
    lines.append("    %c3 = arith.constant 3 : index")
    lines.append("    %c4 = arith.constant 4 : index")
    lines.append(f"    %cN = arith.constant {n_dim} : index")
    lines.append(f"    %cC = arith.constant {c_dim} : index")
    lines.append(f"    %cIH = arith.constant {ih_dim} : index")
    lines.append(f"    %cIW = arith.constant {iw_dim} : index")
    lines.append(f"    %cOH = arith.constant {oh_dim} : index")
    lines.append(f"    %cOW = arith.constant {ow_dim} : index")
    lines.append("    %c0f = arith.constant 0.0 : f32")
    lines.append("    %c1f = arith.constant 1.0 : f32")
    lines.append("    %c2f = arith.constant 2.0 : f32")
    lines.append("    %c3f = arith.constant 3.0 : f32")
    lines.append("    %c4f = arith.constant 4.0 : f32")
    lines.append("    %c5f = arith.constant 5.0 : f32")
    lines.append("    %c8f = arith.constant 8.0 : f32")
    lines.append("    %c0_5 = arith.constant 0.5 : f32")
    lines.append("    %support = arith.constant 2.0 : f32")
    lines.append("    %a = arith.constant -0.5 : f32")
    lines.append("    %a2 = arith.addf %a, %c2f : f32")
    lines.append("    %a3 = arith.addf %a, %c3f : f32")
    lines.append(f"    %rec_h = arith.constant {_f32_lit(float(rec_h))} : f32")
    lines.append(f"    %rec_w = arith.constant {_f32_lit(float(rec_w))} : f32")
    lines.append(f"    %iw_f = arith.constant {_f32_lit(float(iw_dim))} : f32")
    lines.append(f"    %ih_f = arith.constant {_f32_lit(float(ih_dim))} : f32")

    lines.append("    %hw_in = arith.muli %cIH, %cIW : index")
    lines.append("    %hw_out = arith.muli %cOH, %cOW : index")
    lines.append("    scf.for %n = %c0 to %cN step %c1 {")
    lines.append("      scf.for %c = %c0 to %cC step %c1 {")
    lines.append("        %nc = arith.muli %n, %cC : index")
    lines.append("        %nc2 = arith.addi %nc, %c : index")
    lines.append("        %in_base = arith.muli %nc2, %hw_in : index")
    lines.append("        %out_base = arith.muli %nc2, %hw_out : index")
    lines.append("        scf.for %oh = %c0 to %cOH step %c1 {")
    lines.append("          %oh_i32 = arith.index_cast %oh : index to i32")
    lines.append("          %oh_f = arith.sitofp %oh_i32 : i32 to f32")
    lines.append("          %oh_p = arith.addf %oh_f, %c0_5 : f32")
    lines.append("          %center_h = arith.mulf %oh_p, %rec_h : f32")
    lines.append("          %h0 = arith.subf %center_h, %support : f32")
    lines.append("          %h1 = arith.addf %h0, %c0_5 : f32")
    lines.append("          %h2 = arith.maximumf %h1, %c0f : f32")
    lines.append("          %span_h_i32 = arith.fptosi %h2 : f32 to i32")
    lines.append("          %span_h = arith.index_cast %span_h_i32 : i32 to index")
    lines.append("          %span_h_f = arith.sitofp %span_h_i32 : i32 to f32")
    lines.append("          %h3 = arith.addf %center_h, %support : f32")
    lines.append("          %h4 = arith.addf %h3, %c0_5 : f32")
    lines.append("          %h5 = arith.minimumf %h4, %ih_f : f32")
    lines.append("          %h6 = arith.subf %h5, %span_h_f : f32")
    lines.append("          %span_sz_h_i32 = arith.fptosi %h6 : f32 to i32")
    lines.append("          %span_sz_h = arith.index_cast %span_sz_h_i32 : i32 to index")
    lines.append("          %start_minus_center_h = arith.subf %span_h_f, %center_h : f32")
    lines.append("          %out_row = arith.muli %oh, %cOW : index")
    lines.append("          scf.for %ow = %c0 to %cOW step %c1 {")
    lines.append("            %ow_i32 = arith.index_cast %ow : index to i32")
    lines.append("            %ow_f = arith.sitofp %ow_i32 : i32 to f32")
    lines.append("            %ow_p = arith.addf %ow_f, %c0_5 : f32")
    lines.append("            %center_w = arith.mulf %ow_p, %rec_w : f32")
    lines.append("            %w0 = arith.subf %center_w, %support : f32")
    lines.append("            %w1 = arith.addf %w0, %c0_5 : f32")
    lines.append("            %w2 = arith.maximumf %w1, %c0f : f32")
    lines.append("            %span_w_i32 = arith.fptosi %w2 : f32 to i32")
    lines.append("            %span_w = arith.index_cast %span_w_i32 : i32 to index")
    lines.append("            %span_w_f = arith.sitofp %span_w_i32 : i32 to f32")
    lines.append("            %w3 = arith.addf %center_w, %support : f32")
    lines.append("            %w4 = arith.addf %w3, %c0_5 : f32")
    lines.append("            %w5 = arith.minimumf %w4, %iw_f : f32")
    lines.append("            %w6 = arith.subf %w5, %span_w_f : f32")
    lines.append("            %span_sz_w_i32 = arith.fptosi %w6 : f32 to i32")
    lines.append("            %span_sz_w = arith.index_cast %span_sz_w_i32 : i32 to index")
    lines.append("            %start_minus_center_w = arith.subf %span_w_f, %center_w : f32")

    # weights x0..x4 (raw)
    for i in range(5):
        c_idx = "%c0" if i == 0 else ("%c1" if i == 1 else f"%c{i}")
        lines.append(f"            %px{i} = arith.cmpi ult, {c_idx}, %span_sz_w : index")
        lines.append(f"            %x{i}_off = arith.constant {_f32_lit(float(i))} : f32")
        lines.append(f"            %x{i}_t0 = arith.addf %x{i}_off, %start_minus_center_w : f32")
        lines.append(f"            %x{i}_t1 = arith.addf %x{i}_t0, %c0_5 : f32")
        lines.append(f"            %wx{i}_abs = math.absf %x{i}_t1 : f32")
        lines.append(f"            %wx{i}_lt1 = arith.cmpf olt, %wx{i}_abs, %c1f : f32")
        lines.append(f"            %wx{i}_lt2 = arith.cmpf olt, %wx{i}_abs, %c2f : f32")
        lines.append(f"            %wx{i}_u0 = arith.mulf %a2, %wx{i}_abs : f32")
        lines.append(f"            %wx{i}_u1 = arith.subf %wx{i}_u0, %a3 : f32")
        lines.append(f"            %wx{i}_u2 = arith.mulf %wx{i}_u1, %wx{i}_abs : f32")
        lines.append(f"            %wx{i}_u3 = arith.mulf %wx{i}_u2, %wx{i}_abs : f32")
        lines.append(f"            %wx{i}_w1 = arith.addf %wx{i}_u3, %c1f : f32")
        lines.append(f"            %wx{i}_v0 = arith.subf %wx{i}_abs, %c5f : f32")
        lines.append(f"            %wx{i}_v1 = arith.mulf %wx{i}_v0, %wx{i}_abs : f32")
        lines.append(f"            %wx{i}_v2 = arith.addf %wx{i}_v1, %c8f : f32")
        lines.append(f"            %wx{i}_v3 = arith.mulf %wx{i}_v2, %wx{i}_abs : f32")
        lines.append(f"            %wx{i}_v4 = arith.subf %wx{i}_v3, %c4f : f32")
        lines.append(f"            %wx{i}_w2 = arith.mulf %wx{i}_v4, %a : f32")
        lines.append(f"            %wx{i}_s2 = arith.select %wx{i}_lt2, %wx{i}_w2, %c0f : f32")
        lines.append(f"            %wx{i}_s1 = arith.select %wx{i}_lt1, %wx{i}_w1, %wx{i}_s2 : f32")
        lines.append(f"            %xw{i} = arith.select %px{i}, %wx{i}_s1, %c0f : f32")

    lines.append("            %xw_sum0 = arith.addf %xw0, %xw1 : f32")
    lines.append("            %xw_sum1 = arith.addf %xw_sum0, %xw2 : f32")
    lines.append("            %xw_sum2 = arith.addf %xw_sum1, %xw3 : f32")
    lines.append("            %xw_sum = arith.addf %xw_sum2, %xw4 : f32")
    lines.append("            %xw_nz = arith.cmpf une, %xw_sum, %c0f : f32")
    lines.append("            %xw_den = arith.select %xw_nz, %xw_sum, %c1f : f32")
    for i in range(5):
        lines.append(f"            %wx{i} = arith.divf %xw{i}, %xw_den : f32")

    # weights y0..y4 (raw)
    for i in range(5):
        c_idx = "%c0" if i == 0 else ("%c1" if i == 1 else f"%c{i}")
        lines.append(f"            %py{i} = arith.cmpi ult, {c_idx}, %span_sz_h : index")
        lines.append(f"            %y{i}_off = arith.constant {_f32_lit(float(i))} : f32")
        lines.append(f"            %y{i}_t0 = arith.addf %y{i}_off, %start_minus_center_h : f32")
        lines.append(f"            %y{i}_t1 = arith.addf %y{i}_t0, %c0_5 : f32")
        lines.append(f"            %wy{i}_abs = math.absf %y{i}_t1 : f32")
        lines.append(f"            %wy{i}_lt1 = arith.cmpf olt, %wy{i}_abs, %c1f : f32")
        lines.append(f"            %wy{i}_lt2 = arith.cmpf olt, %wy{i}_abs, %c2f : f32")
        lines.append(f"            %wy{i}_u0 = arith.mulf %a2, %wy{i}_abs : f32")
        lines.append(f"            %wy{i}_u1 = arith.subf %wy{i}_u0, %a3 : f32")
        lines.append(f"            %wy{i}_u2 = arith.mulf %wy{i}_u1, %wy{i}_abs : f32")
        lines.append(f"            %wy{i}_u3 = arith.mulf %wy{i}_u2, %wy{i}_abs : f32")
        lines.append(f"            %wy{i}_w1 = arith.addf %wy{i}_u3, %c1f : f32")
        lines.append(f"            %wy{i}_v0 = arith.subf %wy{i}_abs, %c5f : f32")
        lines.append(f"            %wy{i}_v1 = arith.mulf %wy{i}_v0, %wy{i}_abs : f32")
        lines.append(f"            %wy{i}_v2 = arith.addf %wy{i}_v1, %c8f : f32")
        lines.append(f"            %wy{i}_v3 = arith.mulf %wy{i}_v2, %wy{i}_abs : f32")
        lines.append(f"            %wy{i}_v4 = arith.subf %wy{i}_v3, %c4f : f32")
        lines.append(f"            %wy{i}_w2 = arith.mulf %wy{i}_v4, %a : f32")
        lines.append(f"            %wy{i}_s2 = arith.select %wy{i}_lt2, %wy{i}_w2, %c0f : f32")
        lines.append(f"            %wy{i}_s1 = arith.select %wy{i}_lt1, %wy{i}_w1, %wy{i}_s2 : f32")
        lines.append(f"            %yw{i} = arith.select %py{i}, %wy{i}_s1, %c0f : f32")

    lines.append("            %yw_sum0 = arith.addf %yw0, %yw1 : f32")
    lines.append("            %yw_sum1 = arith.addf %yw_sum0, %yw2 : f32")
    lines.append("            %yw_sum2 = arith.addf %yw_sum1, %yw3 : f32")
    lines.append("            %yw_sum = arith.addf %yw_sum2, %yw4 : f32")
    lines.append("            %yw_nz = arith.cmpf une, %yw_sum, %c0f : f32")
    lines.append("            %yw_den = arith.select %yw_nz, %yw_sum, %c1f : f32")
    for i in range(5):
        lines.append(f"            %wy{i} = arith.divf %yw{i}, %yw_den : f32")

    # Neighborhood indices.
    for i in range(5):
        c_idx = "%c0" if i == 0 else ("%c1" if i == 1 else f"%c{i}")
        lines.append(f"            %ih{i} = arith.addi %span_h, {c_idx} : index")
        lines.append(f"            %pyh{i} = arith.cmpi ult, %ih{i}, %cIH : index")
        lines.append(f"            %ih{i}_off = arith.muli %ih{i}, %cIW : index")
        lines.append(f"            %row{i}_base = arith.addi %in_base, %ih{i}_off : index")
        lines.append(f"            %iw{i} = arith.addi %span_w, {c_idx} : index")
        lines.append(f"            %pxw{i} = arith.cmpi ult, %iw{i}, %cIW : index")

    # Load 5x5 neighborhood with guards.
    for y in range(5):
        for x in range(5):
            lines.append(f"            %p{y}{x} = arith.andi %pyh{y}, %pxw{x} : i1")
            lines.append(f"            %v{y}{x} = scf.if %p{y}{x} -> (f32) {{")
            lines.append(f"              %idx{y}{x} = arith.addi %row{y}_base, %iw{x} : index")
            lines.append(f"              %vv{y}{x} = memref.load %{_mlir_ident(in_name)}[%idx{y}{x}] : {in_memref_ty}")
            lines.append(f"              scf.yield %vv{y}{x} : f32")
            lines.append("            } else {")
            lines.append("              scf.yield %c0f : f32")
            lines.append("            }")

    # dataY = sum_x valYx * wxX
    for y in range(5):
        lines.append(f"            %d{y}0 = arith.mulf %v{y}0, %wx0 : f32")
        lines.append(f"            %d{y}1 = arith.mulf %v{y}1, %wx1 : f32")
        lines.append(f"            %d{y}2 = arith.mulf %v{y}2, %wx2 : f32")
        lines.append(f"            %d{y}3 = arith.mulf %v{y}3, %wx3 : f32")
        lines.append(f"            %d{y}4 = arith.mulf %v{y}4, %wx4 : f32")
        lines.append(f"            %d{y}_s0 = arith.addf %d{y}0, %d{y}1 : f32")
        lines.append(f"            %d{y}_s1 = arith.addf %d{y}_s0, %d{y}2 : f32")
        lines.append(f"            %d{y}_s2 = arith.addf %d{y}_s1, %d{y}3 : f32")
        lines.append(f"            %data{y} = arith.addf %d{y}_s2, %d{y}4 : f32")

    # result = sum_y dataY * wyY
    lines.append("            %r0 = arith.mulf %data0, %wy0 : f32")
    lines.append("            %r1 = arith.mulf %data1, %wy1 : f32")
    lines.append("            %r2 = arith.mulf %data2, %wy2 : f32")
    lines.append("            %r3 = arith.mulf %data3, %wy3 : f32")
    lines.append("            %r4 = arith.mulf %data4, %wy4 : f32")
    lines.append("            %r_s0 = arith.addf %r0, %r1 : f32")
    lines.append("            %r_s1 = arith.addf %r_s0, %r2 : f32")
    lines.append("            %r_s2 = arith.addf %r_s1, %r3 : f32")
    lines.append("            %res = arith.addf %r_s2, %r4 : f32")
    lines.append("            %out_off = arith.addi %out_row, %ow : index")
    lines.append("            %out_idx = arith.addi %out_base, %out_off : index")
    lines.append(f"            memref.store %res, %{_mlir_ident(out_name)}[%out_idx] : {out_memref_ty}")

    lines.append("          }")  # ow
    lines.append("        }")  # oh
    lines.append("      }")  # c
    lines.append("    }")  # n
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _maybe_rewrite_relu2d_where_pattern(intent: IntentFunction) -> IntentFunction:
    """
    Canonicalize a common relu2d lowering pattern in expanded intents:

      zero = const(0.0)
      mask = gt(x, zero)
      out  = where(mask, x, zero)

    Our RVV cpu-loops v1 backend only supports a small op set and operates on
    scalar SSA values inside a single `scf.for` loop. Rather than implementing
    general boolean tensors + where, rewrite this pattern into a single `relu`
    op which the emitter supports.
    """
    ops = [op for op in list(intent.ops or []) if op is not None]
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1 or len(ops) != 3:
        return intent

    op0, op1, op2 = ops
    if str(getattr(op0, "op", "")).strip() != "const":
        return intent
    if str(getattr(op1, "op", "")).strip() != "gt":
        return intent
    if str(getattr(op2, "op", "")).strip() != "where":
        return intent

    zero_name = str(getattr(op0, "output", "")).strip()
    mask_name = str(getattr(op1, "output", "")).strip()
    out_name = str(getattr(op2, "output", "")).strip()
    if not zero_name or not mask_name or not out_name:
        return intent
    if out_name != outputs[0]:
        return intent

    attrs0 = dict(getattr(op0, "attrs", {}) or {})
    if str(attrs0.get("dtype") or "f32").strip().lower() != "f32":
        return intent
    try:
        v0 = float(attrs0.get("value"))
    except Exception:
        return intent
    if v0 != 0.0:
        return intent

    ins1 = [str(x).strip() for x in list(getattr(op1, "inputs", []) or []) if str(x).strip()]
    if len(ins1) != 2 or ins1[1] != zero_name:
        return intent
    x_name = str(ins1[0])
    if not x_name:
        return intent

    ins2 = [str(x).strip() for x in list(getattr(op2, "inputs", []) or []) if str(x).strip()]
    if len(ins2) != 3 or ins2[0] != mask_name or ins2[1] != x_name or ins2[2] != zero_name:
        return intent

    x_tt = (intent.tensors or {}).get(x_name)
    out_tt = (intent.tensors or {}).get(out_name)
    zero_tt = (intent.tensors or {}).get(zero_name)
    if x_tt is None or out_tt is None or zero_tt is None:
        return intent
    if str(getattr(x_tt, "dtype", "")).strip().lower() != "f32":
        return intent
    if str(getattr(out_tt, "dtype", "")).strip().lower() != "f32":
        return intent
    if str(getattr(zero_tt, "dtype", "")).strip().lower() != "f32":
        return intent
    if len(list(getattr(zero_tt, "shape", []) or [])) != 0:
        return intent
    if list(getattr(x_tt, "shape", []) or []) != list(getattr(out_tt, "shape", []) or []):
        return intent

    def _dim_json(d: Any) -> int | str:
        raw = getattr(d, "value", d)
        if isinstance(raw, int):
            return int(raw)
        return str(raw)

    return IntentFunction.from_json_dict(
        {
            "name": str(intent.name or "relu2d"),
            "tensors": {
                str(x_name): {
                    "dtype": "f32",
                    "shape": [_dim_json(d) for d in list(getattr(x_tt, "shape", []) or [])],
                    "layout": "row_major",
                },
                str(out_name): {
                    "dtype": "f32",
                    "shape": [_dim_json(d) for d in list(getattr(out_tt, "shape", []) or [])],
                    "layout": "row_major",
                },
            },
            "ops": [{"op": "relu", "inputs": [str(x_name)], "output": str(out_name), "attrs": {}}],
            "outputs": [str(out_name)],
        }
    )


def _maybe_normalize_bf16_to_f32(intent: IntentFunction) -> tuple[IntentFunction, bool]:
    """
    FlagGems runners stage bf16 tensors as f32 (NumPy cannot represent bfloat16
    natively and torch->numpy conversion fails). For RVV cpu-loops v1, normalize
    bf16 tensors to f32 so the emitted ABI matches baseline snapshots.
    """
    tensors = dict(intent.tensors or {})
    has_bf16 = any(str(getattr(t, "dtype", "")).strip().lower() == "bf16" for t in tensors.values())
    if not has_bf16:
        return intent, False

    payload = intent.to_json_dict()
    tensors_json = payload.get("tensors")
    if isinstance(tensors_json, dict):
        for spec in tensors_json.values():
            if not isinstance(spec, dict):
                continue
            if str(spec.get("dtype") or "").strip().lower() == "bf16":
                spec["dtype"] = "f32"

    ops_json = payload.get("ops")
    if isinstance(ops_json, list):
        for op in ops_json:
            if not isinstance(op, dict):
                continue
            op_name = str(op.get("op") or "").strip()
            attrs = op.get("attrs")
            if not isinstance(attrs, dict):
                attrs = {}
                op["attrs"] = attrs
            if op_name == "cast" and str(attrs.get("to") or "").strip().lower() == "bf16":
                attrs["to"] = "f32"
            if op_name == "const" and str(attrs.get("dtype") or "").strip().lower() == "bf16":
                attrs["dtype"] = "f32"

    return IntentFunction.from_json_dict(payload), True


def lower_intent_to_rvv_cpu_kernel(module: IntentMLIRModule, *, backend: str | None = None, **_: object) -> IntentMLIRModule:
    b = str(backend or "").strip().lower()
    if b and not b.startswith("rvv") and b != "riscv":
        return module

    incoming_meta = dict(module.meta or {})
    bindings_raw = incoming_meta.get("shape_bindings")
    if not isinstance(bindings_raw, Mapping) or not bindings_raw:
        raise RuntimeError("rvv cpu-loops v1 requires module.meta['shape_bindings']")
    bindings: dict[str, Any] = {str(k): int(v) for k, v in dict(bindings_raw).items() if str(k).strip()}

    intent = to_intent(module)
    intent = _maybe_rewrite_relu2d_where_pattern(intent)
    intent, normalized_bf16 = _maybe_normalize_bf16_to_f32(intent)
    _ = materialize_missing_op_output_tensors(intent)
    kernel_name = str(
        incoming_meta.get("kernel")
        or incoming_meta.get("spec_name")
        or incoming_meta.get("kernel_name")
        or intent.name
        or "intent"
    ).strip()
    if not kernel_name:
        kernel_name = str(intent.name or "intent")

    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    out_name = str((outputs or [""])[0]).strip()
    out_tt = (intent.tensors or {}).get(out_name) if out_name else None
    out_rank = len(list(getattr(out_tt, "shape", []) or [])) if out_tt is not None else 0
    ops = [op for op in list(intent.ops or []) if op is not None]
    if kernel_name == "softmax_inner":
        module_text = _emit_softmax_inner_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_softmax_inner_v1"
    elif kernel_name == "log_softmax2d":
        module_text = _emit_log_softmax2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_log_softmax2d_v1"
    elif kernel_name == "flash_attention2d":
        module_text = _emit_attention2d_causal_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings, block_kv=32)
        kind = "cpu_loops_flash_attention2d_v1"
    elif kernel_name == "masked_attention2d":
        module_text = _emit_attention2d_causal_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings, block_kv=64)
        kind = "cpu_loops_masked_attention2d_v1"
    elif kernel_name == "_attn_fwd":
        module_text = _emit_attn_fwd_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_attn_fwd_v1"
    elif kernel_name == "upsample_bicubic2d_aa":
        module_text = _emit_upsample_bicubic2d_aa_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_upsample_bicubic2d_aa_v1"
    elif kernel_name == "ai_bench_softmax":
        module_text = _emit_row_softmax2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings, require_mask=False)
        kind = "cpu_loops_row_softmax2d_v1"
    elif kernel_name == "masked_softmax2d":
        module_text = _emit_row_softmax2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings, require_mask=True)
        kind = "cpu_loops_masked_softmax2d_v1"
    elif kernel_name == "gather2d":
        module_text = _emit_gather2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_gather2d_v1"
    elif kernel_name == "grouped_row_sum2d":
        module_text = _emit_grouped_row_sum2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_grouped_row_sum2d_v1"
    elif kernel_name == "any_kernel_dim":
        module_text = _emit_row_reduce_any_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_row_reduce_any_axis1_v1"
    elif kernel_name == "row_mean":
        module_text = _emit_row_mean_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_row_mean_axis1_v1"
    elif kernel_name in {"ai_bench_matmul", "matmul_relu2d", "matmul_bias_relu2d", "matmul_fused_epilogue2d"}:
        relu = kernel_name in {"matmul_relu2d", "matmul_bias_relu2d"}
        require_bias = kernel_name in {"matmul_bias_relu2d", "matmul_fused_epilogue2d"}
        require_row_col_masks = kernel_name in {"matmul_fused_epilogue2d"}
        module_text = _emit_matmul2d_kernel(
            kernel_name=kernel_name,
            intent=intent,
            bindings=bindings,
            relu=relu,
            require_bias=require_bias,
            require_row_col_masks=require_row_col_masks,
        )
        kind = f"cpu_loops_{kernel_name}_v1"
    elif kernel_name == "mlp2d":
        module_text = _emit_mlp2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_mlp2d_v1"
    elif kernel_name == "count_nonzero2d":
        module_text = _emit_count_nonzero2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_count_nonzero2d_v1"
    elif kernel_name == "trace2d":
        module_text = _emit_trace2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_trace2d_v1"
    elif kernel_name == "allclose2d":
        module_text = _emit_allclose2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_allclose2d_v1"
    elif kernel_name == "rms_norm2d":
        module_text = _emit_rms_norm2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "rms_norm2d_rvv_v1"
    elif kernel_name == "rms_norm_residual2d":
        module_text = _emit_rms_norm_residual2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "rms_norm_residual2d_rvv_v1"
    elif kernel_name == "layer_norm_persistent":
        module_text = _emit_layer_norm_persistent_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "layer_norm_rvv_v1"
    elif kernel_name == "layer_norm_residual2d":
        module_text = _emit_layer_norm_residual2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "layer_norm_residual2d_rvv_v1"
    elif kernel_name == "ai_bench_layernorm":
        module_text = _emit_layer_norm_persistent_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "layer_norm_rvv_v1"
    elif kernel_name == "ai_bench_dropout":
        module_text = _emit_ai_bench_dropout_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_ai_bench_dropout_v1"
    elif kernel_name == "ai_bench_rope":
        module_text = _emit_ai_bench_rope_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_ai_bench_rope_v1"
    elif kernel_name == "ai_bench_correlation":
        module_text = _emit_ai_bench_correlation_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_ai_bench_correlation_v1"
    elif kernel_name == "ai_bench_resize":
        module_text = _emit_ai_bench_resize_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_ai_bench_resize_v1"
    elif kernel_name == "ai_bench_warp":
        module_text = _emit_ai_bench_warp_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_ai_bench_warp_v1"
    elif kernel_name == "group_norm_kernel":
        module_text = _emit_group_norm_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "group_norm_rvv_v1"
    elif kernel_name == "arange1d":
        module_text = _emit_arange1d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_arange1d_v1"
    elif kernel_name in {
        "bitwise_and2d",
        "bitwise_or2d",
        "bitwise_not2d",
        "bitwise_left_shift2d",
        "bitwise_right_shift2d",
    }:
        module_text = _emit_bitwise2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_bitwise2d_v1"
    elif kernel_name in {"eq2d", "ne2d", "lt2d", "le2d", "gt2d", "ge2d"}:
        module_text = _emit_cmp2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_cmp2d_v1"
    elif len(ops) == 2:
        op_names = [str(getattr(o, "op", "")).strip() for o in ops]
        if set(op_names) == {"reduce_min", "argmin"} and len(outputs) == 2:
            module_text = _emit_row_reduce_min_argmin_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
            kind = "cpu_loops_row_reduce_min_argmin_axis1_v1"
        else:
            if len(outputs) != 1:
                raise RuntimeError(f"rvv cpu-loops v1 unsupported multi-output intent: name={kernel_name} outputs={outputs}")
            module_text = _emit_elementwise_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
            kind = "cpu_loops_v1"
    elif len(ops) == 1:
        op0 = ops[0]
        op_name = str(getattr(op0, "op", "")).strip()
        attrs = dict(getattr(op0, "attrs", {}) or {})
        perm = list(attrs.get("perm") or []) if isinstance(attrs.get("perm"), list) else []
        if op_name == "transpose" and perm == [1, 0]:
            module_text = _emit_transpose2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
            kind = "cpu_loops_transpose2d_v1"
        elif op_name == "argmax" and out_rank == 1:
            module_text = _emit_row_argminmax_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings, mode="max")
            kind = "cpu_loops_row_argmax_axis1_v1"
        elif op_name == "argmin" and out_rank == 1:
            module_text = _emit_row_argminmax_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings, mode="min")
            kind = "cpu_loops_row_argmin_axis1_v1"
        elif op_name == "reduce_prod" and out_rank == 1:
            module_text = _emit_row_reduce_prod_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
            kind = "cpu_loops_row_reduce_prod_axis1_v1"
        elif op_name == "reduce_min" and out_rank == 0:
            module_text = _emit_reduce_min_all_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
            kind = "cpu_loops_reduce_min_all_v1"
        elif out_rank == 1:
            module_text = _emit_row_reduce_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
            kind = "cpu_loops_row_reduce_v1"
        else:
            module_text = _emit_elementwise_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
            kind = "cpu_loops_v1"
    else:
        if len(outputs) != 1:
            raise RuntimeError(f"rvv cpu-loops v1 unsupported multi-output intent: name={kernel_name} outputs={outputs}")
        module_text = _emit_elementwise_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "cpu_loops_v1"

    out = IntentMLIRModule(
        module_text=str(module_text),
        dialect_version="std_mlir_v1",
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(incoming_meta),
        intent_json=intent.to_json_dict(),
    )
    out.meta["rvv_real_mlir_kernel_kind"] = str(kind)
    out.meta["rvv_real_mlir_kernel_emitted"] = True
    if normalized_bf16:
        out.meta["rvv_dtype_normalized_bf16_to_f32"] = True
    return out


__all__ = ["lower_intent_to_rvv_cpu_kernel"]
