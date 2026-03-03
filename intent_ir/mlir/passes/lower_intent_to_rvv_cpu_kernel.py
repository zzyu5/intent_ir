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
    if s in {"i32"}:
        return "i32"
    if s in {"i64"}:
        return "i64"
    raise RuntimeError(f"rvv cpu-loops v1 supports only f16/f32/bool/u8/i8/i32/i64, got dtype={dt!r}")


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
    lines.append("    %mean = arith.divf %sum, %cNf : f32")
    lines.append(f"    memref.store %mean, {arg_ssa[out_mean]}[%m] : {mean_memref}")
    lines.append("    %sum_sq = scf.for %n2 = %c0 to %cN step %c1 iter_args(%acc = %c0f) -> (f32) {")
    lines.append("      %i2 = arith.addi %base, %n2 : index")
    lines.append(f"      %x2 = memref.load {arg_ssa[inp_name]}[%i2] : {inp_memref}")
    lines.append("      %d = arith.subf %x2, %mean : f32")
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
    lines.append("      %d3 = arith.subf %x3, %mean : f32")
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
    if _dtype(getattr(out_tt, "dtype", "f32")) != "f32":
        raise RuntimeError("rvv cpu-loops v1 supports only f32 outputs")
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
        if name == out_name and elem_ty != "f32":
            raise RuntimeError(f"rvv cpu-loops v1 supports only f32 outputs, got output dtype={elem_ty!r}")
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
    lines.append(f"  func.func @{_mlir_ident(kernel_name)}(")
    lines.append(",\n".join(arg_decls))
    lines.append("  ) {")
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %c1 = arith.constant 1 : index")
    lines.append(f"  %cM = arith.constant {m_dim} : index")
    lines.append(f"  %cN = arith.constant {n_dim} : index")
    lines.append("  %c0f = arith.constant 0.0 : f32")
    lines.append("  %c0i8 = arith.constant 0 : i8")
    lines.append("  %c0i32 = arith.constant 0 : i32")
    lines.extend(scalar_loads)
    lines.append("  scf.for %m = %c0 to %cM step %c1 {")
    lines.append("    %mN = arith.muli %m, %cN : index")
    lines.append("    scf.for %n = %c0 to %cN step %c1 {")
    lines.append("      %i = arith.addi %mN, %n : index")

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
        ssa = f"%{_mlir_ident(name)}_v"
        lines.append(f"    {ssa} = memref.load {arg_ssa[name]}[{idx}] : {memref_ty}")
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
            to = str(attrs.get("to") or attrs.get("dtype") or out_ty).strip().lower()
            if to != out_ty:
                raise RuntimeError(f"rvv cpu-loops v1 cast dtype mismatch: attrs.to={to!r} tensor.dtype={out_ty!r}")
            a = _in(0)
            if in_ty == out_ty:
                computed[out] = a
            elif in_ty == "f16" and out_ty == "f32":
                v = f"%{_mlir_ident(out)}_t"
                lines.append(f"    {v} = arith.extf {a} : f16 to f32")
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

        if name in {"add", "sub", "mul", "div", "max", "min"}:
            if len(ins) != 2 or not out:
                raise RuntimeError(f"invalid binary op: {name} inputs={ins} output={out!r}")
            for nm in [*ins, out]:
                tt = (intent.tensors or {}).get(str(nm))
                if tt is None:
                    raise RuntimeError(f"rvv cpu-loops v1 missing tensor spec: {nm}")
                if _dtype(getattr(tt, "dtype", "f32")) != "f32":
                    raise RuntimeError(
                        f"rvv cpu-loops v1 supports only f32 for op {name}, got tensor={nm} dtype={getattr(tt, 'dtype', '')}"
                    )
            a = _in(0)
            b = _in(1)
            v = f"%{_mlir_ident(out)}_t"
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
            else:
                lines.append(f"    {v} = arith.minimumf {a}, {b} : f32")
            computed[out] = v
            continue

        if name in {"relu", "abs", "sqrt", "rsqrt", "exp", "exp2", "neg", "floor", "ceil", "log", "sin", "cos", "tan"}:
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
            elif name in {"sin", "cos", "tan"}:
                mop = {"sin": "math.sin", "cos": "math.cos", "tan": "math.tan"}[name]
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
    out_memref_ty = f"memref<{total}xf32>"
    lines.append(f"    memref.store {final}, {arg_ssa[out_name]}[%i] : {out_memref_ty}")
    lines.append("    }")
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
    elif kernel_name == "layer_norm_persistent":
        module_text = _emit_layer_norm_persistent_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "layer_norm_rvv_v1"
    elif kernel_name == "group_norm_kernel":
        module_text = _emit_group_norm_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
        kind = "group_norm_rvv_v1"
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
