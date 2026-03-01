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
    raise RuntimeError(f"rvv cpu-loops v1 supports only f16/f32/bool/u8/i8, got dtype={dt!r}")


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
            if _dtype(getattr(cond_tt, "dtype", "bool")) != "i8":
                raise RuntimeError(
                    f"rvv cpu-loops v1 where expects bool/u8 mask, got tensor={cond} dtype={getattr(cond_tt, 'dtype', '')}"
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
            lines.append(f"    {p} = arith.cmpi ne, {m}, %c0i8 : i8")
            v = f"%{_mlir_ident(out)}_t"
            lines.append(f"    {v} = arith.select {p}, {x0}, {x1} : f32")
            computed[out] = v
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

    out_name = str((list(intent.outputs or []) or [""])[0]).strip()
    out_tt = (intent.tensors or {}).get(out_name) if out_name else None
    out_rank = len(list(getattr(out_tt, "shape", []) or [])) if out_tt is not None else 0
    ops = [op for op in list(intent.ops or []) if op is not None]
    if len(ops) == 1:
        op0 = ops[0]
        op_name = str(getattr(op0, "op", "")).strip()
        attrs = dict(getattr(op0, "attrs", {}) or {})
        perm = list(attrs.get("perm") or []) if isinstance(attrs.get("perm"), list) else []
        if op_name == "transpose" and perm == [1, 0]:
            module_text = _emit_transpose2d_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
            kind = "cpu_loops_transpose2d_v1"
        elif out_rank == 1:
            module_text = _emit_row_reduce_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
            kind = "cpu_loops_row_reduce_v1"
        else:
            module_text = _emit_elementwise_kernel(kernel_name=kernel_name, intent=intent, bindings=bindings)
            kind = "cpu_loops_v1"
    else:
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
