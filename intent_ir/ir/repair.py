"""
IntentIR deterministic repair passes.

These passes are *not* intended to change semantics; they only repair common
LLM/schema slips so downstream stages (static_validate / diff / backend) can
operate on a well-formed IntentFunction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

from .ir_types import Dim, IntentFunction, Op, TensorLayout, TensorType


def _strip_trailing_ones(shape: List[object]) -> List[object]:
    out = list(shape)
    while out:
        d = out[-1]
        if getattr(d, "kind", None) != "const":
            break
        try:
            if int(getattr(d, "value", -1)) != 1:
                break
        except Exception:
            break
        out.pop()
    return out


def _tensor_compatible(a: TensorType, b: TensorType) -> bool:
    # Treat i1/bool as compatible.
    da = "bool" if str(a.dtype) == "i1" else str(a.dtype)
    db = "bool" if str(b.dtype) == "i1" else str(b.dtype)
    if da != db:
        return False
    la = a.layout.name if hasattr(a.layout, "name") else str(a.layout)
    lb = b.layout.name if hasattr(b.layout, "name") else str(b.layout)
    if str(la) != str(lb):
        return False

    sa = _strip_trailing_ones(list(a.shape))
    sb = _strip_trailing_ones(list(b.shape))
    if len(sa) != len(sb):
        return False
    for da_, db_ in zip(sa, sb):
        ka = getattr(da_, "kind", None)
        kb = getattr(db_, "kind", None)
        va = getattr(da_, "value", None)
        vb = getattr(db_, "value", None)
        if ka == "const" and kb == "const" and str(va) != str(vb):
            return False
        # sym matches anything (specialization tolerant)
    return True


def _candidate_output_aliases(name: str) -> List[str]:
    """
    Heuristic aliases for missing outputs.

    Examples:
      out_ptr -> out
      out_mean_ptr -> mean
      out_rstd_ptr -> rstd
    """
    raw = str(name)
    s = raw.strip()
    cands: List[str] = []

    def add(x: str) -> None:
        x = str(x)
        if x and x not in cands:
            cands.append(x)

    add(s)
    lower = s.lower()

    # Strip common pointer suffixes.
    for suf in ("_ptr", "ptr"):
        if lower.endswith(suf):
            add(s[: -len(suf)])

    # Strip leading out_/output_ conventions.
    if lower.startswith("out_"):
        add(s[4:])
        base = s[4:]
        base_l = base.lower()
        if base_l.endswith("_ptr"):
            add(base[:-4])
        if base_l.endswith("ptr"):
            add(base[:-3])

    # Norm-specific shorthands.
    if "mean" in lower:
        add("mean")
        add("mean_val")
        add("mean_computed")
    if "rstd" in lower:
        add("rstd")
        add("rstd_val")
        add("rstd_computed")
    if lower in {"out", "output"} or "out" in lower:
        add("out")
        add("output")
        add("y")

    return cands


def repair_missing_outputs(intent: IntentFunction) -> List[str]:
    """
    Repair common LLM slips where declared outputs are not produced by any op.

    Strategy:
      - If an output tensor `out_ptr` is missing, and a produced value `out`
        exists with a compatible tensor type, insert `identity(out) -> out_ptr`.
      - Only perform name-based repairs; do not guess among multiple candidates.
    """
    if not intent.outputs:
        return []
    produced: Set[str] = {op.output for op in intent.ops if op.output}
    actions: List[str] = []

    # Precompute candidate -> type for fast checks.
    types: Dict[str, TensorType] = {k: v for k, v in intent.tensors.items()}

    for out in list(intent.outputs):
        if out in produced:
            continue
        out_tt = types.get(out)
        if out_tt is None:
            continue
        chosen = None
        for cand in _candidate_output_aliases(out):
            if cand not in produced:
                continue
            cand_tt = types.get(cand)
            if cand_tt is None:
                # Many LLM outputs only declare interface tensors; allow a small
                # set of extremely common aliases without relying on types.
                if cand in {"out", "mean", "rstd", "output", "y"}:
                    chosen = cand
                    break
                continue
            if _tensor_compatible(out_tt, cand_tt):
                chosen = cand
                break
        if chosen is None:
            continue
        intent.ops.append(Op(op="identity", inputs=[chosen], output=out, attrs={}))
        produced.add(out)
        actions.append(f"repair_output:{out}<-{chosen}")

    return actions


def _clone_dim(d: Dim) -> Dim:
    return Dim(kind=d.kind, value=d.value)


def _clone_shape(shape: Sequence[Dim]) -> List[Dim]:
    return [_clone_dim(d) for d in shape]


def _clone_layout(layout: TensorLayout) -> TensorLayout:
    return TensorLayout(kind=layout.kind, params=dict(layout.params))


def _dtype_alias(dtype: str) -> str:
    d = str(dtype).strip().lower()
    aliases = {
        "float16": "f16",
        "fp16": "f16",
        "float32": "f32",
        "fp32": "f32",
        "float": "f32",
        "float64": "f64",
        "fp64": "f64",
        "bfloat16": "bf16",
        "int8": "i8",
        "uint8": "u8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "int1": "i1",
        "boolean": "bool",
    }
    return aliases.get(d, d)


def _is_supported_dtype(dtype: str) -> bool:
    return _dtype_alias(dtype) in {"f16", "bf16", "f32", "f64", "i8", "u8", "i16", "i1", "i32", "i64", "bool"}


def _dim_const(v: int) -> Dim:
    return Dim(kind="const", value=int(v))


def _dim_from_raw(v: Any) -> Dim:
    if isinstance(v, int):
        return _dim_const(v)
    return Dim(kind="sym", value=str(v))


def _is_one(d: Dim) -> bool:
    return d.kind == "const" and isinstance(d.value, int) and int(d.value) == 1


def _dim_equal(a: Dim, b: Dim) -> bool:
    if a.kind != b.kind:
        return False
    return str(a.value) == str(b.value)


def _merge_broadcast_dim(a: Dim, b: Dim) -> Dim:
    if _dim_equal(a, b):
        return _clone_dim(a)
    if _is_one(a):
        return _clone_dim(b)
    if _is_one(b):
        return _clone_dim(a)
    if a.kind == "const" and b.kind == "const":
        # Invalid broadcast at runtime, but keep deterministic shape for diagnostics.
        return _clone_dim(a)
    if a.kind == "sym" and b.kind == "const":
        return _clone_dim(a)
    if a.kind == "const" and b.kind == "sym":
        return _clone_dim(b)
    return _clone_dim(a)


def _broadcast_shape(shapes: Sequence[Sequence[Dim]]) -> List[Dim]:
    acc: List[Dim] = []
    for shape in shapes:
        cur = list(shape)
        if not acc:
            acc = _clone_shape(cur)
            continue
        max_rank = max(len(acc), len(cur))
        lhs = [_dim_const(1)] * (max_rank - len(acc)) + _clone_shape(acc)
        rhs = [_dim_const(1)] * (max_rank - len(cur)) + _clone_shape(cur)
        merged: List[Dim] = []
        for a, b in zip(lhs, rhs):
            merged.append(_merge_broadcast_dim(a, b))
        acc = merged
    return acc


def _normalize_axes(attrs: Dict[str, Any], rank: int) -> List[int]:
    raw = attrs.get("dims")
    if raw is None:
        raw = attrs.get("axis")
    if raw is None:
        raw = attrs.get("axes")
    if raw is None:
        return []
    if isinstance(raw, int):
        axes = [int(raw)]
    elif isinstance(raw, list):
        axes = [int(x) for x in raw if isinstance(x, int)]
    else:
        return []
    norm: List[int] = []
    for ax in axes:
        aa = int(ax)
        if aa < 0:
            aa = rank + aa
        if 0 <= aa < rank and aa not in norm:
            norm.append(aa)
    return norm


def _infer_dtype(op: Op, tensors: Dict[str, TensorType]) -> str:
    opname = str(op.op)
    bool_ops = {
        "eq",
        "ne",
        "gt",
        "ge",
        "lt",
        "le",
        "not",
        "and",
        "or",
        "xor",
        "logical_and",
        "logical_or",
        "logical_xor",
        "logical_not",
        "isnan",
        "isinf",
        "isfinite",
        "reduce_any",
        "reduce_all",
    }
    if opname in bool_ops:
        return "bool"
    if opname in {"argmax", "argmin"}:
        return "i32"
    if opname == "cast":
        to_dt = op.attrs.get("to")
        if isinstance(to_dt, str) and _is_supported_dtype(to_dt):
            return _dtype_alias(to_dt)
    if opname == "iota":
        dt = op.attrs.get("dtype")
        if isinstance(dt, str) and _is_supported_dtype(dt):
            return _dtype_alias(dt)
        return "i32"
    if opname == "const":
        dt = op.attrs.get("dtype")
        if isinstance(dt, str) and _is_supported_dtype(dt):
            return _dtype_alias(dt)
        val = op.attrs.get("value")
        if isinstance(val, bool):
            return "bool"
        if isinstance(val, int):
            return "i32"
        if isinstance(val, float):
            return "f32"
        return "f32"
    if opname == "where" and len(op.inputs) >= 2:
        tt = tensors.get(str(op.inputs[1])) or tensors.get(str(op.inputs[-1]))
        if tt is not None:
            return str(tt.dtype)
    for inp in op.inputs:
        tt = tensors.get(str(inp))
        if tt is not None:
            return str(tt.dtype)
    return "f32"


def _infer_shape(op: Op, tensors: Dict[str, TensorType]) -> List[Dim]:
    opname = str(op.op)
    input_tensors = [tensors.get(str(inp)) for inp in op.inputs if isinstance(inp, str)]
    input_tensors = [t for t in input_tensors if t is not None]
    if opname == "const":
        return []
    if opname == "iota":
        shp = op.attrs.get("shape")
        if isinstance(shp, list):
            return [_dim_from_raw(x) for x in shp]
        return []
    if opname in {"reshape", "view"}:
        shp = op.attrs.get("shape") or op.attrs.get("out_shape")
        if isinstance(shp, list):
            return [_dim_from_raw(x) for x in shp]
    if opname == "broadcast_in_dim":
        shp = op.attrs.get("out_shape")
        if isinstance(shp, list):
            return [_dim_from_raw(x) for x in shp]
    if opname == "transpose" and input_tensors:
        base = list(input_tensors[0].shape)
        perm = op.attrs.get("perm")
        if isinstance(perm, list) and len(perm) == len(base) and all(isinstance(x, int) for x in perm):
            out: List[Dim] = []
            for idx in perm:
                if 0 <= int(idx) < len(base):
                    out.append(_clone_dim(base[int(idx)]))
            if len(out) == len(base):
                return out
        return _clone_shape(base)
    if opname in {"reduce_sum", "reduce_prod", "reduce_max", "reduce_min", "reduce_any", "reduce_all", "mean", "var", "std", "argmax", "argmin"}:
        if input_tensors:
            base = list(input_tensors[0].shape)
            axes = _normalize_axes(op.attrs, len(base))
            if not axes:
                return _clone_shape(base)
            keepdims = bool(op.attrs.get("keepdims", op.attrs.get("keep_dims", False)))
            if keepdims:
                out = _clone_shape(base)
                for ax in axes:
                    out[ax] = _dim_const(1)
                return out
            return [_clone_dim(d) for i, d in enumerate(base) if i not in set(axes)]
    if opname == "stack" and input_tensors:
        base = _clone_shape(input_tensors[0].shape)
        axis_raw = op.attrs.get("axis", 0)
        axis = int(axis_raw) if isinstance(axis_raw, int) else 0
        if axis < 0:
            axis += len(base) + 1
        axis = max(0, min(axis, len(base)))
        base.insert(axis, _dim_const(len(op.inputs)))
        return base
    if opname == "concat" and input_tensors:
        base = _clone_shape(input_tensors[0].shape)
        axis_raw = op.attrs.get("axis", 0)
        axis = int(axis_raw) if isinstance(axis_raw, int) else 0
        if axis < 0:
            axis += len(base)
        if 0 <= axis < len(base):
            total_const = 0
            sym_terms: List[str] = []
            for tt in input_tensors:
                if axis >= len(tt.shape):
                    continue
                dim = tt.shape[axis]
                if dim.kind == "const" and isinstance(dim.value, int):
                    total_const += int(dim.value)
                else:
                    sym_terms.append(str(dim.value))
            if sym_terms:
                if total_const:
                    sym_terms.append(str(total_const))
                base[axis] = Dim(kind="sym", value="+".join(sym_terms))
            else:
                base[axis] = _dim_const(total_const)
        return base
    if opname == "pad" and input_tensors:
        base = _clone_shape(input_tensors[0].shape)
        pad_width = op.attrs.get("pad_width")
        pairs = None
        if isinstance(pad_width, dict):
            pairs = pad_width.get("pairs")
        elif isinstance(pad_width, list):
            pairs = pad_width
        if isinstance(pairs, list):
            for idx, p in enumerate(pairs):
                if idx >= len(base) or not (isinstance(p, list) and len(p) == 2):
                    continue
                if not (isinstance(p[0], int) and isinstance(p[1], int)):
                    continue
                d = base[idx]
                if d.kind == "const" and isinstance(d.value, int):
                    base[idx] = _dim_const(int(d.value) + int(p[0]) + int(p[1]))
        return base

    # Generic elementwise shape inference.
    if input_tensors:
        if len(input_tensors) == 1:
            return _clone_shape(input_tensors[0].shape)
        return _broadcast_shape([tt.shape for tt in input_tensors])
    return []


def materialize_missing_op_output_tensors(intent: IntentFunction) -> List[str]:
    """
    Materialize missing tensor declarations for intermediate op outputs.

    Some extraction paths keep interface tensors only (inputs/final outputs) and
    omit SSA intermediates. Backends (especially compiler lowerings) often rely
    on tensor metadata for every op output. This pass fills missing entries with
    deterministic, conservative shape/dtype inference.
    """
    actions: List[str] = []
    tensors = intent.tensors
    for idx, op in enumerate(intent.ops):
        out = str(op.output or "")
        if not out or out in tensors:
            continue
        dtype = _infer_dtype(op, tensors)
        if not _is_supported_dtype(dtype):
            dtype = "f32"
        shape = _infer_shape(op, tensors)
        layout = TensorLayout(kind="row_major", params={})
        for inp in op.inputs:
            inp_tt = tensors.get(str(inp))
            if inp_tt is not None:
                layout = _clone_layout(inp_tt.layout)
                break
        tensors[out] = TensorType(dtype=_dtype_alias(dtype), shape=shape, layout=layout)
        actions.append(f"materialize_tensor:op[{idx}] {op.op}->{out}")
    return actions


__all__ = ["repair_missing_outputs", "materialize_missing_op_output_tensors"]
