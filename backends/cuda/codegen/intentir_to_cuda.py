"""
IntentIR -> CUDA kernel codegen (MVP).

Scope (initial):
- AI-Bench8 kernels used in paper experiments:
  - matmul, dropout, softmax, layernorm, correlation, resize, rope, warp

This codegen intentionally focuses on producing *runnable* CUDA for the paper.
It does not try to cover the full IntentIR op-set yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from intent_ir.ir import IntentFunction, ScheduleSketch

from backends.cuda.runtime import CudaLaunch


class CudaLoweringError(RuntimeError):
    pass


@dataclass(frozen=True)
class CudaLoweredKernel:
    kernel_name: str
    cuda_src: str
    io_spec: Dict[str, Any]
    launch: CudaLaunch
    output_names: list[str]
    # Bindings needed by the runtime to materialize shapes/scalars.
    bindings: Dict[str, Any]


def _as_int(v: Any, *, name: str) -> int:
    try:
        return int(v)
    except Exception as e:
        raise CudaLoweringError(f"expected int for {name}, got {v!r}") from e


def _dim_value(d: Any) -> str | int:
    # `Dim` is defined in intent_ir.ir_types, but we keep this helper generic.
    try:
        # Dim(kind=..., value=...)
        v = d.value  # type: ignore[attr-defined]
    except Exception:
        v = d
    if isinstance(v, (int, str)):
        return v
    return str(v)


def _shape_values(intent: IntentFunction, name: str) -> list[str | int]:
    if name not in intent.tensors:
        raise CudaLoweringError(f"unknown tensor in intent.tensors: {name}")
    t = intent.tensors[name]
    return [_dim_value(d) for d in (t.shape or [])]


def _tensor_io_spec(intent: IntentFunction, name: str) -> Dict[str, Any]:
    t = intent.tensors[name]
    return {"dtype": str(t.dtype), "shape": _shape_values(intent, name)}


def _resolve_dim_int(dim: str | int, bindings: Mapping[str, Any], *, name: str) -> int:
    if isinstance(dim, int):
        return int(dim)
    key = str(dim)
    if key in bindings:
        return _as_int(bindings[key], name=name)
    derived = _derive_binding_value(key, bindings)
    if derived is None:
        raise CudaLoweringError(f"missing binding for dim {name} ({key})")
    return int(derived)


def _binding_int(bindings: Mapping[str, Any], key: str) -> int | None:
    if key not in bindings:
        return None
    try:
        return _as_int(bindings[key], name=key)
    except Exception:
        return None


def _derive_binding_value(key: str, bindings: Mapping[str, Any]) -> int | None:
    c_in = _binding_int(bindings, "C_IN")
    groups = _binding_int(bindings, "GROUPS")
    c_per_g = _binding_int(bindings, "C_PER_G")
    c_out = _binding_int(bindings, "C_OUT")
    mult = _binding_int(bindings, "MULT")
    if key == "C_IN_TOTAL":
        if c_in is not None and groups is not None:
            return c_in * groups
        return c_in
    if key == "C_PER_G":
        if c_per_g is not None:
            return c_per_g
        if c_in is not None and groups is not None and groups > 0 and c_in % groups == 0:
            return c_in // groups
        return c_in
    if key == "C_OUT":
        if c_out is not None:
            return c_out
        if c_in is not None and mult is not None:
            return c_in * mult
        if c_per_g is not None and groups is not None:
            return c_per_g * groups
        return None
    if key == "OH":
        h = _binding_int(bindings, "H")
        ph = _binding_int(bindings, "PH")
        kh = _binding_int(bindings, "KH")
        sh = _binding_int(bindings, "SH")
        dh = _binding_int(bindings, "DH")
        if None not in (h, ph, kh, sh, dh) and sh and sh > 0:
            return (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        return None
    if key == "OW":
        w = _binding_int(bindings, "W")
        pw = _binding_int(bindings, "PW")
        kw = _binding_int(bindings, "KW")
        sw = _binding_int(bindings, "SW")
        dw = _binding_int(bindings, "DW")
        if None not in (w, pw, kw, sw, dw) and sw and sw > 0:
            return (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        return None
    if key == "OD":
        d = _binding_int(bindings, "D")
        pd = _binding_int(bindings, "PD")
        kd = _binding_int(bindings, "KD")
        sd = _binding_int(bindings, "SD")
        dd = _binding_int(bindings, "DD")
        if None not in (d, pd, kd, sd, dd) and sd and sd > 0:
            return (d + 2 * pd - dd * (kd - 1) - 1) // sd + 1
        return None
    return None


def _resolve_attr_int(value: Any, bindings: Mapping[str, Any], *, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    key = str(value)
    if key in bindings:
        return _as_int(bindings[key], name=name)
    try:
        return int(key)
    except Exception as e:
        raise CudaLoweringError(f"missing binding for attr {name} ({key})") from e


def _resolve_attr_tuple(
    value: Any,
    bindings: Mapping[str, Any],
    *,
    name: str,
    rank: int,
    default: int | Sequence[Any],
) -> tuple[int, ...]:
    if value is None:
        if isinstance(default, (list, tuple)):
            if len(default) != rank:
                raise CudaLoweringError(f"{name} default must have length {rank}")
            seq = list(default)
        else:
            seq = [default] * rank
        return tuple(_resolve_attr_int(v, bindings, name=f"{name}[{i}]") for i, v in enumerate(seq))
    if isinstance(value, (list, tuple)):
        if len(value) != rank:
            raise CudaLoweringError(f"{name} must have length {rank}")
        return tuple(_resolve_attr_int(v, bindings, name=f"{name}[{i}]") for i, v in enumerate(value))
    v = _resolve_attr_int(value, bindings, name=name)
    return tuple([v] * rank)


def _is_scalar_tensor(intent: IntentFunction, name: str, *, dtype: str | None = None) -> bool:
    t = intent.tensors.get(name)
    if t is None:
        return False
    if t.shape:
        return False
    if dtype is None:
        return True
    return str(t.dtype) == str(dtype)


def _io_spec_from_args(
    intent: IntentFunction,
    *,
    tensor_args: Sequence[str],
    scalar_args: Mapping[str, str],
    arg_names: Sequence[str],
) -> Dict[str, Any]:
    tensors: Dict[str, Any] = {}
    for n in tensor_args:
        if n not in intent.tensors:
            raise CudaLoweringError(f"io_spec tensor arg missing from intent.tensors: {n}")
        tensors[n] = _tensor_io_spec(intent, n)
    return {"arg_names": list(arg_names), "tensors": tensors, "scalars": dict(scalar_args)}


def _resolve_schedule_int(v: str | int | None, bindings: Mapping[str, Any], *, default: int) -> int:
    if v is None:
        return int(default)
    if isinstance(v, int):
        return int(v)
    key = str(v)
    if key in bindings:
        return _as_int(bindings[key], name=f"schedule.{key}")
    # Accept "BLOCK_M" style names without explicit bindings (fallback to default).
    return int(default)


def _c_ident(name: str) -> str:
    """
    Convert an IntentIR tensor/op name into a C identifier.
    """
    out = []
    for ch in str(name):
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    if not s:
        return "v"
    if ("0" <= s[0] <= "9") or s[0] == "_":
        return "v" + s
    return s


def _c_scalar_literal(dtype: str, value: Any) -> str:
    dt = str(dtype)
    if dt in {"i1", "bool"}:
        return "true" if int(value) != 0 else "false"
    if dt == "i32":
        return str(int(value))
    if dt == "i64":
        return str(int(value)) + "LL"
    if dt == "f32":
        v = float(value)
        s = f"{v:.8g}"
        # Ensure this is parsed as a float literal, not a user-defined literal (e.g., "1f").
        if ("e" not in s) and ("E" not in s) and ("." not in s):
            s += ".0"
        return s + "f"
    if dt == "f16":
        # Minimal support; callers should avoid f16 elementwise for now.
        v = float(value)
        s = f"{v:.8g}"
        if ("e" not in s) and ("E" not in s) and ("." not in s):
            s += ".0"
        return f"__float2half({s}f)"
    raise CudaLoweringError(f"unsupported const dtype for CUDA elementwise: {dtype}")


def _c_type(dtype: str) -> str:
    dt = str(dtype)
    if dt == "f32":
        return "float"
    if dt == "i32":
        return "int"
    if dt == "i64":
        return "int64_t"
    if dt == "u8":
        return "uint8_t"
    if dt == "i8":
        return "int8_t"
    if dt == "i16":
        return "int16_t"
    if dt in {"i1", "bool"}:
        return "bool"
    raise CudaLoweringError(f"unsupported dtype for CUDA elementwise: {dtype}")


def _emit_broadcast_index_expr(
    *,
    out_rank: int,
    in_shape: Sequence[str | int],
    out_idxs: Sequence[str],
    dim_expr: Mapping[str, str],
) -> str:
    """
    Emit a row-major linear index expression into an input tensor, using NumPy-style
    right-aligned broadcasting against the output indices.

    Assumes all tensors are contiguous row-major.
    """
    in_rank = len(in_shape)
    if in_rank == 0:
        return "0"
    if in_rank > out_rank:
        raise CudaLoweringError(f"broadcast: in_rank={in_rank} > out_rank={out_rank}")

    shift = out_rank - in_rank

    # Build per-dim "size" expressions for the input.
    in_sizes: list[str] = []
    for d in in_shape:
        if isinstance(d, int):
            in_sizes.append(str(int(d)))
        else:
            in_sizes.append(dim_expr[str(d)])

    # Row-major strides: stride[j] = prod(in_sizes[j+1:]).
    strides: list[str] = []
    for j in range(in_rank):
        tail = in_sizes[j + 1 :]
        if not tail:
            strides.append("1")
        else:
            strides.append(" * ".join(f"(int64_t){x}" for x in tail))

    terms: list[str] = []
    for j in range(in_rank):
        out_k = j + shift
        idx = out_idxs[out_k]
        size_expr = in_sizes[j]
        if size_expr == "1":
            continue
        terms.append(f"((int64_t){idx}) * ({strides[j]})")
    if not terms:
        return "0"
    return " + ".join(terms)


def _emit_broadcast_in_dim_index_expr(
    *,
    final_out_rank: int,
    op_out_shape: Sequence[str | int],
    in_shape: Sequence[str | int],
    out_idxs: Sequence[str],
    dim_expr: Mapping[str, str],
    broadcast_dims: Sequence[int],
) -> str:
    """
    Emit row-major linear index for broadcast_in_dim(input -> op_out_shape).

    Unlike NumPy-style right alignment, broadcast_in_dim maps each input
    dimension explicitly via `broadcast_dims`.
    """
    in_rank = len(in_shape)
    if in_rank == 0:
        return "0"
    if len(broadcast_dims) != in_rank:
        raise CudaLoweringError("broadcast_in_dim expects len(broadcast_dims) == input rank")

    op_rank = len(op_out_shape)
    shift = int(final_out_rank) - int(op_rank)
    if shift < 0:
        raise CudaLoweringError("broadcast_in_dim output rank cannot exceed fused output rank")

    # Build per-dim "size" expressions for the input.
    in_sizes: list[str] = []
    for d in in_shape:
        if isinstance(d, int):
            in_sizes.append(str(int(d)))
        else:
            in_sizes.append(dim_expr[str(d)])

    # Row-major strides.
    strides: list[str] = []
    for j in range(in_rank):
        tail = in_sizes[j + 1 :]
        if not tail:
            strides.append("1")
        else:
            strides.append(" * ".join(f"(int64_t){x}" for x in tail))

    terms: list[str] = []
    for j, out_dim in enumerate(broadcast_dims):
        k = int(out_dim)
        if k < 0:
            k += op_rank
        if k < 0 or k >= op_rank:
            raise CudaLoweringError("broadcast_in_dim broadcast_dims contains out-of-range axis")
        out_k = shift + k
        if out_k < 0 or out_k >= len(out_idxs):
            raise CudaLoweringError("broadcast_in_dim index mapping exceeds fused output rank")
        size_expr = in_sizes[j]
        if size_expr == "1":
            continue
        terms.append(f"((int64_t){out_idxs[out_k]}) * ({strides[j]})")
    if not terms:
        return "0"
    return " + ".join(terms)


def _kernel_fused_elementwise(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    """
    Generic fused elementwise lowering for small IntentIR graphs.

    Supports:
      - unary/binary float ops: add/sub/mul/div/max/min/relu/abs/exp/cos/erf/floor/rsqrt
      - comparisons -> bool: ne/lt/le/gt/ge
      - bool ops: and/or/not
      - where(cond, a, b)
      - const, identity, broadcast_in_dim

    Limitations (for now):
      - single output tensor
      - contiguous row_major tensors
      - rank <= 4
    """
    if not intent.outputs or len(intent.outputs) != 1:
        raise CudaLoweringError("elementwise lowering requires a single output")
    out_name = str(intent.outputs[0])
    if out_name not in intent.tensors:
        raise CudaLoweringError(f"elementwise lowering: missing output tensor {out_name} in intent.tensors")
    out_t = intent.tensors[out_name]
    if out_t.layout.kind != "row_major":
        raise CudaLoweringError("elementwise lowering supports only row_major tensors")
    out_shape = _shape_values(intent, out_name)
    out_rank = len(out_shape)
    if out_rank > 4:
        raise CudaLoweringError("elementwise lowering supports rank<=4")

    # Collect dim symbols used by involved tensor shapes.
    produced = {o.output for o in (intent.ops or [])}
    outs = set(intent.outputs or [])
    used_tensors: set[str] = {out_name}
    for op in intent.ops or []:
        for inp in op.inputs:
            if str(inp) in intent.tensors:
                used_tensors.add(str(inp))
        if str(op.output) in intent.tensors:
            used_tensors.add(str(op.output))

    dim_syms: set[str] = set()
    for tn in used_tensors:
        t = intent.tensors.get(str(tn))
        if not t:
            continue
        for d in (t.shape or []):
            dv = _dim_value(d)
            if isinstance(dv, str):
                dim_syms.add(dv)

    # Resolve output extents for launch + total element count.
    out_dims_expr: list[str] = []
    for i, d in enumerate(out_shape):
        if isinstance(d, int):
            out_dims_expr.append(str(int(d)))
        else:
            out_dims_expr.append(str(d))

    # Bind dim symbols as either scalar tensors (preferred) or scalar args.
    tensor_args: list[str] = []
    scalar_args: Dict[str, str] = {}
    dim_param: list[str] = []
    dim_load: list[str] = []
    dim_expr: Dict[str, str] = {}
    for sym in sorted(dim_syms):
        if _is_scalar_tensor(intent, sym, dtype="i32"):
            tensor_args.append(sym)
            dim_param.append(f"const int* {sym}_ptr")
            dim_load.append(f"const int {sym} = {sym}_ptr ? {sym}_ptr[0] : 0;")
            dim_expr[sym] = sym
        else:
            scalar_args[sym] = "i32"
            dim_param.append(f"int {sym}")
            dim_expr[sym] = sym

    # Identify external input tensors (non-produced, not outputs) used by ops.
    # Keep scalar tensors too: they are common for modeling parameters (e.g., eps).
    external_inputs: list[str] = []
    for op in intent.ops or []:
        for inp in op.inputs:
            n = str(inp)
            if n not in intent.tensors:
                continue
            if n in produced or n in outs:
                continue
            # If a dim symbol is modeled as scalar tensor (i32), prefer passing it
            # via the explicit dim parameter machinery below.
            if _is_scalar_tensor(intent, n, dtype="i32") and n in dim_syms:
                continue
            if n not in external_inputs:
                external_inputs.append(n)

    tensor_args = [*external_inputs, out_name, *tensor_args]

    # Schedule -> block size.
    sched = intent.schedule or ScheduleSketch()
    hinted = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    block_x = max(32, min(1024, int(hinted)))
    if block_x <= 0:
        block_x = 256

    # Emit output index decomposition.
    # We keep indices as int64 for safety.
    idx_vars: list[str] = []
    idx_code: list[str] = []
    if out_rank == 0:
        idx_vars = []
    elif out_rank == 1:
        idx_vars = ["i0"]
        idx_code.append("const int64_t i0 = tid;")
    elif out_rank == 2:
        idx_vars = ["i0", "i1"]
        d1 = out_dims_expr[1] if isinstance(out_shape[1], int) else dim_expr[str(out_shape[1])]
        idx_code.append(f"const int64_t i0 = tid / (int64_t)({d1});")
        idx_code.append(f"const int64_t i1 = tid - i0 * (int64_t)({d1});")
    elif out_rank == 3:
        idx_vars = ["i0", "i1", "i2"]
        d1 = out_dims_expr[1] if isinstance(out_shape[1], int) else dim_expr[str(out_shape[1])]
        d2 = out_dims_expr[2] if isinstance(out_shape[2], int) else dim_expr[str(out_shape[2])]
        idx_code.append(f"const int64_t i0 = tid / ((int64_t)({d1}) * (int64_t)({d2}));")
        idx_code.append(f"const int64_t rem = tid - i0 * ((int64_t)({d1}) * (int64_t)({d2}));")
        idx_code.append(f"const int64_t i1 = rem / (int64_t)({d2});")
        idx_code.append(f"const int64_t i2 = rem - i1 * (int64_t)({d2});")
    else:
        idx_vars = ["i0", "i1", "i2", "i3"]
        d1 = out_dims_expr[1] if isinstance(out_shape[1], int) else dim_expr[str(out_shape[1])]
        d2 = out_dims_expr[2] if isinstance(out_shape[2], int) else dim_expr[str(out_shape[2])]
        d3 = out_dims_expr[3] if isinstance(out_shape[3], int) else dim_expr[str(out_shape[3])]
        idx_code.append(f"const int64_t i0 = tid / ((int64_t)({d1}) * (int64_t)({d2}) * (int64_t)({d3}));")
        idx_code.append(f"const int64_t rem0 = tid - i0 * ((int64_t)({d1}) * (int64_t)({d2}) * (int64_t)({d3}));")
        idx_code.append(f"const int64_t i1 = rem0 / ((int64_t)({d2}) * (int64_t)({d3}));")
        idx_code.append(f"const int64_t rem1 = rem0 - i1 * ((int64_t)({d2}) * (int64_t)({d3}));")
        idx_code.append(f"const int64_t i2 = rem1 / (int64_t)({d3});")
        idx_code.append(f"const int64_t i3 = rem1 - i2 * (int64_t)({d3});")

    # Emit op evaluation (SSA order). We keep intermediate scalars in registers.
    value_expr: Dict[str, str] = {}
    value_type: Dict[str, str] = {}

    def load_tensor(name: str) -> str:
        t = intent.tensors[name]
        if t.layout.kind != "row_major":
            raise CudaLoweringError("elementwise lowering supports only row_major tensors")
        in_shape = _shape_values(intent, name)
        idx_expr = _emit_broadcast_index_expr(out_rank=out_rank, in_shape=in_shape, out_idxs=idx_vars, dim_expr=dim_expr)
        return f"{name}[(size_t)({idx_expr})]"

    def _dtype_of(name: str) -> str:
        n = str(name)
        if n in value_type:
            return str(value_type[n])
        if n in intent.tensors:
            return str(intent.tensors[n].dtype)
        return "f32"

    def _infer_const_dtype(attrs: Mapping[str, Any]) -> str:
        dt = attrs.get("dtype")
        if isinstance(dt, str) and dt.strip():
            return str(dt).strip()
        value = attrs.get("value", 0)
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "i32"
        return "f32"

    def _normalize_dtype_name(raw: str) -> str:
        dt = str(raw).strip()
        if dt in {"i1", "bool"}:
            return "bool"
        return dt

    def _infer_output_dtype(opname: str, outn: str, inputs: Sequence[str], attrs: Mapping[str, Any]) -> str:
        if outn in intent.tensors:
            return str(intent.tensors[outn].dtype)
        if opname == "cast":
            to = attrs.get("to")
            if isinstance(to, str) and to.strip():
                return _normalize_dtype_name(to)
            return _dtype_of(inputs[0]) if inputs else "f32"
        if opname == "iota":
            dt = attrs.get("dtype")
            if isinstance(dt, str) and dt.strip():
                return _normalize_dtype_name(dt)
            return "i32"
        if opname == "const":
            return _normalize_dtype_name(_infer_const_dtype(attrs))
        if opname in {"eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not"}:
            return "bool"
        if opname == "where":
            return _dtype_of(inputs[1]) if len(inputs) > 1 else "f32"
        return _dtype_of(inputs[0]) if inputs else "f32"

    def _output_shape(opname: str, outn: str, attrs: Mapping[str, Any]) -> list[str | int]:
        if outn in intent.tensors:
            return _shape_values(intent, outn)
        if opname == "iota":
            raw = attrs.get("shape")
            if isinstance(raw, list) and raw:
                return [_dim_value(d) for d in raw]
            raise CudaLoweringError("iota temporary output requires attrs.shape when output tensor is implicit")
        # Fallback for temporary SSA values in fused elementwise lowering.
        return list(out_shape)

    def val(name: str) -> str:
        n = str(name)
        if n in value_expr:
            return value_expr[n]
        if n not in intent.tensors:
            raise CudaLoweringError(f"elementwise: unknown value {n}")
        # Base input tensor load.
        value_type[n] = str(intent.tensors[n].dtype)
        return load_tensor(n)

    code_lines: list[str] = []
    for op in intent.ops or []:
        opname = str(op.op)
        outn = str(op.output)
        op_attrs = dict(op.attrs or {})
        out_dt = _infer_output_dtype(opname, outn, list(op.inputs), op_attrs)
        cty = _c_type(out_dt)

        # Helper: declare + assign.
        def emit_assign(expr: str) -> None:
            # Prefix SSA locals to avoid collisions with tensor argument names (e.g., output "Out").
            vname = "v_" + _c_ident(outn)
            code_lines.append(f"{cty} {vname} = {expr};")
            value_expr[outn] = vname
            value_type[outn] = out_dt

        if opname == "const":
            emit_assign(_c_scalar_literal(out_dt, op_attrs.get("value", 0)))
        elif opname == "iota":
            if len(op.inputs) != 0:
                raise CudaLoweringError("iota expects 0 inputs")
            op_shape = _output_shape(opname, outn, op_attrs)
            op_rank = int(len(op_shape))
            if op_rank <= 0:
                raise CudaLoweringError("iota expects rank>=1 output")
            axis = _as_int(op_attrs.get("axis", 0), name="axis")
            if axis < 0:
                axis += op_rank
            if axis < 0 or axis >= op_rank:
                raise CudaLoweringError("iota axis out of range")
            shift = int(out_rank - op_rank)
            if shift < 0:
                raise CudaLoweringError("iota output rank cannot exceed final output rank")
            emit_assign(f"({cty})({idx_vars[shift + axis]})")
        elif opname == "identity":
            if len(op.inputs) != 1:
                raise CudaLoweringError("identity expects 1 input")
            emit_assign(val(op.inputs[0]))
        elif opname == "broadcast_in_dim":
            if len(op.inputs) != 1:
                raise CudaLoweringError("broadcast_in_dim expects 1 input")
            src = str(op.inputs[0])
            raw_dims = op_attrs.get("broadcast_dims")
            if src in intent.tensors and isinstance(raw_dims, (list, tuple)):
                out_shape_op = _output_shape(opname, outn, op_attrs)
                in_shape = _shape_values(intent, src)
                idx_expr = _emit_broadcast_in_dim_index_expr(
                    final_out_rank=out_rank,
                    op_out_shape=out_shape_op,
                    in_shape=in_shape,
                    out_idxs=idx_vars,
                    dim_expr=dim_expr,
                    broadcast_dims=[int(x) for x in raw_dims],
                )
                emit_assign(f"{src}[(size_t)({idx_expr})]")
            else:
                emit_assign(val(src))
        elif opname in {"add", "sub", "mul", "div", "max", "min"}:
            if len(op.inputs) != 2:
                raise CudaLoweringError(f"{opname} expects 2 inputs")
            a = val(op.inputs[0])
            b = val(op.inputs[1])
            if opname == "add":
                emit_assign(f"({a} + {b})")
            elif opname == "sub":
                emit_assign(f"({a} - {b})")
            elif opname == "mul":
                emit_assign(f"({a} * {b})")
            elif opname == "div":
                emit_assign(f"({a} / {b})")
            elif opname == "max":
                emit_assign(f"fmaxf({a}, {b})")
            else:
                emit_assign(f"fminf({a}, {b})")
        elif opname == "relu":
            if len(op.inputs) != 1:
                raise CudaLoweringError("relu expects 1 input")
            x = val(op.inputs[0])
            emit_assign(f"fmaxf({x}, 0.0f)")
        elif opname == "abs":
            if len(op.inputs) != 1:
                raise CudaLoweringError("abs expects 1 input")
            x = val(op.inputs[0])
            emit_assign(f"fabsf({x})")
        elif opname == "exp":
            if len(op.inputs) != 1:
                raise CudaLoweringError("exp expects 1 input")
            x = val(op.inputs[0])
            base_attr = op_attrs.get("base")
            base_val: float | None = None
            if isinstance(base_attr, (int, float)):
                try:
                    base_val = float(base_attr)
                except Exception:
                    base_val = None
            if base_val is not None and abs(base_val - 2.0) <= 1e-6:
                emit_assign(f"exp2f({x})")
            elif base_val is not None:
                emit_assign(f"powf({base_val:.9g}f, {x})")
            else:
                emit_assign(f"__expf({x})")
        elif opname == "cos":
            if len(op.inputs) != 1:
                raise CudaLoweringError("cos expects 1 input")
            x = val(op.inputs[0])
            emit_assign(f"cosf({x})")
        elif opname == "acos":
            if len(op.inputs) != 1:
                raise CudaLoweringError("acos expects 1 input")
            x = val(op.inputs[0])
            emit_assign(f"acosf({x})")
        elif opname == "atan":
            if len(op.inputs) != 1:
                raise CudaLoweringError("atan expects 1 input")
            x = val(op.inputs[0])
            emit_assign(f"atanf({x})")
        elif opname == "erf":
            if len(op.inputs) != 1:
                raise CudaLoweringError("erf expects 1 input")
            x = val(op.inputs[0])
            emit_assign(f"erff({x})")
        elif opname == "floor":
            if len(op.inputs) != 1:
                raise CudaLoweringError("floor expects 1 input")
            x = val(op.inputs[0])
            emit_assign(f"floorf({x})")
        elif opname == "ceil":
            if len(op.inputs) != 1:
                raise CudaLoweringError("ceil expects 1 input")
            x = val(op.inputs[0])
            emit_assign(f"ceilf({x})")
        elif opname == "rsqrt":
            if len(op.inputs) != 1:
                raise CudaLoweringError("rsqrt expects 1 input")
            x = val(op.inputs[0])
            emit_assign(f"rsqrtf({x})")
        elif opname in {"eq", "ne", "lt", "le", "gt", "ge"}:
            if len(op.inputs) != 2:
                raise CudaLoweringError(f"{opname} expects 2 inputs")
            a = val(op.inputs[0])
            b = val(op.inputs[1])
            op_map = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
            emit_assign(f"({a} {op_map[opname]} {b})")
        elif opname == "bitwise_and":
            if len(op.inputs) != 2:
                raise CudaLoweringError("bitwise_and expects 2 inputs")
            a = val(op.inputs[0])
            b = val(op.inputs[1])
            emit_assign(f"(({a}) & ({b}))")
        elif opname == "bitwise_or":
            if len(op.inputs) != 2:
                raise CudaLoweringError("bitwise_or expects 2 inputs")
            a = val(op.inputs[0])
            b = val(op.inputs[1])
            emit_assign(f"(({a}) | ({b}))")
        elif opname == "bitwise_left_shift":
            if len(op.inputs) != 2:
                raise CudaLoweringError("bitwise_left_shift expects 2 inputs")
            a = val(op.inputs[0])
            b = val(op.inputs[1])
            emit_assign(f"(({a}) << (({b}) & 31))")
        elif opname == "bitwise_right_shift":
            if len(op.inputs) != 2:
                raise CudaLoweringError("bitwise_right_shift expects 2 inputs")
            a = val(op.inputs[0])
            b = val(op.inputs[1])
            emit_assign(f"(({a}) >> (({b}) & 31))")
        elif opname in {"and", "or"}:
            if len(op.inputs) != 2:
                raise CudaLoweringError(f"{opname} expects 2 inputs")
            a = val(op.inputs[0])
            b = val(op.inputs[1])
            out_dtype = str(out_dt)
            if out_dtype in {"bool", "i1"}:
                op_map = {"and": "&&", "or": "||"}
                emit_assign(f"({a} {op_map[opname]} {b})")
            else:
                op_map = {"and": "&", "or": "|"}
                emit_assign(f"(({a}) {op_map[opname]} ({b}))")
        elif opname == "not":
            if len(op.inputs) != 1:
                raise CudaLoweringError("not expects 1 input")
            a = val(op.inputs[0])
            out_dtype = str(out_dt)
            if out_dtype in {"bool", "i1"}:
                emit_assign(f"(!{a})")
            else:
                emit_assign(f"(~({a}))")
        elif opname == "bitwise_not":
            if len(op.inputs) != 1:
                raise CudaLoweringError("bitwise_not expects 1 input")
            a = val(op.inputs[0])
            emit_assign(f"(~({a}))")
        elif opname == "where":
            if len(op.inputs) != 3:
                raise CudaLoweringError("where expects 3 inputs (cond, x, y)")
            cond = val(op.inputs[0])
            a = val(op.inputs[1])
            b = val(op.inputs[2])
            emit_assign(f"({cond} ? {a} : {b})")
        elif opname == "cast":
            if len(op.inputs) != 1:
                raise CudaLoweringError("cast expects 1 input")
            x = val(op.inputs[0])
            emit_assign(f"({cty})({x})")
        else:
            raise CudaLoweringError(f"elementwise lowering unsupported op: {opname}")

    if out_name not in value_expr:
        raise CudaLoweringError("elementwise lowering did not produce the output value")
    out_var = value_expr[out_name]
    out_cty = _c_type(str(out_t.dtype))

    # Total element count expression.
    if out_rank == 0:
        total_expr = "1"
    else:
        parts = []
        for d in out_shape:
            if isinstance(d, int):
                parts.append(f"(int64_t){int(d)}")
            else:
                parts.append(f"(int64_t){dim_expr[str(d)]}")
        total_expr = " * ".join(parts) if parts else "1"

    cuda_src = f"""
#include <math.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(
    {", ".join([f"const {_c_type(intent.tensors[n].dtype)}* __restrict__ {n}" for n in external_inputs])}{"," if external_inputs else ""}
    {_c_type(out_t.dtype)}* __restrict__ {out_name}{"," if dim_param else ""}
    {", ".join(dim_param)}) {{
  {" ".join(dim_load)}
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  const int64_t total = {total_expr};
  if (tid >= total) return;
  {" ".join(idx_code)}
  {" ".join(code_lines)}
  {out_name}[(size_t)tid] = ({out_cty}){out_var};
}}
""".lstrip()

    # Build io_spec for runtime.
    arg_names = [*external_inputs, out_name, *sorted(dim_syms)]
    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    total = 1
    for d in out_shape:
        total *= _resolve_dim_int(d, bindings, name=f"out_dim")
    grid_x = (int(total) + block_x - 1) // block_x
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(
        kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings)
    )


def _kernel_matmul_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or intent.ops[0].op != "matmul":
        raise CudaLoweringError("matmul lowering expects a single matmul op")
    op = intent.ops[0]
    if len(op.inputs) != 2:
        raise CudaLoweringError("matmul expects 2 inputs")
    a, b = op.inputs
    c = op.output
    # Derive dims from tensor shapes to support both:
    # - scalar-tensor dims (AI-Bench)
    # - pure symbolic dims (unit tests)
    a_shape = _shape_values(intent, a)
    b_shape = _shape_values(intent, b)
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise CudaLoweringError("matmul expects rank-2 inputs")
    M_dim, K_dim = a_shape
    K2_dim, N_dim = b_shape
    if str(K_dim) != str(K2_dim):
        raise CudaLoweringError(f"matmul K mismatch: A is {K_dim} but B is {K2_dim}")
    M = _resolve_dim_int(M_dim, bindings, name="M")
    N = _resolve_dim_int(N_dim, bindings, name="N")
    K = _resolve_dim_int(K_dim, bindings, name="K")

    sched = intent.schedule or ScheduleSketch()
    block_y = _resolve_schedule_int(sched.tile_m, bindings, default=16)  # rows
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=16)  # cols
    block_k = _resolve_schedule_int(sched.tile_k, bindings, default=16)  # reduction tile
    # Clamp to sane defaults.
    if block_x <= 0:
        block_x = 16
    if block_y <= 0:
        block_y = 16
    if block_k <= 0:
        block_k = 16
    # Keep X reasonably small (threads along X). If schedule asks for huge tiles,
    # prefer more blocks than a mega-block (better occupancy for small GEMMs).
    if block_x > 64:
        block_x = 64
    # Prefer WMMA TF32 on modern GPUs when explicitly enabled and shapes are friendly.
    # NOTE: TF32 changes numerical semantics vs full f32, so it must be opt-in.
    allow_tf32 = bool(int(bindings.get("ALLOW_TF32", 0))) or bool(op.attrs.get("allow_tf32", False))
    use_wmma = allow_tf32 and (M % 16 == 0) and (N % 16 == 0) and (K % 8 == 0)

    # Use a fixed thread-y tile and let each thread compute multiple rows
    # for the fallback (non-WMMA) kernel.
    thread_m = min(16, block_y)
    if block_x * thread_m > 1024:
        thread_m = max(1, 1024 // max(1, block_x))
    # Ensure BLOCK_K fits both X and the thread-y loader.
    block_k = max(1, min(block_k, block_x, thread_m))
    rows_per_thread = max(1, (block_y + thread_m - 1) // thread_m)

    grid_x = (N + block_x - 1) // block_x
    grid_y = (M + block_y - 1) // block_y

    # Respect scalar-tensor dims if present in the IntentIR signature.
    m_is_tensor = _is_scalar_tensor(intent, str(M_dim), dtype="i32")
    n_is_tensor = _is_scalar_tensor(intent, str(N_dim), dtype="i32")
    k_is_tensor = _is_scalar_tensor(intent, str(K_dim), dtype="i32")
    # Optional specialization: treat resolved dims as compile-time constants.
    # This is useful for performance experiments (fixed shapes) because it:
    #   - removes scalar-tensor loads,
    #   - enables more constant propagation/unrolling in the CUDA compiler.
    specialize_dims = bool(int(bindings.get("CUDA_SPECIALIZE_DIMS", 0) or 0))

    m_param = f"const int* {str(M_dim)}_ptr" if m_is_tensor else "int M_in"
    n_param = f"const int* {str(N_dim)}_ptr" if n_is_tensor else "int N_in"
    k_param = f"const int* {str(K_dim)}_ptr" if k_is_tensor else "int K_in"

    m_init = f"({str(M_dim)}_ptr ? {str(M_dim)}_ptr[0] : 0)" if m_is_tensor else "M_in"
    n_init = f"({str(N_dim)}_ptr ? {str(N_dim)}_ptr[0] : 0)" if n_is_tensor else "N_in"
    k_init = f"({str(K_dim)}_ptr ? {str(K_dim)}_ptr[0] : 0)" if k_is_tensor else "K_in"
    if specialize_dims:
        # Keep the same kernel signature for the runtime, but use compile-time
        # constants in the kernel body. The unused parameters are intentionally
        # ignored (nvcc may warn, but compilation succeeds).
        m_unused = f"{str(M_dim)}_ptr" if m_is_tensor else "M_in"
        n_unused = f"{str(N_dim)}_ptr" if n_is_tensor else "N_in"
        k_unused = f"{str(K_dim)}_ptr" if k_is_tensor else "K_in"
        mnk_load = f"""
  (void){m_unused};
  (void){n_unused};
  (void){k_unused};
  constexpr int M = {int(M)};
  constexpr int N = {int(N)};
  constexpr int K = {int(K)};
""".rstrip()
    else:
        # Load scalar-tensor dims once per CTA (avoids redundant global loads).
        # For our GEMM kernels this also helps keep fast-path predicates consistent.
        if m_is_tensor or n_is_tensor or k_is_tensor:
            mnk_load = f"""
  __shared__ int intentir_M;
  __shared__ int intentir_N;
  __shared__ int intentir_K;
  if ((int)threadIdx.x == 0 && (int)threadIdx.y == 0 && (int)threadIdx.z == 0) {{
    intentir_M = {m_init};
    intentir_N = {n_init};
    intentir_K = {k_init};
  }}
  __syncthreads();
  const int M = intentir_M;
  const int N = intentir_N;
  const int K = intentir_K;
""".rstrip()
        else:
            mnk_load = """
  const int M = M_in;
  const int N = N_in;
  const int K = K_in;
""".rstrip()

    m_load = mnk_load
    n_load = ""
    k_load = ""

    if use_wmma:
        # WMMA tile shape (WARPS_M x WARPS_N warps), each warp computes a 16x16 block.
        #
        # Prefer using the schedule tile sizes when they align with WMMA (multiples of 16),
        # since those tiles are already tuned for the reference Triton kernel in our suite.
        # Keep overrides for manual tuning.
        wmma_warps_m = int(bindings.get("WMMA_WARPS_M", 0) or 0)
        wmma_warps_n = int(bindings.get("WMMA_WARPS_N", 0) or 0)
        warps_m_override = wmma_warps_m > 0
        warps_n_override = wmma_warps_n > 0
        wmma_frag_m = int(bindings.get("WMMA_FRAG_M", 1) or 0)
        wmma_frag_n = int(bindings.get("WMMA_FRAG_N", 1) or 0)
        if wmma_warps_m <= 0 or wmma_warps_n <= 0:
            wmma_warps_m = max(1, min(4, int(block_y) // 16))
            wmma_warps_n = max(1, min(8, int(block_x) // 16))
            # Triton schedules in our suite often use small BLOCK_N (e.g., 16) for tl.dot.
            # In CUDA, widening N tiles reduces launch overhead and improves B reuse within a block.
            if wmma_warps_n <= 1:
                # Heuristic default for N-tiling:
                # - For N>=256, 4 warps in N (TILE_N=64) is a good baseline on modern GPUs.
                # - Wider tiles (e.g., 8 warps, TILE_N=128) can reduce redundancy but may
                #   underutilize the GPU for our medium-sized GEMMs.
                if N >= 256:
                    wmma_warps_n = 4
                elif N >= 64:
                    wmma_warps_n = 2
            # If schedule doesn't express WMMA-friendly tiles, fall back to a simple heuristic.
            if (int(block_y) % 16) != 0:
                wmma_warps_m = 4 if M >= 64 else 2
            if (int(block_x) % 16) != 0:
                wmma_warps_n = 2 if N >= 32 else 1
        wmma_warps_m = max(1, min(4, wmma_warps_m))
        wmma_warps_n = max(1, min(8, wmma_warps_n))

        # Small-GEMM heuristic: for small matrices, too-large tiles can produce too few blocks
        # (especially on large GPUs), under-utilizing the device. Prefer smaller tiles unless
        # the user explicitly overrides the WMMA warp shape.
        if not warps_m_override and M <= 256:
            # For the medium GEMMs in our benchmark suite (e.g., 256x512x256),
            # using only 1 warp in M (TILE_M=16) increases B load redundancy and
            # tends to lose more than it gains from higher CTA counts.
            wmma_warps_m = max(wmma_warps_m, 2)

        # Optional register tiling in M: each warp computes FRAG_M adjacent 16x16 output fragments.
        # This can reduce warps/block (sync/launch overhead) but increases register pressure; keep it opt-in.
        if wmma_frag_m <= 0:
            wmma_frag_m = 1
        if wmma_frag_m not in (1, 2):
            wmma_frag_m = 1
        if (wmma_warps_m % wmma_frag_m) != 0:
            wmma_frag_m = 1
        wmma_warps_m = max(1, wmma_warps_m // wmma_frag_m)

        # Optional register tiling in N: each warp computes FRAG_N adjacent 16x16 output fragments.
        # This can reduce warps/block (launch overhead) but increases register pressure; keep it opt-in.
        if wmma_frag_n <= 0:
            wmma_frag_n = 1
        if wmma_frag_n not in (1, 2):
            wmma_frag_n = 1
        if (wmma_warps_n % wmma_frag_n) != 0:
            wmma_frag_n = 1
        wmma_warps_n = max(1, wmma_warps_n // wmma_frag_n)

        while (wmma_warps_m * wmma_warps_n) > 16:
            if wmma_warps_n > 1:
                wmma_warps_n -= 1
            elif wmma_warps_m > 1:
                wmma_warps_m -= 1
            else:
                break
        # Small-GEMM occupancy heuristic: very tall tiles can leave too few blocks
        # for the AI-Bench matmul shape (M=256, N=512), under-utilizing the GPU.
        if not warps_m_override and M <= 256 and wmma_warps_m > 2:
            wmma_warps_m = 2

        wmma_stage_k = int(bindings.get("WMMA_STAGE_K", 0) or 0)
        if wmma_stage_k <= 0:
            # Default K-stage for WMMA:
            # - Use 64 for larger reductions (reduces pipeline/sync overhead).
            # - Fall back to 32 for smaller K or when shapes are tiny.
            wmma_stage_k = 64 if K >= 256 else 32
        if wmma_stage_k not in (8, 16, 32, 64, 128):
            wmma_stage_k = 32
        # cp.async path uses float4 copies; keep K-stage a multiple of 4.
        if (wmma_stage_k % 8) != 0 or (wmma_stage_k % 4) != 0:
            wmma_stage_k = 32
        if (K % wmma_stage_k) != 0:
            wmma_stage_k = 16
        if (K % wmma_stage_k) != 0:
            wmma_stage_k = 8

        wmma_tile_m = 16 * wmma_warps_m * wmma_frag_m
        wmma_tile_n = 16 * wmma_warps_n * wmma_frag_n
        wmma_grid_x = (N + wmma_tile_n - 1) // wmma_tile_n
        wmma_grid_y = (M + wmma_tile_m - 1) // wmma_tile_m
        # cp.async cache policy heuristics:
        # - A tile is reused across WARPS_N warps.
        # - B tile is reused across WARPS_M warps.
        # Prefer caching the more-reused operand in L1, and streaming the other.
        wmma_cp_a_policy = str(bindings.get("WMMA_CP_A_POLICY", "") or "").strip().lower()
        wmma_cp_b_policy = str(bindings.get("WMMA_CP_B_POLICY", "") or "").strip().lower()
        if wmma_cp_a_policy not in {"ca", "cg"}:
            wmma_cp_a_policy = "ca" if wmma_warps_n >= wmma_warps_m else "cg"
        if wmma_cp_b_policy not in {"ca", "cg"}:
            wmma_cp_b_policy = "ca" if wmma_warps_m > wmma_warps_n else "cg"
        # Whether to use cp.async fast-path. For smaller STAGE_K, we use a
        # synchronous vectorized copy fast-path instead (no cp.async) because it
        # uses less shared memory and avoids cp.async edge cases.
        # IMPORTANT: do NOT use `or -1` here; callers may explicitly pass 0 to
        # disable cp.async. (Python treats 0 as falsy.)
        use_cp_async_raw = bindings.get("WMMA_USE_CP_ASYNC", -1)
        try:
            use_cp_async_raw = int(use_cp_async_raw)
        except Exception:
            use_cp_async_raw = -1
        if int(use_cp_async_raw) < 0:
            # Prefer cp.async on Ampere+ by default. The generated kernel includes a
            # synchronous vector-copy fallback for pre-Ampere architectures.
            wmma_use_cp_async = True
        else:
            wmma_use_cp_async = bool(int(use_cp_async_raw))

        # Pipeline stages (shared-memory buffers).
        # - With cp.async we support 2-stage (double buffer) and 3-stage buffering.
        # - Without cp.async we use a single buffer (stages=1) to keep shared memory low.
        pipe_stages_raw = bindings.get("WMMA_PIPE_STAGES", 0)
        wmma_pipe_stages = int(pipe_stages_raw or 0)
        pipe_stages_override = ("WMMA_PIPE_STAGES" in bindings) and (wmma_pipe_stages > 0)
        if not wmma_use_cp_async:
            wmma_pipe_stages = 1
        else:
            if wmma_pipe_stages <= 0:
                wmma_pipe_stages = 3
            if wmma_pipe_stages not in (2, 3):
                wmma_pipe_stages = 3
        # Use dynamic shared memory for the A/B staging buffers so we can opt into
        # >48KiB shared memory on modern GPUs (required for larger STAGE_K).
        # Default padding reduces shared-memory bank conflicts for WMMA tiles.
        # Can be overridden via bindings when doing autotune sweeps.
        wmma_as_pad = int(bindings.get("WMMA_AS_PAD", 8) or 0)
        wmma_bs_pad = int(bindings.get("WMMA_BS_PAD", 8) or 0)
        # Keep padding small; it exists to reduce shared-memory bank conflicts.
        if wmma_as_pad < 0:
            wmma_as_pad = 0
        if wmma_bs_pad < 0:
            wmma_bs_pad = 0
        if wmma_as_pad > 32:
            wmma_as_pad = 32
        if wmma_bs_pad > 32:
            wmma_bs_pad = 32
        # Ensure 16B alignment for float4 copies: leading dims must be multiples of 4.
        if (wmma_as_pad % 4) != 0:
            wmma_as_pad = int((wmma_as_pad // 4) * 4)
        if (wmma_bs_pad % 4) != 0:
            wmma_bs_pad = int((wmma_bs_pad // 4) * 4)
        max_smem_optin = int(bindings.get("CUDA_MAX_SMEM_OPTIN", 0) or 0)
        if max_smem_optin <= 0:
            # Safe default across recent NVIDIA architectures; can be overridden by
            # CUDA_MAX_SMEM_OPTIN if needed.
            max_smem_optin = 96 * 1024

        def _wmma_smem_bytes(tile_m: int, tile_n: int, stage_k: int, pipe_stages: int) -> int:
            as_ld = stage_k + wmma_as_pad
            bs_ld = tile_n + wmma_bs_pad
            return 4 * (pipe_stages * tile_m * as_ld + pipe_stages * stage_k * bs_ld)

        shared_bytes = _wmma_smem_bytes(wmma_tile_m, wmma_tile_n, wmma_stage_k, wmma_pipe_stages)
        # If 3-stage buffering doesn't fit, fall back to double buffering.
        if wmma_pipe_stages > 2 and shared_bytes > max_smem_optin:
            wmma_pipe_stages = 2
            shared_bytes = _wmma_smem_bytes(wmma_tile_m, wmma_tile_n, wmma_stage_k, wmma_pipe_stages)
        # Clamp the K-stage if shared memory exceeds our opt-in budget.
        if shared_bytes > max_smem_optin:
            for cand in (64, 32, 16, 8):
                if cand >= wmma_stage_k:
                    continue
                if (K % cand) != 0:
                    continue
                if _wmma_smem_bytes(wmma_tile_m, wmma_tile_n, cand, wmma_pipe_stages) <= max_smem_optin:
                    wmma_stage_k = cand
                    shared_bytes = _wmma_smem_bytes(wmma_tile_m, wmma_tile_n, wmma_stage_k, wmma_pipe_stages)
                    break
        wmma_force_sync = False
        if (not pipe_stages_override) and wmma_pipe_stages == 2:
            bytes3 = _wmma_smem_bytes(wmma_tile_m, wmma_tile_n, wmma_stage_k, 3)
            if bytes3 <= max_smem_optin:
                wmma_pipe_stages = 3
                shared_bytes = bytes3

        # Round up to 16B for alignment.
        shared_bytes = ((shared_bytes + 15) // 16) * 16
        wmma_disable_fastpath = wmma_force_sync or bool(int(bindings.get("WMMA_DISABLE_FASTPATH", 0) or 0))
        specialize_full_tile = (
            specialize_dims
            and (M % wmma_tile_m == 0)
            and (N % wmma_tile_n == 0)
            and (K % wmma_stage_k == 0)
            and ((K & 3) == 0)
            and ((N & 3) == 0)
            and (not wmma_disable_fastpath)
        )
        wmma_cp_a_enum = "intentir_cuda::CpAsyncPolicy::CA" if wmma_cp_a_policy == "ca" else "intentir_cuda::CpAsyncPolicy::CG"
        wmma_cp_b_enum = "intentir_cuda::CpAsyncPolicy::CA" if wmma_cp_b_policy == "ca" else "intentir_cuda::CpAsyncPolicy::CG"
        use_cp_async_const = "true" if wmma_use_cp_async else "false"
        enable_fastpath_const = "false" if wmma_disable_fastpath else "true"
        specialize_full_tile_const = "true" if specialize_full_tile else "false"
        cuda_src = f"""
#include "kernels/wmma_matmul.cuh"

extern "C" __global__ void {intent.name}(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    {m_param}, {n_param}, {k_param}) {{
  {m_load}
  {n_load}
  {k_load}

	  // Assumes M,N are multiples of 16 and K is a multiple of 8 (enforced by host-side lowering).
	  constexpr int WARPS_M = {wmma_warps_m};
	  constexpr int WARPS_N = {wmma_warps_n};
	  constexpr int FRAG_M = {wmma_frag_m};
	  constexpr int FRAG_N = {wmma_frag_n};
	  constexpr int STAGE_K = {wmma_stage_k};
	  constexpr int AS_PAD = {wmma_as_pad};
	  constexpr int BS_PAD = {wmma_bs_pad};
	  constexpr int PIPE_STAGES = {wmma_pipe_stages};

	  intentir_cuda::wmma_matmul_f32_tf32<
	      WARPS_M,
	      WARPS_N,
	      FRAG_M,
	      FRAG_N,
	      STAGE_K,
	      AS_PAD,
	      BS_PAD,
      PIPE_STAGES,
      {use_cp_async_const},
      {wmma_cp_a_enum},
      {wmma_cp_b_enum},
      {enable_fastpath_const},
      {specialize_full_tile_const}>(A, B, C, M, N, K);
}}
""".lstrip()
        launch = CudaLaunch(grid=(wmma_grid_x, wmma_grid_y, 1), block=(32 * wmma_warps_m * wmma_warps_n, 1, 1), shared_mem=int(shared_bytes))
    else:
        cuda_src = f"""
extern "C" __global__ void {intent.name}(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    {m_param}, {n_param}, {k_param}) {{
  {m_load}
  {n_load}
  {k_load}
  constexpr int BLOCK_M = {block_y};
  constexpr int BLOCK_N = {block_x};
  constexpr int BLOCK_K = {block_k};
  constexpr int THREAD_M = {thread_m};
  constexpr int ROWS_PER_THREAD = {rows_per_thread};
  __shared__ float As[BLOCK_M][BLOCK_K];
  __shared__ float Bs[BLOCK_K][BLOCK_N];

  const int tx = (int)threadIdx.x;
  const int ty = (int)threadIdx.y;
  const int col = (int)(blockIdx.x * BLOCK_N + tx);
  const int block_row = (int)(blockIdx.y * BLOCK_M);
  const int row0 = block_row + ty;

  float acc[ROWS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < ROWS_PER_THREAD; ++i) acc[i] = 0.0f;
  for (int kt = 0; kt < K; kt += BLOCK_K) {{
    // Cooperative load (guarded).
    if (tx < BLOCK_K) {{
      #pragma unroll
      for (int i = 0; i < ROWS_PER_THREAD; ++i) {{
        const int r = ty + i * THREAD_M;
        if (r < BLOCK_M) {{
          const int row = block_row + r;
          if (row < M && (kt + tx) < K) As[r][tx] = A[row * K + (kt + tx)];
          else As[r][tx] = 0.0f;
        }}
      }}
    }}
    if (ty < BLOCK_K) {{
      if (col < N && (kt + ty) < K) Bs[ty][tx] = B[(kt + ty) * N + col];
      else Bs[ty][tx] = 0.0f;
    }}
    __syncthreads();
    #pragma unroll
    for (int k0 = 0; k0 < BLOCK_K; ++k0) {{
      const float b0 = Bs[k0][tx];
      #pragma unroll
      for (int i = 0; i < ROWS_PER_THREAD; ++i) {{
        const int r = ty + i * THREAD_M;
        if (r < BLOCK_M) acc[i] = fmaf(As[r][k0], b0, acc[i]);
      }}
    }}
    __syncthreads();
  }}
  if (col < N) {{
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {{
      const int row = row0 + i * THREAD_M;
      if (row < M) C[row * N + col] = acc[i];
    }}
  }}
}}
""".lstrip()

    tensor_args = [a, b, c]
    scalar_args: Dict[str, str] = {}
    arg_names = [a, b, c]
    if m_is_tensor:
        tensor_args.append(str(M_dim))
        arg_names.append(str(M_dim))
    else:
        scalar_args[str(M_dim)] = "i32"
        arg_names.append(str(M_dim))
    if n_is_tensor:
        tensor_args.append(str(N_dim))
        arg_names.append(str(N_dim))
    else:
        scalar_args[str(N_dim)] = "i32"
        arg_names.append(str(N_dim))
    if k_is_tensor:
        tensor_args.append(str(K_dim))
        arg_names.append(str(K_dim))
    else:
        scalar_args[str(K_dim)] = "i32"
        arg_names.append(str(K_dim))

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    if not use_wmma:
        launch = CudaLaunch(grid=(grid_x, grid_y, 1), block=(block_x, thread_m, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[c], bindings=dict(bindings))


def _kernel_dropout_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or intent.ops[0].op != "dropout":
        raise CudaLoweringError("dropout lowering expects a single dropout op")
    op = intent.ops[0]
    if len(op.inputs) != 3:
        raise CudaLoweringError("dropout expects 3 inputs (X,p,seed)")
    X, p_name, seed_name = op.inputs
    Y = op.output
    n = _as_int(bindings.get("n_elements"), name="n_elements")

    # Default to 10 Philox rounds (matches Triton reference).
    rounds = int(op.attrs.get("n_rounds") or 10)
    if rounds <= 0:
        rounds = 10
    if rounds > 10:
        rounds = 10

    # Use descriptor-like block size if present.
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    respect_schedule = bool(int(bindings.get("CUDA_RESPECT_SCHEDULE", 0) or 0))
    # Optional explicit override for backend tuning.
    tuned_block = bindings.get("DROPOUT_THREADS")
    if isinstance(tuned_block, int) and 0 < tuned_block <= 1024:
        block_x = int(tuned_block)
    elif not respect_schedule:
        # Backend heuristic: ignore small frontend defaults (often 32) and pick a
        # throughput-oriented block size.
        block_x = 256 if n >= (1 << 20) else 128
    # Keep block size warp-aligned.
    if block_x < 32:
        block_x = 32
    if (block_x % 32) != 0:
        block_x = int(((block_x + 31) // 32) * 32)
    if block_x > 1024:
        block_x = 1024

    # Process multiple elements per thread to trade ILP vs occupancy.
    #
    # Large vectors on high-end GPUs typically prefer higher occupancy (smaller EPT),
    # while smaller vectors can benefit from a bit more ILP.
    ept = int(op.attrs.get("elements_per_thread") or bindings.get("DROPOUT_EPT") or 0)
    if ept <= 0:
        ept = 1 if n >= (1 << 20) else 4
    if ept > 8:
        ept = 8

    grid_x = (n + (block_x * ept) - 1) // (block_x * ept)

    specialize_dims = bool(int(bindings.get("CUDA_SPECIALIZE_DIMS", 0) or 0))
    p_is_scalar = _is_scalar_tensor(intent, str(p_name), dtype="f32")
    seed_is_scalar = _is_scalar_tensor(intent, str(seed_name), dtype="i32")
    n_decl = (
        f"(void)n_elements_in;\n  constexpr int64_t n_elements = {int(n)}LL;"
        if specialize_dims
        else "const int64_t n_elements = n_elements_in;"
    )
    if p_is_scalar and seed_is_scalar:
        cuda_src = f"""
#include "kernels/dropout.cuh"

extern "C" __global__ void {intent.name}(const float* X, float p, int seed, float* Y, int64_t n_elements_in) {{
  {n_decl}
  constexpr int BLOCK_THREADS = {block_x};
  constexpr int EPT = {ept};
  constexpr int N_ROUNDS = {rounds};
  const int64_t tile = (int64_t)BLOCK_THREADS * (int64_t)EPT;
  const bool full_tile = (tile > 0) ? ((n_elements % tile) == 0) : false;
  if (full_tile) {{
    intentir_cuda::dropout_f32<EPT, N_ROUNDS, true>(X, p, (uint32_t)seed, Y, n_elements);
  }} else {{
    intentir_cuda::dropout_f32<EPT, N_ROUNDS, false>(X, p, (uint32_t)seed, Y, n_elements);
  }}
}}
""".lstrip()
        io_spec = _io_spec_from_args(
            intent,
            tensor_args=[X, Y],
            scalar_args={str(p_name): "f32", str(seed_name): "i32", "n_elements": "i64"},
            arg_names=[X, str(p_name), str(seed_name), Y, "n_elements"],
        )
    else:
        cuda_src = f"""
#include "kernels/dropout.cuh"

extern "C" __global__ void {intent.name}(const float* X, const float* p_ptr, const int* seed_ptr, float* Y, int64_t n_elements_in) {{
  {n_decl}
  constexpr int BLOCK_THREADS = {block_x};
  constexpr int EPT = {ept};
  constexpr int N_ROUNDS = {rounds};
  const int64_t tile = (int64_t)BLOCK_THREADS * (int64_t)EPT;
  const bool full_tile = (tile > 0) ? ((n_elements % tile) == 0) : false;
  if (full_tile) {{
    intentir_cuda::dropout_f32<EPT, N_ROUNDS, true>(X, p_ptr, seed_ptr, Y, n_elements);
  }} else {{
    intentir_cuda::dropout_f32<EPT, N_ROUNDS, false>(X, p_ptr, seed_ptr, Y, n_elements);
  }}
}}
""".lstrip()
        io_spec = _io_spec_from_args(
            intent,
            tensor_args=[X, p_name, seed_name, Y],
            scalar_args={"n_elements": "i64"},
            arg_names=[X, p_name, seed_name, Y, "n_elements"],
        )
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    out_bindings: Dict[str, int] = dict(bindings)
    out_bindings.setdefault("n_elements", n)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[Y], bindings=out_bindings)


def _kernel_softmax_2d_last_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    # Softmax kernel: one block per row, reduce over last dimension.
    # Identify the input matrix by tracing the reduce_max input (robust against extra tensors).
    out_name = str(intent.outputs[0]) if intent.outputs else "out"
    in_name = None
    for op in intent.ops or []:
        if op.op == "reduce_max" and op.inputs:
            in_name = str(op.inputs[0])
            break
    if not in_name:
        # Fallback: any rank-2 f32 tensor that is not produced by ops and not the output.
        produced = {o.output for o in (intent.ops or [])}
        for tn, tt in intent.tensors.items():
            if tn == out_name or tn in produced:
                continue
            if str(tt.dtype) == "f32" and len(tt.shape or []) == 2:
                in_name = str(tn)
                break
    if not in_name:
        raise CudaLoweringError("softmax lowering failed to identify input tensor")

    in_shape = _shape_values(intent, in_name)
    if len(in_shape) != 2:
        raise CudaLoweringError("softmax expects rank-2 input")
    R_dim, C_dim = in_shape
    R = _resolve_dim_int(R_dim, bindings, name="R")
    C = _resolve_dim_int(C_dim, bindings, name="C")

    # Thread block size heuristic:
    # - Triton uses BLOCK_SIZE=next_pow2(C) for reductions, but the actual CUDA threadcount
    #   is typically much smaller (num_warps), with each thread processing multiple lanes.
    # - Here we pick a block size that keeps "elements per thread" small (so we can keep
    #   exp(x) in registers across the sum reduction), avoiding a large shared exp buffer.
    def _next_pow2(x: int) -> int:
        if x <= 1:
            return 1
        return 1 << (int(x - 1).bit_length())

    sched = intent.schedule or ScheduleSketch()
    default_block = min(1024, _next_pow2(C))
    hinted = _resolve_schedule_int(sched.tile_n, bindings, default=default_block)
    block_x = hinted if 0 < hinted <= 1024 else default_block
    if block_x <= 0:
        block_x = default_block
    if block_x > 1024:
        block_x = 1024
    # Make it a multiple of 32.
    if block_x < 32:
        block_x = 32
    if (block_x % 32) != 0:
        block_x = int(((block_x + 31) // 32) * 32)
    if block_x > 1024:
        block_x = 1024
    if C > 1024:
        raise CudaLoweringError("softmax MVP supports only C<=1024")

    # Choose block threads for f32 softmax:
    # - We prefer fewer threads with higher "elements-per-thread" (EPT) to reduce
    #   reduction overhead while keeping enough ILP to hide exp latency.
    # - For AI-Bench softmax (C=781), 128 threads (EPT=7) is near parity with Triton.
    hinted_threads = _resolve_schedule_int(sched.tile_m, bindings, default=0)
    if hinted_threads and 0 < hinted_threads <= 1024:
        block_threads = int(hinted_threads)
    else:
        block_threads = 128 if C >= 128 else 64
    if block_threads < 32:
        block_threads = 32
    if (block_threads % 32) != 0:
        block_threads = int(((block_threads + 31) // 32) * 32)
    if block_threads > 1024:
        block_threads = 1024

    # Explicit tuning hook (bench scripts can set this via --bind SOFTMAX_THREADS=...).
    tuned_threads = bindings.get("SOFTMAX_THREADS")
    if isinstance(tuned_threads, int) and 0 < tuned_threads <= 1024:
        block_threads = int(tuned_threads)
        if block_threads < 32:
            block_threads = 32
        if (block_threads % 32) != 0:
            block_threads = int(((block_threads + 31) // 32) * 32)
        if block_threads > 1024:
            block_threads = 1024
    # We keep exp values in registers (expv[EPT]). Larger EPT increases register
    # pressure, but in practice softmax rows are small (<=1024) and fewer threads
    # can reduce reduction/sync overhead. Allow EPT up to 16 by default.
    max_ept = int(bindings.get("SOFTMAX_MAX_EPT") or 16)
    if max_ept < 4:
        max_ept = 4
    if max_ept > 16:
        max_ept = 16
    # If the user requests too few threads (which would require EPT>max_ept),
    # bump threads up to the minimum that still covers the whole row.
    min_threads = max(32, (C + max_ept - 1) // max_ept)
    if (min_threads % 32) != 0:
        min_threads = int(((min_threads + 31) // 32) * 32)
    if block_threads < min_threads:
        block_threads = min_threads
    ept = max(1, (C + block_threads - 1) // block_threads)

    specialize_dims = bool(int(bindings.get("CUDA_SPECIALIZE_DIMS", 0) or 0))
    r_is_tensor = _is_scalar_tensor(intent, str(R_dim), dtype="i32")
    c_is_tensor = _is_scalar_tensor(intent, str(C_dim), dtype="i32")
    # If the row/col dims are modeled as scalar tensors but we have concrete
    # bindings (and are specializing dims), pass them as by-value scalars to
    # avoid extra global loads.
    if specialize_dims and r_is_tensor and (str(R_dim) in bindings):
        r_is_tensor = False
    if specialize_dims and c_is_tensor and (str(C_dim) in bindings):
        c_is_tensor = False
    r_param = f"const int* {str(R_dim)}_ptr" if r_is_tensor else "int R"
    c_param = f"const int* {str(C_dim)}_ptr" if c_is_tensor else "int C"
    r_load = f"const int R = {str(R_dim)}_ptr ? {str(R_dim)}_ptr[0] : 0;" if r_is_tensor else ""
    c_load = f"const int C = {str(C_dim)}_ptr ? {str(C_dim)}_ptr[0] : 0;" if c_is_tensor else ""

    cuda_src = f"""
#include "kernels/softmax.cuh"

extern "C" __global__ void {intent.name}(const float* __restrict__ {in_name}, float* __restrict__ {out_name}, {r_param}, {c_param}) {{
  {r_load}
  {c_load}
  constexpr int BLOCK_THREADS = {block_threads};
  constexpr int EPT = {ept};
  intentir_cuda::softmax_2d_last_f32<BLOCK_THREADS, EPT>({in_name}, {out_name}, R, C);
}}
""".lstrip()

    # NOTE: We allocate a fixed 1024 shared array; block_x must be <=1024 (enforced).
    tensor_args = [in_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [in_name, out_name]
    if r_is_tensor:
        tensor_args.append(str(R_dim))
        arg_names.append(str(R_dim))
    else:
        scalar_args[str(R_dim)] = "i32"
        arg_names.append(str(R_dim))
    if c_is_tensor:
        tensor_args.append(str(C_dim))
        arg_names.append(str(C_dim))
    else:
        scalar_args[str(C_dim)] = "i32"
        arg_names.append(str(C_dim))
    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(R, 1, 1), block=(block_threads, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_layernorm_2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    # Identify inputs from the true input set (ignore intermediates present in `intent.tensors`).
    produced = {o.output for o in (intent.ops or [])}
    outs = set(intent.outputs or [])
    inputs = [n for n in intent.tensors.keys() if n not in produced and n not in outs]

    X_name = "X" if "X" in inputs else None
    if not X_name:
        for n in inputs:
            if str(intent.tensors[n].dtype) == "f32" and len(intent.tensors[n].shape or []) == 2:
                X_name = n
                break
    if not X_name:
        raise CudaLoweringError("layernorm lowering cannot find input X")

    # Prefer canonical names for gamma/beta if present.
    W_name = "W" if "W" in inputs else None
    B_name = "B" if "B" in inputs else None
    if not W_name or not B_name:
        rank1 = [n for n in inputs if str(intent.tensors[n].dtype) == "f32" and len(intent.tensors[n].shape or []) == 1]
        if not W_name and rank1:
            W_name = rank1[0]
        if not B_name and len(rank1) > 1:
            B_name = rank1[1]
    if not W_name or not B_name:
        raise CudaLoweringError("layernorm lowering cannot find W/B inputs")

    out_names = list(intent.outputs) if intent.outputs else ["Y", "Mean", "Rstd"]
    if len(out_names) != 3:
        raise CudaLoweringError("layernorm lowering expects 3 outputs (Y, Mean, Rstd)")
    Y_name, Mean_name, Rstd_name = (str(x) for x in out_names)

    x_shape = _shape_values(intent, X_name)
    if len(x_shape) != 2:
        raise CudaLoweringError("layernorm expects rank-2 X")
    M_dim, N_dim = x_shape
    M = _resolve_dim_int(M_dim, bindings, name="M")
    N = _resolve_dim_int(N_dim, bindings, name="N")

    # eps from const() if present, else fallback.
    eps = None
    for op in intent.ops or []:
        if op.op == "const" and op.output == "eps":
            try:
                eps = float(op.attrs.get("value"))
            except Exception:
                eps = None
            break
    if eps is None:
        eps = float(bindings.get("eps", 1e-5))

    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    if (block_x & (block_x - 1)) != 0:
        b = 1 << (int(block_x).bit_length() - 1)
        block_x = max(1, min(1024, b))

    cuda_src = f"""
#include "kernels/layernorm.cuh"

extern "C" __global__ void {intent.name}(
    const float* __restrict__ {X_name},
    float* __restrict__ {Y_name},
    const float* __restrict__ {W_name},
    const float* __restrict__ {B_name},
    float* __restrict__ {Mean_name},
    float* __restrict__ {Rstd_name},
    int M,
    int N,
    float eps) {{
  constexpr int BLOCK_THREADS = {block_x};
  intentir_cuda::layernorm_2d_f32<BLOCK_THREADS, false>({X_name}, {Y_name}, {W_name}, {B_name}, {Mean_name}, {Rstd_name}, M, N, eps);
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[X_name, Y_name, W_name, B_name, Mean_name, Rstd_name],
        scalar_args={"M": "i32", "N": "i32", "eps": "f32"},
        arg_names=[X_name, Y_name, W_name, B_name, Mean_name, Rstd_name, "M", "N", "eps"],
    )
    launch = CudaLaunch(grid=(M, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    out_bindings: Dict[str, Any] = dict(bindings)
    out_bindings.setdefault("eps", float(eps))
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[Y_name, Mean_name, Rstd_name], bindings=out_bindings)


def _kernel_glu_2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "glu":
        raise CudaLoweringError("glu lowering expects a single glu op")
    op = intent.ops[0]
    if len(op.inputs) != 1:
        raise CudaLoweringError("glu expects one input tensor")
    x_name = str(op.inputs[0])
    out_name = str(op.output)
    axis = int((op.attrs or {}).get("axis", -1))

    x_shape = _shape_values(intent, x_name)
    out_shape = _shape_values(intent, out_name)
    if len(x_shape) != 2 or len(out_shape) != 2:
        raise CudaLoweringError("glu lowering currently supports rank-2 tensors")
    axis_norm = axis if axis >= 0 else (axis + len(x_shape))
    if axis_norm != 1:
        raise CudaLoweringError("glu lowering currently supports axis=1")

    M = _resolve_dim_int(x_shape[0], bindings, name="M")
    N = _resolve_dim_int(x_shape[1], bindings, name="N")
    if (N % 2) != 0:
        raise CudaLoweringError("glu expects even input extent on split axis")
    N_out = N // 2
    try:
        M_decl = _resolve_dim_int(out_shape[0], bindings, name="M_out")
        if M_decl != M:
            raise CudaLoweringError("glu output M mismatch")
    except Exception:
        pass
    try:
        N_decl = _resolve_dim_int(out_shape[1], bindings, name="N_out")
        if N_decl not in {N_out, N}:
            raise CudaLoweringError("glu output axis extent incompatible with input")
    except Exception:
        pass

    total = int(M) * int(N_out)
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    grid_x = (total + block_x - 1) // block_x

    cuda_src = f"""
#include <math.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(const float* __restrict__ {x_name},
                                                                       float* __restrict__ {out_name}) {{
  const int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  const int total = {total};
  if (tid >= total) return;
  const int m = tid / {N_out};
  const int n = tid % {N_out};
  const int base = m * {N};
  const float a = {x_name}[base + n];
  const float b = {x_name}[base + n + {N_out}];
  const float sig = 1.0f / (1.0f + expf(-b));
  {out_name}[tid] = a * sig;
}}
""".lstrip()

    io_spec = _io_spec_from_args(intent, tensor_args=[x_name, out_name], scalar_args={}, arg_names=[x_name, out_name])
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update({"M": int(M), "N": int(N), "N_out": int(N_out)})
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=lowered_bindings)


def _kernel_group_norm_3d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    outs = list(intent.outputs or [])
    if len(outs) != 3:
        raise CudaLoweringError("group_norm lowering expects outputs [Y, Mean, Rstd]")
    Y_name, Mean_name, Rstd_name = (str(outs[0]), str(outs[1]), str(outs[2]))

    produced = {o.output for o in (intent.ops or []) if o.output}
    ext_inputs = [n for n in intent.tensors.keys() if n not in produced and n not in set(outs)]
    x_candidates = [n for n in ext_inputs if len(_shape_values(intent, n)) == 3 and str(intent.tensors[n].dtype) == "f32"]
    if not x_candidates:
        raise CudaLoweringError("group_norm lowering cannot find rank-3 input X")
    X_name = str(x_candidates[0])
    vec_candidates = [n for n in ext_inputs if len(_shape_values(intent, n)) == 1 and str(intent.tensors[n].dtype) == "f32"]
    if len(vec_candidates) < 2:
        raise CudaLoweringError("group_norm lowering cannot find W/B vectors")
    w_guess = next((n for n in vec_candidates if str(n).lower() in {"w", "weight"}), None)
    b_guess = next((n for n in vec_candidates if str(n).lower() in {"b", "bias"}), None)
    W_name = str(w_guess or vec_candidates[0])
    B_name = str(b_guess or (vec_candidates[1] if str(vec_candidates[0]) == W_name else vec_candidates[0]))
    if W_name == B_name:
        raise CudaLoweringError("group_norm lowering failed to disambiguate W/B inputs")

    x_shape = _shape_values(intent, X_name)
    y_shape = _shape_values(intent, Y_name)
    mean_shape = _shape_values(intent, Mean_name)
    rstd_shape = _shape_values(intent, Rstd_name)
    if len(x_shape) != 3 or len(y_shape) != 3 or len(mean_shape) != 2 or len(rstd_shape) != 2:
        raise CudaLoweringError("group_norm lowering expects X/Y rank-3 and Mean/Rstd rank-2 tensors")
    N = _resolve_dim_int(x_shape[0], bindings, name="N")
    C = _resolve_dim_int(x_shape[1], bindings, name="C")
    HW = _resolve_dim_int(x_shape[2], bindings, name="HW")
    if (
        _resolve_dim_int(y_shape[0], bindings, name="YN") != N
        or _resolve_dim_int(y_shape[1], bindings, name="YC") != C
        or _resolve_dim_int(y_shape[2], bindings, name="YHW") != HW
    ):
        raise CudaLoweringError("group_norm output Y shape mismatch")

    num_groups = int(bindings.get("num_groups", bindings.get("G", 0)) or 0)
    if num_groups <= 0:
        num_groups = _resolve_dim_int(mean_shape[1], bindings, name="num_groups")
    if num_groups <= 0:
        raise CudaLoweringError("group_norm requires positive num_groups")
    if C % num_groups != 0:
        raise CudaLoweringError("group_norm requires C divisible by num_groups")
    group_size = int(bindings.get("group_size", C // num_groups))
    if group_size <= 0 or group_size * num_groups != C:
        group_size = C // num_groups

    if (
        _resolve_dim_int(mean_shape[0], bindings, name="MeanN") != N
        or _resolve_dim_int(mean_shape[1], bindings, name="MeanG") != num_groups
        or _resolve_dim_int(rstd_shape[0], bindings, name="RstdN") != N
        or _resolve_dim_int(rstd_shape[1], bindings, name="RstdG") != num_groups
    ):
        raise CudaLoweringError("group_norm Mean/Rstd shape mismatch")

    eps = 1e-5
    for op in intent.ops or []:
        if op.op == "const" and str(op.output).lower() == "eps":
            try:
                eps = float(op.attrs.get("value"))
            except Exception:
                eps = 1e-5
            break

    total_groups = int(N) * int(num_groups)
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=128)
    if block_x <= 0:
        block_x = 128
    if block_x > 1024:
        block_x = 1024
    grid_x = (total_groups + block_x - 1) // block_x

    cuda_src = f"""
#include <math.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(const float* __restrict__ {X_name},
                                                                       const float* __restrict__ {W_name},
                                                                       const float* __restrict__ {B_name},
                                                                       float* __restrict__ {Y_name},
                                                                       float* __restrict__ {Mean_name},
                                                                       float* __restrict__ {Rstd_name}) {{
  const int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (tid >= {total_groups}) return;
  const int n = tid / {num_groups};
  const int g = tid % {num_groups};
  const int c0 = g * {group_size};
  const int elems = {group_size} * {HW};

  float sum = 0.0f;
  float sq = 0.0f;
  for (int gc = 0; gc < {group_size}; ++gc) {{
    const int c = c0 + gc;
    for (int hw = 0; hw < {HW}; ++hw) {{
      const int idx = ((n * {C} + c) * {HW}) + hw;
      const float x = {X_name}[idx];
      sum += x;
      sq += x * x;
    }}
  }}
  const float mean = sum / (float)elems;
  const float var = fmaxf((sq / (float)elems) - (mean * mean), 0.0f);
  const float rstd = rsqrtf(var + {eps:.9g}f);
  {Mean_name}[n * {num_groups} + g] = mean;
  {Rstd_name}[n * {num_groups} + g] = rstd;

  for (int gc = 0; gc < {group_size}; ++gc) {{
    const int c = c0 + gc;
    const float w = {W_name}[c];
    const float b = {B_name}[c];
    for (int hw = 0; hw < {HW}; ++hw) {{
      const int idx = ((n * {C} + c) * {HW}) + hw;
      const float x = {X_name}[idx];
      {Y_name}[idx] = ((x - mean) * rstd) * w + b;
    }}
  }}
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[X_name, W_name, B_name, Y_name, Mean_name, Rstd_name],
        scalar_args={},
        arg_names=[X_name, W_name, B_name, Y_name, Mean_name, Rstd_name],
    )
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    out_bindings: Dict[str, Any] = dict(bindings)
    out_bindings.setdefault("num_groups", int(num_groups))
    out_bindings.setdefault("group_size", int(group_size))
    out_bindings.setdefault("eps", float(eps))
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=[Y_name, Mean_name, Rstd_name],
        bindings=out_bindings,
    )


def _kernel_rope_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    # Respect scalar-tensor dims when present; fall back to scalar args otherwise.
    SEQ = _as_int(bindings.get("SEQ_LEN"), name="SEQ_LEN")
    B = _as_int(bindings.get("BATCH_NUM"), name="BATCH_NUM")
    H = _as_int(bindings.get("HEAD_NUM"), name="HEAD_NUM")
    D = _as_int(bindings.get("HEAD_DIM"), name="HEAD_DIM")
    if (D & 1) != 0:
        raise CudaLoweringError("rope expects even HEAD_DIM")
    half = D // 2

    # Rope is typically memory-bandwidth bound.
    #
    # We support grouping multiple heads per block for the same (batch, seq) to reuse
    # cos/sin across heads. In practice (AI-Bench shapes), the canonical mapping
    # (one head per block) is usually best; keep it as default but leave the knob for
    # future tuning.
    heads_per_block = _as_int(bindings.get("ROPE_HEADS_PER_BLOCK", 1), name="ROPE_HEADS_PER_BLOCK")
    if heads_per_block <= 0:
        heads_per_block = 1
    if heads_per_block > 16:
        # Keep the per-thread inner loop small (register pressure).
        heads_per_block = 16
    if heads_per_block > H:
        heads_per_block = H

    # Choose a compile-time vectorization width for RoPE. Using float4 reduces loop
    # iterations but increases register pressure; for the large HEAD_DIM in AI-Bench,
    # float4 is generally fine. Keep this as a knob to enable fair tuning when needed.
    rope_vec = _as_int(bindings.get("ROPE_VEC", 4), name="ROPE_VEC")
    if rope_vec not in (1, 2, 4):
        rope_vec = 4
    # Ensure the chosen vector width is compatible with the problem.
    if rope_vec == 4 and (half & 3) != 0:
        rope_vec = 2 if (half & 1) == 0 else 1
    if rope_vec == 2 and (half & 1) != 0:
        rope_vec = 1

    # Pre-unroll count for the chosen vector width.
    if rope_vec == 4:
        packs = half // 4
    elif rope_vec == 2:
        packs = half // 2
    else:
        packs = half

    # CUDA thread block size.
    #
    # Important: the Triton "BLOCK_SIZE" used in the reference kernels is a logical
    # vector width for tl.arange, not the number of CUDA threads. Using a 32-thread
    # block here (one warp) often under-utilizes memory bandwidth. We therefore
    # choose threads based on the work size, and expose an override knob.
    block_x = int(bindings.get("ROPE_THREADS", 0) or 0)
    if block_x <= 0:
        # Heuristic: keep ~4 vector packs per thread to increase ILP (important for
        # memory-bound RoPE), rather than maximizing threads.
        target = max(1, packs // 4)
        block_x = 1 << int(target - 1).bit_length()
        block_x = max(32, min(256, block_x))
    # Make it a multiple of 32.
    if block_x < 32:
        block_x = 32
    if (block_x % 32) != 0:
        block_x = int(((block_x + 31) // 32) * 32)
    if block_x > 1024:
        block_x = 1024

    # Indexing: use 32-bit offsets when safe (reduces 64-bit mul/add overhead).
    total_elems = int(SEQ) * int(B) * int(H) * int(D)
    use_i32 = total_elems <= (2**31 - 1)
    idx_t = "int" if use_i32 else "size_t"

    # Historically this intent was a single `rope` op, but newer pipeline versions
    # may prepend derived scalars as unused `const` ops (e.g. HEAD_DIM_DIV2 =
    # HEAD_DIM // 2). Treat `const* + rope` as equivalent.
    if not intent.ops:
        raise CudaLoweringError("rope lowering expects a rope op")
    if intent.ops[-1].op != "rope" or any(o.op != "const" for o in intent.ops[:-1]):
        raise CudaLoweringError("rope lowering expects `const* + rope` (rope last)")
    op = intent.ops[-1]
    if len(op.inputs) != 3:
        raise CudaLoweringError("rope expects 3 inputs (input, cos, sin)")
    in_name, cos_name, sin_name = op.inputs
    out_name = op.output

    # Scalar-tensor dims (AI-Bench) if present.
    seq_is_tensor = _is_scalar_tensor(intent, "SEQ_LEN", dtype="i32")
    b_is_tensor = _is_scalar_tensor(intent, "BATCH_NUM", dtype="i32")
    h_is_tensor = _is_scalar_tensor(intent, "HEAD_NUM", dtype="i32")
    d_is_tensor = _is_scalar_tensor(intent, "HEAD_DIM", dtype="i32")

    def _dim_param(name: str) -> str:
        return f"const int* {name}_ptr" if _is_scalar_tensor(intent, name, dtype="i32") else f"int {name}"

    def _dim_load(name: str) -> str:
        if _is_scalar_tensor(intent, name, dtype="i32"):
            return f"const int {name} = {name}_ptr ? {name}_ptr[0] : 0;"
        return ""

    # Pre-unroll count for the chosen vector width and block size.
    iters = max(1, (packs + block_x - 1) // block_x)

    cuda_src = f"""
#include "kernels/rope.cuh"

extern "C" __global__ void {intent.name}(
    const float* __restrict__ {in_name}, const float* __restrict__ {cos_name}, const float* __restrict__ {sin_name}, float* __restrict__ {out_name},
    {_dim_param("SEQ_LEN")}, {_dim_param("BATCH_NUM")}, {_dim_param("HEAD_NUM")}, {_dim_param("HEAD_DIM")}) {{
  {_dim_load("SEQ_LEN")}
  {_dim_load("BATCH_NUM")}
  {_dim_load("HEAD_NUM")}
  {_dim_load("HEAD_DIM")}
  constexpr int HEADS_PER_BLOCK = {heads_per_block};
  constexpr int ROPE_VEC = {rope_vec};
  constexpr int BLOCK_X = {block_x};
  constexpr int ITERS = {iters};
  using idx_t = {idx_t};
  intentir_cuda::rope_f32<HEADS_PER_BLOCK, ROPE_VEC, BLOCK_X, ITERS, false, false, idx_t>(
      {in_name}, {cos_name}, {sin_name}, {out_name}, SEQ_LEN, BATCH_NUM, HEAD_NUM, HEAD_DIM);
}}
""".lstrip()

    tensor_args = [in_name, cos_name, sin_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [in_name, cos_name, sin_name, out_name]
    for dim_name in ["SEQ_LEN", "BATCH_NUM", "HEAD_NUM", "HEAD_DIM"]:
        if _is_scalar_tensor(intent, dim_name, dtype="i32"):
            tensor_args.append(dim_name)
            arg_names.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
            arg_names.append(dim_name)

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    # Provide derived symbol for cos/sin shapes (e.g., HEAD_DIM_DIV_2).
    bindings = dict(bindings)
    try:
        cos_shape = _shape_values(intent, cos_name)
        if len(cos_shape) >= 2 and isinstance(cos_shape[1], str):
            bindings.setdefault(str(cos_shape[1]), half)
    except Exception:
        bindings.setdefault("HEAD_DIM_DIV_2", half)
    grid_x = (H + heads_per_block - 1) // heads_per_block
    launch = CudaLaunch(grid=(grid_x, B, SEQ), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=bindings)


def _kernel_transpose_2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "transpose":
        raise CudaLoweringError("transpose lowering expects a single transpose op")
    op = intent.ops[0]
    if len(op.inputs) != 1:
        raise CudaLoweringError("transpose expects 1 input")
    inp_name = str(op.inputs[0])
    out_name = str(op.output)
    perm = (op.attrs or {}).get("perm")
    if perm not in ([1, 0], (1, 0)):
        raise CudaLoweringError(f"transpose MVP supports only perm=[1,0], got {perm!r}")

    M = _as_int(bindings.get("M"), name="M")
    N = _as_int(bindings.get("N"), name="N")

    # Tile size: prefer schedule knobs; default to 16 to avoid 1024-thread blocks.
    sched = intent.schedule or ScheduleSketch()
    tile_n = _resolve_schedule_int(sched.tile_n, bindings, default=16)
    tile_m = _resolve_schedule_int(sched.tile_m, bindings, default=tile_n)
    tile = int(max(8, min(32, min(tile_m, tile_n))))

    block = (tile, tile, 1)
    grid = ((N + tile - 1) // tile, (M + tile - 1) // tile, 1)

    m_is_tensor = _is_scalar_tensor(intent, "M", dtype="i32")
    n_is_tensor = _is_scalar_tensor(intent, "N", dtype="i32")
    m_param = "const int* M_ptr" if m_is_tensor else "int M"
    n_param = "const int* N_ptr" if n_is_tensor else "int N"
    m_load = "const int M = M_ptr ? M_ptr[0] : 0;" if m_is_tensor else ""
    n_load = "const int N = N_ptr ? N_ptr[0] : 0;" if n_is_tensor else ""

    cuda_src = f"""
extern "C" __global__ void {intent.name}(const float* __restrict__ {inp_name}, float* __restrict__ {out_name}, {m_param}, {n_param}) {{
  {m_load}
  {n_load}
  __shared__ float tile[{tile}][{tile} + 1];
  const int x = (int)blockIdx.x * {tile} + (int)threadIdx.x;  // input col
  const int y = (int)blockIdx.y * {tile} + (int)threadIdx.y;  // input row
  if (x < N && y < M) {{
    tile[(int)threadIdx.y][(int)threadIdx.x] = {inp_name}[(size_t)y * (size_t)N + (size_t)x];
  }}
  __syncthreads();
  const int ox = (int)blockIdx.y * {tile} + (int)threadIdx.x;  // output col (input row)
  const int oy = (int)blockIdx.x * {tile} + (int)threadIdx.y;  // output row (input col)
  if (ox < M && oy < N) {{
    {out_name}[(size_t)oy * (size_t)M + (size_t)ox] = tile[(int)threadIdx.x][(int)threadIdx.y];
  }}
}}
""".lstrip()

    tensor_args = [inp_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [inp_name, out_name]
    for dim_name in ["M", "N"]:
        if _is_scalar_tensor(intent, dim_name, dtype="i32"):
            tensor_args.append(dim_name)
            arg_names.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
            arg_names.append(dim_name)

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=grid, block=block, shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_reduce_sum_2d_axis1_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "reduce_sum":
        raise CudaLoweringError("reduce_sum lowering expects a single reduce_sum op")
    op = intent.ops[0]
    if len(op.inputs) != 1:
        raise CudaLoweringError("reduce_sum expects 1 input")
    inp_name = str(op.inputs[0])
    out_name = str(op.output)
    dims = (op.attrs or {}).get("dims")
    axis = (op.attrs or {}).get("axis")
    if dims not in ([1], (1,)) and axis not in ([1], 1, "1"):
        raise CudaLoweringError("reduce_sum MVP supports only axis=1 for 2D tensors")

    M = _as_int(bindings.get("M"), name="M")
    N = _as_int(bindings.get("N"), name="N")
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    # Ensure a power-of-two block size for the reduction.
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    if (block_x & (block_x - 1)) != 0:
        block_x = 1 << int(block_x - 1).bit_length()
    block_x = max(32, min(1024, int(block_x)))

    m_is_tensor = _is_scalar_tensor(intent, "M", dtype="i32")
    n_is_tensor = _is_scalar_tensor(intent, "N", dtype="i32")
    m_param = "const int* M_ptr" if m_is_tensor else "int M"
    n_param = "const int* N_ptr" if n_is_tensor else "int N"
    m_load = "const int M = M_ptr ? M_ptr[0] : 0;" if m_is_tensor else ""
    n_load = "const int N = N_ptr ? N_ptr[0] : 0;" if n_is_tensor else ""

    cuda_src = f"""
#include <stdint.h>

#include "intentir_cuda_ops.cuh"
#include "kernels/reduce.cuh"

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(const float* __restrict__ {inp_name}, float* __restrict__ {out_name}, {m_param}, {n_param}) {{
  {m_load}
  {n_load}
  const int m = (int)blockIdx.x;
  if (m >= M) return;
  constexpr int BLOCK_THREADS = {block_x};
  __shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red;
  float acc = 0.0f;
  const float* row = {inp_name} + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) acc += intentir_ldg_f32(row + (size_t)n);
  const float sum = intentir_cuda::block_allreduce_sum<BLOCK_THREADS>(acc, &red);
  if ((int)threadIdx.x == 0) {out_name}[m] = sum;
}}
""".lstrip()

    tensor_args = [inp_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [inp_name, out_name]
    for dim_name in ["M", "N"]:
        if _is_scalar_tensor(intent, dim_name, dtype="i32"):
            tensor_args.append(dim_name)
            arg_names.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
            arg_names.append(dim_name)

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(M, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_reduce_max_2d_axis1_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "reduce_max":
        raise CudaLoweringError("reduce_max lowering expects a single reduce_max op")
    op = intent.ops[0]
    if len(op.inputs) != 1:
        raise CudaLoweringError("reduce_max expects 1 input")
    inp_name = str(op.inputs[0])
    out_name = str(op.output)
    dims = (op.attrs or {}).get("dims")
    axis = (op.attrs or {}).get("axis")
    if dims not in ([1], (1,)) and axis not in ([1], 1, "1"):
        raise CudaLoweringError("reduce_max MVP supports only axis=1 for 2D tensors")

    M = _as_int(bindings.get("M"), name="M")
    N = _as_int(bindings.get("N"), name="N")
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    if (block_x & (block_x - 1)) != 0:
        block_x = 1 << int(block_x - 1).bit_length()
    block_x = max(32, min(1024, int(block_x)))

    m_is_tensor = _is_scalar_tensor(intent, "M", dtype="i32")
    n_is_tensor = _is_scalar_tensor(intent, "N", dtype="i32")
    m_param = "const int* M_ptr" if m_is_tensor else "int M"
    n_param = "const int* N_ptr" if n_is_tensor else "int N"
    m_load = "const int M = M_ptr ? M_ptr[0] : 0;" if m_is_tensor else ""
    n_load = "const int N = N_ptr ? N_ptr[0] : 0;" if n_is_tensor else ""

    cuda_src = f"""
#include <math.h>
#include <stdint.h>

#include "intentir_cuda_ops.cuh"
#include "kernels/reduce.cuh"

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(const float* __restrict__ {inp_name}, float* __restrict__ {out_name}, {m_param}, {n_param}) {{
  {m_load}
  {n_load}
  const int m = (int)blockIdx.x;
  if (m >= M) return;
  constexpr int BLOCK_THREADS = {block_x};
  __shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red;
  float acc = -INFINITY;
  const float* row = {inp_name} + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) acc = fmaxf(acc, intentir_ldg_f32(row + (size_t)n));
  const float mx = intentir_cuda::block_allreduce_max<BLOCK_THREADS>(acc, &red);
  if ((int)threadIdx.x == 0) {out_name}[m] = mx;
}}
""".lstrip()

    tensor_args = [inp_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [inp_name, out_name]
    for dim_name in ["M", "N"]:
        if _is_scalar_tensor(intent, dim_name, dtype="i32"):
            tensor_args.append(dim_name)
            arg_names.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
            arg_names.append(dim_name)

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(M, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_reduce_min_2d_axis1_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "reduce_min":
        raise CudaLoweringError("reduce_min lowering expects a single reduce_min op")
    op = intent.ops[0]
    if len(op.inputs) != 1:
        raise CudaLoweringError("reduce_min expects 1 input")
    inp_name = str(op.inputs[0])
    out_name = str(op.output)
    dims = (op.attrs or {}).get("dims")
    axis = (op.attrs or {}).get("axis")
    if dims not in ([1], (1,)) and axis not in ([1], 1, "1"):
        raise CudaLoweringError("reduce_min MVP supports only axis=1 for 2D tensors")

    M = _as_int(bindings.get("M"), name="M")
    N = _as_int(bindings.get("N"), name="N")
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    if (block_x & (block_x - 1)) != 0:
        block_x = 1 << int(block_x - 1).bit_length()
    block_x = max(32, min(1024, int(block_x)))

    m_is_tensor = _is_scalar_tensor(intent, "M", dtype="i32")
    n_is_tensor = _is_scalar_tensor(intent, "N", dtype="i32")
    m_param = "const int* M_ptr" if m_is_tensor else "int M"
    n_param = "const int* N_ptr" if n_is_tensor else "int N"
    m_load = "const int M = M_ptr ? M_ptr[0] : 0;" if m_is_tensor else ""
    n_load = "const int N = N_ptr ? N_ptr[0] : 0;" if n_is_tensor else ""

    cuda_src = f"""
#include <math.h>
#include <stdint.h>

#include "intentir_cuda_ops.cuh"
#include "kernels/reduce.cuh"

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(const float* __restrict__ {inp_name}, float* __restrict__ {out_name}, {m_param}, {n_param}) {{
  {m_load}
  {n_load}
  const int m = (int)blockIdx.x;
  if (m >= M) return;
  constexpr int BLOCK_THREADS = {block_x};
  __shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red;
  float acc = INFINITY;
  const float* row = {inp_name} + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) acc = fminf(acc, intentir_ldg_f32(row + (size_t)n));
  const float mn = intentir_cuda::block_allreduce_min<BLOCK_THREADS>(acc, &red);
  if ((int)threadIdx.x == 0) {out_name}[m] = mn;
}}
""".lstrip()

    tensor_args = [inp_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [inp_name, out_name]
    for dim_name in ["M", "N"]:
        if _is_scalar_tensor(intent, dim_name, dtype="i32"):
            tensor_args.append(dim_name)
            arg_names.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
            arg_names.append(dim_name)

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(M, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_any_dim_f32_to_i1(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    """
    Pattern family:
      - const(z) + ne(inp, z) + reduce_any(axis=1)
      - const(z) + eq(inp, z) + reduce_any(axis=1) + not

    Supports keepdim and reduction combine_fn={or,and}. The second pattern is
    used by row_all-style semantics: not(any(inp == 0)).
    """
    if not intent.ops or len(intent.ops) not in {3, 4}:
        raise CudaLoweringError("any-dim lowering expects 3 or 4 ops")
    c0 = intent.ops[0]
    cmp0 = intent.ops[1]
    r0 = intent.ops[2]
    tail_not = intent.ops[3] if len(intent.ops) == 4 else None
    if c0.op != "const" or cmp0.op not in {"ne", "eq"} or r0.op != "reduce_any":
        raise CudaLoweringError("any-dim lowering expects const->(eq|ne)->reduce_any")
    if tail_not is not None and tail_not.op != "not":
        raise CudaLoweringError("any-dim lowering expects optional trailing not")
    if len(cmp0.inputs) != 2 or len(r0.inputs) != 1:
        raise CudaLoweringError("any-dim lowering invalid op arity")
    if tail_not is not None and len(tail_not.inputs) != 1:
        raise CudaLoweringError("any-dim lowering trailing not expects 1 input")
    const_out = str(c0.output)
    a0, b0 = (str(x) for x in cmp0.inputs)
    if a0 == const_out and b0 != const_out:
        inp_name = b0
    elif b0 == const_out and a0 != const_out:
        inp_name = a0
    else:
        raise CudaLoweringError("any-dim lowering expects compare(inp, const)")
    if tail_not is not None and str(tail_not.inputs[0]) != str(r0.output):
        raise CudaLoweringError("any-dim lowering trailing not must consume reduce_any output")
    out_name = str(tail_not.output) if tail_not is not None else str(r0.output)
    cmp_is_ne = str(cmp0.op) == "ne"
    invert_output = tail_not is not None
    z = float((c0.attrs or {}).get("value", 0.0))

    combine_fn = str((r0.attrs or {}).get("combine_fn", "or")).strip().lower()
    init_value = bool((r0.attrs or {}).get("init_value", False))
    reduce_is_and = combine_fn in {"and", "all"} or init_value

    # Require reduce axis=1.
    dims = (r0.attrs or {}).get("dims")
    axis = (r0.attrs or {}).get("axis")
    if dims not in ([1], (1,)) and axis not in ([1], 1, "1"):
        raise CudaLoweringError("any-dim MVP supports only axis=1 for 2D tensors")

    keepdim = bool((r0.attrs or {}).get("keepdims", False) or (r0.attrs or {}).get("keepdim", False))
    out_shape = _shape_values(intent, out_name)
    if keepdim:
        if len(out_shape) != 2:
            raise CudaLoweringError("any-dim keepdim expects rank-2 output")
        if str(out_shape[0]) != "M" or str(out_shape[1]) not in {"1", "1.0"}:
            raise CudaLoweringError("any-dim keepdim expects output shape [M,1]")
    else:
        if len(out_shape) != 1:
            raise CudaLoweringError("any-dim non-keepdim expects rank-1 output")
        if str(out_shape[0]) != "M":
            raise CudaLoweringError("any-dim non-keepdim expects output shape [M]")

    M = _as_int(bindings.get("M"), name="M")
    N = _as_int(bindings.get("N"), name="N")
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    if (block_x & (block_x - 1)) != 0:
        block_x = 1 << int(block_x - 1).bit_length()
    block_x = max(32, min(1024, int(block_x)))

    m_is_tensor = _is_scalar_tensor(intent, "M", dtype="i32")
    n_is_tensor = _is_scalar_tensor(intent, "N", dtype="i32")
    m_param = "const int* M_ptr" if m_is_tensor else "int M"
    n_param = "const int* N_ptr" if n_is_tensor else "int N"
    m_load = "const int M = M_ptr ? M_ptr[0] : 0;" if m_is_tensor else ""
    n_load = "const int N = N_ptr ? N_ptr[0] : 0;" if n_is_tensor else ""

    z_lit = _c_scalar_literal("f32", z)
    cmp_pred = "!=" if cmp_is_ne else "=="
    reduce_op = "|" if not reduce_is_and else "&"
    reduce_init = "0" if not reduce_is_and else "1"
    reduce_call = "block_allreduce_max" if not reduce_is_and else "block_allreduce_min"
    out_expr = "((any != 0) ? false : true)" if invert_output else "(any != 0)"
    cuda_src = f"""
#include <stdint.h>

#include "intentir_cuda_ops.cuh"
#include "kernels/reduce.cuh"

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(const float* __restrict__ {inp_name}, bool* __restrict__ {out_name}, {m_param}, {n_param}) {{
  {m_load}
  {n_load}
  const int m = (int)blockIdx.x;
  if (m >= M) return;
  constexpr int BLOCK_THREADS = {block_x};
  __shared__ intentir_cuda::BlockAllreduceI32<BLOCK_THREADS> red;
  int anyv = {reduce_init};
  const float* row = {inp_name} + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) {{
    const int pred = (intentir_ldg_f32(row + (size_t)n) {cmp_pred} {z_lit}) ? 1 : 0;
    anyv {reduce_op}= pred;
  }}
  const int any = intentir_cuda::{reduce_call}<BLOCK_THREADS>(anyv, &red);
  if ((int)threadIdx.x == 0) {out_name}[m] = {out_expr};
}}
""".lstrip()

    tensor_args = [inp_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [inp_name, out_name]
    for dim_name in ["M", "N"]:
        if _is_scalar_tensor(intent, dim_name, dtype="i32"):
            tensor_args.append(dim_name)
            arg_names.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
            arg_names.append(dim_name)

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(M, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_addmm_2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    """
    Pattern: matmul + alpha/beta scaling + add (+ optional cast).
    """
    ops = list(intent.ops or [])
    if len(ops) not in {4, 5}:
        raise CudaLoweringError("addmm lowering expects 4 or 5 ops")
    if ops[0].op != "matmul" or ops[1].op != "mul" or ops[2].op != "mul" or ops[3].op != "add":
        raise CudaLoweringError("addmm lowering expects matmul->mul->mul->add chain")
    mat, mul_a, mul_b, add0 = ops[:4]
    cast0 = ops[4] if len(ops) == 5 else None
    if cast0 is not None and cast0.op != "cast":
        raise CudaLoweringError("addmm optional 5th op must be cast")

    if len(mat.inputs) != 2:
        raise CudaLoweringError("addmm matmul expects 2 inputs")
    a_name = str(mat.inputs[0])
    b_name = str(mat.inputs[1])
    mm_out = str(mat.output)

    if len(mul_a.inputs) != 2 or len(mul_b.inputs) != 2:
        raise CudaLoweringError("addmm mul ops expect 2 inputs each")

    mul_a_in = [str(x) for x in mul_a.inputs]
    if mm_out in mul_a_in:
        alpha_name = mul_a_in[1] if mul_a_in[0] == mm_out else mul_a_in[0]
        scaled_mm = str(mul_a.output)
    else:
        raise CudaLoweringError("addmm first mul must consume matmul output")

    mul_b_in = [str(x) for x in mul_b.inputs]
    beta_name = None
    bias_name = None
    for x in mul_b_in:
        if _is_scalar_tensor(intent, x, dtype="f32"):
            beta_name = x
        else:
            bias_name = x
    if beta_name is None or bias_name is None:
        raise CudaLoweringError("addmm second mul must be bias * beta")
    scaled_bias = str(mul_b.output)

    add_in = [str(x) for x in add0.inputs]
    if len(add_in) != 2 or sorted(add_in) != sorted([scaled_mm, scaled_bias]):
        raise CudaLoweringError("addmm add op must consume scaled matmul and scaled bias")
    add_out = str(add0.output)

    out_name = str(intent.outputs[0])
    if cast0 is not None:
        if len(cast0.inputs) != 1 or str(cast0.inputs[0]) != add_out or str(cast0.output) != out_name:
            raise CudaLoweringError("addmm cast must consume add output and produce final output")
    elif add_out != out_name:
        raise CudaLoweringError("addmm add output must be final output when cast is absent")

    a_shape = _shape_values(intent, a_name)
    b_shape = _shape_values(intent, b_name)
    bias_shape = _shape_values(intent, bias_name)
    out_shape = _shape_values(intent, out_name)
    if len(a_shape) != 2 or len(b_shape) != 2 or len(bias_shape) != 2 or len(out_shape) != 2:
        raise CudaLoweringError("addmm lowering expects rank-2 tensors")
    M_dim, K_dim = a_shape
    K2_dim, N_dim = b_shape
    if str(K_dim) != str(K2_dim):
        raise CudaLoweringError("addmm K mismatch")
    if str(bias_shape[0]) != str(M_dim) or str(bias_shape[1]) != str(N_dim):
        raise CudaLoweringError("addmm bias shape mismatch")
    if str(out_shape[0]) != str(M_dim) or str(out_shape[1]) != str(N_dim):
        raise CudaLoweringError("addmm output shape mismatch")

    ta = bool((mat.attrs or {}).get("transpose_a", False))
    tb = bool((mat.attrs or {}).get("transpose_b", False))
    M = _resolve_dim_int(M_dim, bindings, name="M")
    N = _resolve_dim_int(N_dim, bindings, name="N")
    K = _resolve_dim_int(K_dim, bindings, name="K")

    sched = intent.schedule or ScheduleSketch()
    block_y = _resolve_schedule_int(sched.tile_m, bindings, default=16)
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=16)
    if block_x <= 0:
        block_x = 16
    if block_y <= 0:
        block_y = 16
    block_x = max(8, min(32, int(block_x)))
    block_y = max(8, min(32, int(block_y)))
    if block_x * block_y > 1024:
        block_y = max(1, 1024 // block_x)
    grid_x = (N + block_x - 1) // block_x
    grid_y = (M + block_y - 1) // block_y

    m_is_tensor = _is_scalar_tensor(intent, str(M_dim), dtype="i32")
    n_is_tensor = _is_scalar_tensor(intent, str(N_dim), dtype="i32")
    k_is_tensor = _is_scalar_tensor(intent, str(K_dim), dtype="i32")
    m_param = f"const int* {str(M_dim)}_ptr" if m_is_tensor else "int M_in"
    n_param = f"const int* {str(N_dim)}_ptr" if n_is_tensor else "int N_in"
    k_param = f"const int* {str(K_dim)}_ptr" if k_is_tensor else "int K_in"
    m_load = f"const int M = {str(M_dim)}_ptr ? {str(M_dim)}_ptr[0] : 0;" if m_is_tensor else "const int M = M_in;"
    n_load = f"const int N = {str(N_dim)}_ptr ? {str(N_dim)}_ptr[0] : 0;" if n_is_tensor else "const int N = N_in;"
    k_load = f"const int K = {str(K_dim)}_ptr ? {str(K_dim)}_ptr[0] : 0;" if k_is_tensor else "const int K = K_in;"

    a_idx = "((size_t)row * (size_t)K + (size_t)k)" if not ta else "((size_t)k * (size_t)M + (size_t)row)"
    b_idx = "((size_t)k * (size_t)N + (size_t)col)" if not tb else "((size_t)col * (size_t)K + (size_t)k)"
    out_idx = "((size_t)row * (size_t)N + (size_t)col)"
    alpha_ptr_name = f"{alpha_name}_ptr"
    beta_ptr_name = f"{beta_name}_ptr"
    cuda_src = f"""
#include <math.h>
#include <stdint.h>

#include "intentir_cuda_ops.cuh"

extern "C" __global__ __launch_bounds__({block_x * block_y}) void {intent.name}(
    const float* __restrict__ {a_name},
    const float* __restrict__ {b_name},
    const float* __restrict__ {bias_name},
    const float* __restrict__ {alpha_ptr_name},
    const float* __restrict__ {beta_ptr_name},
    float* __restrict__ {out_name},
    {m_param},
    {n_param},
    {k_param}) {{
  {m_load}
  {n_load}
  {k_load}
  const int row = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
  const int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (row >= M || col >= N) return;
  const float alpha = {alpha_ptr_name} ? {alpha_ptr_name}[0] : 1.0f;
  const float beta = {beta_ptr_name} ? {beta_ptr_name}[0] : 1.0f;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {{
    const float av = intentir_ldg_f32({a_name} + {a_idx});
    const float bv = intentir_ldg_f32({b_name} + {b_idx});
    acc += av * bv;
  }}
  const float bias_val = intentir_ldg_f32({bias_name} + {out_idx});
  {out_name}[{out_idx}] = alpha * acc + beta * bias_val;
}}
""".lstrip()

    tensor_args = [a_name, b_name, bias_name, alpha_name, beta_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [a_name, b_name, bias_name, alpha_name, beta_name, out_name]
    for dim_name, is_tensor in ((str(M_dim), m_is_tensor), (str(N_dim), n_is_tensor), (str(K_dim), k_is_tensor)):
        if is_tensor:
            tensor_args.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
        arg_names.append(dim_name)
    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(grid_x, grid_y, 1), block=(block_x, block_y, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_addmv_2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    """
    Pattern: matmul(A[N,M], B[M]) + alpha/beta scaling + add.
    """
    ops = list(intent.ops or [])
    if len(ops) != 4 or [o.op for o in ops] != ["matmul", "mul", "mul", "add"]:
        raise CudaLoweringError("addmv lowering expects matmul->mul->mul->add chain")
    mat, mul_a, mul_b, add0 = ops
    if len(mat.inputs) != 2:
        raise CudaLoweringError("addmv matmul expects 2 inputs")
    a_name = str(mat.inputs[0])
    b_name = str(mat.inputs[1])
    mm_out = str(mat.output)

    mul_a_in = [str(x) for x in mul_a.inputs]
    if mm_out in mul_a_in:
        alpha_name = mul_a_in[1] if mul_a_in[0] == mm_out else mul_a_in[0]
        scaled_mm = str(mul_a.output)
    else:
        raise CudaLoweringError("addmv first mul must consume matmul output")

    mul_b_in = [str(x) for x in mul_b.inputs]
    beta_name = None
    inp_name = None
    for x in mul_b_in:
        if _is_scalar_tensor(intent, x, dtype="f32"):
            beta_name = x
        else:
            inp_name = x
    if beta_name is None or inp_name is None:
        raise CudaLoweringError("addmv second mul must be input * beta")
    scaled_inp = str(mul_b.output)

    add_in = [str(x) for x in add0.inputs]
    if len(add_in) != 2 or sorted(add_in) != sorted([scaled_mm, scaled_inp]):
        raise CudaLoweringError("addmv add op must consume scaled matmul and scaled input")
    out_name = str(intent.outputs[0])
    if str(add0.output) != out_name:
        raise CudaLoweringError("addmv add output must be final output")

    a_shape = _shape_values(intent, a_name)
    b_shape = _shape_values(intent, b_name)
    inp_shape = _shape_values(intent, inp_name)
    out_shape = _shape_values(intent, out_name)
    if len(a_shape) != 2 or len(b_shape) != 1 or len(inp_shape) != 1 or len(out_shape) != 1:
        raise CudaLoweringError("addmv lowering expects A[N,M], B[M], input[N], out[N]")
    N_dim, M_dim = a_shape
    if str(b_shape[0]) != str(M_dim):
        raise CudaLoweringError("addmv shape mismatch between A and B")
    if str(inp_shape[0]) != str(N_dim) or str(out_shape[0]) != str(N_dim):
        raise CudaLoweringError("addmv shape mismatch for input/output")

    N = _resolve_dim_int(N_dim, bindings, name="N")
    M = _resolve_dim_int(M_dim, bindings, name="M")
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    block_x = max(32, min(1024, int(block_x)))
    if (block_x & (block_x - 1)) != 0:
        block_x = 1 << int(block_x - 1).bit_length()
    grid_x = (N + block_x - 1) // block_x

    n_is_tensor = _is_scalar_tensor(intent, str(N_dim), dtype="i32")
    m_is_tensor = _is_scalar_tensor(intent, str(M_dim), dtype="i32")
    n_param = f"const int* {str(N_dim)}_ptr" if n_is_tensor else "int N_in"
    m_param = f"const int* {str(M_dim)}_ptr" if m_is_tensor else "int M_in"
    n_load = f"const int N = {str(N_dim)}_ptr ? {str(N_dim)}_ptr[0] : 0;" if n_is_tensor else "const int N = N_in;"
    m_load = f"const int M = {str(M_dim)}_ptr ? {str(M_dim)}_ptr[0] : 0;" if m_is_tensor else "const int M = M_in;"
    alpha_ptr_name = f"{alpha_name}_ptr"
    beta_ptr_name = f"{beta_name}_ptr"

    cuda_src = f"""
#include <stdint.h>

#include "intentir_cuda_ops.cuh"

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(
    const float* __restrict__ {a_name},
    const float* __restrict__ {b_name},
    const float* __restrict__ {inp_name},
    const float* __restrict__ {alpha_ptr_name},
    const float* __restrict__ {beta_ptr_name},
    float* __restrict__ {out_name},
    {n_param},
    {m_param}) {{
  {n_load}
  {m_load}
  const int n = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (n >= N) return;
  const float alpha = {alpha_ptr_name} ? {alpha_ptr_name}[0] : 1.0f;
  const float beta = {beta_ptr_name} ? {beta_ptr_name}[0] : 1.0f;
  float acc = 0.0f;
  const float* row = {a_name} + (size_t)n * (size_t)M;
  for (int k = 0; k < M; ++k) {{
    acc += intentir_ldg_f32(row + (size_t)k) * intentir_ldg_f32({b_name} + (size_t)k);
  }}
  {out_name}[(size_t)n] = alpha * acc + beta * intentir_ldg_f32({inp_name} + (size_t)n);
}}
""".lstrip()

    tensor_args = [a_name, b_name, inp_name, alpha_name, beta_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [a_name, b_name, inp_name, alpha_name, beta_name, out_name]
    for dim_name, is_tensor in ((str(N_dim), n_is_tensor), (str(M_dim), m_is_tensor)):
        if is_tensor:
            tensor_args.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
        arg_names.append(dim_name)
    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_bmm_3d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "matmul":
        raise CudaLoweringError("bmm lowering expects a single matmul op")
    mat = intent.ops[0]
    if len(mat.inputs) != 2:
        raise CudaLoweringError("bmm matmul expects 2 inputs")

    a_name = str(mat.inputs[0])
    b_name = str(mat.inputs[1])
    out_name = str(intent.outputs[0])

    a_shape = _shape_values(intent, a_name)
    b_shape = _shape_values(intent, b_name)
    out_shape = _shape_values(intent, out_name)
    if len(a_shape) != 3 or len(b_shape) != 3 or len(out_shape) != 3:
        raise CudaLoweringError("bmm lowering expects rank-3 tensors")

    batch_dim, a1_dim, a2_dim = a_shape
    b_batch_dim, b1_dim, b2_dim = b_shape
    ta = bool((mat.attrs or {}).get("transpose_a", False))
    tb = bool((mat.attrs or {}).get("transpose_b", False))

    M_dim = a2_dim if ta else a1_dim
    K_dim = a1_dim if ta else a2_dim
    K2_dim = b2_dim if tb else b1_dim
    N_dim = b1_dim if tb else b2_dim
    if str(batch_dim) != str(b_batch_dim):
        raise CudaLoweringError("bmm batch mismatch between A and B")
    if str(K_dim) != str(K2_dim):
        raise CudaLoweringError("bmm K mismatch")
    if [str(x) for x in out_shape] != [str(batch_dim), str(M_dim), str(N_dim)]:
        raise CudaLoweringError("bmm output shape mismatch")

    BATCH = _resolve_dim_int(batch_dim, bindings, name="BATCH")
    M = _resolve_dim_int(M_dim, bindings, name="M")
    N = _resolve_dim_int(N_dim, bindings, name="N")
    K = _resolve_dim_int(K_dim, bindings, name="K")

    sched = intent.schedule or ScheduleSketch()
    block_y = _resolve_schedule_int(sched.tile_m, bindings, default=16)
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=16)
    if block_x <= 0:
        block_x = 16
    if block_y <= 0:
        block_y = 16
    block_x = max(8, min(32, int(block_x)))
    block_y = max(8, min(32, int(block_y)))
    if block_x * block_y > 1024:
        block_y = max(1, 1024 // block_x)

    grid_x = (N + block_x - 1) // block_x
    grid_y = (M + block_y - 1) // block_y
    grid_z = max(1, int(BATCH))

    b_is_tensor = _is_scalar_tensor(intent, str(batch_dim), dtype="i32")
    m_is_tensor = _is_scalar_tensor(intent, str(M_dim), dtype="i32")
    n_is_tensor = _is_scalar_tensor(intent, str(N_dim), dtype="i32")
    k_is_tensor = _is_scalar_tensor(intent, str(K_dim), dtype="i32")
    b_param = f"const int* {str(batch_dim)}_ptr" if b_is_tensor else "int BATCH_in"
    m_param = f"const int* {str(M_dim)}_ptr" if m_is_tensor else "int M_in"
    n_param = f"const int* {str(N_dim)}_ptr" if n_is_tensor else "int N_in"
    k_param = f"const int* {str(K_dim)}_ptr" if k_is_tensor else "int K_in"
    b_load = f"const int BATCH = {str(batch_dim)}_ptr ? {str(batch_dim)}_ptr[0] : 0;" if b_is_tensor else "const int BATCH = BATCH_in;"
    m_load = f"const int M = {str(M_dim)}_ptr ? {str(M_dim)}_ptr[0] : 0;" if m_is_tensor else "const int M = M_in;"
    n_load = f"const int N = {str(N_dim)}_ptr ? {str(N_dim)}_ptr[0] : 0;" if n_is_tensor else "const int N = N_in;"
    k_load = f"const int K = {str(K_dim)}_ptr ? {str(K_dim)}_ptr[0] : 0;" if k_is_tensor else "const int K = K_in;"

    a1 = _resolve_dim_int(a1_dim, bindings, name="A_dim1")
    a2 = _resolve_dim_int(a2_dim, bindings, name="A_dim2")
    b1 = _resolve_dim_int(b1_dim, bindings, name="B_dim1")
    b2 = _resolve_dim_int(b2_dim, bindings, name="B_dim2")

    a_idx = "((size_t)row * (size_t)A_dim2 + (size_t)k)" if not ta else "((size_t)k * (size_t)A_dim2 + (size_t)row)"
    b_idx = "((size_t)k * (size_t)B_dim2 + (size_t)col)" if not tb else "((size_t)col * (size_t)B_dim2 + (size_t)k)"
    out_idx = "((size_t)row * (size_t)N + (size_t)col)"
    cuda_src = f"""
#include <stdint.h>

#include "intentir_cuda_ops.cuh"

extern "C" __global__ __launch_bounds__({block_x * block_y}) void {intent.name}(
    const float* __restrict__ {a_name},
    const float* __restrict__ {b_name},
    float* __restrict__ {out_name},
    {b_param},
    {m_param},
    {n_param},
    {k_param}) {{
  {b_load}
  {m_load}
  {n_load}
  {k_load}
  const int b = (int)blockIdx.z;
  const int row = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
  const int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (b >= BATCH || row >= M || col >= N) return;
  const int A_dim1 = {a1};
  const int A_dim2 = {a2};
  const int B_dim1 = {b1};
  const int B_dim2 = {b2};
  const float* A_base = {a_name} + (size_t)b * (size_t)A_dim1 * (size_t)A_dim2;
  const float* B_base = {b_name} + (size_t)b * (size_t)B_dim1 * (size_t)B_dim2;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {{
    const float av = intentir_ldg_f32(A_base + {a_idx});
    const float bv = intentir_ldg_f32(B_base + {b_idx});
    acc += av * bv;
  }}
  {out_name}[(size_t)b * (size_t)M * (size_t)N + {out_idx}] = acc;
}}
""".lstrip()

    tensor_args = [a_name, b_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [a_name, b_name, out_name]
    for dim_name, is_tensor in (
        (str(batch_dim), b_is_tensor),
        (str(M_dim), m_is_tensor),
        (str(N_dim), n_is_tensor),
        (str(K_dim), k_is_tensor),
    ):
        if is_tensor:
            tensor_args.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
        arg_names.append(dim_name)
    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(grid_x, grid_y, grid_z), block=(block_x, block_y, 1), shared_mem=0)
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=[out_name],
        bindings=dict(bindings),
    )


def _kernel_baddbmm_3d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    """
    Pattern: batched addmm decomposition
      matmul + mul(alpha) + mul(beta on bias) + add (+ optional cast)
    with rank-3 tensors [B, M, N].
    """
    ops = list(intent.ops or [])
    if len(ops) not in {4, 5}:
        raise CudaLoweringError("baddbmm lowering expects 4 or 5 ops")
    if [o.op for o in ops[:4]] != ["matmul", "mul", "mul", "add"]:
        raise CudaLoweringError("baddbmm lowering expects matmul->mul->mul->add chain")
    mat, mul_a, mul_b, add0 = ops[:4]
    cast0 = ops[4] if len(ops) == 5 else None
    if cast0 is not None and cast0.op != "cast":
        raise CudaLoweringError("baddbmm optional 5th op must be cast")

    if len(mat.inputs) != 2:
        raise CudaLoweringError("baddbmm matmul expects 2 inputs")
    a_name = str(mat.inputs[0])
    b_name = str(mat.inputs[1])
    mm_out = str(mat.output)

    mul_a_in = [str(x) for x in mul_a.inputs]
    if mm_out in mul_a_in:
        alpha_name = mul_a_in[1] if mul_a_in[0] == mm_out else mul_a_in[0]
        scaled_mm = str(mul_a.output)
    else:
        raise CudaLoweringError("baddbmm first mul must consume matmul output")

    mul_b_in = [str(x) for x in mul_b.inputs]
    beta_name = None
    bias_name = None
    for x in mul_b_in:
        if _is_scalar_tensor(intent, x, dtype="f32"):
            beta_name = x
        else:
            bias_name = x
    if beta_name is None or bias_name is None:
        raise CudaLoweringError("baddbmm second mul must be bias * beta")
    scaled_bias = str(mul_b.output)

    add_in = [str(x) for x in add0.inputs]
    if len(add_in) != 2 or sorted(add_in) != sorted([scaled_mm, scaled_bias]):
        raise CudaLoweringError("baddbmm add must consume scaled matmul and scaled bias")
    add_out = str(add0.output)

    out_name = str(intent.outputs[0])
    if cast0 is not None:
        if len(cast0.inputs) != 1 or str(cast0.inputs[0]) != add_out or str(cast0.output) != out_name:
            raise CudaLoweringError("baddbmm cast must consume add output and produce final output")
    elif add_out != out_name:
        raise CudaLoweringError("baddbmm add output must be final output when cast is absent")

    a_shape = _shape_values(intent, a_name)
    b_shape = _shape_values(intent, b_name)
    bias_shape = _shape_values(intent, bias_name)
    out_shape = _shape_values(intent, out_name)
    if len(a_shape) != 3 or len(b_shape) != 3 or len(bias_shape) != 3 or len(out_shape) != 3:
        raise CudaLoweringError("baddbmm lowering expects rank-3 tensors")

    batch_dim, a1_dim, a2_dim = a_shape
    b_batch_dim, b1_dim, b2_dim = b_shape
    ta = bool((mat.attrs or {}).get("transpose_a", False))
    tb = bool((mat.attrs or {}).get("transpose_b", False))

    M_dim = a2_dim if ta else a1_dim
    K_dim = a1_dim if ta else a2_dim
    K2_dim = b2_dim if tb else b1_dim
    N_dim = b1_dim if tb else b2_dim

    if str(batch_dim) != str(b_batch_dim):
        raise CudaLoweringError("baddbmm batch mismatch between A and B")
    if str(K_dim) != str(K2_dim):
        raise CudaLoweringError("baddbmm K mismatch")
    if [str(x) for x in bias_shape] != [str(batch_dim), str(M_dim), str(N_dim)]:
        raise CudaLoweringError("baddbmm bias shape mismatch")
    if [str(x) for x in out_shape] != [str(batch_dim), str(M_dim), str(N_dim)]:
        raise CudaLoweringError("baddbmm output shape mismatch")

    BATCH = _resolve_dim_int(batch_dim, bindings, name="BATCH")
    M = _resolve_dim_int(M_dim, bindings, name="M")
    N = _resolve_dim_int(N_dim, bindings, name="N")
    K = _resolve_dim_int(K_dim, bindings, name="K")

    sched = intent.schedule or ScheduleSketch()
    block_y = _resolve_schedule_int(sched.tile_m, bindings, default=16)
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=16)
    if block_x <= 0:
        block_x = 16
    if block_y <= 0:
        block_y = 16
    block_x = max(8, min(32, int(block_x)))
    block_y = max(8, min(32, int(block_y)))
    if block_x * block_y > 1024:
        block_y = max(1, 1024 // block_x)

    grid_x = (N + block_x - 1) // block_x
    grid_y = (M + block_y - 1) // block_y
    grid_z = max(1, int(BATCH))

    b_is_tensor = _is_scalar_tensor(intent, str(batch_dim), dtype="i32")
    m_is_tensor = _is_scalar_tensor(intent, str(M_dim), dtype="i32")
    n_is_tensor = _is_scalar_tensor(intent, str(N_dim), dtype="i32")
    k_is_tensor = _is_scalar_tensor(intent, str(K_dim), dtype="i32")
    b_param = f"const int* {str(batch_dim)}_ptr" if b_is_tensor else "int BATCH_in"
    m_param = f"const int* {str(M_dim)}_ptr" if m_is_tensor else "int M_in"
    n_param = f"const int* {str(N_dim)}_ptr" if n_is_tensor else "int N_in"
    k_param = f"const int* {str(K_dim)}_ptr" if k_is_tensor else "int K_in"
    b_load = f"const int BATCH = {str(batch_dim)}_ptr ? {str(batch_dim)}_ptr[0] : 0;" if b_is_tensor else "const int BATCH = BATCH_in;"
    m_load = f"const int M = {str(M_dim)}_ptr ? {str(M_dim)}_ptr[0] : 0;" if m_is_tensor else "const int M = M_in;"
    n_load = f"const int N = {str(N_dim)}_ptr ? {str(N_dim)}_ptr[0] : 0;" if n_is_tensor else "const int N = N_in;"
    k_load = f"const int K = {str(K_dim)}_ptr ? {str(K_dim)}_ptr[0] : 0;" if k_is_tensor else "const int K = K_in;"

    if ta:
        a_idx = "((size_t)batch * (size_t)K * (size_t)M + (size_t)k * (size_t)M + (size_t)row)"
    else:
        a_idx = "((size_t)batch * (size_t)M * (size_t)K + (size_t)row * (size_t)K + (size_t)k)"
    if tb:
        b_idx = "((size_t)batch * (size_t)N * (size_t)K + (size_t)col * (size_t)K + (size_t)k)"
    else:
        b_idx = "((size_t)batch * (size_t)K * (size_t)N + (size_t)k * (size_t)N + (size_t)col)"
    out_idx = "((size_t)batch * (size_t)M * (size_t)N + (size_t)row * (size_t)N + (size_t)col)"

    alpha_ptr_name = f"{alpha_name}_ptr"
    beta_ptr_name = f"{beta_name}_ptr"
    cuda_src = f"""
#include <stdint.h>

#include "intentir_cuda_ops.cuh"

extern "C" __global__ __launch_bounds__({block_x * block_y}) void {intent.name}(
    const float* __restrict__ {a_name},
    const float* __restrict__ {b_name},
    const float* __restrict__ {bias_name},
    const float* __restrict__ {alpha_ptr_name},
    const float* __restrict__ {beta_ptr_name},
    float* __restrict__ {out_name},
    {b_param},
    {m_param},
    {n_param},
    {k_param}) {{
  {b_load}
  {m_load}
  {n_load}
  {k_load}
  const int batch = (int)blockIdx.z;
  const int row = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
  const int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (batch >= BATCH || row >= M || col >= N) return;
  const float alpha = {alpha_ptr_name} ? {alpha_ptr_name}[0] : 1.0f;
  const float beta = {beta_ptr_name} ? {beta_ptr_name}[0] : 1.0f;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {{
    const float av = intentir_ldg_f32({a_name} + {a_idx});
    const float bv = intentir_ldg_f32({b_name} + {b_idx});
    acc += av * bv;
  }}
  const float bias_val = intentir_ldg_f32({bias_name} + {out_idx});
  {out_name}[{out_idx}] = alpha * acc + beta * bias_val;
}}
""".lstrip()

    tensor_args = [a_name, b_name, bias_name, alpha_name, beta_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [a_name, b_name, bias_name, alpha_name, beta_name, out_name]
    for dim_name, is_tensor in (
        (str(batch_dim), b_is_tensor),
        (str(M_dim), m_is_tensor),
        (str(N_dim), n_is_tensor),
        (str(K_dim), k_is_tensor),
    ):
        if is_tensor:
            tensor_args.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
        arg_names.append(dim_name)

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(grid_x, grid_y, grid_z), block=(block_x, block_y, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_batch_norm2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    """
    Canonical batch_norm2d lowering for input shape [N, C, HW].
    Produces 5 outputs:
      output_1, mean, inv_std, running_mean_out, running_var_out
    """
    produced = {o.output for o in (intent.ops or [])}
    outs = [str(x) for x in (intent.outputs or [])]
    out_set = set(outs)
    required_outs = {"output_1", "mean", "inv_std", "running_mean_out", "running_var_out"}
    if not required_outs.issubset(out_set):
        raise CudaLoweringError("batch_norm lowering expects outputs: output_1/mean/inv_std/running_mean_out/running_var_out")

    inputs = [n for n in intent.tensors.keys() if n not in produced and n not in out_set]
    required_inputs = ["input", "weight", "bias", "running_mean", "running_var", "eps", "momentum"]
    for name in required_inputs:
        if name not in inputs:
            raise CudaLoweringError(f"batch_norm lowering missing required input: {name}")

    x_name = "input"
    w_name = "weight"
    b_name = "bias"
    rm_name = "running_mean"
    rv_name = "running_var"
    eps_name = "eps"
    momentum_name = "momentum"
    eps_ptr_name = f"{eps_name}_ptr"
    momentum_ptr_name = f"{momentum_name}_ptr"
    n_elements_name = "n_elements" if "n_elements" in inputs else None
    n_minus_1_name = "n_minus_1" if "n_minus_1" in inputs else None
    n_elements_ptr_name = f"{n_elements_name}_ptr" if n_elements_name is not None else None
    n_minus_1_ptr_name = f"{n_minus_1_name}_ptr" if n_minus_1_name is not None else None

    y_name = "output_1"
    mean_name = "mean"
    inv_std_name = "inv_std"
    rm_out_name = "running_mean_out"
    rv_out_name = "running_var_out"

    x_shape = _shape_values(intent, x_name)
    if len(x_shape) != 3:
        raise CudaLoweringError("batch_norm lowering expects rank-3 input [N,C,HW]")
    N_dim, C_dim, HW_dim = x_shape
    N = _resolve_dim_int(N_dim, bindings, name="N")
    C = _resolve_dim_int(C_dim, bindings, name="C")
    HW = _resolve_dim_int(HW_dim, bindings, name="HW")

    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    block_x = max(32, min(1024, int(block_x)))
    if (block_x & (block_x - 1)) != 0:
        block_x = 1 << int(block_x - 1).bit_length()

    n_is_tensor = _is_scalar_tensor(intent, str(N_dim), dtype="i32")
    c_is_tensor = _is_scalar_tensor(intent, str(C_dim), dtype="i32")
    hw_is_tensor = _is_scalar_tensor(intent, str(HW_dim), dtype="i32")
    n_param = f"const int* {str(N_dim)}_ptr" if n_is_tensor else "int N_in"
    c_param = f"const int* {str(C_dim)}_ptr" if c_is_tensor else "int C_in"
    hw_param = f"const int* {str(HW_dim)}_ptr" if hw_is_tensor else "int HW_in"
    n_load = f"const int N = {str(N_dim)}_ptr ? {str(N_dim)}_ptr[0] : 0;" if n_is_tensor else "const int N = N_in;"
    c_load = f"const int C = {str(C_dim)}_ptr ? {str(C_dim)}_ptr[0] : 0;" if c_is_tensor else "const int C = C_in;"
    hw_load = f"const int HW = {str(HW_dim)}_ptr ? {str(HW_dim)}_ptr[0] : 0;" if hw_is_tensor else "const int HW = HW_in;"

    n_elements_load = f"{n_elements_ptr_name}[0]" if n_elements_ptr_name is not None else "(float)(N * HW)"
    n_minus_1_load = f"{n_minus_1_ptr_name}[0]" if n_minus_1_ptr_name is not None else "(float)((N * HW) > 1 ? (N * HW - 1) : 1)"

    n_elements_param = f", const float* __restrict__ {n_elements_ptr_name}" if n_elements_ptr_name is not None else ""
    n_minus_1_param = f", const float* __restrict__ {n_minus_1_ptr_name}" if n_minus_1_ptr_name is not None else ""
    tensor_args = [x_name, w_name, b_name, rm_name, rv_name, eps_name, momentum_name]
    arg_names = [x_name, w_name, b_name, rm_name, rv_name, eps_name, momentum_name]
    if n_elements_name is not None:
        tensor_args.append(n_elements_name)
        arg_names.append(n_elements_name)
    if n_minus_1_name is not None:
        tensor_args.append(n_minus_1_name)
        arg_names.append(n_minus_1_name)

    tensor_args.extend([y_name, mean_name, inv_std_name, rm_out_name, rv_out_name])
    arg_names.extend([y_name, mean_name, inv_std_name, rm_out_name, rv_out_name])

    scalar_args: Dict[str, str] = {}
    for dim_name, is_tensor in ((str(N_dim), n_is_tensor), (str(C_dim), c_is_tensor), (str(HW_dim), hw_is_tensor)):
        if is_tensor:
            tensor_args.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
        arg_names.append(dim_name)

    cuda_src = f"""
#include <math.h>
#include <stdint.h>

#include "intentir_cuda_ops.cuh"
#include "kernels/reduce.cuh"

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(
    const float* __restrict__ {x_name},
    const float* __restrict__ {w_name},
    const float* __restrict__ {b_name},
    const float* __restrict__ {rm_name},
    const float* __restrict__ {rv_name},
    const float* __restrict__ {eps_ptr_name},
    const float* __restrict__ {momentum_ptr_name}{n_elements_param}{n_minus_1_param},
    float* __restrict__ {y_name},
    float* __restrict__ {mean_name},
    float* __restrict__ {inv_std_name},
    float* __restrict__ {rm_out_name},
    float* __restrict__ {rv_out_name},
    {n_param},
    {c_param},
    {hw_param}) {{
  {n_load}
  {c_load}
  {hw_load}
  const int ch = (int)blockIdx.x;
  if (ch >= C) return;
  constexpr int BLOCK_THREADS = {block_x};
  __shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red_sum;
  __shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red_sqs;
  __shared__ float s_mean;
  __shared__ float s_inv_std;
  __shared__ float s_w;
  __shared__ float s_b;
  const int nhw_i = N * HW;
  const float n_elements = fmaxf({n_elements_load}, 1.0f);
  const float n_minus_1 = fmaxf({n_minus_1_load}, 1.0f);
  float local_sum = 0.0f;
  float local_sqs = 0.0f;
  for (int idx = (int)threadIdx.x; idx < nhw_i; idx += (int)blockDim.x) {{
    const int n = idx / HW;
    const int hw = idx - n * HW;
    const size_t off = ((size_t)n * (size_t)C + (size_t)ch) * (size_t)HW + (size_t)hw;
    const float xv = intentir_ldg_f32({x_name} + off);
    local_sum += xv;
    local_sqs += xv * xv;
  }}
  const float sum = intentir_cuda::block_allreduce_sum<BLOCK_THREADS>(local_sum, &red_sum);
  const float sqs = intentir_cuda::block_allreduce_sum<BLOCK_THREADS>(local_sqs, &red_sqs);
  const float eps = {eps_ptr_name} ? {eps_ptr_name}[0] : 1e-5f;
  const float momentum = {momentum_ptr_name} ? {momentum_ptr_name}[0] : 0.1f;
  if ((int)threadIdx.x == 0) {{
    const float mu = sum / n_elements;
    float var = sqs / n_elements - mu * mu;
    if (var < 0.0f) var = 0.0f;
    const float inv = rsqrtf(var + eps);
    s_mean = mu;
    s_inv_std = inv;
    s_w = intentir_ldg_f32({w_name} + (size_t)ch);
    s_b = intentir_ldg_f32({b_name} + (size_t)ch);
    {mean_name}[(size_t)ch] = mu;
    {inv_std_name}[(size_t)ch] = inv;
    const float one_minus = 1.0f - momentum;
    {rm_out_name}[(size_t)ch] = one_minus * intentir_ldg_f32({rm_name} + (size_t)ch) + momentum * mu;
    const float unbiased_var = var * (n_elements / n_minus_1);
    {rv_out_name}[(size_t)ch] = one_minus * intentir_ldg_f32({rv_name} + (size_t)ch) + momentum * unbiased_var;
  }}
  __syncthreads();
  for (int idx = (int)threadIdx.x; idx < nhw_i; idx += (int)blockDim.x) {{
    const int n = idx / HW;
    const int hw = idx - n * HW;
    const size_t off = ((size_t)n * (size_t)C + (size_t)ch) * (size_t)HW + (size_t)hw;
    const float xv = intentir_ldg_f32({x_name} + off);
    const float yv = ((xv - s_mean) * s_inv_std) * s_w + s_b;
    {y_name}[off] = yv;
  }}
}}
""".lstrip()

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(max(1, int(C)), 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=[y_name, mean_name, inv_std_name, rm_out_name, rv_out_name],
        bindings=dict(bindings),
    )


def _kernel_allclose_2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    """
    Pattern: allclose decomposition:
      sub, abs, abs, mul, add, le, not, reduce_any, not -> scalar bool.
    """
    ops = list(intent.ops or [])
    expected = ["sub", "abs", "abs", "mul", "add", "le", "not", "reduce_any", "not"]
    if [o.op for o in ops] != expected:
        raise CudaLoweringError("allclose lowering expects canonical op chain")
    sub0, abs0, abs1, mul0, add0, le0, not0, red0, not1 = ops
    if len(sub0.inputs) != 2:
        raise CudaLoweringError("allclose sub expects 2 inputs")
    a_name = str(sub0.inputs[0])
    b_name = str(sub0.inputs[1])
    if len(abs0.inputs) != 1 or str(abs0.inputs[0]) != str(sub0.output):
        raise CudaLoweringError("allclose abs(diff) pattern mismatch")
    if len(abs1.inputs) != 1 or str(abs1.inputs[0]) != b_name:
        raise CudaLoweringError("allclose abs(b) pattern mismatch")
    if len(mul0.inputs) != 2:
        raise CudaLoweringError("allclose mul expects 2 inputs")
    rtol_name = None
    abs_b_tmp = str(abs1.output)
    for x in (str(mul0.inputs[0]), str(mul0.inputs[1])):
        if _is_scalar_tensor(intent, x, dtype="f32"):
            rtol_name = x
    if rtol_name is None:
        raise CudaLoweringError("allclose mul must consume scalar rtol")
    if abs_b_tmp not in {str(mul0.inputs[0]), str(mul0.inputs[1])}:
        raise CudaLoweringError("allclose mul must consume abs(B)")
    if len(add0.inputs) != 2:
        raise CudaLoweringError("allclose add expects 2 inputs")
    atol_name = None
    for x in (str(add0.inputs[0]), str(add0.inputs[1])):
        if _is_scalar_tensor(intent, x, dtype="f32"):
            atol_name = x
    if atol_name is None:
        raise CudaLoweringError("allclose add must consume scalar atol")
    if str(mul0.output) not in {str(add0.inputs[0]), str(add0.inputs[1])}:
        raise CudaLoweringError("allclose add must consume mul(rtol, abs(B))")
    if len(le0.inputs) != 2 or str(le0.inputs[0]) != str(abs0.output) or str(le0.inputs[1]) != str(add0.output):
        raise CudaLoweringError("allclose le pattern mismatch")
    if len(not0.inputs) != 1 or str(not0.inputs[0]) != str(le0.output):
        raise CudaLoweringError("allclose first not pattern mismatch")
    if len(red0.inputs) != 1 or str(red0.inputs[0]) != str(not0.output):
        raise CudaLoweringError("allclose reduce_any pattern mismatch")
    if len(not1.inputs) != 1 or str(not1.inputs[0]) != str(red0.output):
        raise CudaLoweringError("allclose final not pattern mismatch")

    out_name = str(intent.outputs[0])
    if str(not1.output) != out_name:
        raise CudaLoweringError("allclose final output mismatch")

    dims = (red0.attrs or {}).get("dims")
    if dims not in ([0, 1], [1, 0], (0, 1), (1, 0)):
        raise CudaLoweringError("allclose reduce_any expects dims=[0,1]")

    a_shape = _shape_values(intent, a_name)
    b_shape = _shape_values(intent, b_name)
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise CudaLoweringError("allclose expects rank-2 A/B")
    if str(a_shape[0]) != str(b_shape[0]) or str(a_shape[1]) != str(b_shape[1]):
        raise CudaLoweringError("allclose shape mismatch between A/B")
    M_dim, N_dim = a_shape
    M = _resolve_dim_int(M_dim, bindings, name="M")
    N = _resolve_dim_int(N_dim, bindings, name="N")

    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    if (block_x & (block_x - 1)) != 0:
        block_x = 1 << int(block_x - 1).bit_length()
    block_x = max(32, min(1024, int(block_x)))

    m_is_tensor = _is_scalar_tensor(intent, str(M_dim), dtype="i32")
    n_is_tensor = _is_scalar_tensor(intent, str(N_dim), dtype="i32")
    m_param = f"const int* {str(M_dim)}_ptr" if m_is_tensor else "int M_in"
    n_param = f"const int* {str(N_dim)}_ptr" if n_is_tensor else "int N_in"
    m_load = f"const int M = {str(M_dim)}_ptr ? {str(M_dim)}_ptr[0] : 0;" if m_is_tensor else "const int M = M_in;"
    n_load = f"const int N = {str(N_dim)}_ptr ? {str(N_dim)}_ptr[0] : 0;" if n_is_tensor else "const int N = N_in;"
    rtol_ptr_name = f"{rtol_name}_ptr"
    atol_ptr_name = f"{atol_name}_ptr"

    cuda_src = f"""
#include <math.h>
#include <stdint.h>

#include "intentir_cuda_ops.cuh"
#include "kernels/reduce.cuh"

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(
    const float* __restrict__ {a_name},
    const float* __restrict__ {b_name},
    const float* __restrict__ {rtol_ptr_name},
    const float* __restrict__ {atol_ptr_name},
    bool* __restrict__ {out_name},
    {m_param},
    {n_param}) {{
  {m_load}
  {n_load}
  constexpr int BLOCK_THREADS = {block_x};
  __shared__ intentir_cuda::BlockAllreduceI32<BLOCK_THREADS> red;
  const float rtol = {rtol_ptr_name} ? {rtol_ptr_name}[0] : 1.0e-5f;
  const float atol = {atol_ptr_name} ? {atol_ptr_name}[0] : 1.0e-8f;
  const int64_t total = (int64_t)M * (int64_t)N;
  int any_not_close = 0;
  for (int64_t idx = (int64_t)threadIdx.x; idx < total; idx += (int64_t)blockDim.x) {{
    const float av = intentir_ldg_f32({a_name} + (size_t)idx);
    const float bv = intentir_ldg_f32({b_name} + (size_t)idx);
    const float diff = fabsf(av - bv);
    const float tol = atol + rtol * fabsf(bv);
    any_not_close |= (diff > tol) ? 1 : 0;
  }}
  const int any = intentir_cuda::block_allreduce_max<BLOCK_THREADS>(any_not_close, &red);
  if ((int)threadIdx.x == 0) {out_name}[0] = (any == 0);
}}
""".lstrip()

    tensor_args = [a_name, b_name, rtol_name, atol_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [a_name, b_name, rtol_name, atol_name, out_name]
    for dim_name, is_tensor in ((str(M_dim), m_is_tensor), (str(N_dim), n_is_tensor)):
        if is_tensor:
            tensor_args.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
        arg_names.append(dim_name)
    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(1, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_gather2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    """
    Pattern: gather(inp[M,N], row_idx[...], col_idx[...]) -> out[...]
    MVP supports output rank 1 or 2 and flattens row/col/output indexing.
    """
    gather_ops = [o for o in (intent.ops or []) if o.op == "gather"]
    if len(gather_ops) != 1:
        raise CudaLoweringError("gather2d lowering expects exactly one gather op")
    op = gather_ops[0]
    if len(op.inputs) != 3:
        raise CudaLoweringError("gather expects inputs (inp, row_idx, col_idx)")
    inp_name, row_name, col_name = (str(x) for x in op.inputs)
    out_name = str(op.output)
    other = float((op.attrs or {}).get("other_value", 0.0))

    M = _as_int(bindings.get("M"), name="M")
    N = _as_int(bindings.get("N"), name="N")
    out_shape = _shape_values(intent, out_name)
    if len(out_shape) == 1:
        total = _resolve_dim_int(out_shape[0], bindings, name="L")
    elif len(out_shape) == 2:
        d0 = _resolve_dim_int(out_shape[0], bindings, name="out_dim0")
        d1 = _resolve_dim_int(out_shape[1], bindings, name="out_dim1")
        total = int(d0) * int(d1)
    else:
        raise CudaLoweringError("gather2d MVP supports output rank<=2")
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    grid_x = (total + block_x - 1) // block_x

    other_lit = _c_scalar_literal("f32", other)
    cuda_src = f"""
#include <stdint.h>
extern "C" __global__ void {intent.name}(
    const float* __restrict__ {inp_name},
    const int* __restrict__ {row_name},
    const int* __restrict__ {col_name},
    float* __restrict__ {out_name},
    int M, int N, int TOTAL) {{
  const int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (tid >= TOTAL) return;
  const int r = {row_name}[tid];
  const int c = {col_name}[tid];
  float v = {other_lit};
  if ((unsigned)r < (unsigned)M && (unsigned)c < (unsigned)N) {{
    v = {inp_name}[(size_t)r * (size_t)N + (size_t)c];
  }}
  {out_name}[tid] = v;
}}
""".lstrip()

    lowered_bindings = dict(bindings)
    lowered_bindings["TOTAL"] = int(total)
    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[inp_name, row_name, col_name, out_name],
        scalar_args={"M": "i32", "N": "i32", "TOTAL": "i32"},
        arg_names=[inp_name, row_name, col_name, out_name, "M", "N", "TOTAL"],
    )
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=lowered_bindings)


def _kernel_diag_2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    ops = list(intent.ops or [])
    if len(ops) != 2 or [o.op for o in ops] != ["iota", "gather"]:
        raise CudaLoweringError("diag lowering expects iota->gather pattern")
    g = ops[1]
    if len(g.inputs) != 3:
        raise CudaLoweringError("diag gather expects inputs [data, idx, idx]")
    data_name = str(g.inputs[0])
    idx0_name = str(g.inputs[1])
    idx1_name = str(g.inputs[2])
    out_name = str(g.output)
    if idx0_name != idx1_name:
        raise CudaLoweringError("diag gather requires row/col indices to match")
    data_shape = _shape_values(intent, data_name)
    out_shape = _shape_values(intent, out_name)
    if len(data_shape) != 2 or len(out_shape) != 1:
        raise CudaLoweringError("diag lowering expects data rank-2 and output rank-1")
    M = _resolve_dim_int(data_shape[0], bindings, name="M")
    N = _resolve_dim_int(data_shape[1], bindings, name="N")
    D = _resolve_dim_int(out_shape[0], bindings, name="D")

    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    block_x = max(32, min(1024, int(block_x) if block_x > 0 else 256))
    grid_x = (D + block_x - 1) // block_x

    cuda_src = f"""
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(
    const float* __restrict__ {data_name},
    float* __restrict__ {out_name},
    int M, int N, int D) {{
  const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (i >= D) return;
  int r = i;
  int c = r;
  if ((unsigned)r < (unsigned)M && (unsigned)c < (unsigned)N) {{
    {out_name}[i] = {data_name}[(size_t)r * (size_t)N + (size_t)c];
  }} else {{
    {out_name}[i] = 0.0f;
  }}
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[data_name, out_name],
        scalar_args={"M": "i32", "N": "i32", "D": "i32"},
        arg_names=[data_name, out_name, "M", "N", "D"],
    )
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update({"M": int(M), "N": int(N), "D": int(D)})
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=[out_name],
        bindings=lowered_bindings,
    )


def _kernel_count_nonzero_2d_i64(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    ops = list(intent.ops or [])
    if len(ops) != 4 or [o.op for o in ops] != ["const", "ne", "cast", "reduce_sum"]:
        raise CudaLoweringError("count_nonzero lowering expects const->ne->cast->reduce_sum pattern")
    const0, ne0, cast0, red0 = ops
    const_name = str(const0.output)
    ne_in = [str(x) for x in ne0.inputs]
    if len(ne_in) != 2 or const_name not in ne_in:
        raise CudaLoweringError("count_nonzero ne must compare input tensor with const zero")
    inp_name = ne_in[1] if ne_in[0] == const_name else ne_in[0]
    if str(cast0.inputs[0]) != str(ne0.output):
        raise CudaLoweringError("count_nonzero cast must consume ne output")
    if str(red0.inputs[0]) != str(cast0.output):
        raise CudaLoweringError("count_nonzero reduce_sum must consume cast output")
    out_name = str(intent.outputs[0])
    in_shape = _shape_values(intent, inp_name)
    out_shape = _shape_values(intent, out_name)
    if len(in_shape) != 2:
        raise CudaLoweringError("count_nonzero lowering currently supports rank-2 input")
    if len(out_shape) > 1 or (len(out_shape) == 1 and _resolve_dim_int(out_shape[0], bindings, name="OUT") != 1):
        raise CudaLoweringError("count_nonzero lowering expects scalar output")
    out_dt = str(intent.tensors[out_name].dtype)
    if out_dt not in {"i64", "i32"}:
        raise CudaLoweringError("count_nonzero lowering supports i64/i32 output")
    M = _resolve_dim_int(in_shape[0], bindings, name="M")
    N = _resolve_dim_int(in_shape[1], bindings, name="N")
    zero = float((const0.attrs or {}).get("value", 0.0))
    out_ct = "int64_t" if out_dt == "i64" else "int32_t"

    cuda_src = f"""
#include <stdint.h>

extern "C" __global__ void {intent.name}(
    const float* __restrict__ {inp_name},
    {out_ct}* __restrict__ {out_name},
    int M, int N) {{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int64_t cnt = 0;
  for (int i = 0; i < M; ++i) {{
    for (int j = 0; j < N; ++j) {{
      const float v = {inp_name}[(size_t)i * (size_t)N + (size_t)j];
      if (v != (float){zero}) cnt += 1;
    }}
  }}
  {out_name}[0] = ({out_ct})cnt;
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[inp_name, out_name],
        scalar_args={"M": "i32", "N": "i32"},
        arg_names=[inp_name, out_name, "M", "N"],
    )
    launch = CudaLaunch(grid=(1, 1, 1), block=(1, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update({"M": int(M), "N": int(N)})
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=[out_name],
        bindings=lowered_bindings,
    )


def _kernel_cumsum_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "cumsum":
        raise CudaLoweringError("cumsum lowering expects a single cumsum op")
    op = intent.ops[0]
    if len(op.inputs) != 1:
        raise CudaLoweringError("cumsum expects one input")
    inp_name = str(op.inputs[0])
    out_name = str(op.output)
    in_shape = _shape_values(intent, inp_name)
    out_shape = _shape_values(intent, out_name)
    if [str(x) for x in in_shape] != [str(x) for x in out_shape]:
        raise CudaLoweringError("cumsum input/output shape mismatch")
    axis = int((op.attrs or {}).get("axis", 0))

    if len(in_shape) == 1:
        D = _resolve_dim_int(in_shape[0], bindings, name="D")
        if axis < 0:
            axis += 1
        if axis != 0:
            raise CudaLoweringError("cumsum rank-1 supports axis=0 only")
        cuda_src = f"""
#include <stdint.h>

extern "C" __global__ void {intent.name}(const float* __restrict__ {inp_name}, float* __restrict__ {out_name}, int D) {{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float acc = 0.0f;
  for (int i = 0; i < D; ++i) {{
    acc += {inp_name}[i];
    {out_name}[i] = acc;
  }}
}}
""".lstrip()
        io_spec = _io_spec_from_args(intent, tensor_args=[inp_name, out_name], scalar_args={"D": "i32"}, arg_names=[inp_name, out_name, "D"])
        launch = CudaLaunch(grid=(1, 1, 1), block=(1, 1, 1), shared_mem=0)
        lowered_bindings = dict(bindings)
        lowered_bindings["D"] = int(D)
        return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=lowered_bindings)

    if len(in_shape) == 2:
        M = _resolve_dim_int(in_shape[0], bindings, name="M")
        N = _resolve_dim_int(in_shape[1], bindings, name="N")
        if axis < 0:
            axis += 2
        if axis not in {0, 1}:
            raise CudaLoweringError("cumsum rank-2 supports axis=0/1 only")
        cuda_src = f"""
#include <stdint.h>

extern "C" __global__ void {intent.name}(const float* __restrict__ {inp_name}, float* __restrict__ {out_name}, int M, int N) {{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if ({axis} == 0) {{
    for (int j = 0; j < N; ++j) {{
      float acc = 0.0f;
      for (int i = 0; i < M; ++i) {{
        const int idx = i * N + j;
        acc += {inp_name}[idx];
        {out_name}[idx] = acc;
      }}
    }}
  }} else {{
    for (int i = 0; i < M; ++i) {{
      float acc = 0.0f;
      for (int j = 0; j < N; ++j) {{
        const int idx = i * N + j;
        acc += {inp_name}[idx];
        {out_name}[idx] = acc;
      }}
    }}
  }}
}}
""".lstrip()
        io_spec = _io_spec_from_args(
            intent,
            tensor_args=[inp_name, out_name],
            scalar_args={"M": "i32", "N": "i32"},
            arg_names=[inp_name, out_name, "M", "N"],
        )
        launch = CudaLaunch(grid=(1, 1, 1), block=(1, 1, 1), shared_mem=0)
        lowered_bindings = dict(bindings)
        lowered_bindings.update({"M": int(M), "N": int(N)})
        return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=lowered_bindings)

    raise CudaLoweringError("cumsum lowering currently supports rank-1/2 only")


def _kernel_cumext_1d_f32(intent: IntentFunction, bindings: Dict[str, int], *, is_max: bool) -> CudaLoweredKernel:
    opname = "cummax" if is_max else "cummin"
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != opname:
        raise CudaLoweringError(f"{opname} lowering expects a single {opname} op")
    op = intent.ops[0]
    if len(op.inputs) != 1:
        raise CudaLoweringError(f"{opname} expects one input")
    inp_name = str(op.inputs[0])
    out_name = str(op.output)
    in_shape = _shape_values(intent, inp_name)
    out_shape = _shape_values(intent, out_name)
    if len(in_shape) != 1 or len(out_shape) != 1:
        raise CudaLoweringError(f"{opname} lowering currently supports rank-1 only")
    if str(in_shape[0]) != str(out_shape[0]):
        raise CudaLoweringError(f"{opname} input/output shape mismatch")
    axis = int((op.attrs or {}).get("axis", 0))
    if axis < 0:
        axis += 1
    if axis != 0:
        raise CudaLoweringError(f"{opname} rank-1 supports axis=0 only")
    D = _resolve_dim_int(in_shape[0], bindings, name="D")
    fun = "fmaxf" if is_max else "fminf"
    cuda_src = f"""
#include <math.h>
#include <stdint.h>

extern "C" __global__ void {intent.name}(const float* __restrict__ {inp_name}, float* __restrict__ {out_name}, int D) {{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (D <= 0) return;
  float best = {inp_name}[0];
  {out_name}[0] = best;
  for (int i = 1; i < D; ++i) {{
    best = {fun}(best, {inp_name}[i]);
    {out_name}[i] = best;
  }}
}}
""".lstrip()
    io_spec = _io_spec_from_args(intent, tensor_args=[inp_name, out_name], scalar_args={"D": "i32"}, arg_names=[inp_name, out_name, "D"])
    launch = CudaLaunch(grid=(1, 1, 1), block=(1, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings["D"] = int(D)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=lowered_bindings)


def _kernel_concat_2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "concat":
        raise CudaLoweringError("concat lowering expects a single concat op")
    op = intent.ops[0]
    if len(op.inputs) != 2:
        raise CudaLoweringError("concat lowering currently supports exactly 2 inputs")
    a_name = str(op.inputs[0])
    b_name = str(op.inputs[1])
    out_name = str(op.output)
    a_shape = _shape_values(intent, a_name)
    b_shape = _shape_values(intent, b_name)
    out_shape = _shape_values(intent, out_name)
    if len(a_shape) != 2 or len(b_shape) != 2 or len(out_shape) != 2:
        raise CudaLoweringError("concat lowering currently supports rank-2 tensors")

    axis = int((op.attrs or {}).get("axis", 0))
    if axis < 0:
        axis += 2
    if axis not in {0, 1}:
        raise CudaLoweringError("concat lowering currently supports axis 0/1")

    M0 = _resolve_dim_int(a_shape[0], bindings, name="M0")
    N0 = _resolve_dim_int(a_shape[1], bindings, name="N0")
    M1 = _resolve_dim_int(b_shape[0], bindings, name="M1")
    N1 = _resolve_dim_int(b_shape[1], bindings, name="N1")
    MO = _resolve_dim_int(out_shape[0], bindings, name="MO")
    NO = _resolve_dim_int(out_shape[1], bindings, name="NO")
    if axis == 1:
        if M0 != M1 or MO != M0 or NO != (N0 + N1):
            raise CudaLoweringError("concat axis=1 shape mismatch")
    else:
        if N0 != N1 or NO != N0 or MO != (M0 + M1):
            raise CudaLoweringError("concat axis=0 shape mismatch")

    sched = intent.schedule or ScheduleSketch()
    block_y = _resolve_schedule_int(sched.tile_m, bindings, default=16)
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=16)
    block_x = max(8, min(32, int(block_x) if block_x > 0 else 16))
    block_y = max(8, min(32, int(block_y) if block_y > 0 else 16))
    if block_x * block_y > 1024:
        block_y = max(1, 1024 // block_x)
    grid_x = (NO + block_x - 1) // block_x
    grid_y = (MO + block_y - 1) // block_y

    cuda_src = f"""
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x * block_y}) void {intent.name}(
    const float* __restrict__ {a_name},
    const float* __restrict__ {b_name},
    float* __restrict__ {out_name},
    int M0, int N0, int M1, int N1, int MO, int NO) {{
  const int row = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
  const int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (row >= MO || col >= NO) return;
  float v = 0.0f;
  if ({axis} == 1) {{
    if (col < N0) v = {a_name}[(size_t)row * (size_t)N0 + (size_t)col];
    else v = {b_name}[(size_t)row * (size_t)N1 + (size_t)(col - N0)];
  }} else {{
    if (row < M0) v = {a_name}[(size_t)row * (size_t)N0 + (size_t)col];
    else v = {b_name}[(size_t)(row - M0) * (size_t)N1 + (size_t)col];
  }}
  {out_name}[(size_t)row * (size_t)NO + (size_t)col] = v;
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[a_name, b_name, out_name],
        scalar_args={"M0": "i32", "N0": "i32", "M1": "i32", "N1": "i32", "MO": "i32", "NO": "i32"},
        arg_names=[a_name, b_name, out_name, "M0", "N0", "M1", "N1", "MO", "NO"],
    )
    launch = CudaLaunch(grid=(grid_x, grid_y, 1), block=(block_x, block_y, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update({"M0": int(M0), "N0": int(N0), "M1": int(M1), "N1": int(N1), "MO": int(MO), "NO": int(NO)})
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=[out_name],
        bindings=lowered_bindings,
    )


def _kernel_pad_2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "pad":
        raise CudaLoweringError("pad lowering expects a single pad op")
    op = intent.ops[0]
    if len(op.inputs) != 1:
        raise CudaLoweringError("pad lowering expects exactly 1 input")
    inp_name = str(op.inputs[0])
    out_name = str(op.output)
    in_shape = _shape_values(intent, inp_name)
    out_shape = _shape_values(intent, out_name)
    if len(in_shape) != 2 or len(out_shape) != 2:
        raise CudaLoweringError("pad lowering currently supports rank-2 tensors")
    mode = str((op.attrs or {}).get("mode", "constant"))
    if mode != "constant":
        raise CudaLoweringError("pad lowering currently supports mode=constant")

    pad_attr = (op.attrs or {}).get("pad_width")
    if isinstance(pad_attr, dict):
        pad_attr = pad_attr.get("pairs")
    if not (isinstance(pad_attr, list) and len(pad_attr) == 2):
        raise CudaLoweringError("pad lowering expects attrs.pad_width with 2 pairs")
    try:
        pad_top = int(pad_attr[0][0])
        pad_bottom = int(pad_attr[0][1])
        pad_left = int(pad_attr[1][0])
        pad_right = int(pad_attr[1][1])
    except Exception as exc:
        raise CudaLoweringError("pad lowering expects integer pad_width pairs") from exc
    pad_value = float((op.attrs or {}).get("value", 0.0))

    M = _resolve_dim_int(in_shape[0], bindings, name="M")
    N = _resolve_dim_int(in_shape[1], bindings, name="N")
    MO = _resolve_dim_int(out_shape[0], bindings, name="MO")
    NO = _resolve_dim_int(out_shape[1], bindings, name="NO")
    if MO != M + pad_top + pad_bottom or NO != N + pad_left + pad_right:
        raise CudaLoweringError("pad output shape mismatch with pad_width")

    sched = intent.schedule or ScheduleSketch()
    block_y = _resolve_schedule_int(sched.tile_m, bindings, default=16)
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=16)
    block_x = max(8, min(32, int(block_x) if block_x > 0 else 16))
    block_y = max(8, min(32, int(block_y) if block_y > 0 else 16))
    if block_x * block_y > 1024:
        block_y = max(1, 1024 // block_x)
    grid_x = (NO + block_x - 1) // block_x
    grid_y = (MO + block_y - 1) // block_y

    cuda_src = f"""
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x * block_y}) void {intent.name}(
    const float* __restrict__ {inp_name},
    float* __restrict__ {out_name},
    int M, int N, int MO, int NO) {{
  const int row = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
  const int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (row >= MO || col >= NO) return;
  const int src_row = row - {pad_top};
  const int src_col = col - {pad_left};
  float v = (float){pad_value};
  if ((unsigned)src_row < (unsigned)M && (unsigned)src_col < (unsigned)N) {{
    v = {inp_name}[(size_t)src_row * (size_t)N + (size_t)src_col];
  }}
  {out_name}[(size_t)row * (size_t)NO + (size_t)col] = v;
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[inp_name, out_name],
        scalar_args={"M": "i32", "N": "i32", "MO": "i32", "NO": "i32"},
        arg_names=[inp_name, out_name, "M", "N", "MO", "NO"],
    )
    launch = CudaLaunch(grid=(grid_x, grid_y, 1), block=(block_x, block_y, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update({"M": int(M), "N": int(N), "MO": int(MO), "NO": int(NO)})
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=[out_name],
        bindings=lowered_bindings,
    )


def _kernel_arg_reduce_2d_axis1_i32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op not in {"argmax", "argmin"}:
        raise CudaLoweringError("arg-reduce lowering expects a single argmax/argmin op")
    op = intent.ops[0]
    if len(op.inputs) != 1:
        raise CudaLoweringError(f"{op.op} expects 1 input")
    inp_name = str(op.inputs[0])
    out_name = str(op.output)
    in_shape = _shape_values(intent, inp_name)
    out_shape = _shape_values(intent, out_name)
    if len(in_shape) != 2 or len(out_shape) != 1:
        raise CudaLoweringError(f"{op.op} MVP supports rank-2 input and rank-1 output")
    axis = op.attrs.get("axis", op.attrs.get("dim", -1)) if op.attrs else -1
    try:
        axis = int(axis)
    except Exception:
        axis = -1
    if axis < 0:
        axis += 2
    if axis != 1:
        raise CudaLoweringError(f"{op.op} MVP supports only axis=1")

    M = _resolve_dim_int(in_shape[0], bindings, name="M")
    N = _resolve_dim_int(in_shape[1], bindings, name="N")
    if _resolve_dim_int(out_shape[0], bindings, name="OUT_M") != M:
        raise CudaLoweringError(f"{op.op} output shape mismatch")

    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    if (block_x & (block_x - 1)) != 0:
        block_x = 1 << int(block_x - 1).bit_length()
    block_x = max(32, min(1024, int(block_x)))
    cmp = ">" if op.op == "argmax" else "<"

    m_is_tensor = _is_scalar_tensor(intent, "M", dtype="i32")
    n_is_tensor = _is_scalar_tensor(intent, "N", dtype="i32")
    m_param = "const int* M_ptr" if m_is_tensor else "int M"
    n_param = "const int* N_ptr" if n_is_tensor else "int N"
    m_load = "const int M = M_ptr ? M_ptr[0] : 0;" if m_is_tensor else ""
    n_load = "const int N = N_ptr ? N_ptr[0] : 0;" if n_is_tensor else ""

    cuda_src = f"""
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(const float* __restrict__ {inp_name}, int* __restrict__ {out_name}, {m_param}, {n_param}) {{
  {m_load}
  {n_load}
  const int row = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (row >= M) return;
  int best_idx = 0;
  float best_val = {inp_name}[(size_t)row * (size_t)N];
  for (int col = 1; col < N; ++col) {{
    const float v = {inp_name}[(size_t)row * (size_t)N + (size_t)col];
    if (v {cmp} best_val) {{
      best_val = v;
      best_idx = col;
    }}
  }}
  {out_name}[row] = best_idx;
}}
""".lstrip()

    tensor_args = [inp_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [inp_name, out_name]
    for dim_name in ["M", "N"]:
        if _is_scalar_tensor(intent, dim_name, dtype="i32"):
            tensor_args.append(dim_name)
            arg_names.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
            arg_names.append(dim_name)

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(M, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.setdefault("M", int(M))
    lowered_bindings.setdefault("N", int(N))
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=lowered_bindings)


def _kernel_conv1d_ncl_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "conv1d":
        raise CudaLoweringError("conv1d lowering expects a single conv1d op")
    op = intent.ops[0]
    if len(op.inputs) != 3:
        raise CudaLoweringError("conv1d lowering expects inputs [input, weight, bias]")
    inp_name = str(op.inputs[0])
    weight_name = str(op.inputs[1])
    bias_name = str(op.inputs[2])
    out_name = str(op.output)

    in_shape = _shape_values(intent, inp_name)
    w_shape = _shape_values(intent, weight_name)
    b_shape = _shape_values(intent, bias_name)
    out_shape = _shape_values(intent, out_name)
    if len(in_shape) != 3 or len(w_shape) != 3 or len(b_shape) != 1 or len(out_shape) != 3:
        raise CudaLoweringError("conv1d lowering expects input[N,C,L], weight[CO,C_PER_G,K], bias[CO], out[N,CO,OL]")

    N_dim, C_in_dim, L_dim = in_shape
    C_out_dim, C_per_g_dim, K_dim = w_shape
    if str(b_shape[0]) != str(C_out_dim):
        raise CudaLoweringError("conv1d bias shape mismatch")
    if str(out_shape[0]) != str(N_dim) or str(out_shape[1]) != str(C_out_dim):
        raise CudaLoweringError("conv1d output N/CO mismatch")

    stride = int((op.attrs or {}).get("stride", 1))
    padding = int((op.attrs or {}).get("padding", 0))
    dilation = int((op.attrs or {}).get("dilation", 1))
    groups = int((op.attrs or {}).get("groups", 1))
    if stride <= 0 or dilation <= 0 or groups <= 0:
        raise CudaLoweringError("conv1d attrs stride/dilation/groups must be positive")

    N = _resolve_dim_int(N_dim, bindings, name="N")
    C_IN = _resolve_dim_int(C_in_dim, bindings, name="C_IN")
    L = _resolve_dim_int(L_dim, bindings, name="L")
    C_OUT = _resolve_dim_int(C_out_dim, bindings, name="C_OUT")
    C_PER_G = _resolve_dim_int(C_per_g_dim, bindings, name="C_PER_G")
    K = _resolve_dim_int(K_dim, bindings, name="K")
    OL = _resolve_dim_int(out_shape[2], bindings, name="OL")
    if C_IN != C_PER_G * groups:
        raise CudaLoweringError("conv1d channel/group mismatch: C_IN must equal C_PER_G * groups")
    if C_OUT % groups != 0:
        raise CudaLoweringError("conv1d C_OUT must be divisible by groups")
    expected_ol = (L + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    if expected_ol != OL:
        raise CudaLoweringError(f"conv1d output length mismatch: expected {expected_ol}, got {OL}")

    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=64)
    block_y = _resolve_schedule_int(sched.tile_m, bindings, default=4)
    block_x = max(16, min(256, int(block_x) if block_x > 0 else 64))
    block_y = max(1, min(8, int(block_y) if block_y > 0 else 4))
    if block_x * block_y > 1024:
        block_y = max(1, 1024 // block_x)
    grid_x = (OL + block_x - 1) // block_x
    grid_y = (C_OUT + block_y - 1) // block_y
    grid_z = max(1, int(N))

    cuda_src = f"""
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x * block_y}) void {intent.name}(
    const float* __restrict__ {inp_name},
    const float* __restrict__ {weight_name},
    const float* __restrict__ {bias_name},
    float* __restrict__ {out_name},
    int N, int C_IN, int L, int C_OUT, int C_PER_G, int K, int OL) {{
  const int ol = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  const int co = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
  const int n = (int)blockIdx.z;
  if (n >= N || co >= C_OUT || ol >= OL) return;
  const int co_per_g = C_OUT / {groups};
  const int g = co / co_per_g;
  const int c_start = g * C_PER_G;
  float acc = {bias_name}[(size_t)co];
  for (int ci = 0; ci < C_PER_G; ++ci) {{
    const int c = c_start + ci;
    for (int k = 0; k < K; ++k) {{
      const int li = ol * {stride} - {padding} + k * {dilation};
      if ((unsigned)li < (unsigned)L) {{
        const size_t x_idx = ((size_t)n * (size_t)C_IN + (size_t)c) * (size_t)L + (size_t)li;
        const size_t w_idx = ((size_t)co * (size_t)C_PER_G + (size_t)ci) * (size_t)K + (size_t)k;
        acc += {inp_name}[x_idx] * {weight_name}[w_idx];
      }}
    }}
  }}
  const size_t y_idx = ((size_t)n * (size_t)C_OUT + (size_t)co) * (size_t)OL + (size_t)ol;
  {out_name}[y_idx] = acc;
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[inp_name, weight_name, bias_name, out_name],
        scalar_args={"N": "i32", "C_IN": "i32", "L": "i32", "C_OUT": "i32", "C_PER_G": "i32", "K": "i32", "OL": "i32"},
        arg_names=[inp_name, weight_name, bias_name, out_name, "N", "C_IN", "L", "C_OUT", "C_PER_G", "K", "OL"],
    )
    launch = CudaLaunch(grid=(grid_x, grid_y, grid_z), block=(block_x, block_y, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update({"N": int(N), "C_IN": int(C_IN), "L": int(L), "C_OUT": int(C_OUT), "C_PER_G": int(C_PER_G), "K": int(K), "OL": int(OL)})
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=[out_name],
        bindings=lowered_bindings,
    )


def _kernel_conv2d_nchw_impl_f32(
    intent: IntentFunction,
    bindings: Dict[str, int],
    *,
    inp_name: str,
    weight_name: str,
    out_name: str,
    attrs: Mapping[str, Any],
    bias_name: str | None,
) -> CudaLoweredKernel:
    in_shape = _shape_values(intent, inp_name)
    w_shape = _shape_values(intent, weight_name)
    out_shape = _shape_values(intent, out_name)
    if len(in_shape) != 4 or len(w_shape) != 4 or len(out_shape) != 4:
        raise CudaLoweringError("conv2d lowering expects input[N,C,H,W], weight[CO,C_PER_G,KH,KW], out[N,CO,OH,OW]")
    if bias_name is not None:
        b_shape = _shape_values(intent, bias_name)
        if len(b_shape) != 1:
            raise CudaLoweringError("conv2d bias must be rank-1")
    else:
        b_shape = []

    N_dim, C_in_dim, H_dim, W_dim = in_shape
    C_out_dim, C_per_g_dim, KH_dim, KW_dim = w_shape
    ON_dim, OC_dim, OH_dim, OW_dim = out_shape
    if str(ON_dim) != str(N_dim) or str(OC_dim) != str(C_out_dim):
        raise CudaLoweringError("conv2d output N/C mismatch")
    if bias_name is not None and str(b_shape[0]) != str(C_out_dim):
        raise CudaLoweringError("conv2d bias shape mismatch")

    sh, sw = _resolve_attr_tuple(attrs.get("stride"), bindings, name="conv2d.stride", rank=2, default=1)
    ph, pw = _resolve_attr_tuple(attrs.get("padding"), bindings, name="conv2d.padding", rank=2, default=0)
    dh, dw = _resolve_attr_tuple(attrs.get("dilation"), bindings, name="conv2d.dilation", rank=2, default=1)
    groups = _resolve_attr_int(attrs.get("groups", 1), bindings, name="conv2d.groups")
    if sh <= 0 or sw <= 0 or dh <= 0 or dw <= 0 or groups <= 0:
        raise CudaLoweringError("conv2d attrs stride/dilation/groups must be positive")

    N = _resolve_dim_int(N_dim, bindings, name="N")
    C_IN_TOTAL = _resolve_dim_int(C_in_dim, bindings, name="C_IN_TOTAL")
    H = _resolve_dim_int(H_dim, bindings, name="H")
    W = _resolve_dim_int(W_dim, bindings, name="W")
    C_OUT = _resolve_dim_int(C_out_dim, bindings, name="C_OUT")
    C_PER_G = _resolve_dim_int(C_per_g_dim, bindings, name="C_PER_G")
    KH = _resolve_dim_int(KH_dim, bindings, name="KH")
    KW = _resolve_dim_int(KW_dim, bindings, name="KW")
    OH = _resolve_dim_int(OH_dim, bindings, name="OH")
    OW = _resolve_dim_int(OW_dim, bindings, name="OW")
    if C_IN_TOTAL != C_PER_G * groups:
        raise CudaLoweringError("conv2d channel/group mismatch: C_IN_TOTAL must equal C_PER_G * groups")
    if C_OUT % groups != 0:
        raise CudaLoweringError("conv2d C_OUT must be divisible by groups")
    expected_oh = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    expected_ow = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    if expected_oh != OH or expected_ow != OW:
        raise CudaLoweringError(f"conv2d output shape mismatch: expected ({expected_oh}, {expected_ow}), got ({OH}, {OW})")

    total = int(N) * int(C_OUT) * int(OH) * int(OW)
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=128)
    block_x = max(32, min(1024, int(block_x) if block_x > 0 else 128))
    grid_x = (total + block_x - 1) // block_x

    bias_param = f", const float* __restrict__ {bias_name}" if bias_name is not None else ""
    bias_init = f"float acc = {bias_name}[(size_t)co];" if bias_name is not None else "float acc = 0.0f;"
    cuda_src = f"""
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(
    const float* __restrict__ {inp_name},
    const float* __restrict__ {weight_name}{bias_param},
    float* __restrict__ {out_name},
    int N, int C_IN_TOTAL, int H, int W, int C_OUT, int C_PER_G, int KH, int KW, int OH, int OW) {{
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  const int64_t total = (int64_t)N * (int64_t)C_OUT * (int64_t)OH * (int64_t)OW;
  if (tid >= total) return;
  int64_t t = tid;
  const int ow = (int)(t % OW); t /= OW;
  const int oh = (int)(t % OH); t /= OH;
  const int co = (int)(t % C_OUT); t /= C_OUT;
  const int n = (int)t;
  const int co_per_g = C_OUT / {groups};
  const int g = co / co_per_g;
  const int c_start = g * C_PER_G;
  {bias_init}
  for (int ci = 0; ci < C_PER_G; ++ci) {{
    const int c = c_start + ci;
    for (int kh = 0; kh < KH; ++kh) {{
      const int ih = oh * {sh} - {ph} + kh * {dh};
      if ((unsigned)ih >= (unsigned)H) continue;
      for (int kw = 0; kw < KW; ++kw) {{
        const int iw = ow * {sw} - {pw} + kw * {dw};
        if ((unsigned)iw >= (unsigned)W) continue;
        const size_t x_idx = (((size_t)n * (size_t)C_IN_TOTAL + (size_t)c) * (size_t)H + (size_t)ih) * (size_t)W + (size_t)iw;
        const size_t w_idx = (((size_t)co * (size_t)C_PER_G + (size_t)ci) * (size_t)KH + (size_t)kh) * (size_t)KW + (size_t)kw;
        acc += {inp_name}[x_idx] * {weight_name}[w_idx];
      }}
    }}
  }}
  {out_name}[tid] = acc;
}}
""".lstrip()

    tensor_args = [inp_name, weight_name]
    arg_names = [inp_name, weight_name]
    if bias_name is not None:
        tensor_args.append(bias_name)
        arg_names.append(bias_name)
    tensor_args.append(out_name)
    arg_names.append(out_name)
    scalar_args = {
        "N": "i32",
        "C_IN_TOTAL": "i32",
        "H": "i32",
        "W": "i32",
        "C_OUT": "i32",
        "C_PER_G": "i32",
        "KH": "i32",
        "KW": "i32",
        "OH": "i32",
        "OW": "i32",
    }
    arg_names.extend(["N", "C_IN_TOTAL", "H", "W", "C_OUT", "C_PER_G", "KH", "KW", "OH", "OW"])
    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update(
        {
            "N": int(N),
            "C_IN_TOTAL": int(C_IN_TOTAL),
            "H": int(H),
            "W": int(W),
            "C_OUT": int(C_OUT),
            "C_PER_G": int(C_PER_G),
            "KH": int(KH),
            "KW": int(KW),
            "OH": int(OH),
            "OW": int(OW),
        }
    )
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=[out_name],
        bindings=lowered_bindings,
    )


def _kernel_conv2d_nchw_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "conv2d":
        raise CudaLoweringError("conv2d lowering expects a single conv2d op")
    op = intent.ops[0]
    if len(op.inputs) not in {2, 3}:
        raise CudaLoweringError("conv2d lowering expects inputs [input, weight] or [input, weight, bias]")
    inp_name = str(op.inputs[0])
    weight_name = str(op.inputs[1])
    bias_name = str(op.inputs[2]) if len(op.inputs) == 3 else None
    return _kernel_conv2d_nchw_impl_f32(
        intent,
        bindings,
        inp_name=inp_name,
        weight_name=weight_name,
        out_name=str(op.output),
        attrs=(op.attrs or {}),
        bias_name=bias_name,
    )


def _kernel_conv2d_nchw_bias_pattern_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    ops = list(intent.ops or [])
    if len(ops) != 3 or [o.op for o in ops] != ["conv2d", "broadcast_in_dim", "add"]:
        raise CudaLoweringError("conv2d+bias lowering expects [conv2d, broadcast_in_dim, add]")
    conv_op, bcast_op, add_op = ops
    if len(conv_op.inputs) != 2:
        raise CudaLoweringError("conv2d+bias lowering expects conv2d inputs [input, weight]")
    if len(bcast_op.inputs) != 1:
        raise CudaLoweringError("conv2d+bias lowering expects broadcast_in_dim input [bias]")
    if len(add_op.inputs) != 2:
        raise CudaLoweringError("conv2d+bias lowering expects add with two inputs")
    conv_out = str(conv_op.output)
    bcast_out = str(bcast_op.output)
    add_in = [str(x) for x in add_op.inputs]
    if conv_out not in add_in or bcast_out not in add_in:
        raise CudaLoweringError("conv2d+bias lowering expects add(conv_out, bias_bcast)")
    bcast_dims = (bcast_op.attrs or {}).get("broadcast_dims")
    if bcast_dims not in ([1], (1,)):
        raise CudaLoweringError("conv2d+bias lowering expects bias broadcast_dims=[1]")
    return _kernel_conv2d_nchw_impl_f32(
        intent,
        bindings,
        inp_name=str(conv_op.inputs[0]),
        weight_name=str(conv_op.inputs[1]),
        out_name=str(add_op.output),
        attrs=(conv_op.attrs or {}),
        bias_name=str(bcast_op.inputs[0]),
    )


def _kernel_conv3d_ncdhw_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "conv3d":
        raise CudaLoweringError("conv3d lowering expects a single conv3d op")
    op = intent.ops[0]
    if len(op.inputs) not in {2, 3}:
        raise CudaLoweringError("conv3d lowering expects inputs [input, weight] or [input, weight, bias]")
    inp_name = str(op.inputs[0])
    weight_name = str(op.inputs[1])
    bias_name = str(op.inputs[2]) if len(op.inputs) == 3 else None
    out_name = str(op.output)

    in_shape = _shape_values(intent, inp_name)
    w_shape = _shape_values(intent, weight_name)
    out_shape = _shape_values(intent, out_name)
    if len(in_shape) != 5 or len(w_shape) != 5 or len(out_shape) != 5:
        raise CudaLoweringError("conv3d lowering expects input[N,C,D,H,W], weight[CO,C_PER_G,KD,KH,KW], out[N,CO,OD,OH,OW]")
    if bias_name is not None:
        b_shape = _shape_values(intent, bias_name)
        if len(b_shape) != 1:
            raise CudaLoweringError("conv3d bias must be rank-1")
    else:
        b_shape = []

    N_dim, C_in_dim, D_dim, H_dim, W_dim = in_shape
    C_out_dim, C_per_g_dim, KD_dim, KH_dim, KW_dim = w_shape
    ON_dim, OC_dim, OD_dim, OH_dim, OW_dim = out_shape
    if str(ON_dim) != str(N_dim) or str(OC_dim) != str(C_out_dim):
        raise CudaLoweringError("conv3d output N/C mismatch")
    if bias_name is not None and str(b_shape[0]) != str(C_out_dim):
        raise CudaLoweringError("conv3d bias shape mismatch")

    sd, sh, sw = _resolve_attr_tuple((op.attrs or {}).get("stride"), bindings, name="conv3d.stride", rank=3, default=1)
    pd, ph, pw = _resolve_attr_tuple((op.attrs or {}).get("padding"), bindings, name="conv3d.padding", rank=3, default=0)
    dd, dh, dw = _resolve_attr_tuple((op.attrs or {}).get("dilation"), bindings, name="conv3d.dilation", rank=3, default=1)
    groups = _resolve_attr_int((op.attrs or {}).get("groups", 1), bindings, name="conv3d.groups")
    if min(sd, sh, sw, dd, dh, dw, groups) <= 0:
        raise CudaLoweringError("conv3d attrs stride/dilation/groups must be positive")

    N = _resolve_dim_int(N_dim, bindings, name="N")
    C_IN_TOTAL = _resolve_dim_int(C_in_dim, bindings, name="C_IN_TOTAL")
    D = _resolve_dim_int(D_dim, bindings, name="D")
    H = _resolve_dim_int(H_dim, bindings, name="H")
    W = _resolve_dim_int(W_dim, bindings, name="W")
    C_OUT = _resolve_dim_int(C_out_dim, bindings, name="C_OUT")
    C_PER_G = _resolve_dim_int(C_per_g_dim, bindings, name="C_PER_G")
    KD = _resolve_dim_int(KD_dim, bindings, name="KD")
    KH = _resolve_dim_int(KH_dim, bindings, name="KH")
    KW = _resolve_dim_int(KW_dim, bindings, name="KW")
    OD = _resolve_dim_int(OD_dim, bindings, name="OD")
    OH = _resolve_dim_int(OH_dim, bindings, name="OH")
    OW = _resolve_dim_int(OW_dim, bindings, name="OW")
    if C_IN_TOTAL != C_PER_G * groups:
        raise CudaLoweringError("conv3d channel/group mismatch: C_IN_TOTAL must equal C_PER_G * groups")
    if C_OUT % groups != 0:
        raise CudaLoweringError("conv3d C_OUT must be divisible by groups")
    expected_od = (D + 2 * pd - dd * (KD - 1) - 1) // sd + 1
    expected_oh = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    expected_ow = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    if expected_od != OD or expected_oh != OH or expected_ow != OW:
        raise CudaLoweringError(
            f"conv3d output shape mismatch: expected ({expected_od}, {expected_oh}, {expected_ow}), got ({OD}, {OH}, {OW})"
        )

    total = int(N) * int(C_OUT) * int(OD) * int(OH) * int(OW)
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=64)
    block_x = max(32, min(1024, int(block_x) if block_x > 0 else 64))
    grid_x = (total + block_x - 1) // block_x

    bias_param = f", const float* __restrict__ {bias_name}" if bias_name is not None else ""
    bias_init = f"float acc = {bias_name}[(size_t)co];" if bias_name is not None else "float acc = 0.0f;"
    cuda_src = f"""
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(
    const float* __restrict__ {inp_name},
    const float* __restrict__ {weight_name}{bias_param},
    float* __restrict__ {out_name},
    int N, int C_IN_TOTAL, int D, int H, int W, int C_OUT, int C_PER_G, int KD, int KH, int KW, int OD, int OH, int OW) {{
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  const int64_t total = (int64_t)N * (int64_t)C_OUT * (int64_t)OD * (int64_t)OH * (int64_t)OW;
  if (tid >= total) return;
  int64_t t = tid;
  const int ow = (int)(t % OW); t /= OW;
  const int oh = (int)(t % OH); t /= OH;
  const int od = (int)(t % OD); t /= OD;
  const int co = (int)(t % C_OUT); t /= C_OUT;
  const int n = (int)t;
  const int co_per_g = C_OUT / {groups};
  const int g = co / co_per_g;
  const int c_start = g * C_PER_G;
  {bias_init}
  for (int ci = 0; ci < C_PER_G; ++ci) {{
    const int c = c_start + ci;
    for (int kd = 0; kd < KD; ++kd) {{
      const int id = od * {sd} - {pd} + kd * {dd};
      if ((unsigned)id >= (unsigned)D) continue;
      for (int kh = 0; kh < KH; ++kh) {{
        const int ih = oh * {sh} - {ph} + kh * {dh};
        if ((unsigned)ih >= (unsigned)H) continue;
        for (int kw = 0; kw < KW; ++kw) {{
          const int iw = ow * {sw} - {pw} + kw * {dw};
          if ((unsigned)iw >= (unsigned)W) continue;
          const size_t x_idx = ((((size_t)n * (size_t)C_IN_TOTAL + (size_t)c) * (size_t)D + (size_t)id) * (size_t)H + (size_t)ih) * (size_t)W + (size_t)iw;
          const size_t w_idx = ((((size_t)co * (size_t)C_PER_G + (size_t)ci) * (size_t)KD + (size_t)kd) * (size_t)KH + (size_t)kh) * (size_t)KW + (size_t)kw;
          acc += {inp_name}[x_idx] * {weight_name}[w_idx];
        }}
      }}
    }}
  }}
  {out_name}[tid] = acc;
}}
""".lstrip()

    tensor_args = [inp_name, weight_name]
    arg_names = [inp_name, weight_name]
    if bias_name is not None:
        tensor_args.append(bias_name)
        arg_names.append(bias_name)
    tensor_args.append(out_name)
    arg_names.append(out_name)
    scalar_args = {
        "N": "i32",
        "C_IN_TOTAL": "i32",
        "D": "i32",
        "H": "i32",
        "W": "i32",
        "C_OUT": "i32",
        "C_PER_G": "i32",
        "KD": "i32",
        "KH": "i32",
        "KW": "i32",
        "OD": "i32",
        "OH": "i32",
        "OW": "i32",
    }
    arg_names.extend(["N", "C_IN_TOTAL", "D", "H", "W", "C_OUT", "C_PER_G", "KD", "KH", "KW", "OD", "OH", "OW"])
    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update(
        {
            "N": int(N),
            "C_IN_TOTAL": int(C_IN_TOTAL),
            "D": int(D),
            "H": int(H),
            "W": int(W),
            "C_OUT": int(C_OUT),
            "C_PER_G": int(C_PER_G),
            "KD": int(KD),
            "KH": int(KH),
            "KW": int(KW),
            "OD": int(OD),
            "OH": int(OH),
            "OW": int(OW),
        }
    )
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=[out_name],
        bindings=lowered_bindings,
    )


def _kernel_conv_depthwise2d_nchw_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "conv_depthwise2d":
        raise CudaLoweringError("conv_depthwise2d lowering expects a single conv_depthwise2d op")
    op = intent.ops[0]
    if len(op.inputs) not in {2, 3}:
        raise CudaLoweringError("conv_depthwise2d lowering expects inputs [input, weight] or [input, weight, bias]")
    inp_name = str(op.inputs[0])
    weight_name = str(op.inputs[1])
    bias_name = str(op.inputs[2]) if len(op.inputs) == 3 else None
    out_name = str(op.output)

    in_shape = _shape_values(intent, inp_name)
    w_shape = _shape_values(intent, weight_name)
    out_shape = _shape_values(intent, out_name)
    if len(in_shape) != 4 or len(w_shape) != 4 or len(out_shape) != 4:
        raise CudaLoweringError("conv_depthwise2d expects input[N,C,H,W], weight[C_OUT,1,KH,KW], out[N,C_OUT,OH,OW]")
    if bias_name is not None:
        b_shape = _shape_values(intent, bias_name)
        if len(b_shape) != 1:
            raise CudaLoweringError("conv_depthwise2d bias must be rank-1")
    else:
        b_shape = []

    N_dim, C_in_dim, H_dim, W_dim = in_shape
    C_out_dim, C1_dim, KH_dim, KW_dim = w_shape
    ON_dim, OC_dim, OH_dim, OW_dim = out_shape
    if str(ON_dim) != str(N_dim) or str(OC_dim) != str(C_out_dim):
        raise CudaLoweringError("conv_depthwise2d output N/C mismatch")
    if bias_name is not None and str(b_shape[0]) != str(C_out_dim):
        raise CudaLoweringError("conv_depthwise2d bias shape mismatch")
    if _resolve_dim_int(C1_dim, bindings, name="C1") != 1:
        raise CudaLoweringError("conv_depthwise2d expects weight second dim == 1")

    sh, sw = _resolve_attr_tuple((op.attrs or {}).get("stride"), bindings, name="conv_depthwise2d.stride", rank=2, default=1)
    ph, pw = _resolve_attr_tuple((op.attrs or {}).get("padding"), bindings, name="conv_depthwise2d.padding", rank=2, default=0)
    dh, dw = _resolve_attr_tuple((op.attrs or {}).get("dilation"), bindings, name="conv_depthwise2d.dilation", rank=2, default=1)
    if min(sh, sw, dh, dw) <= 0:
        raise CudaLoweringError("conv_depthwise2d stride/dilation must be positive")

    N = _resolve_dim_int(N_dim, bindings, name="N")
    C_IN = _resolve_dim_int(C_in_dim, bindings, name="C_IN")
    H = _resolve_dim_int(H_dim, bindings, name="H")
    W = _resolve_dim_int(W_dim, bindings, name="W")
    C_OUT = _resolve_dim_int(C_out_dim, bindings, name="C_OUT")
    KH = _resolve_dim_int(KH_dim, bindings, name="KH")
    KW = _resolve_dim_int(KW_dim, bindings, name="KW")
    OH = _resolve_dim_int(OH_dim, bindings, name="OH")
    OW = _resolve_dim_int(OW_dim, bindings, name="OW")
    if C_IN <= 0 or C_OUT <= 0 or C_OUT % C_IN != 0:
        raise CudaLoweringError("conv_depthwise2d channel multiplier mismatch")
    channel_multiplier = C_OUT // C_IN
    expected_oh = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    expected_ow = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    if expected_oh != OH or expected_ow != OW:
        raise CudaLoweringError(f"conv_depthwise2d output shape mismatch: expected ({expected_oh}, {expected_ow}), got ({OH}, {OW})")

    total = int(N) * int(C_OUT) * int(OH) * int(OW)
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=128)
    block_x = max(32, min(1024, int(block_x) if block_x > 0 else 128))
    grid_x = (total + block_x - 1) // block_x

    bias_param = f", const float* __restrict__ {bias_name}" if bias_name is not None else ""
    bias_init = f"float acc = {bias_name}[(size_t)co];" if bias_name is not None else "float acc = 0.0f;"
    cuda_src = f"""
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(
    const float* __restrict__ {inp_name},
    const float* __restrict__ {weight_name}{bias_param},
    float* __restrict__ {out_name},
    int N, int C_IN, int H, int W, int C_OUT, int KH, int KW, int OH, int OW) {{
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  const int64_t total = (int64_t)N * (int64_t)C_OUT * (int64_t)OH * (int64_t)OW;
  if (tid >= total) return;
  int64_t t = tid;
  const int ow = (int)(t % OW); t /= OW;
  const int oh = (int)(t % OH); t /= OH;
  const int co = (int)(t % C_OUT); t /= C_OUT;
  const int n = (int)t;
  const int ci = co / {channel_multiplier};
  {bias_init}
  for (int kh = 0; kh < KH; ++kh) {{
    const int ih = oh * {sh} - {ph} + kh * {dh};
    if ((unsigned)ih >= (unsigned)H) continue;
    for (int kw = 0; kw < KW; ++kw) {{
      const int iw = ow * {sw} - {pw} + kw * {dw};
      if ((unsigned)iw >= (unsigned)W) continue;
      const size_t x_idx = (((size_t)n * (size_t)C_IN + (size_t)ci) * (size_t)H + (size_t)ih) * (size_t)W + (size_t)iw;
      const size_t w_idx = ((size_t)co * (size_t)KH + (size_t)kh) * (size_t)KW + (size_t)kw;
      acc += {inp_name}[x_idx] * {weight_name}[w_idx];
    }}
  }}
  {out_name}[tid] = acc;
}}
""".lstrip()

    tensor_args = [inp_name, weight_name]
    arg_names = [inp_name, weight_name]
    if bias_name is not None:
        tensor_args.append(bias_name)
        arg_names.append(bias_name)
    tensor_args.append(out_name)
    arg_names.append(out_name)
    scalar_args = {"N": "i32", "C_IN": "i32", "H": "i32", "W": "i32", "C_OUT": "i32", "KH": "i32", "KW": "i32", "OH": "i32", "OW": "i32"}
    arg_names.extend(["N", "C_IN", "H", "W", "C_OUT", "KH", "KW", "OH", "OW"])
    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update({"N": int(N), "C_IN": int(C_IN), "H": int(H), "W": int(W), "C_OUT": int(C_OUT), "KH": int(KH), "KW": int(KW), "OH": int(OH), "OW": int(OW)})
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=lowered_bindings)


def _kernel_dot_1d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    ops = list(intent.ops or [])
    if len(ops) != 4 or [o.op for o in ops] != ["cast", "cast", "mul", "reduce_sum"]:
        raise CudaLoweringError("dot lowering expects cast->cast->mul->reduce_sum pattern")
    cast_x, cast_y, mul_op, red_op = ops
    if len(cast_x.inputs) != 1 or len(cast_y.inputs) != 1:
        raise CudaLoweringError("dot lowering expects cast ops with single input")
    x_name = str(cast_x.inputs[0])
    y_name = str(cast_y.inputs[0])
    if [str(v) for v in mul_op.inputs] != [str(cast_x.output), str(cast_y.output)]:
        raise CudaLoweringError("dot lowering expects mul(cast_x, cast_y)")
    if len(red_op.inputs) != 1 or str(red_op.inputs[0]) != str(mul_op.output):
        raise CudaLoweringError("dot lowering expects reduce_sum over mul output")
    out_name = str(red_op.output)
    x_shape = _shape_values(intent, x_name)
    y_shape = _shape_values(intent, y_name)
    out_shape = _shape_values(intent, out_name)
    if len(x_shape) != 1 or len(y_shape) != 1:
        raise CudaLoweringError("dot lowering expects rank-1 x/y")
    if len(out_shape) != 0:
        raise CudaLoweringError("dot lowering expects scalar output")
    N = _resolve_dim_int(x_shape[0], bindings, name="N")
    Ny = _resolve_dim_int(y_shape[0], bindings, name="N_y")
    if N != Ny:
        raise CudaLoweringError("dot lowering expects x/y same length")

    cuda_src = f"""
#include <stdint.h>

extern "C" __global__ void {intent.name}(
    const float* __restrict__ {x_name},
    const float* __restrict__ {y_name},
    float* __restrict__ {out_name},
    int N) {{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float acc = 0.0f;
  for (int i = 0; i < N; ++i) {{
    acc += {x_name}[i] * {y_name}[i];
  }}
  {out_name}[0] = acc;
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[x_name, y_name, out_name],
        scalar_args={"N": "i32"},
        arg_names=[x_name, y_name, out_name, "N"],
    )
    launch = CudaLaunch(grid=(1, 1, 1), block=(1, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings["N"] = int(N)
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=[out_name],
        bindings=lowered_bindings,
    )


def _kernel_diag_embed_2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    ops = list(intent.ops or [])
    expected = ["const", "broadcast_in_dim", "iota", "iota", "iota", "ne", "not", "gather", "where"]
    if len(ops) != len(expected) or [o.op for o in ops] != expected:
        raise CudaLoweringError("diag_embed lowering expects const->broadcast->iota*3->ne->not->gather->where pattern")
    bcast_op = ops[1]
    gather_op = ops[7]
    where_op = ops[8]
    if len(gather_op.inputs) != 3:
        raise CudaLoweringError("diag_embed gather expects inputs [x, idx_b, idx_col]")
    x_name = str(gather_op.inputs[0])
    out_name = str(where_op.output)
    x_shape = _shape_values(intent, x_name)
    out_shape = _shape_values(intent, out_name)
    if len(x_shape) != 2 or len(out_shape) != 3:
        raise CudaLoweringError("diag_embed lowering expects x[B,N] -> y[B,N,N]")
    B = _resolve_dim_int(x_shape[0], bindings, name="B")
    N = _resolve_dim_int(x_shape[1], bindings, name="N")
    if _resolve_dim_int(out_shape[0], bindings, name="OB") != B:
        raise CudaLoweringError("diag_embed output B mismatch")
    if _resolve_dim_int(out_shape[1], bindings, name="ON1") != N or _resolve_dim_int(out_shape[2], bindings, name="ON2") != N:
        raise CudaLoweringError("diag_embed output N mismatch")

    bcast_shape = (bcast_op.attrs or {}).get("out_shape")
    if not isinstance(bcast_shape, list) or len(bcast_shape) != 3:
        raise CudaLoweringError("diag_embed broadcast_in_dim must provide out_shape=[B,N,N]")

    total = int(B) * int(N) * int(N)
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    block_x = max(32, min(1024, int(block_x) if block_x > 0 else 256))
    grid_x = (total + block_x - 1) // block_x

    cuda_src = f"""
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(
    const float* __restrict__ {x_name},
    float* __restrict__ {out_name},
    int B, int N) {{
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  const int64_t total = (int64_t)B * (int64_t)N * (int64_t)N;
  if (tid >= total) return;
  int64_t t = tid;
  const int col = (int)(t % N); t /= N;
  const int row = (int)(t % N); t /= N;
  const int b = (int)t;
  if (row == col) {{
    {out_name}[tid] = {x_name}[(size_t)b * (size_t)N + (size_t)col];
  }} else {{
    {out_name}[tid] = 0.0f;
  }}
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[x_name, out_name],
        scalar_args={"B": "i32", "N": "i32"},
        arg_names=[x_name, out_name, "B", "N"],
    )
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update({"B": int(B), "N": int(N)})
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=[out_name],
        bindings=lowered_bindings,
    )


def _kernel_avg_pool2d_nchw_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "avg_pool2d":
        raise CudaLoweringError("avg_pool2d lowering expects a single avg_pool2d op")
    op = intent.ops[0]
    if len(op.inputs) != 1:
        raise CudaLoweringError("avg_pool2d expects 1 input")
    inp_name = str(op.inputs[0])
    out_name = str(op.output)
    in_shape = _shape_values(intent, inp_name)
    out_shape = _shape_values(intent, out_name)
    if len(in_shape) != 4 or len(out_shape) != 4:
        raise CudaLoweringError("avg_pool2d MVP expects rank-4 NCHW tensors")

    N = _resolve_dim_int(in_shape[0], bindings, name="N")
    C = _resolve_dim_int(in_shape[1], bindings, name="C")
    H = _resolve_dim_int(in_shape[2], bindings, name="H")
    W = _resolve_dim_int(in_shape[3], bindings, name="W")
    ON = _resolve_dim_int(out_shape[0], bindings, name="ON")
    OC = _resolve_dim_int(out_shape[1], bindings, name="OC")
    OH = _resolve_dim_int(out_shape[2], bindings, name="OH")
    OW = _resolve_dim_int(out_shape[3], bindings, name="OW")
    if N != ON or C != OC:
        raise CudaLoweringError("avg_pool2d output N/C mismatch")

    def _pair(val: Any, default: int) -> tuple[int, int]:
        if val is None:
            return (default, default)
        if isinstance(val, (list, tuple)):
            if len(val) != 2:
                raise CudaLoweringError("avg_pool2d attrs pair must have length 2")
            return (int(val[0]), int(val[1]))
        v = int(val)
        return (v, v)

    attrs = op.attrs or {}
    kh, kw = _pair(attrs.get("kernel_size"), 2)
    sh, sw = _pair(attrs.get("stride"), kh)
    ph, pw = _pair(attrs.get("padding"), 0)
    count_include_pad = bool(attrs.get("count_include_pad", True))
    if kh <= 0 or kw <= 0 or sh <= 0 or sw <= 0:
        raise CudaLoweringError("avg_pool2d attrs must be positive")

    total = int(N) * int(C) * int(OH) * int(OW)
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    grid_x = (total + block_x - 1) // block_x
    include_pad = "true" if count_include_pad else "false"

    cuda_src = f"""
#include <math.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(const float* __restrict__ {inp_name}, float* __restrict__ {out_name}) {{
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  const int64_t total = (int64_t){total};
  if (tid >= total) return;
  int64_t t = tid;
  const int ow = (int)(t % {OW}); t /= {OW};
  const int oh = (int)(t % {OH}); t /= {OH};
  const int c = (int)(t % {C}); t /= {C};
  const int n = (int)t;
  const int h_start = oh * {sh} - {ph};
  const int w_start = ow * {sw} - {pw};
  const int h_end = h_start + {kh};
  const int w_end = w_start + {kw};
  float sum = 0.0f;
  int cnt = 0;
  for (int ih = h_start; ih < h_end; ++ih) {{
    if ((unsigned)ih >= (unsigned){H}) continue;
    for (int iw = w_start; iw < w_end; ++iw) {{
      if ((unsigned)iw >= (unsigned){W}) continue;
      const int64_t in_idx = ((((int64_t)n * {C} + c) * {H} + ih) * {W}) + iw;
      sum += {inp_name}[in_idx];
      cnt += 1;
    }}
  }}
  const float denom = {include_pad} ? (float)({kh} * {kw}) : (float)(cnt > 0 ? cnt : 1);
  {out_name}[tid] = sum / denom;
}}
""".lstrip()

    io_spec = _io_spec_from_args(intent, tensor_args=[inp_name, out_name], scalar_args={}, arg_names=[inp_name, out_name])
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update({"N": int(N), "C": int(C), "H": int(H), "W": int(W), "OH": int(OH), "OW": int(OW)})
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=lowered_bindings)


def _kernel_scaled_dot_product_attention_bhsd_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or len(intent.ops) != 1 or intent.ops[0].op != "scaled_dot_product_attention":
        raise CudaLoweringError("sdpa lowering expects a single scaled_dot_product_attention op")
    op = intent.ops[0]
    if len(op.inputs) < 3:
        raise CudaLoweringError("scaled_dot_product_attention expects query/key/value inputs")
    q_name, k_name, v_name = (str(op.inputs[0]), str(op.inputs[1]), str(op.inputs[2]))
    out_name = str(op.output)

    q_shape = _shape_values(intent, q_name)
    k_shape = _shape_values(intent, k_name)
    v_shape = _shape_values(intent, v_name)
    out_shape = _shape_values(intent, out_name)
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4 or len(out_shape) != 4:
        raise CudaLoweringError("sdpa MVP expects rank-4 BHQD/BHKD tensors")

    B = _resolve_dim_int(q_shape[0], bindings, name="B")
    Hh = _resolve_dim_int(q_shape[1], bindings, name="H")
    Q = _resolve_dim_int(q_shape[2], bindings, name="Q")
    D = _resolve_dim_int(q_shape[3], bindings, name="D")
    BK = _resolve_dim_int(k_shape[0], bindings, name="BK")
    HK = _resolve_dim_int(k_shape[1], bindings, name="HK")
    K = _resolve_dim_int(k_shape[2], bindings, name="K")
    DK = _resolve_dim_int(k_shape[3], bindings, name="DK")
    BV = _resolve_dim_int(v_shape[0], bindings, name="BV")
    HV = _resolve_dim_int(v_shape[1], bindings, name="HV")
    KV = _resolve_dim_int(v_shape[2], bindings, name="KV")
    DV = _resolve_dim_int(v_shape[3], bindings, name="DV")
    if BK != B or HK != Hh or DK != D:
        raise CudaLoweringError("sdpa shape mismatch between query and key")
    if BV != B or HV != Hh or KV != K:
        raise CudaLoweringError("sdpa shape mismatch between key and value")
    if _resolve_dim_int(out_shape[0], bindings, name="OB") != B or _resolve_dim_int(out_shape[1], bindings, name="OH") != Hh:
        raise CudaLoweringError("sdpa output BH mismatch")
    if _resolve_dim_int(out_shape[2], bindings, name="OQ") != Q or _resolve_dim_int(out_shape[3], bindings, name="OD") != DV:
        raise CudaLoweringError("sdpa output QD mismatch")

    is_causal = bool((op.attrs or {}).get("is_causal", False))
    causal_guard = "if (k > q) continue;" if is_causal else ""

    total = int(B) * int(Hh) * int(Q) * int(DV)
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=128)
    if block_x <= 0:
        block_x = 128
    if block_x > 1024:
        block_x = 1024
    grid_x = (total + block_x - 1) // block_x

    cuda_src = f"""
#include <math.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(const float* __restrict__ {q_name},
                                                                       const float* __restrict__ {k_name},
                                                                       const float* __restrict__ {v_name},
                                                                       float* __restrict__ {out_name}) {{
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  const int64_t total = (int64_t){total};
  if (tid >= total) return;

  int64_t t = tid;
  const int d = (int)(t % {DV}); t /= {DV};
  const int q = (int)(t % {Q}); t /= {Q};
  const int h = (int)(t % {Hh}); t /= {Hh};
  const int b = (int)t;

  const float inv_sqrt_d = rsqrtf((float){D});
  float max_score = -INFINITY;
  for (int k = 0; k < {K}; ++k) {{
    {causal_guard}
    float dot = 0.0f;
    for (int x = 0; x < {D}; ++x) {{
      const int64_t qidx = ((((int64_t)b * {Hh} + h) * {Q} + q) * {D}) + x;
      const int64_t kidx = ((((int64_t)b * {Hh} + h) * {K} + k) * {D}) + x;
      dot += {q_name}[qidx] * {k_name}[kidx];
    }}
    const float score = dot * inv_sqrt_d;
    max_score = fmaxf(max_score, score);
  }}

  float denom = 0.0f;
  float numer = 0.0f;
  for (int k = 0; k < {K}; ++k) {{
    {causal_guard}
    float dot = 0.0f;
    for (int x = 0; x < {D}; ++x) {{
      const int64_t qidx = ((((int64_t)b * {Hh} + h) * {Q} + q) * {D}) + x;
      const int64_t kidx = ((((int64_t)b * {Hh} + h) * {K} + k) * {D}) + x;
      dot += {q_name}[qidx] * {k_name}[kidx];
    }}
    const float wgt = expf(dot * inv_sqrt_d - max_score);
    denom += wgt;
    const int64_t vidx = ((((int64_t)b * {Hh} + h) * {K} + k) * {DV}) + d;
    numer += wgt * {v_name}[vidx];
  }}

  {out_name}[tid] = (denom > 0.0f) ? (numer / denom) : 0.0f;
}}
""".lstrip()

    io_spec = _io_spec_from_args(intent, tensor_args=[q_name, k_name, v_name, out_name], scalar_args={}, arg_names=[q_name, k_name, v_name, out_name])
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update({"B": int(B), "H": int(Hh), "Q": int(Q), "K": int(K), "D": int(D), "DV": int(DV)})
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=lowered_bindings)


def _kernel_flash_attn_varlen_decomposed_bhsd_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    ops = list(intent.ops or [])
    if [o.op for o in ops] != ["transpose", "matmul", "mul", "add", "softmax", "matmul"]:
        raise CudaLoweringError("flash-attn decomposition lowering expects transpose->matmul->mul->add->softmax->matmul")
    op_t, op_qk, op_mul, op_add, op_softmax, op_out = ops

    if len(op_t.inputs) != 1:
        raise CudaLoweringError("flash-attn transpose expects 1 input")
    k_name = str(op_t.inputs[0])
    k_t_name = str(op_t.output)

    if len(op_qk.inputs) != 2 or str(op_qk.inputs[1]) != k_t_name:
        raise CudaLoweringError("flash-attn first matmul must consume transpose(K)")
    q_name = str(op_qk.inputs[0])
    qk_name = str(op_qk.output)

    if len(op_mul.inputs) != 2:
        raise CudaLoweringError("flash-attn mul expects 2 inputs")
    if str(op_mul.inputs[0]) == qk_name:
        sm_scale_name = str(op_mul.inputs[1])
    elif str(op_mul.inputs[1]) == qk_name:
        sm_scale_name = str(op_mul.inputs[0])
    else:
        raise CudaLoweringError("flash-attn mul must consume first matmul output")
    qk_scaled_name = str(op_mul.output)

    if len(op_add.inputs) != 2:
        raise CudaLoweringError("flash-attn add expects 2 inputs")
    if str(op_add.inputs[0]) == qk_scaled_name:
        attn_mask_name = str(op_add.inputs[1])
    elif str(op_add.inputs[1]) == qk_scaled_name:
        attn_mask_name = str(op_add.inputs[0])
    else:
        raise CudaLoweringError("flash-attn add must consume scaled scores")
    qk_masked_name = str(op_add.output)

    if len(op_softmax.inputs) != 1 or str(op_softmax.inputs[0]) != qk_masked_name:
        raise CudaLoweringError("flash-attn softmax must consume masked scores")
    attn_weights_name = str(op_softmax.output)

    if len(op_out.inputs) != 2:
        raise CudaLoweringError("flash-attn output matmul expects 2 inputs")
    if str(op_out.inputs[0]) == attn_weights_name:
        v_name = str(op_out.inputs[1])
    elif str(op_out.inputs[1]) == attn_weights_name:
        v_name = str(op_out.inputs[0])
    else:
        raise CudaLoweringError("flash-attn output matmul must consume softmax output")
    out_name = str(op_out.output)

    q_shape = _shape_values(intent, q_name)
    k_shape = _shape_values(intent, k_name)
    v_shape = _shape_values(intent, v_name)
    mask_shape = _shape_values(intent, attn_mask_name)
    out_shape = _shape_values(intent, out_name)
    scale_shape = _shape_values(intent, sm_scale_name)
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4 or len(mask_shape) != 4 or len(out_shape) != 4:
        raise CudaLoweringError("flash-attn decomposition expects rank-4 Q/K/V/mask/out tensors")
    if len(scale_shape) != 0:
        raise CudaLoweringError("flash-attn sm_scale must be scalar tensor")

    B = _resolve_dim_int(q_shape[0], bindings, name="B")
    Hh = _resolve_dim_int(q_shape[1], bindings, name="H")
    Q = _resolve_dim_int(q_shape[2], bindings, name="Q")
    D = _resolve_dim_int(q_shape[3], bindings, name="D")
    BK = _resolve_dim_int(k_shape[0], bindings, name="BK")
    HK = _resolve_dim_int(k_shape[1], bindings, name="HK")
    K = _resolve_dim_int(k_shape[2], bindings, name="K")
    DK = _resolve_dim_int(k_shape[3], bindings, name="DK")
    BV = _resolve_dim_int(v_shape[0], bindings, name="BV")
    HV = _resolve_dim_int(v_shape[1], bindings, name="HV")
    KV = _resolve_dim_int(v_shape[2], bindings, name="KV")
    DV = _resolve_dim_int(v_shape[3], bindings, name="DV")
    if BK != B or HK != Hh or DK != D:
        raise CudaLoweringError("flash-attn shape mismatch between query and key")
    if BV != B or HV != Hh or KV != K:
        raise CudaLoweringError("flash-attn shape mismatch between key and value")
    if (
        _resolve_dim_int(mask_shape[0], bindings, name="MB") != B
        or _resolve_dim_int(mask_shape[1], bindings, name="MH") != Hh
        or _resolve_dim_int(mask_shape[2], bindings, name="MQ") != Q
        or _resolve_dim_int(mask_shape[3], bindings, name="MK") != K
    ):
        raise CudaLoweringError("flash-attn mask shape mismatch")
    if (
        _resolve_dim_int(out_shape[0], bindings, name="OB") != B
        or _resolve_dim_int(out_shape[1], bindings, name="OH") != Hh
        or _resolve_dim_int(out_shape[2], bindings, name="OQ") != Q
        or _resolve_dim_int(out_shape[3], bindings, name="OD") != DV
    ):
        raise CudaLoweringError("flash-attn output shape mismatch")

    lse_name: str | None = None
    for cand in list(intent.outputs or []):
        cc = str(cand)
        if cc != out_name and cc in intent.tensors:
            lse_name = cc
            break
    if lse_name is not None:
        lse_shape = _shape_values(intent, lse_name)
        if len(lse_shape) != 3:
            raise CudaLoweringError("flash-attn softmax_lse output must be rank-3 [B,H,Q]")
        if (
            _resolve_dim_int(lse_shape[0], bindings, name="LB") != B
            or _resolve_dim_int(lse_shape[1], bindings, name="LH") != Hh
            or _resolve_dim_int(lse_shape[2], bindings, name="LQ") != Q
        ):
            raise CudaLoweringError("flash-attn softmax_lse shape mismatch")

    total = int(B) * int(Hh) * int(Q) * int(DV)
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=128)
    if block_x <= 0:
        block_x = 128
    if block_x > 1024:
        block_x = 1024
    grid_x = (total + block_x - 1) // block_x

    lse_param = f", float* __restrict__ {lse_name}" if lse_name is not None else ""
    lse_write = (
        f"if (d == 0) {{ const int64_t lidx = (((int64_t)b * {Hh} + h) * {Q}) + q; {lse_name}[lidx] = (denom > 0.0f) ? (max_score + logf(denom)) : -INFINITY; }}"
        if lse_name is not None
        else ""
    )
    cuda_src = f"""
#include <math.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__({block_x}) void {intent.name}(const float* __restrict__ {q_name},
                                                                       const float* __restrict__ {k_name},
                                                                       const float* __restrict__ {v_name},
                                                                       const float* __restrict__ {attn_mask_name},
                                                                       const float* __restrict__ {sm_scale_name},
                                                                       float* __restrict__ {out_name}{lse_param}) {{
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  const int64_t total = (int64_t){total};
  if (tid >= total) return;

  int64_t t = tid;
  const int d = (int)(t % {DV}); t /= {DV};
  const int q = (int)(t % {Q}); t /= {Q};
  const int h = (int)(t % {Hh}); t /= {Hh};
  const int b = (int)t;

  const float scale = {sm_scale_name}[0];
  float max_score = -INFINITY;
  for (int k = 0; k < {K}; ++k) {{
    float dot = 0.0f;
    for (int x = 0; x < {D}; ++x) {{
      const int64_t qidx = ((((int64_t)b * {Hh} + h) * {Q} + q) * {D}) + x;
      const int64_t kidx = ((((int64_t)b * {Hh} + h) * {K} + k) * {D}) + x;
      dot += {q_name}[qidx] * {k_name}[kidx];
    }}
    const int64_t midx = ((((int64_t)b * {Hh} + h) * {Q} + q) * {K}) + k;
    const float score = dot * scale + {attn_mask_name}[midx];
    max_score = fmaxf(max_score, score);
  }}

  float denom = 0.0f;
  float numer = 0.0f;
  for (int k = 0; k < {K}; ++k) {{
    float dot = 0.0f;
    for (int x = 0; x < {D}; ++x) {{
      const int64_t qidx = ((((int64_t)b * {Hh} + h) * {Q} + q) * {D}) + x;
      const int64_t kidx = ((((int64_t)b * {Hh} + h) * {K} + k) * {D}) + x;
      dot += {q_name}[qidx] * {k_name}[kidx];
    }}
    const int64_t midx = ((((int64_t)b * {Hh} + h) * {Q} + q) * {K}) + k;
    const float score = dot * scale + {attn_mask_name}[midx];
    const float wgt = expf(score - max_score);
    denom += wgt;
    const int64_t vidx = ((((int64_t)b * {Hh} + h) * {K} + k) * {DV}) + d;
    numer += wgt * {v_name}[vidx];
  }}

  {out_name}[tid] = (denom > 0.0f) ? (numer / denom) : 0.0f;
  {lse_write}
}}
""".lstrip()

    tensor_args = [q_name, k_name, v_name, attn_mask_name, sm_scale_name, out_name]
    if lse_name is not None:
        tensor_args.append(lse_name)
    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args={}, arg_names=tensor_args)
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    lowered_bindings = dict(bindings)
    lowered_bindings.update({"B": int(B), "H": int(Hh), "Q": int(Q), "K": int(K), "D": int(D), "DV": int(DV)})
    output_names = [out_name] + ([lse_name] if lse_name is not None else [])
    return CudaLoweredKernel(
        kernel_name=intent.name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        output_names=output_names,
        bindings=lowered_bindings,
    )


def _kernel_resize_bilinear2x_i8(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or intent.ops[0].op != "resize":
        raise CudaLoweringError("resize lowering expects a single resize op")
    op = intent.ops[0]
    if len(op.inputs) != 1:
        raise CudaLoweringError("resize expects 1 input")
    src_name = op.inputs[0]
    out_name = op.output

    C = _as_int(bindings.get("C"), name="C")
    H = _as_int(bindings.get("H"), name="H")
    W = _as_int(bindings.get("W"), name="W")
    OH = _as_int(bindings.get("OH", 2 * H), name="OH")
    OW = _as_int(bindings.get("OW", 2 * W), name="OW")
    if OH != 2 * H or OW != 2 * W:
        raise CudaLoweringError("resize MVP supports only 2x upsample")
    # This kernel models the AI-Bench resize: 2x bilinear with fixed-point `hw_fl=7`.
    # The IntentIR already carries `hw_fl` as an attribute, not a runtime scalar.
    hw_fl = 7
    try:
        hw_fl = int(op.attrs.get("hw_fl", 7))
    except Exception:
        hw_fl = 7
    if hw_fl != 7:
        raise CudaLoweringError("resize MVP expects hw_fl=7")

    # Prefer the Triton constexpr BLOCK_W when available.
    sched = intent.schedule or ScheduleSketch()
    block_w = _as_int(bindings.get("BLOCK_W", 128), name="BLOCK_W")
    hinted = _resolve_schedule_int(sched.tile_n, bindings, default=block_w)
    block_w = hinted if 0 < hinted <= 1024 else block_w
    if block_w <= 0:
        block_w = 128
    if block_w > 1024:
        block_w = 1024
    # We map one thread to one input-x (x0) and write two output pixels (2x upsample),
    # which reduces global loads and launch overhead compared to 1 thread per output pixel.
    grid_w = (W + block_w - 1) // block_w

    c_is_tensor = _is_scalar_tensor(intent, "C", dtype="i32")
    h_is_tensor = _is_scalar_tensor(intent, "H", dtype="i32")
    w_is_tensor = _is_scalar_tensor(intent, "W", dtype="i32")

    c_param = "const int* C_ptr" if c_is_tensor else "int C"
    h_param = "const int* H_ptr" if h_is_tensor else "int H"
    w_param = "const int* W_ptr" if w_is_tensor else "int W"
    c_load = "const int C = C_ptr ? C_ptr[0] : 0;" if c_is_tensor else ""
    h_load = "const int H = H_ptr ? H_ptr[0] : 0;" if h_is_tensor else ""
    w_load = "const int W = W_ptr ? W_ptr[0] : 0;" if w_is_tensor else ""

    cuda_src = f"""
#include "kernels/resize.cuh"

extern "C" __global__ void {intent.name}(const int8_t* __restrict__ {src_name}, int8_t* __restrict__ {out_name}, {c_param}, {h_param}, {w_param}) {{
  {c_load}
  {h_load}
  {w_load}
  constexpr int BLOCK_W = {block_w};
  intentir_cuda::resize_bilinear2x_i8<BLOCK_W, false>({src_name}, {out_name}, C, H, W);
}}
""".lstrip()

    # Allow scalar-tensor dims if present (AI-Bench uses scalar tensors for C/H/W).
    tensor_args = [src_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [src_name, out_name]
    for dim_name in ["C", "H", "W"]:
        if _is_scalar_tensor(intent, dim_name, dtype="i32"):
            tensor_args.append(dim_name)
            arg_names.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
            arg_names.append(dim_name)

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    bindings = dict(bindings)
    bindings.setdefault("OH", OH)
    bindings.setdefault("OW", OW)
    launch = CudaLaunch(grid=(grid_w, OH, C), block=(block_w, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=bindings)


def _kernel_warp_q8_8_i8_i16(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or intent.ops[0].op != "warp":
        raise CudaLoweringError("warp lowering expects a single warp op")
    op = intent.ops[0]
    if len(op.inputs) != 2:
        raise CudaLoweringError("warp expects 2 inputs (src, offset)")
    src_name, offset_name = op.inputs
    out_name = op.output

    C = _as_int(bindings.get("C"), name="C")
    H = _as_int(bindings.get("H"), name="H")
    W = _as_int(bindings.get("W"), name="W")
    sched = intent.schedule or ScheduleSketch()
    # Prefer a 3D mapping (like the reference Triton kernel) to avoid div/mod per element.
    block_w = _as_int(bindings.get("BLOCK_W", 128), name="BLOCK_W")
    hinted = _resolve_schedule_int(sched.tile_n, bindings, default=block_w)
    block_w = hinted if 0 < hinted <= 1024 else block_w
    if block_w <= 0:
        block_w = 128
    if block_w > 1024:
        block_w = 1024
    grid_w = (W + block_w - 1) // block_w

    specialize_dims = bool(int(bindings.get("CUDA_SPECIALIZE_DIMS", 0) or 0))
    dim_decl = (
        f"(void)C_in; (void)H_in; (void)W_in;\n  constexpr int C = {int(C)};\n  constexpr int H = {int(H)};\n  constexpr int W = {int(W)};"
        if specialize_dims
        else "const int C = C_in;\n  const int H = H_in;\n  const int W = W_in;"
    )

    cuda_src = f"""
#include "kernels/warp.cuh"

extern "C" __global__ void {intent.name}(const int8_t* __restrict__ {src_name}, const int16_t* __restrict__ {offset_name}, int8_t* __restrict__ {out_name}, int C_in, int H_in, int W_in) {{
  constexpr int BLOCK_W = {block_w};
  {dim_decl}
  const bool full_w = ((W % BLOCK_W) == 0);
  if (full_w) {{
    intentir_cuda::warp_q8_8_i8_i16<BLOCK_W, true>({src_name}, {offset_name}, {out_name}, C, H, W);
  }} else {{
    intentir_cuda::warp_q8_8_i8_i16<BLOCK_W, false>({src_name}, {offset_name}, {out_name}, C, H, W);
  }}
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[src_name, offset_name, out_name],
        scalar_args={"C": "i32", "H": "i32", "W": "i32"},
        arg_names=[src_name, offset_name, out_name, "C", "H", "W"],
    )
    launch = CudaLaunch(grid=(grid_w, H, C), block=(block_w, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_correlation_i8(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or intent.ops[0].op != "correlation":
        raise CudaLoweringError("correlation lowering expects a single correlation op")
    op = intent.ops[0]
    if len(op.inputs) != 3:
        raise CudaLoweringError("correlation expects 3 inputs (src0,src1,out_shift)")
    src0_name, src1_name, out_shift_name = op.inputs
    out_name = op.output

    OC = _as_int(bindings.get("out_channel"), name="out_channel")
    IC = _as_int(bindings.get("in_channel"), name="in_channel")
    H = _as_int(bindings.get("height"), name="height")
    W = _as_int(bindings.get("width"), name="width")
    total = OC * H * W
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=128)
    if block_x <= 0:
        block_x = 128
    if block_x > 1024:
        block_x = 1024
    grid_x = (total + block_x - 1) // block_x

    oc_is_tensor = _is_scalar_tensor(intent, "out_channel", dtype="i32")
    ic_is_tensor = _is_scalar_tensor(intent, "in_channel", dtype="i32")
    h_is_tensor = _is_scalar_tensor(intent, "height", dtype="i32")
    w_is_tensor = _is_scalar_tensor(intent, "width", dtype="i32")
    sh_is_tensor = _is_scalar_tensor(intent, "out_shift", dtype="i32")

    oc_param = "const int* out_channel_ptr" if oc_is_tensor else "int out_channel"
    ic_param = "const int* in_channel_ptr" if ic_is_tensor else "int in_channel"
    h_param = "const int* height_ptr" if h_is_tensor else "int height"
    w_param = "const int* width_ptr" if w_is_tensor else "int width"
    sh_param = "const int* out_shift_ptr" if sh_is_tensor else "int out_shift"

    oc_load = "const int out_channel = out_channel_ptr ? out_channel_ptr[0] : 0;" if oc_is_tensor else ""
    ic_load = "const int in_channel = in_channel_ptr ? in_channel_ptr[0] : 0;" if ic_is_tensor else ""
    h_load = "const int height = height_ptr ? height_ptr[0] : 0;" if h_is_tensor else ""
    w_load = "const int width = width_ptr ? width_ptr[0] : 0;" if w_is_tensor else ""
    sh_load = "const int out_shift = out_shift_ptr ? out_shift_ptr[0] : 0;" if sh_is_tensor else ""

    cuda_src = f"""
#include "kernels/correlation.cuh"

extern "C" __global__ void {intent.name}(
    const int8_t* __restrict__ {src0_name},
    const int8_t* __restrict__ {src1_name},
    int8_t* __restrict__ {out_name},
    {oc_param}, {ic_param}, {h_param}, {w_param}, {sh_param}) {{
  {oc_load}
  {ic_load}
  {h_load}
  {w_load}
  {sh_load}
  constexpr int BLOCK_THREADS = {block_x};
  intentir_cuda::correlation_i8<BLOCK_THREADS, false>({src0_name}, {src1_name}, {out_name}, out_channel, in_channel, height, width, out_shift);
}}
""".lstrip()

    # Prefer scalar tensors if present in the IntentIR signature (AI-Bench).
    tensor_args = [src0_name, src1_name, out_name]
    arg_names = [src0_name, src1_name, out_name]
    scalar_args: Dict[str, str] = {}
    for dim_name in ["out_channel", "in_channel", "height", "width", "out_shift"]:
        if _is_scalar_tensor(intent, dim_name, dtype="i32"):
            tensor_args.append(dim_name)
            arg_names.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
            arg_names.append(dim_name)

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def lower_intent_to_cuda_kernel(
    intent: IntentFunction,
    *,
    shape_bindings: Mapping[str, Any],
    schedule_override: ScheduleSketch | Dict[str, Any] | None = None,
) -> CudaLoweredKernel:
    """
    Lower an IntentFunction into a single CUDA kernel (MVP).

    Note: The lowering currently targets the AI-Bench8 kernels and a small set
    of common patterns (softmax2d-last, layernorm2d).
    """
    bindings: Dict[str, Any] = {}
    for k, v in dict(shape_bindings).items():
        key = str(k)
        if isinstance(v, bool):
            bindings[key] = int(v)
            continue
        if isinstance(v, int):
            bindings[key] = int(v)
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        bindings[key] = int(fv) if float(fv).is_integer() else float(fv)

    # When a caller provides an explicit schedule override (e.g., freeze/retune
    # experiments), treat the schedule as authoritative. Otherwise, treat the
    # recovered schedule as a hint and allow backend heuristics to override it.
    if schedule_override is not None:
        bindings.setdefault("CUDA_RESPECT_SCHEDULE", 1)
    else:
        bindings.setdefault("CUDA_RESPECT_SCHEDULE", 0)

    # Allow caller overrides for schedule knobs (used by tuning / freeze-vs-retune).
    if schedule_override is not None:
        if isinstance(schedule_override, ScheduleSketch):
            intent.schedule = schedule_override
        elif isinstance(schedule_override, dict):
            intent.schedule = ScheduleSketch(
                tile_m=schedule_override.get("tile_m"),
                tile_n=schedule_override.get("tile_n"),
                tile_k=schedule_override.get("tile_k"),
                vec_width=schedule_override.get("vec_width"),
                pipeline_depth=schedule_override.get("pipeline_depth"),
                axis_bindings=dict(schedule_override.get("axis_bindings") or {}),
                vec_axis=(schedule_override.get("vec_axis") if isinstance(schedule_override.get("vec_axis"), str) else None),
                parallel_axes=[str(x) for x in (schedule_override.get("parallel_axes") or [])],
                memory_hint=dict(schedule_override.get("memory_hint") or {}),
            )

    # Default to the C++ codegen tool (real backend feel). Set INTENTIR_CUDA_CODEGEN=py
    # to force the legacy Python codegen, or INTENTIR_CUDA_CODEGEN_STRICT=1 to
    # disable fallback.
    raw_codegen = os.getenv("INTENTIR_CUDA_CODEGEN", "cpp").strip().lower()
    force_python_codegen = raw_codegen in {"0", "false", "no", "n", "py", "python"}
    use_cpp_codegen = not force_python_codegen
    strict_cpp = os.getenv("INTENTIR_CUDA_CODEGEN_STRICT", "0").strip().lower() in {"1", "true", "yes", "y"}
    if use_cpp_codegen:
        try:
            from .cpp_driver import lower_intent_to_cuda_kernel_cpp  # noqa: PLC0415

            j = lower_intent_to_cuda_kernel_cpp(intent, bindings=bindings)
            launch_j = j.get("launch") if isinstance(j.get("launch"), dict) else {}
            grid = launch_j.get("grid")
            block = launch_j.get("block")
            shared_mem = launch_j.get("shared_mem", 0)
            if not (isinstance(grid, list) and len(grid) == 3 and isinstance(block, list) and len(block) == 3):
                raise CudaLoweringError("cuda cpp codegen returned invalid launch config")
            launch = CudaLaunch(grid=(int(grid[0]), int(grid[1]), int(grid[2])), block=(int(block[0]), int(block[1]), int(block[2])), shared_mem=int(shared_mem))
            return CudaLoweredKernel(
                kernel_name=str(j.get("kernel_name") or intent.name),
                cuda_src=str(j.get("cuda_src") or ""),
                io_spec=j.get("io_spec") if isinstance(j.get("io_spec"), dict) else {},
                launch=launch,
                output_names=[str(x) for x in (j.get("output_names") or [])],
                bindings=j.get("bindings") if isinstance(j.get("bindings"), dict) else dict(bindings),
            )
        except Exception:
            if strict_cpp:
                raise
            # Fallback to the Python codegen for kernels not yet supported by the C++ tool.

    # 1) Direct kernels.
    if intent.ops:
        # Robustness across pipeline versions: allow `const* + rope` (rope last).
        if intent.ops[-1].op == "rope" and all(o.op == "const" for o in intent.ops[:-1]):
            return _kernel_rope_f32(intent, bindings)

        if len(intent.ops) == 1:
            op0 = intent.ops[0].op
            if op0 == "matmul":
                out_name = str((intent.outputs or [""])[0])
                out_shape = _shape_values(intent, out_name)
                if len(out_shape) == 3:
                    return _kernel_bmm_3d_f32(intent, bindings)
                return _kernel_matmul_f32(intent, bindings)
            if op0 == "conv1d":
                return _kernel_conv1d_ncl_f32(intent, bindings)
            if op0 == "conv2d":
                return _kernel_conv2d_nchw_f32(intent, bindings)
            if op0 == "conv3d":
                return _kernel_conv3d_ncdhw_f32(intent, bindings)
            if op0 == "conv_depthwise2d":
                return _kernel_conv_depthwise2d_nchw_f32(intent, bindings)
            if op0 == "dropout":
                return _kernel_dropout_f32(intent, bindings)
            if op0 == "cumsum":
                return _kernel_cumsum_f32(intent, bindings)
            if op0 == "cummax":
                return _kernel_cumext_1d_f32(intent, bindings, is_max=True)
            if op0 == "cummin":
                return _kernel_cumext_1d_f32(intent, bindings, is_max=False)
            if op0 == "correlation":
                return _kernel_correlation_i8(intent, bindings)
            if op0 == "resize":
                return _kernel_resize_bilinear2x_i8(intent, bindings)
            if op0 == "warp":
                return _kernel_warp_q8_8_i8_i16(intent, bindings)
            if op0 == "transpose":
                return _kernel_transpose_2d_f32(intent, bindings)
            if op0 == "concat":
                return _kernel_concat_2d_f32(intent, bindings)
            if op0 == "pad":
                return _kernel_pad_2d_f32(intent, bindings)
            if op0 == "glu":
                return _kernel_glu_2d_f32(intent, bindings)
            if op0 == "reduce_sum":
                return _kernel_reduce_sum_2d_axis1_f32(intent, bindings)
            if op0 == "reduce_max":
                return _kernel_reduce_max_2d_axis1_f32(intent, bindings)
            if op0 == "reduce_min":
                return _kernel_reduce_min_2d_axis1_f32(intent, bindings)
            if op0 == "gather":
                return _kernel_gather2d_f32(intent, bindings)
            if op0 in {"argmax", "argmin"}:
                return _kernel_arg_reduce_2d_axis1_i32(intent, bindings)
            if op0 == "avg_pool2d":
                return _kernel_avg_pool2d_nchw_f32(intent, bindings)
            if op0 == "scaled_dot_product_attention":
                return _kernel_scaled_dot_product_attention_bhsd_f32(intent, bindings)

    # 2) Pattern-based kernels (fused).
    outs = set(intent.outputs or [])
    op_names = {o.op for o in (intent.ops or [])}
    # any/all row-reduction family.
    op_seq = [o.op for o in (intent.ops or [])]
    if op_seq in (["const", "ne", "reduce_any"], ["const", "eq", "reduce_any"], ["const", "eq", "reduce_any", "not"], ["const", "ne", "reduce_any", "not"]):
        return _kernel_any_dim_f32_to_i1(intent, bindings)
    if op_seq == ["const", "ne", "cast", "reduce_sum"]:
        return _kernel_count_nonzero_2d_i64(intent, bindings)
    if op_seq == ["iota", "gather"]:
        return _kernel_diag_2d_f32(intent, bindings)
    if op_seq == ["cast", "cast", "mul", "reduce_sum"]:
        return _kernel_dot_1d_f32(intent, bindings)
    if op_seq == ["const", "broadcast_in_dim", "iota", "iota", "iota", "ne", "not", "gather", "where"]:
        return _kernel_diag_embed_2d_f32(intent, bindings)
    if op_seq == ["transpose", "matmul", "mul", "add", "softmax", "matmul"]:
        return _kernel_flash_attn_varlen_decomposed_bhsd_f32(intent, bindings)
    if op_seq == ["conv2d", "broadcast_in_dim", "add"]:
        return _kernel_conv2d_nchw_bias_pattern_f32(intent, bindings)
    if op_seq in (["matmul", "mul", "mul", "add", "cast"], ["matmul", "mul", "mul", "add"]):
        out_shape = _shape_values(intent, str((intent.outputs or [""])[0]))
        if len(out_shape) == 2:
            return _kernel_addmm_2d_f32(intent, bindings)
        if len(out_shape) == 1:
            return _kernel_addmv_2d_f32(intent, bindings)
        if len(out_shape) == 3:
            return _kernel_baddbmm_3d_f32(intent, bindings)
    if op_seq == ["sub", "abs", "abs", "mul", "add", "le", "not", "reduce_any", "not"]:
        return _kernel_allclose_2d_f32(intent, bindings)
    if {"output_1", "mean", "inv_std", "running_mean_out", "running_var_out"}.issubset(outs) and {
        "reduce_sum",
        "broadcast_in_dim",
        "rsqrt",
    }.issubset(op_names):
        return _kernel_batch_norm2d_f32(intent, bindings)
    if str(intent.name).lower() in {"group_norm_kernel", "group_norm"}:
        return _kernel_group_norm_3d_f32(intent, bindings)
    if {"Y", "Mean", "Rstd"}.issubset(outs):
        return _kernel_layernorm_2d_f32(intent, bindings)
    # Softmax: recognize by name or by presence of reduce_max + exp + reduce_sum.
    if ("softmax" in str(intent.name).lower()) or ({"reduce_max", "reduce_sum", "exp", "div"}.issubset(op_names)):
        return _kernel_softmax_2d_last_f32(intent, bindings)

    # 3) Generic fused elementwise lowering.
    # This is intentionally conservative: only elementwise/broadcast/const/where/cast ops.
    elem_ops = {
        "const",
        "iota",
        "identity",
        "broadcast_in_dim",
        "cast",
        "add",
        "sub",
        "mul",
        "div",
        "max",
        "min",
        "relu",
        "abs",
        "exp",
        "acos",
        "atan",
        "cos",
        "erf",
        "floor",
        "ceil",
        "rsqrt",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "bitwise_and",
        "bitwise_or",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "bitwise_not",
        "and",
        "or",
        "not",
        "where",
    }
    if intent.ops and all(str(o.op) in elem_ops for o in intent.ops):
        return _kernel_fused_elementwise(intent, bindings)

    raise CudaLoweringError(f"CUDA lowering unsupported for intent: name={intent.name} ops={sorted(op_names)}")


__all__ = ["CudaLoweringError", "CudaLoweredKernel", "lower_intent_to_cuda_kernel"]
