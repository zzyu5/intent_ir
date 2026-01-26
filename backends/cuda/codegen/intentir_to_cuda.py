"""
IntentIR -> CUDA kernel codegen (MVP).

Scope (initial):
- AI-Bench8 kernels used in paper experiments:
  - matmul, dropout, softmax, layernorm, correlation, resize, rope, warp

This codegen intentionally focuses on producing *runnable* CUDA for the paper.
It does not try to cover the full IntentIR op-set yet.
"""

from __future__ import annotations

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
    if key not in bindings:
        raise CudaLoweringError(f"missing binding for dim {name} ({key})")
    return _as_int(bindings[key], name=name)


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


def _kernel_fused_elementwise(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    """
    Generic fused elementwise lowering for small IntentIR graphs.

    Supports:
      - unary/binary float ops: add/sub/mul/div/max/min/relu/abs/exp/floor/rsqrt
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
        dt = str(t.dtype)
        if t.layout.kind != "row_major":
            raise CudaLoweringError("elementwise lowering supports only row_major tensors")
        in_shape = _shape_values(intent, name)
        idx_expr = _emit_broadcast_index_expr(out_rank=out_rank, in_shape=in_shape, out_idxs=idx_vars, dim_expr=dim_expr)
        return f"{name}[(size_t)({idx_expr})]"

    def val(name: str) -> str:
        n = str(name)
        if n in value_expr:
            return value_expr[n]
        if n not in intent.tensors:
            raise CudaLoweringError(f"elementwise: unknown value {n}")
        # Base input tensor load.
        return load_tensor(n)

    code_lines: list[str] = []
    for op in intent.ops or []:
        opname = str(op.op)
        outn = str(op.output)
        if outn not in intent.tensors:
            raise CudaLoweringError(f"elementwise: op output missing from tensors: {outn}")
        out_dt = str(intent.tensors[outn].dtype)
        cty = _c_type(out_dt)

        # Helper: declare + assign.
        def emit_assign(expr: str) -> None:
            # Prefix SSA locals to avoid collisions with tensor argument names (e.g., output "Out").
            vname = "v_" + _c_ident(outn)
            code_lines.append(f"{cty} {vname} = {expr};")
            value_expr[outn] = vname
            value_type[outn] = out_dt

        if opname == "const":
            emit_assign(_c_scalar_literal(out_dt, (op.attrs or {}).get("value", 0)))
        elif opname == "identity":
            if len(op.inputs) != 1:
                raise CudaLoweringError("identity expects 1 input")
            emit_assign(val(op.inputs[0]))
        elif opname == "broadcast_in_dim":
            if len(op.inputs) != 1:
                raise CudaLoweringError("broadcast_in_dim expects 1 input")
            emit_assign(val(op.inputs[0]))
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
            emit_assign(f"__expf({x})")
        elif opname == "floor":
            if len(op.inputs) != 1:
                raise CudaLoweringError("floor expects 1 input")
            x = val(op.inputs[0])
            emit_assign(f"floorf({x})")
        elif opname == "rsqrt":
            if len(op.inputs) != 1:
                raise CudaLoweringError("rsqrt expects 1 input")
            x = val(op.inputs[0])
            emit_assign(f"rsqrtf({x})")
        elif opname in {"ne", "lt", "le", "gt", "ge"}:
            if len(op.inputs) != 2:
                raise CudaLoweringError(f"{opname} expects 2 inputs")
            a = val(op.inputs[0])
            b = val(op.inputs[1])
            op_map = {"ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
            emit_assign(f"({a} {op_map[opname]} {b})")
        elif opname in {"and", "or"}:
            if len(op.inputs) != 2:
                raise CudaLoweringError(f"{opname} expects 2 inputs")
            a = val(op.inputs[0])
            b = val(op.inputs[1])
            op_map = {"and": "&&", "or": "||"}
            emit_assign(f"({a} {op_map[opname]} {b})")
        elif opname == "not":
            if len(op.inputs) != 1:
                raise CudaLoweringError("not expects 1 input")
            a = val(op.inputs[0])
            emit_assign(f"(!{a})")
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

        wmma_tile_m = 16 * wmma_warps_m
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
        wmma_pipe_stages = int(bindings.get("WMMA_PIPE_STAGES", 0) or 0)
        if not wmma_use_cp_async:
            wmma_pipe_stages = 1
        else:
            if wmma_pipe_stages <= 0:
                # Our cp.async pipeline is validated for triple buffering.
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
        # NOTE: our cp.async pipeline implementation is only validated for PIPE_STAGES==3.
        # If we ended up at PIPE_STAGES==2 due to shared-memory pressure, first try to
        # re-enable triple buffering after STAGE_K clamping; if that still doesn't fit,
        # fall back to the (correct but slower) synchronous path.
        wmma_force_sync = False
        # NOTE: PIPE_STAGES==2 is not validated for correctness yet; prefer
        # triple-buffering when possible, otherwise fall back to a synchronous
        # path to preserve correctness.
        if wmma_pipe_stages == 2:
            bytes3 = _wmma_smem_bytes(wmma_tile_m, wmma_tile_n, wmma_stage_k, 3)
            if bytes3 <= max_smem_optin:
                wmma_pipe_stages = 3
                shared_bytes = bytes3
            else:
                wmma_force_sync = True
                wmma_pipe_stages = 1
                shared_bytes = _wmma_smem_bytes(wmma_tile_m, wmma_tile_n, wmma_stage_k, wmma_pipe_stages)

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
        bounds_guard = "" if specialize_full_tile else "if (row0 >= M || col0 >= N) return;"
        full_tile_expr = (
            "true"
            if specialize_full_tile
            else f"(row0 + TILE_M <= M) && (col0 + TILE_N <= N) && ((K % STAGE_K) == 0) && ((K & 3) == 0) && ((N & 3) == 0) && ({0 if wmma_disable_fastpath else 1} != 0)"
        )
        cuda_src = f"""
#include <mma.h>
#include "intentir_cuda_ops.cuh"
using namespace nvcuda::wmma;

template <int TILE_M, int TILE_N, int STAGE_K, int AS_PAD, int BS_PAD>
__device__ __forceinline__ void intentir_cp_async_tile_f32(
    int buf,
    int k_base,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ As,
    float* __restrict__ Bs,
    int row0,
    int col0,
    int K,
    int N) {{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  constexpr int AS_LD = STAGE_K + AS_PAD;
  constexpr int BS_LD = TILE_N + BS_PAD;
  // Copy a TILE_M x STAGE_K slice of A and a STAGE_K x TILE_N slice of B.
  // Use 16-byte copies (float4) for coalesced global memory access.
  constexpr int VEC = 4;
  constexpr int A_EL = TILE_M * STAGE_K;
  constexpr int B_EL = STAGE_K * TILE_N;
  constexpr int A_V = A_EL / VEC;
  constexpr int B_V = B_EL / VEC;
  const int tid = (int)threadIdx.x;
  #pragma unroll
  for (int idx = tid; idx < A_V; idx += (int)blockDim.x) {{
    const int off = idx * VEC;
    const int r = off / STAGE_K;
    const int kk = off - r * STAGE_K;
    const int gr = row0 + r;
    const int gk = k_base + kk;
    // A tile is reused across WARPS_N warps; cache when WARPS_N dominates.
    intentir_cp_async_{wmma_cp_a_policy}_16(As + (size_t)buf * (size_t)(TILE_M * AS_LD) + (size_t)r * (size_t)AS_LD + (size_t)kk,
                         A + (size_t)gr * (size_t)K + (size_t)gk);
  }}
  #pragma unroll
  for (int bidx = tid; bidx < B_V; bidx += (int)blockDim.x) {{
    const int off = bidx * VEC;
    const int kk = off / TILE_N;
    const int n = off - kk * TILE_N;
    const int gn = col0 + n;
    const int gk = k_base + kk;
    // B tile is reused across WARPS_M warps; cache when WARPS_M dominates.
    intentir_cp_async_{wmma_cp_b_policy}_16(Bs + (size_t)buf * (size_t)(STAGE_K * BS_LD) + (size_t)kk * (size_t)BS_LD + (size_t)n,
                         B + (size_t)gk * (size_t)N + (size_t)gn);
  }}
#else
  (void)buf;
  (void)k_base;
  (void)A;
  (void)B;
  (void)As;
  (void)Bs;
  (void)row0;
  (void)col0;
  (void)K;
  (void)N;
#endif
	}}

	__device__ __forceinline__ void intentir_mma_tf32_m16n16k8_rr(
	    fragment<accumulator, 16, 16, 8, float>& acc,
	    const fragment<matrix_a, 16, 16, 8, precision::tf32, row_major>& a_frag,
	    const fragment<matrix_b, 16, 16, 8, precision::tf32, row_major>& b_frag) {{
	#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
	  const int a0 = __float_as_int(a_frag.x[0]);
	  const int a1 = __float_as_int(a_frag.x[1]);
	  const int a2 = __float_as_int(a_frag.x[2]);
	  const int a3 = __float_as_int(a_frag.x[3]);
	  const int b0 = __float_as_int(b_frag.x[0]);
	  const int b1 = __float_as_int(b_frag.x[1]);
	  const int b2 = __float_as_int(b_frag.x[2]);
	  const int b3 = __float_as_int(b_frag.x[3]);

	  float d0 = acc.x[0];
	  float d1 = acc.x[1];
	  float d2 = acc.x[2];
	  float d3 = acc.x[3];
	  float d4 = acc.x[4];
	  float d5 = acc.x[5];
	  float d6 = acc.x[6];
	  float d7 = acc.x[7];

	  // Compute the 16x16 accumulator as two 16x8 MMA operations in N.
	  asm volatile(
	      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
	      "{{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%0, %1, %2, %3}};\\n"
	      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
	      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
	  asm volatile(
	      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
	      "{{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%0, %1, %2, %3}};\\n"
	      : "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
	      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));

	  acc.x[0] = d0;
	  acc.x[1] = d1;
	  acc.x[2] = d2;
	  acc.x[3] = d3;
	  acc.x[4] = d4;
	  acc.x[5] = d5;
	  acc.x[6] = d6;
	  acc.x[7] = d7;
	#else
	  mma_sync(acc, a_frag, b_frag, acc);
	#endif
	}}

extern "C" __global__ void {intent.name}(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    {m_param}, {n_param}, {k_param}) {{
  {m_load}
  {n_load}
  {k_load}

  // Assumes M,N are multiples of 16 and K is a multiple of 8 (enforced by host-side lowering).
  constexpr int WARPS_M = {wmma_warps_m};
  constexpr int WARPS_N = {wmma_warps_n};
  constexpr int FRAG_N = {wmma_frag_n};
  constexpr int TILE_M = 16 * WARPS_M;
  constexpr int TILE_N = 16 * WARPS_N * FRAG_N;
  constexpr int STAGE_K = {wmma_stage_k};
  constexpr int AS_PAD = {wmma_as_pad};
  constexpr int BS_PAD = {wmma_bs_pad};
  constexpr int AS_LD = STAGE_K + AS_PAD;
  constexpr int BS_LD = TILE_N + BS_PAD;

  // WARPS_M * WARPS_N warps compute a TILE_M x TILE_N output tile.
  // Stage A/B tiles into shared memory. For the fast-path, we use cp.async with
  // pipelined buffering to overlap global->shared copies with MMA compute.
  extern __shared__ __align__(16) float intentir_smem[];
  float* __restrict__ As = intentir_smem;
  constexpr int PIPE_STAGES = {wmma_pipe_stages};
  float* __restrict__ Bs = intentir_smem + (size_t)PIPE_STAGES * (size_t)TILE_M * (size_t)AS_LD;

  const int warp = (int)(threadIdx.x >> 5);  // 0..(WARPS_M*WARPS_N-1)
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp - warp_m * WARPS_N;

  const int row0 = (int)blockIdx.y * TILE_M;
  const int col0 = (int)blockIdx.x * TILE_N;
  {bounds_guard}
  // Fast-path condition: this CTA is fully inside bounds and K tiles exactly.
  // When CUDA_SPECIALIZE_DIMS is enabled and the shape is divisible, we
  // constant-fold this to `true` to compile out the guarded fallback path.
  const bool full_tile = {full_tile_expr};

  fragment<accumulator, 16, 16, 8, float> acc0;
  fill_fragment(acc0, 0.0f);
#if {wmma_frag_n} > 1
  fragment<accumulator, 16, 16, 8, float> acc1;
  fill_fragment(acc1, 0.0f);
#endif

  if (full_tile) {{
    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
    // Fast path:
    //   - cp.async pipelining on Ampere+ (when enabled)
    //   - otherwise, synchronous float4 vectorized copies (still no bounds checks)
#if {1 if wmma_use_cp_async else 0}
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // cp.async pipelining (Ampere+). Use 2-stage (double-buffer) or 3-stage buffering.
#if PIPE_STAGES == 2
    // Prefetch stage 0.
    intentir_cp_async_tile_f32<TILE_M, TILE_N, STAGE_K, AS_PAD, BS_PAD>(0, 0, A, B, As, Bs, row0, col0, K, N);
    intentir_cp_async_commit();
    intentir_cp_async_wait_all();
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += STAGE_K) {{
      const int next_k0 = k0 + STAGE_K;
      const int next_buf = buf ^ 1;
      if (next_k0 < K) {{
        intentir_cp_async_tile_f32<TILE_M, TILE_N, STAGE_K, AS_PAD, BS_PAD>(next_buf, next_k0, A, B, As, Bs, row0, col0, K, N);
        intentir_cp_async_commit();
      }}

      #pragma unroll
      for (int kk0 = 0; kk0 < STAGE_K; kk0 += 8) {{
        load_matrix_sync(a_frag,
                         As + (size_t)buf * (size_t)(TILE_M * AS_LD) + (size_t)(warp_m * 16) * (size_t)AS_LD + (size_t)kk0,
                         AS_LD);
	        load_matrix_sync(b_frag,
	                         Bs + (size_t)buf * (size_t)(STAGE_K * BS_LD) + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 0) * 16),
	                         BS_LD);
	        intentir_mma_tf32_m16n16k8_rr(acc0, a_frag, b_frag);
	#if {wmma_frag_n} > 1
	        load_matrix_sync(b_frag,
	                         Bs + (size_t)buf * (size_t)(STAGE_K * BS_LD) + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 1) * 16),
	                         BS_LD);
	        intentir_mma_tf32_m16n16k8_rr(acc1, a_frag, b_frag);
	#endif
      }}

      if (next_k0 < K) {{
        intentir_cp_async_wait_all();
        __syncthreads();
      }}
      buf = next_buf;
    }}
#else
    // Triple-buffering. Prefetch stages 0 and 1, then keep one stage in flight
    // while computing on the current stage.
    const int num_tiles = K / STAGE_K;
    intentir_cp_async_tile_f32<TILE_M, TILE_N, STAGE_K, AS_PAD, BS_PAD>(0, 0, A, B, As, Bs, row0, col0, K, N);
    intentir_cp_async_commit();
    if (num_tiles > 1) {{
      intentir_cp_async_tile_f32<TILE_M, TILE_N, STAGE_K, AS_PAD, BS_PAD>(1, STAGE_K, A, B, As, Bs, row0, col0, K, N);
      intentir_cp_async_commit();
    }}
    // Wait until stage 0 is ready (leave stage 1 possibly in flight).
    intentir_cp_async_wait_group<1>();
    __syncthreads();

    for (int stage = 0; stage < num_tiles; ++stage) {{
      const int buf = stage % PIPE_STAGES;
      // Prefetch stage+2 (if any) to overlap with compute on `buf`.
      const int pf_stage = stage + 2;
      if (pf_stage < num_tiles) {{
        const int pf_buf = pf_stage % PIPE_STAGES;
        intentir_cp_async_tile_f32<TILE_M, TILE_N, STAGE_K, AS_PAD, BS_PAD>(
            pf_buf, pf_stage * STAGE_K, A, B, As, Bs, row0, col0, K, N);
        intentir_cp_async_commit();
      }}

      #pragma unroll
      for (int kk0 = 0; kk0 < STAGE_K; kk0 += 8) {{
        load_matrix_sync(a_frag,
                         As + (size_t)buf * (size_t)(TILE_M * AS_LD) + (size_t)(warp_m * 16) * (size_t)AS_LD + (size_t)kk0,
                         AS_LD);
	        load_matrix_sync(b_frag,
	                         Bs + (size_t)buf * (size_t)(STAGE_K * BS_LD) + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 0) * 16),
	                         BS_LD);
	        intentir_mma_tf32_m16n16k8_rr(acc0, a_frag, b_frag);
	#if {wmma_frag_n} > 1
	        load_matrix_sync(b_frag,
	                         Bs + (size_t)buf * (size_t)(STAGE_K * BS_LD) + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 1) * 16),
	                         BS_LD);
	        intentir_mma_tf32_m16n16k8_rr(acc1, a_frag, b_frag);
	#endif
      }}

      // Ensure the next stage is ready before advancing.
      if (stage + 1 < num_tiles) {{
        if (stage + 2 < num_tiles) {{
          // Leave one stage in flight.
          intentir_cp_async_wait_group<1>();
        }} else {{
          // Drain the pipeline for the tail.
          intentir_cp_async_wait_group<0>();
        }}
        __syncthreads();
      }}
    }}
#endif
#else
    // Pre-Ampere fallback for the "cp.async" build: use synchronous vector copies.
    constexpr int VEC = 4;
    constexpr int A_EL = TILE_M * STAGE_K;
    constexpr int B_EL = STAGE_K * TILE_N;
    constexpr int A_V = A_EL / VEC;
    constexpr int B_V = B_EL / VEC;
    const int tid = (int)threadIdx.x;
    for (int k0 = 0; k0 < K; k0 += STAGE_K) {{
      for (int idx = tid; idx < A_V; idx += (int)blockDim.x) {{
        const int off = idx * VEC;
        const int r = off / STAGE_K;
        const int kk = off - r * STAGE_K;
        const int gr = row0 + r;
        const int gk = k0 + kk;
        *reinterpret_cast<float4*>(As + (size_t)r * (size_t)AS_LD + (size_t)kk) =
            *reinterpret_cast<const float4*>(A + (size_t)gr * (size_t)K + (size_t)gk);
      }}
      for (int bidx = tid; bidx < B_V; bidx += (int)blockDim.x) {{
        const int off = bidx * VEC;
        const int kk = off / TILE_N;
        const int n = off - kk * TILE_N;
        const int gn = col0 + n;
        const int gk = k0 + kk;
        *reinterpret_cast<float4*>(Bs + (size_t)kk * (size_t)BS_LD + (size_t)n) =
            *reinterpret_cast<const float4*>(B + (size_t)gk * (size_t)N + (size_t)gn);
      }}
      __syncthreads();
      #pragma unroll
      for (int kk0 = 0; kk0 < STAGE_K; kk0 += 8) {{
        load_matrix_sync(a_frag,
                         As + (size_t)(warp_m * 16) * (size_t)AS_LD + (size_t)kk0,
                         AS_LD);
	        load_matrix_sync(b_frag,
	                         Bs + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 0) * 16),
	                         BS_LD);
	        intentir_mma_tf32_m16n16k8_rr(acc0, a_frag, b_frag);
	#if {wmma_frag_n} > 1
	        load_matrix_sync(b_frag,
	                         Bs + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 1) * 16),
	                         BS_LD);
	        intentir_mma_tf32_m16n16k8_rr(acc1, a_frag, b_frag);
	#endif
      }}
      __syncthreads();
    }}
#endif  // __CUDA_ARCH__ >= 800
#else
    // Synchronous float4 vectorized copies (no cp.async).
    constexpr int VEC = 4;
    constexpr int A_EL = TILE_M * STAGE_K;
    constexpr int B_EL = STAGE_K * TILE_N;
    constexpr int A_V = A_EL / VEC;
    constexpr int B_V = B_EL / VEC;
    const int tid = (int)threadIdx.x;
    for (int k0 = 0; k0 < K; k0 += STAGE_K) {{
      for (int idx = tid; idx < A_V; idx += (int)blockDim.x) {{
        const int off = idx * VEC;
        const int r = off / STAGE_K;
        const int kk = off - r * STAGE_K;
        const int gr = row0 + r;
        const int gk = k0 + kk;
        *reinterpret_cast<float4*>(As + (size_t)r * (size_t)AS_LD + (size_t)kk) =
            *reinterpret_cast<const float4*>(A + (size_t)gr * (size_t)K + (size_t)gk);
      }}
      for (int bidx = tid; bidx < B_V; bidx += (int)blockDim.x) {{
        const int off = bidx * VEC;
        const int kk = off / TILE_N;
        const int n = off - kk * TILE_N;
        const int gn = col0 + n;
        const int gk = k0 + kk;
        *reinterpret_cast<float4*>(Bs + (size_t)kk * (size_t)BS_LD + (size_t)n) =
            *reinterpret_cast<const float4*>(B + (size_t)gk * (size_t)N + (size_t)gn);
      }}
      __syncthreads();
      #pragma unroll
      for (int kk0 = 0; kk0 < STAGE_K; kk0 += 8) {{
        load_matrix_sync(a_frag,
                         As + (size_t)(warp_m * 16) * (size_t)AS_LD + (size_t)kk0,
                         AS_LD);
	        load_matrix_sync(b_frag,
	                         Bs + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 0) * 16),
	                         BS_LD);
	        intentir_mma_tf32_m16n16k8_rr(acc0, a_frag, b_frag);
	#if {wmma_frag_n} > 1
	        load_matrix_sync(b_frag,
	                         Bs + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 1) * 16),
	                         BS_LD);
	        intentir_mma_tf32_m16n16k8_rr(acc1, a_frag, b_frag);
	#endif
      }}
      __syncthreads();
    }}
#endif

    const int out_r = row0 + warp_m * 16;
    const int out_c0 = col0 + (warp_n * FRAG_N + 0) * 16;
    store_matrix_sync(C + (size_t)out_r * (size_t)N + (size_t)out_c0, acc0, (unsigned)N, mem_row_major);
#if {wmma_frag_n} > 1
    const int out_c1 = col0 + (warp_n * FRAG_N + 1) * 16;
    store_matrix_sync(C + (size_t)out_r * (size_t)N + (size_t)out_c1, acc1, (unsigned)N, mem_row_major);
#endif
    return;
  }}

  // Fallback: guarded synchronous loads (keeps correctness for non-multiple shapes).
  fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
  fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
  for (int k0 = 0; k0 < K; k0 += STAGE_K) {{
    const int tid = (int)threadIdx.x;
    const int total = TILE_M * STAGE_K + STAGE_K * TILE_N;
    for (int idx = tid; idx < total; idx += (int)blockDim.x) {{
      if (idx < TILE_M * STAGE_K) {{
        const int r = idx / STAGE_K;
        const int kk = idx - r * STAGE_K;
        const int gr = row0 + r;
        const int gk = k0 + kk;
        As[(size_t)0 * (size_t)(TILE_M * AS_LD) + (size_t)r * (size_t)AS_LD + (size_t)kk] =
            (gr < M && gk < K) ? intentir_ldg_f32(A + (size_t)gr * (size_t)K + (size_t)gk) : 0.0f;
      }} else {{
        const int bidx = idx - TILE_M * STAGE_K;
        const int kk = bidx / TILE_N;
        const int n = bidx - kk * TILE_N;
        const int gn = col0 + n;
        const int gk = k0 + kk;
        Bs[(size_t)0 * (size_t)(STAGE_K * BS_LD) + (size_t)kk * (size_t)BS_LD + (size_t)n] =
            (gn < N && gk < K) ? intentir_ldg_f32(B + (size_t)gk * (size_t)N + (size_t)gn) : 0.0f;
      }}
    }}
    __syncthreads();
    #pragma unroll
    for (int kk0 = 0; kk0 < STAGE_K; kk0 += 8) {{
      load_matrix_sync(a_frag,
                       As + (size_t)0 * (size_t)(TILE_M * AS_LD) + (size_t)(warp_m * 16) * (size_t)AS_LD + (size_t)kk0,
                       AS_LD);
      load_matrix_sync(b_frag,
                       Bs + (size_t)0 * (size_t)(STAGE_K * BS_LD) + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 0) * 16),
                       BS_LD);
      intentir_mma_tf32_m16n16k8_rr(acc0, a_frag, b_frag);
#if {wmma_frag_n} > 1
      load_matrix_sync(b_frag,
                       Bs + (size_t)0 * (size_t)(STAGE_K * BS_LD) + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 1) * 16),
                       BS_LD);
      intentir_mma_tf32_m16n16k8_rr(acc1, a_frag, b_frag);
#endif
    }}
    __syncthreads();
  }}

  const int out_r = row0 + warp_m * 16;
  const int out_c0 = col0 + (warp_n * FRAG_N + 0) * 16;
  if (out_r < M && out_c0 < N) {{
    store_matrix_sync(C + (size_t)out_r * (size_t)N + (size_t)out_c0, acc0, (unsigned)N, mem_row_major);
  }}
#if {wmma_frag_n} > 1
  const int out_c1 = col0 + (warp_n * FRAG_N + 1) * 16;
  if (out_r < M && out_c1 < N) {{
    store_matrix_sync(C + (size_t)out_r * (size_t)N + (size_t)out_c1, acc1, (unsigned)N, mem_row_major);
  }}
#endif
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

    # Use descriptor-like block size if present.
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024

    # Process multiple elements per thread to amortize Philox cost.
    #
    # AI-Bench dropout is bandwidth + RNG bound; using a larger EPT reduces launch
    # overhead and improves global memory coalescing when we map each thread to a
    # contiguous EPT segment.
    ept = int(op.attrs.get("elements_per_thread") or bindings.get("DROPOUT_EPT") or 8)
    if ept <= 0:
        ept = 1
    if ept > 8:
        ept = 8

    grid_x = (n + (block_x * ept) - 1) // (block_x * ept)

    cuda_src = f"""
#include <stdint.h>
#include <math.h>
#include "intentir_cuda_ops.cuh"

extern "C" __global__ void {intent.name}(const float* X, const float* p_ptr, const int* seed_ptr, float* Y, int64_t n_elements) {{
  constexpr int EPT = {ept};
  const int tid = (int)threadIdx.x;
  // Strided EPT mapping keeps each iteration's global loads fully coalesced:
  // for a fixed `e`, threads in a warp access consecutive indices.
  const int64_t base = (int64_t)blockIdx.x * (int64_t)blockDim.x * (int64_t)EPT + (int64_t)tid;
  // Loading these scalars per-thread avoids an unconditional __syncthreads().
  const float p = p_ptr ? p_ptr[0] : 0.0f;
  const uint64_t seed = (uint64_t)(seed_ptr ? (uint32_t)seed_ptr[0] : 0u);
  if (p <= 0.0f) {{
    #pragma unroll
    for (int e = 0; e < EPT; ++e) {{
      const int64_t i = base + (int64_t)e * (int64_t)blockDim.x;
      if (i >= n_elements) break;
      Y[i] = intentir_ldg_f32(X + i);
    }}
    return;
  }}
  if (p >= 1.0f) {{
    #pragma unroll
    for (int e = 0; e < EPT; ++e) {{
      const int64_t i = base + (int64_t)e * (int64_t)blockDim.x;
      if (i >= n_elements) break;
      Y[i] = 0.0f;
    }}
    return;
  }}
  const float inv_keep = __fdividef(1.0f, (1.0f - p));
  #pragma unroll
  for (int e = 0; e < EPT; ++e) {{
    const int64_t i = base + (int64_t)e * (int64_t)blockDim.x;
    if (i >= n_elements) break;
    const float x = intentir_ldg_f32(X + i);
    const uint32_t ctr = (uint32_t)i;
    const float r = intentir_uint_to_uniform_float_u32(intentir_philox_randint_u32(seed, ctr, {rounds}));
    Y[i] = (r > p) ? (x * inv_keep) : 0.0f;
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

    r_is_tensor = _is_scalar_tensor(intent, str(R_dim), dtype="i32")
    c_is_tensor = _is_scalar_tensor(intent, str(C_dim), dtype="i32")
    r_param = f"const int* {str(R_dim)}_ptr" if r_is_tensor else "int R"
    c_param = f"const int* {str(C_dim)}_ptr" if c_is_tensor else "int C"
    r_load = f"const int R = {str(R_dim)}_ptr ? {str(R_dim)}_ptr[0] : 0;" if r_is_tensor else ""
    c_load = f"const int C = {str(C_dim)}_ptr ? {str(C_dim)}_ptr[0] : 0;" if c_is_tensor else ""

    cuda_src = f"""
#include <math.h>
#include "intentir_cuda_ops.cuh"

__device__ __forceinline__ float intentir_warp_reduce_max(float v) {{
  for (int off = 16; off > 0; off >>= 1) {{
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
  }}
  return v;
}}

__device__ __forceinline__ float intentir_warp_reduce_sum(float v) {{
  for (int off = 16; off > 0; off >>= 1) {{
    v += __shfl_down_sync(0xffffffff, v, off);
  }}
  return v;
}}

__device__ __forceinline__ float intentir_block_allreduce_max(float v) {{
  __shared__ float shared[32];
  const int lane = (int)threadIdx.x & 31;
  const int warp = (int)threadIdx.x >> 5;
  const int num_warps = ((int)blockDim.x + 31) >> 5;
  v = intentir_warp_reduce_max(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  v = (warp == 0) ? ((lane < num_warps) ? shared[lane] : -INFINITY) : -INFINITY;
  if (warp == 0) v = intentir_warp_reduce_max(v);
  if ((int)threadIdx.x == 0) shared[0] = v;
  __syncthreads();
  return shared[0];
}}

__device__ __forceinline__ float intentir_block_allreduce_sum(float v) {{
  __shared__ float shared[32];
  const int lane = (int)threadIdx.x & 31;
  const int warp = (int)threadIdx.x >> 5;
  const int num_warps = ((int)blockDim.x + 31) >> 5;
  v = intentir_warp_reduce_sum(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  v = (warp == 0) ? ((lane < num_warps) ? shared[lane] : 0.0f) : 0.0f;
  if (warp == 0) v = intentir_warp_reduce_sum(v);
  if ((int)threadIdx.x == 0) shared[0] = v;
  __syncthreads();
  return shared[0];
}}

extern "C" __global__ void {intent.name}(const float* {in_name}, float* {out_name}, {r_param}, {c_param}) {{
  {r_load}
  {c_load}
  const int r = (int)blockIdx.x;
  if (r >= R) return;
  const float* __restrict__ in_row = {in_name} + (size_t)r * (size_t)C;
  float* __restrict__ out_row = {out_name} + (size_t)r * (size_t)C;
  const int tid = (int)threadIdx.x;
  constexpr int BLOCK_THREADS = {block_threads};
  constexpr int EPT = {ept};

  float tmax = -INFINITY;
  float expv[EPT];
  #pragma unroll
  for (int i = 0; i < EPT; ++i) {{
    const int c = tid + i * BLOCK_THREADS;
    float v = -INFINITY;
    if (c < C) v = intentir_ldg_f32(in_row + (size_t)c);
    expv[i] = v;
    tmax = fmaxf(tmax, v);
  }}
  const float mx = intentir_block_allreduce_max(tmax);

  float tsum = 0.0f;
  #pragma unroll
  for (int i = 0; i < EPT; ++i) {{
    const int c = tid + i * BLOCK_THREADS;
    float e = 0.0f;
    if (c < C) {{
      e = __expf(expv[i] - mx);
      tsum += e;
    }}
    expv[i] = e;
  }}
  const float sum = intentir_block_allreduce_sum(tsum);
  const float inv = __fdividef(1.0f, sum);
  #pragma unroll
  for (int i = 0; i < EPT; ++i) {{
    const int c = tid + i * BLOCK_THREADS;
    if (c < C) {{
      out_row[(size_t)c] = expv[i] * inv;
    }}
  }}
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
#include <math.h>

extern "C" __global__ void {intent.name}(
    const float* {X_name}, float* {Y_name}, const float* {W_name}, const float* {B_name}, float* {Mean_name}, float* {Rstd_name},
    int M, int N, float eps) {{
  const int m = (int)blockIdx.x;
  if (m >= M) return;
  __shared__ float smem[1024];
  float tsum = 0.0f;
  const float* xrow = {X_name} + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) {{
    tsum += xrow[n];
  }}
  smem[threadIdx.x] = tsum;
  __syncthreads();
  for (int off = ((int)blockDim.x >> 1); off > 0; off >>= 1) {{
    if ((int)threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
    __syncthreads();
  }}
  const float mean = smem[0] / (float)N;

  float tsq = 0.0f;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) {{
    float c = xrow[n] - mean;
    tsq += c * c;
  }}
  smem[threadIdx.x] = tsq;
  __syncthreads();
  for (int off = ((int)blockDim.x >> 1); off > 0; off >>= 1) {{
    if ((int)threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
    __syncthreads();
  }}
  const float var = smem[0] / (float)N;
  const float rstd = rsqrtf(var + eps);
  if ((int)threadIdx.x == 0) {{
    {Mean_name}[m] = mean;
    {Rstd_name}[m] = rstd;
  }}
  float* yrow = {Y_name} + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) {{
    float c = xrow[n] - mean;
    yrow[n] = (c * rstd) * {W_name}[n] + {B_name}[n];
  }}
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

    if not intent.ops or intent.ops[0].op != "rope":
        raise CudaLoweringError("rope lowering expects a single rope op")
    op = intent.ops[0]
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
#include "intentir_cuda_ops.cuh"

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
  const int pid_head_group = (int)blockIdx.x;
  const int pid_batch = (int)blockIdx.y;
  const int pid_seq = (int)blockIdx.z;
  // Grid is chosen from bindings; for the canonical mapping (HEADS_PER_BLOCK=1),
  // (pid_seq, pid_batch, pid_head) are always in range, so these checks are redundant.
  // Keep them only for non-canonical head grouping.
  if constexpr (HEADS_PER_BLOCK != 1) {{
    if (pid_batch >= BATCH_NUM || pid_seq >= SEQ_LEN) return;
  }}
  const int half = (int)(HEAD_DIM >> 1);
  const int head0 = pid_head_group * HEADS_PER_BLOCK;
  if constexpr (HEADS_PER_BLOCK != 1) {{
    if (head0 >= HEAD_NUM) return;
  }}

  using idx_t = {idx_t};
  const idx_t base0 = (idx_t)(((idx_t)pid_seq * (idx_t)BATCH_NUM + (idx_t)pid_batch) * (idx_t)HEAD_NUM) * (idx_t)HEAD_DIM;
  const idx_t cb0 = (idx_t)pid_seq * (idx_t)half;
  const int tid = (int)threadIdx.x;

  if constexpr (ROPE_VEC == 4) {{
    const int half4 = (int)(half >> 2);  // # of float4 packs
    const float4* __restrict__ cos4 = (const float4* __restrict__)({cos_name} + cb0);
    const float4* __restrict__ sin4 = (const float4* __restrict__)({sin_name} + cb0);
    if constexpr (HEADS_PER_BLOCK == 1) {{
      const int head = head0;
      const idx_t base = base0 + (idx_t)head * (idx_t)HEAD_DIM;
      const float4* __restrict__ x14 = (const float4* __restrict__)({in_name} + base);
      const float4* __restrict__ x24 = (const float4* __restrict__)({in_name} + base + (idx_t)half);
      float4* __restrict__ y14 = (float4* __restrict__)({out_name} + base);
      float4* __restrict__ y24 = (float4* __restrict__)({out_name} + base + (idx_t)half);
      #pragma unroll
      for (int k = 0; k < {iters}; ++k) {{
        const int j4 = tid + k * BLOCK_X;
        if (j4 >= half4) break;
        const float4 c4 = cos4[j4];
        const float4 s4 = sin4[j4];
        const float4 a = x14[j4];
        const float4 b = x24[j4];
        float4 y1;
        float4 y2;
        y1.x = __fmaf_rn(-b.x, s4.x, a.x * c4.x);
        y1.y = __fmaf_rn(-b.y, s4.y, a.y * c4.y);
        y1.z = __fmaf_rn(-b.z, s4.z, a.z * c4.z);
        y1.w = __fmaf_rn(-b.w, s4.w, a.w * c4.w);
        y2.x = __fmaf_rn(a.x, s4.x, b.x * c4.x);
        y2.y = __fmaf_rn(a.y, s4.y, b.y * c4.y);
        y2.z = __fmaf_rn(a.z, s4.z, b.z * c4.z);
        y2.w = __fmaf_rn(a.w, s4.w, b.w * c4.w);
        y14[j4] = y1;
        y24[j4] = y2;
      }}
    }} else {{
      for (int j4 = tid; j4 < half4; j4 += BLOCK_X) {{
        const float4 c4 = cos4[j4];
        const float4 s4 = sin4[j4];
        #pragma unroll
        for (int gh = 0; gh < HEADS_PER_BLOCK; ++gh) {{
          const int head = head0 + gh;
          if (head >= HEAD_NUM) break;
          const idx_t base = base0 + (idx_t)head * (idx_t)HEAD_DIM;
          const float4* __restrict__ x14 = (const float4* __restrict__)({in_name} + base);
          const float4* __restrict__ x24 = (const float4* __restrict__)({in_name} + base + (idx_t)half);
          float4* __restrict__ y14 = (float4* __restrict__)({out_name} + base);
          float4* __restrict__ y24 = (float4* __restrict__)({out_name} + base + (idx_t)half);
          const float4 a = x14[j4];
          const float4 b = x24[j4];
          float4 y1;
          float4 y2;
          y1.x = __fmaf_rn(-b.x, s4.x, a.x * c4.x);
          y1.y = __fmaf_rn(-b.y, s4.y, a.y * c4.y);
          y1.z = __fmaf_rn(-b.z, s4.z, a.z * c4.z);
          y1.w = __fmaf_rn(-b.w, s4.w, a.w * c4.w);
          y2.x = __fmaf_rn(a.x, s4.x, b.x * c4.x);
          y2.y = __fmaf_rn(a.y, s4.y, b.y * c4.y);
          y2.z = __fmaf_rn(a.z, s4.z, b.z * c4.z);
          y2.w = __fmaf_rn(a.w, s4.w, b.w * c4.w);
          y14[j4] = y1;
          y24[j4] = y2;
        }}
      }}
    }}
  }} else if constexpr (ROPE_VEC == 2) {{
    const int half2 = (int)(half >> 1);  // # of float2 packs
    const float2* __restrict__ cos2 = (const float2* __restrict__)({cos_name} + cb0);
    const float2* __restrict__ sin2 = (const float2* __restrict__)({sin_name} + cb0);
    if constexpr (HEADS_PER_BLOCK == 1) {{
      const int head = head0;
      const idx_t base = base0 + (idx_t)head * (idx_t)HEAD_DIM;
      const float2* __restrict__ x12 = (const float2* __restrict__)({in_name} + base);
      const float2* __restrict__ x22 = (const float2* __restrict__)({in_name} + base + (idx_t)half);
      float2* __restrict__ y12 = (float2* __restrict__)({out_name} + base);
      float2* __restrict__ y22 = (float2* __restrict__)({out_name} + base + (idx_t)half);
      #pragma unroll
      for (int k = 0; k < {iters}; ++k) {{
        const int j2 = tid + k * BLOCK_X;
        if (j2 >= half2) break;
        const float2 c2 = cos2[j2];
        const float2 s2 = sin2[j2];
        const float2 a = x12[j2];
        const float2 b = x22[j2];
        float2 y1;
        float2 y2;
        y1.x = __fmaf_rn(-b.x, s2.x, a.x * c2.x);
        y1.y = __fmaf_rn(-b.y, s2.y, a.y * c2.y);
        y2.x = __fmaf_rn(a.x, s2.x, b.x * c2.x);
        y2.y = __fmaf_rn(a.y, s2.y, b.y * c2.y);
        y12[j2] = y1;
        y22[j2] = y2;
      }}
    }} else {{
      for (int j2 = tid; j2 < half2; j2 += BLOCK_X) {{
        const float2 c2 = cos2[j2];
        const float2 s2 = sin2[j2];
        #pragma unroll
        for (int gh = 0; gh < HEADS_PER_BLOCK; ++gh) {{
          const int head = head0 + gh;
          if (head >= HEAD_NUM) break;
          const idx_t base = base0 + (idx_t)head * (idx_t)HEAD_DIM;
          const float2* __restrict__ x12 = (const float2* __restrict__)({in_name} + base);
          const float2* __restrict__ x22 = (const float2* __restrict__)({in_name} + base + (idx_t)half);
          float2* __restrict__ y12 = (float2* __restrict__)({out_name} + base);
          float2* __restrict__ y22 = (float2* __restrict__)({out_name} + base + (idx_t)half);
          const float2 a = x12[j2];
          const float2 b = x22[j2];
          float2 y1;
          float2 y2;
          y1.x = __fmaf_rn(-b.x, s2.x, a.x * c2.x);
          y1.y = __fmaf_rn(-b.y, s2.y, a.y * c2.y);
          y2.x = __fmaf_rn(a.x, s2.x, b.x * c2.x);
          y2.y = __fmaf_rn(a.y, s2.y, b.y * c2.y);
          y12[j2] = y1;
          y22[j2] = y2;
        }}
      }}
    }}
  }} else {{
    if constexpr (HEADS_PER_BLOCK == 1) {{
      const int head = head0;
      const idx_t base = base0 + (idx_t)head * (idx_t)HEAD_DIM;
      #pragma unroll
      for (int k = 0; k < {iters}; ++k) {{
        const int j = tid + k * BLOCK_X;
        if (j >= half) break;
        const idx_t cb = cb0 + (idx_t)j;
        const float c = intentir_ldg_f32(&{cos_name}[cb]);
        const float s0 = intentir_ldg_f32(&{sin_name}[cb]);
        const float x1 = intentir_ldg_f32(&{in_name}[base + (idx_t)j]);
        const float x2 = intentir_ldg_f32(&{in_name}[base + (idx_t)half + (idx_t)j]);
        {out_name}[base + (idx_t)j] = __fmaf_rn(-x2, s0, x1 * c);
        {out_name}[base + (idx_t)half + (idx_t)j] = __fmaf_rn(x1, s0, x2 * c);
      }}
    }} else {{
      for (int j = tid; j < half; j += BLOCK_X) {{
        const idx_t cb = cb0 + (idx_t)j;
        const float c = intentir_ldg_f32(&{cos_name}[cb]);
        const float s0 = intentir_ldg_f32(&{sin_name}[cb]);
        #pragma unroll
        for (int gh = 0; gh < HEADS_PER_BLOCK; ++gh) {{
          const int head = head0 + gh;
          if (head >= HEAD_NUM) break;
          const idx_t base = base0 + (idx_t)head * (idx_t)HEAD_DIM;
          const float x1 = intentir_ldg_f32(&{in_name}[base + (idx_t)j]);
          const float x2 = intentir_ldg_f32(&{in_name}[base + (idx_t)half + (idx_t)j]);
          {out_name}[base + (idx_t)j] = __fmaf_rn(-x2, s0, x1 * c);
          {out_name}[base + (idx_t)half + (idx_t)j] = __fmaf_rn(x1, s0, x2 * c);
        }}
      }}
    }}
  }}
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
extern "C" __global__ void {intent.name}(const float* __restrict__ {inp_name}, float* __restrict__ {out_name}, {m_param}, {n_param}) {{
  {m_load}
  {n_load}
  const int m = (int)blockIdx.x;
  if (m >= M) return;
  __shared__ float smem[{block_x}];
  float acc = 0.0f;
  const float* row = {inp_name} + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) acc += row[n];
  smem[(int)threadIdx.x] = acc;
  __syncthreads();
  for (int off = ((int)blockDim.x >> 1); off > 0; off >>= 1) {{
    if ((int)threadIdx.x < off) smem[(int)threadIdx.x] += smem[(int)threadIdx.x + off];
    __syncthreads();
  }}
  if ((int)threadIdx.x == 0) {out_name}[m] = smem[0];
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
extern "C" __global__ void {intent.name}(const float* __restrict__ {inp_name}, float* __restrict__ {out_name}, {m_param}, {n_param}) {{
  {m_load}
  {n_load}
  const int m = (int)blockIdx.x;
  if (m >= M) return;
  __shared__ float smem[{block_x}];
  float acc = -INFINITY;
  const float* row = {inp_name} + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) acc = fmaxf(acc, row[n]);
  smem[(int)threadIdx.x] = acc;
  __syncthreads();
  for (int off = ((int)blockDim.x >> 1); off > 0; off >>= 1) {{
    if ((int)threadIdx.x < off) smem[(int)threadIdx.x] = fmaxf(smem[(int)threadIdx.x], smem[(int)threadIdx.x + off]);
    __syncthreads();
  }}
  if ((int)threadIdx.x == 0) {out_name}[m] = smem[0];
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
    Pattern: const(z) + ne(inp, z) + reduce_any(axis=1).

    Common coverage kernel: `any_kernel_dim`.
    """
    if not intent.ops or len(intent.ops) != 3:
        raise CudaLoweringError("any-dim lowering expects 3 ops (const, ne, reduce_any)")
    c0, ne0, r0 = intent.ops
    if c0.op != "const" or ne0.op != "ne" or r0.op != "reduce_any":
        raise CudaLoweringError("any-dim lowering expects ops const->ne->reduce_any")
    if len(ne0.inputs) != 2 or len(r0.inputs) != 1:
        raise CudaLoweringError("any-dim lowering invalid op arity")
    const_out = str(c0.output)
    a0, b0 = (str(x) for x in ne0.inputs)
    if a0 == const_out and b0 != const_out:
        inp_name = b0
    elif b0 == const_out and a0 != const_out:
        inp_name = a0
    else:
        raise CudaLoweringError("any-dim lowering expects ne(inp, const)")
    out_name = str(r0.output)
    z = float((c0.attrs or {}).get("value", 0.0))
    # Require reduce axis=1.
    dims = (r0.attrs or {}).get("dims")
    axis = (r0.attrs or {}).get("axis")
    if dims not in ([1], (1,)) and axis not in ([1], 1, "1"):
        raise CudaLoweringError("any-dim MVP supports only axis=1 for 2D tensors")

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
    cuda_src = f"""
#include <stdint.h>
extern "C" __global__ void {intent.name}(const float* __restrict__ {inp_name}, bool* __restrict__ {out_name}, {m_param}, {n_param}) {{
  {m_load}
  {n_load}
  const int m = (int)blockIdx.x;
  if (m >= M) return;
  __shared__ int smem[{block_x}];
  int anyv = 0;
  const float* row = {inp_name} + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) anyv |= (row[n] != {z_lit});
  smem[(int)threadIdx.x] = anyv;
  __syncthreads();
  for (int off = ((int)blockDim.x >> 1); off > 0; off >>= 1) {{
    if ((int)threadIdx.x < off) smem[(int)threadIdx.x] |= smem[(int)threadIdx.x + off];
    __syncthreads();
  }}
  if ((int)threadIdx.x == 0) {out_name}[m] = (smem[0] != 0);
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


def _kernel_gather2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    """
    Pattern: gather(inp[M,N], row_idx[L], col_idx[L]) -> out[L]
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
    L = _as_int(bindings.get("L"), name="L")
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    grid_x = (L + block_x - 1) // block_x

    other_lit = _c_scalar_literal("f32", other)
    cuda_src = f"""
#include <stdint.h>
extern "C" __global__ void {intent.name}(
    const float* __restrict__ {inp_name},
    const int* __restrict__ {row_name},
    const int* __restrict__ {col_name},
    float* __restrict__ {out_name},
    int M, int N, int L) {{
  const int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (tid >= L) return;
  const int r = {row_name}[tid];
  const int c = {col_name}[tid];
  float v = {other_lit};
  if ((unsigned)r < (unsigned)M && (unsigned)c < (unsigned)N) {{
    v = {inp_name}[(size_t)r * (size_t)N + (size_t)c];
  }}
  {out_name}[tid] = v;
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[inp_name, row_name, col_name, out_name],
        scalar_args={"M": "i32", "N": "i32", "L": "i32"},
        arg_names=[inp_name, row_name, col_name, out_name, "M", "N", "L"],
    )
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


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
#include <stdint.h>
extern "C" __global__ void {intent.name}(const int8_t* __restrict__ {src_name}, int8_t* __restrict__ {out_name}, {c_param}, {h_param}, {w_param}) {{
  {c_load}
  {h_load}
  {w_load}
  const int OH = H * 2;
  const int OW = W * 2;

  const int h_idx = (int)blockIdx.y;
  const int c = (int)blockIdx.z;
  const int x0 = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (c >= C || h_idx >= OH || x0 >= W) return;
  const int y0 = h_idx >> 1;
  const int x1 = (x0 + 1 < W) ? (x0 + 1) : (W - 1);
  const int y1 = (y0 + 1 < H) ? (y0 + 1) : (H - 1);

  const int64_t src_hw = (int64_t)H * (int64_t)W;
  const int64_t dst_hw = (int64_t)OH * (int64_t)OW;
  const int64_t src_base = (int64_t)c * src_hw;
  const int64_t dst_base = (int64_t)c * dst_hw + (int64_t)h_idx * (int64_t)OW;

  const int64_t row0 = src_base + (int64_t)y0 * (int64_t)W;
  const int64_t row1 = src_base + (int64_t)y1 * (int64_t)W;

  const int16_t a = (int16_t){src_name}[row0 + x0];
  const int16_t b = (int16_t){src_name}[row0 + x1];
  const int16_t c0 = (int16_t){src_name}[row1 + x0];
  const int16_t d = (int16_t){src_name}[row1 + x1];

  // Compute 2 output pixels: w=2*x0 (even) and w=2*x0+1 (odd).
  const int y_odd = (h_idx & 1);
  const int32_t sum1_even = (int32_t)a;
  const int32_t sum2_even = (int32_t)c0;
  const int32_t sum1_odd = (((int32_t)a + (int32_t)b) >> 1);
  const int32_t sum2_odd = (((int32_t)c0 + (int32_t)d) >> 1);
  const int32_t out_even = y_odd ? ((sum1_even + sum2_even) >> 1) : sum1_even;
  const int32_t out_odd = y_odd ? ((sum1_odd + sum2_odd) >> 1) : sum1_odd;
  const int w_even = x0 << 1;
  const int w_odd = w_even + 1;
  {out_name}[dst_base + (int64_t)w_even] = (int8_t)out_even;
  {out_name}[dst_base + (int64_t)w_odd] = (int8_t)out_odd;
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

    cuda_src = f"""
#include <stdint.h>
extern "C" __global__ void {intent.name}(const int8_t* {src_name}, const int16_t* {offset_name}, int8_t* {out_name}, int C, int H, int W) {{
  constexpr int BLOCK_W = {block_w};
  const int h = (int)blockIdx.y;
  const int c = (int)blockIdx.z;
  const int w = (int)blockIdx.x * BLOCK_W + (int)threadIdx.x;
  if (h >= H || c >= C || w >= W) return;
  const int64_t hw = (int64_t)H * (int64_t)W;
  const int64_t row_base = (int64_t)c * hw + (int64_t)h * (int64_t)W;
  const int64_t off_base = (int64_t)h * (int64_t)W;
  const int16_t ov = {offset_name}[off_base + w];
  const int8_t offset_int = (int8_t)(ov >> 8);
  const int8_t offset_frac = (int8_t)(((int16_t)(ov << 8)) >> 8);
  const int8_t indvar = (int8_t)w;
  const int8_t right_i8 = (int8_t)(indvar - offset_int);
  const int8_t left_i8 = (int8_t)(right_i8 - 1);
  const int right = (int)right_i8;
  const int left = (int)left_i8;
  int8_t right_val = 0;
  int8_t left_val = 0;
  if (right >= 0 && right < W) right_val = {src_name}[row_base + (int64_t)right];
  if (left >= 0 && left < W) left_val = {src_name}[row_base + (int64_t)left];
  int16_t outv = (int16_t)((int16_t)right_val << 8);
  outv = (int16_t)(outv + (int16_t)((int16_t)(left_val - right_val) * (int16_t)offset_frac));
  outv = (int16_t)(outv >> 8);
  {out_name}[row_base + w] = (int8_t)outv;
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
#include <stdint.h>
extern "C" __global__ void {intent.name}(
    const int8_t* {src0_name}, const int8_t* {src1_name}, int8_t* {out_name},
    {oc_param}, {ic_param}, {h_param}, {w_param}, {sh_param}) {{
  {oc_load}
  {ic_load}
  {h_load}
  {w_load}
  {sh_load}
  const int64_t hw = (int64_t)height * (int64_t)width;
  const int64_t total = (int64_t)out_channel * hw;
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (tid >= total) return;
  const int oc = (int)(tid / hw);
  const int64_t rem = tid - (int64_t)oc * hw;
  const int h = (int)(rem / (int64_t)width);
  const int w = (int)(rem - (int64_t)h * (int64_t)width);

  int sh = out_shift;
  if (sh < 0) sh = 0;
  if (sh > 30) sh = 30;

  if (oc >= width || w < oc) {{
    out[(size_t)tid] = 0;
    return;
  }}

  int32_t acc = 0;
  const int64_t off0 = (int64_t)h * (int64_t)width + (int64_t)w;
  const int64_t off1 = (int64_t)h * (int64_t)width + (int64_t)(w - oc);
  for (int k = 0; k < in_channel; ++k) {{
    const int64_t base = (int64_t)k * hw;
    acc += (int32_t){src0_name}[base + off0] * (int32_t){src1_name}[base + off1];
  }}
  {out_name}[(size_t)tid] = (int8_t)(acc >> sh);
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

    # 1) Direct single-op kernels.
    if intent.ops and len(intent.ops) == 1:
        op0 = intent.ops[0].op
        if op0 == "matmul":
            return _kernel_matmul_f32(intent, bindings)
        if op0 == "dropout":
            return _kernel_dropout_f32(intent, bindings)
        if op0 == "correlation":
            return _kernel_correlation_i8(intent, bindings)
        if op0 == "resize":
            return _kernel_resize_bilinear2x_i8(intent, bindings)
        if op0 == "warp":
            return _kernel_warp_q8_8_i8_i16(intent, bindings)
        if op0 == "rope":
            return _kernel_rope_f32(intent, bindings)
        if op0 == "transpose":
            return _kernel_transpose_2d_f32(intent, bindings)
        if op0 == "reduce_sum":
            return _kernel_reduce_sum_2d_axis1_f32(intent, bindings)
        if op0 == "reduce_max":
            return _kernel_reduce_max_2d_axis1_f32(intent, bindings)
        if op0 == "gather":
            return _kernel_gather2d_f32(intent, bindings)

    # 2) Pattern-based kernels (fused).
    outs = set(intent.outputs or [])
    # any_kernel_dim (coverage): const + ne + reduce_any(axis=1)
    if intent.ops and len(intent.ops) == 3 and [o.op for o in intent.ops] == ["const", "ne", "reduce_any"]:
        return _kernel_any_dim_f32_to_i1(intent, bindings)
    if {"Y", "Mean", "Rstd"}.issubset(outs):
        return _kernel_layernorm_2d_f32(intent, bindings)
    # Softmax: recognize by name or by presence of reduce_max + exp + reduce_sum.
    op_names = {o.op for o in (intent.ops or [])}
    if ("softmax" in str(intent.name).lower()) or ({"reduce_max", "reduce_sum", "exp", "div"}.issubset(op_names)):
        return _kernel_softmax_2d_last_f32(intent, bindings)

    # 3) Generic fused elementwise lowering.
    # This is intentionally conservative: only elementwise/broadcast/const/where/cast ops.
    elem_ops = {
        "const",
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
        "floor",
        "rsqrt",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "and",
        "or",
        "not",
        "where",
    }
    if intent.ops and all(str(o.op) in elem_ops for o in intent.ops):
        return _kernel_fused_elementwise(intent, bindings)

    raise CudaLoweringError(f"CUDA lowering unsupported for intent: name={intent.name} ops={sorted(op_names)}")


__all__ = ["CudaLoweringError", "CudaLoweredKernel", "lower_intent_to_cuda_kernel"]
