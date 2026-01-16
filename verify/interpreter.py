"""
Numpy-based interpreter for Intent-IR v1.1.
Supports matmul (+epilogue), broadcast_in_dim, transpose, reshape, layout_cast (no-op),
reduce_sum, softmax, and basic elemwise ops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from intent_ir.ir import IntentFunction, Op


NUM_BIN_OPS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "max": np.maximum,
    "min": np.minimum,
}

NUM_UNARY_OPS = {
    "relu": lambda x: np.maximum(x, 0),
    "exp": np.exp,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "abs": np.abs,
    "floor": np.floor,
    "identity": lambda x: x,
}

CMP_BIN_OPS = {
    "ne": np.not_equal,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
}

BOOL_BIN_OPS = {
    "and": np.logical_and,
    "or": np.logical_or,
}

BOOL_UNARY_OPS = {
    "not": np.logical_not,
}

INTERPRETER_SUPPORTED_OPS: set[str] = set().union(
    set(NUM_BIN_OPS.keys()),
    set(NUM_UNARY_OPS.keys()),
    set(CMP_BIN_OPS.keys()),
    set(BOOL_BIN_OPS.keys()),
    set(BOOL_UNARY_OPS.keys()),
    {
        "broadcast_in_dim",
        "transpose",
        "reshape",
        "layout_cast",
        "cast",
        "iota",
        "gather",
        "where",
        "dropout",
        "reduce_sum",
        "reduce_any",
        "reduce_max",
        "softmax",
        "matmul",
        "const",
    },
)


def _np_dtype(dtype: str | None) -> Any:
    if dtype is None:
        return None
    dt = str(dtype)
    if dt in {"f16"}:
        return np.float16
    if dt in {"bf16"}:
        # NumPy has no native bfloat16 in older versions; use fp16 for interpreter purposes.
        return np.float16
    if dt in {"f32"}:
        return np.float32
    if dt in {"f64"}:
        return np.float64
    if dt in {"i32"}:
        return np.int32
    if dt in {"i64"}:
        return np.int64
    if dt in {"i8"}:
        return np.int8
    if dt in {"i16"}:
        return np.int16
    if dt in {"u8"}:
        return np.uint8
    if dt in {"bool", "i1"}:
        return np.bool_
    raise ValueError(f"unsupported dtype for interpreter: {dtype}")


def execute_intent(intent: IntentFunction, inputs: Dict[str, np.ndarray], shape_bindings: Dict[str, int] | None = None) -> Dict[str, np.ndarray]:
    env: Dict[str, np.ndarray] = {}
    env.update(inputs)
    bindings = shape_bindings or _infer_shape_bindings(intent, inputs)
    for op in intent.ops:
        env[op.output] = _execute_op(intent, op, env, bindings)
    return {name: env[name] for name in intent.outputs}


@dataclass(frozen=True)
class TensorSummary:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    size: int
    nan: int
    inf: int
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    sample: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class OpTrace:
    op_index: int
    op: Dict[str, Any]
    inputs: Dict[str, TensorSummary]
    output: TensorSummary


@dataclass
class InterpreterTrace:
    op_traces: List[OpTrace] = field(default_factory=list)


def _summarize_tensor(name: str, arr: np.ndarray, *, sample_elems: int = 16) -> TensorSummary:
    a = np.asarray(arr)
    flat = a.reshape(-1)
    # Handle bool/integers separately (still cast to float for summaries).
    a_float = flat.astype(np.float64, copy=False) if flat.size else flat.astype(np.float64, copy=False)
    nan = int(np.isnan(a_float).sum()) if a_float.size else 0
    inf = int(np.isinf(a_float).sum()) if a_float.size else 0
    if a_float.size:
        try:
            amin = float(np.nanmin(a_float))
            amax = float(np.nanmax(a_float))
            mean = float(np.nanmean(a_float))
            std = float(np.nanstd(a_float))
        except Exception:
            amin = amax = mean = std = None
    else:
        amin = amax = mean = std = None
    sample = [float(x) for x in a_float[: max(0, int(sample_elems))].tolist()] if a_float.size else []
    return TensorSummary(
        name=str(name),
        shape=tuple(int(x) for x in a.shape),
        dtype=str(a.dtype),
        size=int(a.size),
        nan=nan,
        inf=inf,
        min=amin,
        max=amax,
        mean=mean,
        std=std,
        sample=sample,
    )


def execute_intent_with_trace(
    intent: IntentFunction,
    inputs: Dict[str, np.ndarray],
    shape_bindings: Dict[str, int] | None = None,
    *,
    sample_elems: int = 16,
) -> Tuple[Dict[str, np.ndarray], InterpreterTrace, Dict[str, np.ndarray]]:
    """
    Execute IntentIR and record per-op input/output summaries.

    Returns (outputs, trace, full_env).
    """
    env: Dict[str, np.ndarray] = {}
    env.update(inputs)
    bindings = shape_bindings or _infer_shape_bindings(intent, inputs)
    trace = InterpreterTrace()
    for idx, op in enumerate(intent.ops):
        op_inputs = {k: env[k] for k in op.inputs if k in env}
        out = _execute_op(intent, op, env, bindings)
        env[op.output] = out
        trace.op_traces.append(
            OpTrace(
                op_index=int(idx),
                op={"op": op.op, "inputs": list(op.inputs), "output": op.output, "attrs": dict(op.attrs or {})},
                inputs={k: _summarize_tensor(k, v, sample_elems=sample_elems) for k, v in op_inputs.items()},
                output=_summarize_tensor(op.output, out, sample_elems=sample_elems),
            )
        )
    outputs = {name: env[name] for name in intent.outputs}
    return outputs, trace, env


def _execute_op(intent: IntentFunction, op: Op, env: Dict[str, np.ndarray], shape_bindings: Dict[str, int]) -> np.ndarray:
    if op.op in NUM_BIN_OPS:
        input_names = list(op.inputs)
        args = [_get(env, name) for name in input_names]
        if len(args) == 1:
            # allow implicit second operand from attrs (common in LLM outputs)
            if op.op == "div" and "divisor" in op.attrs:
                args.append(_resolve_value(op.attrs["divisor"], env, shape_bindings))
            elif op.op == "add" and "addend" in op.attrs:
                args.append(_resolve_value(op.attrs["addend"], env, shape_bindings))
            elif op.op == "sub" and "subtract" in op.attrs:
                args.append(_resolve_value(op.attrs["subtract"], env, shape_bindings))
            elif op.op == "mul" and "mul_factor" in op.attrs:
                args.append(_resolve_value(op.attrs["mul_factor"], env, shape_bindings))
            elif op.op in {"max", "min"} and "other" in op.attrs:
                args.append(_resolve_value(op.attrs["other"], env, shape_bindings))
        if len(args) != 2:
            raise ValueError(f"{op.op} requires 2 inputs (or 1+attrs)")
        a_name = input_names[0] if input_names else None
        b_name = input_names[1] if len(input_names) > 1 else None
        a, b = _align_shapes_for_elemwise_named(intent, a_name, args[0], b_name, args[1], shape_bindings)
        return NUM_BIN_OPS[op.op](a, b)
    if op.op in NUM_UNARY_OPS:
        x = _get(env, op.inputs[0])
        return NUM_UNARY_OPS[op.op](x)
    if op.op in CMP_BIN_OPS:
        a_name = op.inputs[0]
        b_name = op.inputs[1]
        a = _get(env, a_name)
        b = _get(env, b_name)
        a, b = _align_shapes_for_elemwise_named(intent, a_name, a, b_name, b, shape_bindings)
        return CMP_BIN_OPS[op.op](a, b)
    if op.op in BOOL_BIN_OPS:
        a_name = op.inputs[0]
        b_name = op.inputs[1]
        a = _get(env, a_name).astype(bool)
        b = _get(env, b_name).astype(bool)
        a, b = _align_shapes_for_elemwise_named(intent, a_name, a, b_name, b, shape_bindings)
        return BOOL_BIN_OPS[op.op](a, b)
    if op.op in BOOL_UNARY_OPS:
        x = _get(env, op.inputs[0]).astype(bool)
        return BOOL_UNARY_OPS[op.op](x)
    if op.op == "broadcast_in_dim":
        x = _get(env, op.inputs[0])
        out_shape = _shape_from_attr(op.attrs.get("out_shape"), shape_bindings)
        bcast_dims = op.attrs.get("broadcast_dims", [])
        return _broadcast_in_dim(x, out_shape, bcast_dims)
    if op.op == "transpose":
        x = _get(env, op.inputs[0])
        perm = op.attrs.get("perm")
        if perm is None:
            raise ValueError("transpose requires attrs.perm")
        return np.transpose(x, axes=perm)
    if op.op == "reshape":
        x = _get(env, op.inputs[0])
        shape = _shape_from_attr(op.attrs.get("shape"), shape_bindings)
        return np.reshape(x, shape)
    if op.op == "layout_cast":
        return _get(env, op.inputs[0])
    if op.op == "cast":
        x = _get(env, op.inputs[0])
        to = op.attrs.get("to")
        return x.astype(_np_dtype(to))
    if op.op == "iota":
        shp = _shape_from_attr(op.attrs.get("shape"), shape_bindings)
        axis = int(op.attrs.get("axis", 0))
        dt = _np_dtype(op.attrs.get("dtype", "i32"))
        if axis < 0 or axis >= len(shp):
            raise ValueError(f"iota axis out of range: axis={axis} rank={len(shp)}")
        ar = np.arange(int(shp[axis]), dtype=dt)
        view_shape = [1] * len(shp)
        view_shape[axis] = int(shp[axis])
        return np.broadcast_to(ar.reshape(view_shape), shp)
    if op.op == "gather":
        data = _get(env, op.inputs[0])
        idxs = [_get(env, n) for n in op.inputs[1:]]
        if not idxs:
            raise ValueError("gather requires at least one index tensor")
        idxs_b = np.broadcast_arrays(*idxs)
        idxs_b = [np.asarray(ix, dtype=np.int64) for ix in idxs_b]
        return data[tuple(idxs_b)]
    if op.op == "where":
        cond_name = op.inputs[0]
        x_name = op.inputs[1]
        y_name = op.inputs[2]
        cond = _get(env, cond_name).astype(bool)
        x = _get(env, x_name)
        y = _get(env, y_name)
        cond, x = _align_shapes_for_elemwise_named(intent, cond_name, cond, x_name, x, shape_bindings)
        cond, y = _align_shapes_for_elemwise_named(intent, cond_name, cond, y_name, y, shape_bindings)
        x, y = _align_shapes_for_elemwise_named(intent, x_name, x, y_name, y, shape_bindings)
        return np.where(cond, x, y)
    if op.op == "dropout":
        # Match Triton's tl.rand(seed, offsets) semantics (Philox + uint_to_uniform_float).
        if len(op.inputs) != 3:
            raise ValueError("dropout requires 3 inputs (X, p, seed)")
        x = _get(env, op.inputs[0])
        p_raw = _get(env, op.inputs[1])
        seed_raw = _get(env, op.inputs[2])
        p = float(np.asarray(p_raw).reshape(()))
        seed = int(np.asarray(seed_raw).reshape(()))
        n_rounds = int(op.attrs.get("n_rounds", 10))
        if n_rounds <= 0:
            n_rounds = 10
        x_f32 = np.asarray(x, dtype=np.float32, order="C")
        n = int(x_f32.size)
        if n == 0:
            return np.asarray(x, dtype=np.float32)
        keep_prob = float(1.0 - p)
        if keep_prob <= 0.0:
            return np.zeros_like(x_f32)
        if p <= 0.0:
            return x_f32
        offsets = np.arange(n, dtype=np.uint32)
        rnd_u32 = _philox_randint_u32(seed, offsets, n_rounds=n_rounds)
        rnd = _uint_to_uniform_float_u32(rnd_u32).reshape(x_f32.shape)
        keep = rnd > np.float32(p)
        out = np.where(keep, x_f32 / np.float32(keep_prob), np.float32(0.0))
        return out.astype(np.float32, copy=False)
    if op.op == "reduce_sum":
        x = _get(env, op.inputs[0])
        dims_raw = op.attrs.get("axes", op.attrs.get("dims", op.attrs.get("axis")))
        dims = _resolve_dims(dims_raw, x)
        keepdims = bool(op.attrs.get("keepdims", False))
        if op.attrs.get("reduction_type") == "any":
            out = np.any(x, axis=dims, keepdims=keepdims)
        else:
            out = np.sum(x, axis=dims, keepdims=keepdims)
        out = _apply_reduce_scale(out, op.attrs.get("scale"), shape_bindings)
        return out
    if op.op == "reduce_any":
        x = _get(env, op.inputs[0])
        dims_raw = op.attrs.get("axes", op.attrs.get("dims", op.attrs.get("axis")))
        dims = _resolve_dims(dims_raw, x)
        keepdims = bool(op.attrs.get("keepdims", False))
        return np.any(x, axis=dims, keepdims=keepdims)
    if op.op == "reduce_max":
        x = _get(env, op.inputs[0])
        dims_raw = op.attrs.get("axes", op.attrs.get("dims", op.attrs.get("axis")))
        dims = _resolve_dims(dims_raw, x)
        keepdims = bool(op.attrs.get("keepdims", False))
        return np.max(x, axis=dims, keepdims=keepdims)
    if op.op == "softmax":
        x = _get(env, op.inputs[0])
        axis = op.attrs.get("axis", -1)
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=axis, keepdims=True)
    if op.op == "matmul":
        a = _get(env, op.inputs[0])
        b = _get(env, op.inputs[1])
        ta = bool(op.attrs.get("transpose_a", False))
        tb = bool(op.attrs.get("transpose_b", False))
        if ta:
            a = np.swapaxes(a, -1, -2)
        if tb:
            b = np.swapaxes(b, -1, -2)
        acc = np.matmul(a, b)
        epilogue = op.attrs.get("epilogue")
        if epilogue:
            acc = _eval_epilogue(epilogue, env | {"$x": acc})
        return acc
    if op.op == "const":
        val = op.attrs.get("value")
        dtype = op.attrs.get("dtype")
        if isinstance(val, str):
            if val.startswith("placeholder:"):
                raise ValueError(f"placeholder const is not a valid semantic value: {val}")
            if val in env:
                val = _get(env, val)
            elif val in shape_bindings:
                val = shape_bindings[val]
            else:
                try:
                    val = float(val)
                except Exception:
                    if val == "eps":
                        val = 1e-5
                    else:
                        try:
                            # try to eval simple expressions with bindings
                            val = eval(val, {}, dict(shape_bindings))
                        except Exception:
                            raise ValueError(f"unresolved const value: {val}")
        np_dt = _np_dtype(dtype) if dtype is not None else None
        return np.array(val, dtype=np_dt)
    raise ValueError(f"Unsupported op: {op.op}")


def _eval_epilogue(node: Any, env: Dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(node, str):
        return _get(env, node)
    op = node.get("op")
    inputs = node.get("inputs", [])
    args = [_eval_epilogue(inp, env) if isinstance(inp, dict) else _get(env, inp) for inp in inputs]
    if op in NUM_BIN_OPS:
        if len(args) != 2:
            raise ValueError(f"epilogue {op} expects 2 inputs")
        a, b = _align_shapes_for_elemwise(args[0], args[1])
        return NUM_BIN_OPS[op](a, b)
    if op in NUM_UNARY_OPS:
        if len(args) != 1:
            raise ValueError(f"epilogue {op} expects 1 input")
        return NUM_UNARY_OPS[op](args[0])
    if op not in CMP_BIN_OPS and op not in BOOL_BIN_OPS and op not in BOOL_UNARY_OPS:
        raise ValueError(f"Unsupported epilogue op: {op}")
    if op in CMP_BIN_OPS:
        a, b = _align_shapes_for_elemwise(args[0], args[1])
        return CMP_BIN_OPS[op](a, b)
    if op in BOOL_BIN_OPS:
        a, b = _align_shapes_for_elemwise(args[0].astype(bool), args[1].astype(bool))
        return BOOL_BIN_OPS[op](a, b)
    # not
    return BOOL_UNARY_OPS[op](args[0].astype(bool))


def _broadcast_in_dim(x: np.ndarray, out_shape, broadcast_dims):
    if not isinstance(broadcast_dims, list):
        raise ValueError("broadcast_in_dim requires attrs.broadcast_dims as list[int]")
    out_shape = tuple(int(v) for v in out_shape)
    temp = np.reshape(x, _expand_shape_for_broadcast(x.shape, out_shape, broadcast_dims))
    return np.broadcast_to(temp, out_shape)


def _expand_shape_for_broadcast(in_shape, out_shape, broadcast_dims):
    # out rank = len(out_shape); place input dims at broadcast_dims positions
    new_shape = [1] * len(out_shape)
    for in_dim, out_dim in enumerate(broadcast_dims):
        if in_dim < len(in_shape):
            new_shape[out_dim] = in_shape[in_dim]
    return new_shape


def _shape_from_attr(shape_attr, bindings: Dict[str, int]):
    if shape_attr is None:
        return ()
    out = []
    for s in shape_attr:
        if isinstance(s, (int, np.integer)) or str(s).isdigit():
            out.append(int(s))
        elif isinstance(s, str) and s in bindings:
            out.append(bindings[s])
        else:
            raise ValueError(f"unbound symbolic dim in shape: {s}")
    return tuple(out)


def _get(env: Dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "$x":
        return env[name]
    if name not in env:
        raise KeyError(f"undefined value referenced: {name}")
    return env[name]


def _resolve_value(val, env: Dict[str, np.ndarray], bindings: Dict[str, int]):
    if isinstance(val, str):
        if val in env:
            return _get(env, val)
        if val in bindings:
            return np.array(bindings[val])
        try:
            return np.array(float(val))
        except Exception:
            raise ValueError(f"unresolved scalar value: {val}")
    return np.array(val)


def _apply_reduce_scale(out: np.ndarray, scale: Any, bindings: Dict[str, int]) -> np.ndarray:
    """
    Optional post-reduction scaling, used by some LLM outputs (e.g., mean = sum * (1/num_elements)).
    Accepts numeric scale or string expressions evaluable from shape bindings.
    """
    if scale is None:
        return out
    factor: float
    if isinstance(scale, (int, float, np.number)):
        factor = float(scale)
    elif isinstance(scale, str):
        # Allow common derived names.
        locals_dict: Dict[str, Any] = dict(bindings)
        if "num_elements" not in locals_dict:
            gs = locals_dict.get("group_size")
            hw = locals_dict.get("HW")
            if isinstance(gs, int) and isinstance(hw, int):
                locals_dict["num_elements"] = gs * hw
        if scale in locals_dict:
            factor = float(locals_dict[scale])
        else:
            try:
                factor = float(eval(scale, {}, locals_dict))
            except Exception as e:
                raise ValueError(f"unresolved reduce scale: {scale}") from e
    else:
        raise ValueError(f"unsupported reduce scale type: {type(scale)}")
    return out * factor


def _resolve_dims(dims_raw, x: np.ndarray):
    if dims_raw is None:
        return tuple(range(x.ndim))
    if isinstance(dims_raw, (int, np.integer)):
        dims = [int(dims_raw)]
    else:
        dims = list(dims_raw)
    resolved = []
    for d in dims:
        if isinstance(d, (int, np.integer)):
            resolved.append(int(d))
            continue
        if isinstance(d, str) and d.isdigit():
            resolved.append(int(d))
            continue
    if not resolved:
        # Fall back to a common Triton pattern: reduce over all non-leading axes.
        resolved = list(range(1, x.ndim))
    return tuple(resolved)


def _philox_randint_u32(seed: int, offsets: np.ndarray, *, n_rounds: int = 10) -> np.ndarray:
    """
    Triton-compatible philox (32-bit) for tl.rand/tl.randint.

    Mirrors triton.language.random.randint(seed, offset) which calls:
      philox(seed, offset, 0, 0, 0, n_rounds) and returns c0 as uint32.
    """
    off = np.asarray(offsets, dtype=np.uint32)
    c0 = off
    c1 = np.zeros_like(c0, dtype=np.uint32)
    c2 = np.zeros_like(c0, dtype=np.uint32)
    c3 = np.zeros_like(c0, dtype=np.uint32)
    k0 = np.uint32(int(seed) & 0xFFFFFFFF)
    k1 = np.uint32((int(seed) >> 32) & 0xFFFFFFFF)
    KEY_A = np.uint32(0x9E3779B9)
    KEY_B = np.uint32(0xBB67AE85)
    ROUND_A = np.uint32(0xD2511F53)
    ROUND_B = np.uint32(0xCD9E8D57)
    A64 = np.uint64(int(ROUND_A))
    B64 = np.uint64(int(ROUND_B))
    for _ in range(int(n_rounds)):
        _c0 = c0
        _c2 = c2
        hi_B_c2 = ((B64 * _c2.astype(np.uint64)) >> 32).astype(np.uint32)
        hi_A_c0 = ((A64 * _c0.astype(np.uint64)) >> 32).astype(np.uint32)
        c0 = hi_B_c2 ^ c1 ^ k0
        c2 = hi_A_c0 ^ c3 ^ k1
        c1 = (B64 * _c2.astype(np.uint64)).astype(np.uint32)
        c3 = (A64 * _c0.astype(np.uint64)).astype(np.uint32)
        k0 = k0 + KEY_A
        k1 = k1 + KEY_B
    return c0


def _uint_to_uniform_float_u32(x: np.ndarray) -> np.ndarray:
    """
    Triton-compatible uint_to_uniform_float for uint32/int32 source.

    Matches triton.language.random.uint_to_uniform_float for 32-bit inputs:
      x = bitcast to int32; x = where(x < 0, -x-1, x); return x * 4.6566127342e-10
    Note: for int32, -x-1 is equivalent to bitwise_not(x) in two's complement.
    """
    u = np.asarray(x, dtype=np.uint32)
    xi = u.view(np.int32)
    xi = np.where(xi < 0, np.bitwise_not(xi), xi)
    return xi.astype(np.float32) * np.float32(4.6566127342e-10)


def _align_shapes_for_elemwise(a: np.ndarray, b: np.ndarray):
    # Kept for backward compatibility with older call sites. Prefer
    # `_align_shapes_for_elemwise_named`, which can use IntentIR shape symbols to
    # disambiguate 1D-vs-2D broadcasting.
    return a, b


def _align_shapes_for_elemwise_named(
    intent: IntentFunction,
    a_name: str | None,
    a: np.ndarray,
    b_name: str | None,
    b: np.ndarray,
    shape_bindings: Dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Disambiguate broadcasting between a 1D vector and a higher-rank tensor using
    IntentIR tensor shapes (symbol names / constants).

    Motivation: NumPy aligns 1D on the trailing axis. For patterns like softmax:
      mx: [M] = reduce_max(x[M,N], axis=1, keepdims=false)
      x - mx   # intended mx -> [M,1] (broadcast across N)
    the IR's declared shape carries the intent ("M" matches axis0), so we can
    reshape mx safely without breaking cases like weights [N] for [M,N].
    """

    def decl_tokens(name: str | None) -> list[object] | None:
        if not name or name not in intent.tensors:
            return None
        t = intent.tensors[name]
        toks: list[object] = []
        for d in t.shape:
            # Dim(kind="sym"/"const", value=...)
            if hasattr(d, "kind") and hasattr(d, "value"):
                if getattr(d, "kind") == "sym":
                    toks.append(str(getattr(d, "value")))
                elif getattr(d, "kind") == "const":
                    toks.append(int(getattr(d, "value")))
                else:
                    toks.append(None)
                continue
            if isinstance(d, str):
                toks.append(d)
            elif isinstance(d, int):
                toks.append(int(d))
            else:
                toks.append(None)
        return toks

    def resolved(tok: object | None) -> int | None:
        if tok is None:
            return None
        if isinstance(tok, int):
            return int(tok)
        if isinstance(tok, str) and tok in shape_bindings:
            return int(shape_bindings[tok])
        return None

    def match_axis(vec_name: str | None, tensor_name: str | None, tensor_rank: int) -> int | None:
        v = decl_tokens(vec_name)
        t = decl_tokens(tensor_name)
        if not v or len(v) != 1 or not t or len(t) != tensor_rank:
            return None
        v0 = v[0]
        # 1) Exact symbol match (strongest signal).
        if isinstance(v0, str):
            hits = [i for i, tt in enumerate(t) if tt == v0]
            if len(hits) == 1:
                return int(hits[0])
        # 2) Resolved numeric match (weaker; can be ambiguous when symbols bind equal).
        v_num = resolved(v0)
        if v_num is None:
            return None
        hits = [i for i, tt in enumerate(t) if resolved(tt) == v_num]
        if len(hits) == 1:
            return int(hits[0])
        return None

    def reshape_vec_for_axis(vec: np.ndarray, axis: int, target_rank: int, target_shape: tuple[int, ...]) -> np.ndarray:
        if vec.ndim != 1 or axis < 0 or axis >= target_rank:
            return vec
        if vec.shape[0] != int(target_shape[axis]):
            return vec
        new_shape = [1] * target_rank
        new_shape[axis] = int(vec.shape[0])
        return vec.reshape(tuple(new_shape))

    # Align 1D -> higher rank using declared shapes.
    if a.ndim == 1 and b.ndim >= 2:
        ax = match_axis(a_name, b_name, b.ndim)
        if ax is not None:
            a = reshape_vec_for_axis(a, ax, b.ndim, b.shape)
    if b.ndim == 1 and a.ndim >= 2:
        ax = match_axis(b_name, a_name, a.ndim)
        if ax is not None:
            b = reshape_vec_for_axis(b, ax, a.ndim, a.shape)
    return a, b


def _infer_shape_bindings(intent: IntentFunction, inputs: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Infer symbolic dim bindings from provided tensor shapes."""
    bindings: Dict[str, int] = {}
    for name, tensor in intent.tensors.items():
        if name not in inputs:
            continue
        actual_shape = inputs[name].shape
        for idx, dim in enumerate(tensor.shape):
            sym = None
            if hasattr(dim, "kind") and getattr(dim, "kind") == "sym":
                sym = getattr(dim, "value")
            elif isinstance(dim, str):
                sym = dim
            if sym is None:
                continue
            val = actual_shape[idx]
            if sym in bindings and bindings[sym] != val:
                raise ValueError(f"Inconsistent binding for {sym}: {bindings[sym]} vs {val}")
            bindings[sym] = val
    return bindings


__all__ = ["execute_intent", "execute_intent_with_trace", "InterpreterTrace", "INTERPRETER_SUPPORTED_OPS"]
