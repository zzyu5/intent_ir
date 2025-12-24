"""
Numpy-based interpreter for Intent-IR v1.1.
Supports matmul (+epilogue), broadcast_in_dim, transpose, reshape, layout_cast (no-op),
reduce_sum, softmax, and basic elemwise ops.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict

from intent_ir.ir_types import IntentFunction, Op


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


def _execute_op(intent: IntentFunction, op: Op, env: Dict[str, np.ndarray], shape_bindings: Dict[str, int]) -> np.ndarray:
    if op.op in NUM_BIN_OPS:
        args = [_get(env, name) for name in op.inputs]
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
        a, b = _align_shapes_for_elemwise(args[0], args[1])
        return NUM_BIN_OPS[op.op](a, b)
    if op.op in NUM_UNARY_OPS:
        x = _get(env, op.inputs[0])
        return NUM_UNARY_OPS[op.op](x)
    if op.op in CMP_BIN_OPS:
        a = _get(env, op.inputs[0])
        b = _get(env, op.inputs[1])
        a, b = _align_shapes_for_elemwise(a, b)
        return CMP_BIN_OPS[op.op](a, b)
    if op.op in BOOL_BIN_OPS:
        a = _get(env, op.inputs[0]).astype(bool)
        b = _get(env, op.inputs[1]).astype(bool)
        a, b = _align_shapes_for_elemwise(a, b)
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
        cond = _get(env, op.inputs[0]).astype(bool)
        x = _get(env, op.inputs[1])
        y = _get(env, op.inputs[2])
        # Help some LLM outputs that forget singleton dims.
        cond, x = _align_shapes_for_elemwise(cond, x)
        cond, y = _align_shapes_for_elemwise(cond, y)
        x, y = _align_shapes_for_elemwise(x, y)
        return np.where(cond, x, y)
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


def _align_shapes_for_elemwise(a: np.ndarray, b: np.ndarray):
    # Conservative heuristic for some LLM outputs that drop singleton dims:
    # only reshape when one side is 1D and the leading dimension matches.
    # (Avoid breaking numpy-style trailing-dim broadcasting, e.g. [OH,OW] with [N,C,OH,OW].)
    if a.shape and b.shape and a.shape[0] == b.shape[0] and (min(a.ndim, b.ndim) == 1):
        max_nd = max(a.ndim, b.ndim)
        if a.ndim < max_nd:
            a = a.reshape(a.shape + (1,) * (max_nd - a.ndim))
        if b.ndim < max_nd:
            b = b.reshape(b.shape + (1,) * (max_nd - b.ndim))
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


__all__ = ["execute_intent"]
