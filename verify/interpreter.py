"""
Numpy-based interpreter for Intent-IR v1.1.
Supports matmul (+epilogue), broadcast_in_dim, transpose, reshape, layout_cast (no-op),
reduce_sum, softmax, and basic elemwise ops.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict

from intent_ir.ir_types import IntentFunction, Op


ELEM_OPS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "max": np.maximum,
    "min": np.minimum,
    "or": np.maximum,
    "relu": lambda x: np.maximum(x, 0),
    "exp": np.exp,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "ne": np.not_equal,
    "identity": lambda x: x,
}


def execute_intent(intent: IntentFunction, inputs: Dict[str, np.ndarray], shape_bindings: Dict[str, int] | None = None) -> Dict[str, np.ndarray]:
    env: Dict[str, np.ndarray] = {}
    env.update(inputs)
    bindings = shape_bindings or _infer_shape_bindings(intent, inputs)
    for op in intent.ops:
        env[op.output] = _execute_op(intent, op, env, bindings)
    return {name: env[name] for name in intent.outputs}


def _execute_op(intent: IntentFunction, op: Op, env: Dict[str, np.ndarray], shape_bindings: Dict[str, int]) -> np.ndarray:
    if op.op in ELEM_OPS:
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
        if len(args) == 2:
            args = list(_align_shapes_for_elemwise(args[0], args[1]))
        return ELEM_OPS[op.op](*args)
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
    if op.op == "custom_call":
        callee = op.attrs.get("callee")
        if callee == "upsample_bicubic2d_aa":
            # Model the Triton `upsample_bicubic2d_aa_kernel` semantics directly.
            # Expected:
            #   Input: [N,C,IH,IW]
            #   Output: [N,C,OH,OW]
            # Inputs optionally include scalar tensors `reciprocal_scale_h`/`reciprocal_scale_w`.
            x = _get(env, op.inputs[0])
            if x.ndim != 4:
                raise ValueError("upsample_bicubic2d_aa expects Input rank-4 (N,C,IH,IW)")
            N, C, IH, IW = (int(s) for s in x.shape)
            # Resolve reciprocal scales (prefer scalar tensors provided as inputs).
            rs_h = None
            rs_w = None
            for name in op.inputs[1:]:
                if "reciprocal_scale_h" in str(name):
                    rs_h = float(np.asarray(_get(env, name)).reshape(()))
                if "reciprocal_scale_w" in str(name):
                    rs_w = float(np.asarray(_get(env, name)).reshape(()))
            if rs_h is None:
                rs_h = float(_resolve_value(op.attrs.get("reciprocal_scale_h", IH / max(1, shape_bindings.get("OH", IH))), env, shape_bindings).reshape(()))
            if rs_w is None:
                rs_w = float(_resolve_value(op.attrs.get("reciprocal_scale_w", IW / max(1, shape_bindings.get("OW", IW))), env, shape_bindings).reshape(()))
            # Output sizes from bindings if present, else infer as "same size".
            OH = int(shape_bindings.get("OH", IH))
            OW = int(shape_bindings.get("OW", IW))
            a = float(op.attrs.get("a", -0.5))
            return _upsample_bicubic2d_aa_numpy(x, OH=OH, OW=OW, rs_h=rs_h, rs_w=rs_w, a=a)
        raise ValueError(f"Unsupported custom_call callee: {callee}")
    # Note: legacy op name `upsample_bicubic2d_aa` is normalized to custom_call by Task2 parser.
    if op.op == "const":
        val = op.attrs.get("value")
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
        return np.array(val)
    raise ValueError(f"Unsupported op: {op.op}")


def _cubic_weight(t: np.ndarray, a: float) -> np.ndarray:
    t = np.abs(t).astype(np.float32)
    w1 = ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0
    w2 = (((t - 5.0) * t + 8.0) * t - 4.0) * a
    return np.where(t < 1.0, w1, np.where(t < 2.0, w2, 0.0)).astype(np.float32)


def _upsample_bicubic2d_aa_numpy(x: np.ndarray, *, OH: int, OW: int, rs_h: float, rs_w: float, a: float) -> np.ndarray:
    """
    Numpy implementation matching `tritonops.upsample_bicubic2d_aa.upsample_bicubic2d_aa_kernel`.
    This is the "upsample only" path where support=2.0 and invscale=1.0.
    """
    x = np.asarray(x, dtype=np.float32)
    N, C, IH, IW = (int(s) for s in x.shape)
    out = np.zeros((N, C, int(OH), int(OW)), dtype=np.float32)
    support_w = 2.0
    support_h = 2.0
    invscale_w = 1.0
    invscale_h = 1.0

    for oh in range(int(OH)):
        center_h = (oh + 0.5) * float(rs_h)
        span_start_h = int(max(center_h - support_h + 0.5, 0.0))
        span_size_h = int(min(center_h + support_h + 0.5, float(IH)) - span_start_h)
        start_minus_center_h = float(span_start_h) - float(center_h)
        wy = np.zeros((5,), dtype=np.float32)
        for y in range(5):
            if y < span_size_h:
                t = (y + start_minus_center_h + 0.5) * invscale_h
                wy[y] = float(_cubic_weight(np.array(t, dtype=np.float32), a))
        s = float(np.sum(wy))
        if s == 0.0:
            s = 1.0
        wy /= s

        for ow in range(int(OW)):
            center_w = (ow + 0.5) * float(rs_w)
            span_start_w = int(max(center_w - support_w + 0.5, 0.0))
            span_size_w = int(min(center_w + support_w + 0.5, float(IW)) - span_start_w)
            start_minus_center_w = float(span_start_w) - float(center_w)
            wx = np.zeros((5,), dtype=np.float32)
            for xk in range(5):
                if xk < span_size_w:
                    t = (xk + start_minus_center_w + 0.5) * invscale_w
                    wx[xk] = float(_cubic_weight(np.array(t, dtype=np.float32), a))
            sx = float(np.sum(wx))
            if sx == 0.0:
                sx = 1.0
            wx /= sx

            for n in range(N):
                for c in range(C):
                    acc = 0.0
                    for y in range(5):
                        iy = span_start_h + y
                        if iy < 0 or iy >= IH:
                            continue
                        wyv = float(wy[y])
                        if wyv == 0.0:
                            continue
                        for xk in range(5):
                            ix = span_start_w + xk
                            if ix < 0 or ix >= IW:
                                continue
                            wxv = float(wx[xk])
                            if wxv == 0.0:
                                continue
                            acc += float(x[n, c, iy, ix]) * wyv * wxv
                    out[n, c, oh, ow] = np.float32(acc)
    return out


def _eval_epilogue(node: Any, env: Dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(node, str):
        return _get(env, node)
    op = node.get("op")
    inputs = node.get("inputs", [])
    args = [_eval_epilogue(inp, env) if isinstance(inp, dict) else _get(env, inp) for inp in inputs]
    if op not in ELEM_OPS:
        raise ValueError(f"Unsupported epilogue op: {op}")
    return ELEM_OPS[op](*args)


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
    # Heuristic: if leading dimension matches and ndims differ, pad trailing ones on the smaller
    if a.shape and b.shape and a.shape[0] == b.shape[0]:
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
