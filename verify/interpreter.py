"""
Numpy-based interpreter for Intent-IR v1.1.
Supports matmul (+epilogue), broadcast_in_dim, transpose, reshape, layout_cast (no-op),
reduce_sum, softmax, and basic elemwise ops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os

# Avoid MKL/OpenMP threading-layer conflicts when importing torch after numpy in
# some environments (e.g., upsample_* ops use torch.nn.functional.interpolate).
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

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
    "remainder": np.remainder,
    "pow": np.power,
}

NUM_UNARY_OPS = {
    "relu": lambda x: np.maximum(x, 0),
    "exp": np.exp,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "abs": np.abs,
    "floor": np.floor,
    "ceil": np.ceil,
    "sqrt": np.sqrt,
    "neg": np.negative,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "acos": np.arccos,
    "atan": np.arctan,
    "angle": np.angle,
    "tan": np.tan,
    "erf": lambda x: np.vectorize(math.erf, otypes=[np.float64])(x),
    "identity": lambda x: x,
}

CMP_BIN_OPS = {
    "ne": np.not_equal,
    "eq": np.equal,
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

BITWISE_BIN_OPS = {
    "bitwise_and": np.bitwise_and,
    "bitwise_or": np.bitwise_or,
    "bitwise_left_shift": np.left_shift,
    "bitwise_right_shift": np.right_shift,
}

BITWISE_UNARY_OPS = {
    "bitwise_not": np.bitwise_not,
}

INTERPRETER_SUPPORTED_OPS: set[str] = set().union(
    set(NUM_BIN_OPS.keys()),
    set(NUM_UNARY_OPS.keys()),
    set(CMP_BIN_OPS.keys()),
    set(BOOL_BIN_OPS.keys()),
    set(BOOL_UNARY_OPS.keys()),
    set(BITWISE_BIN_OPS.keys()),
    set(BITWISE_UNARY_OPS.keys()),
    {
        "broadcast_in_dim",
        "concat",
        "stack",
        "transpose",
        "reshape",
        "tile",
        "repeat",
        "repeat_interleave",
        "pad",
        "sort",
        "topk",
        "unique",
        "nonzero",
        "count_nonzero",
        "trace",
        "triu",
        "diag",
        "diag_embed",
        "scatter",
        "select_scatter",
        "slice_scatter",
        "quantile",
        "polar",
        "scaled_dot_product_attention",
        "weight_norm_interface",
        "per_token_group_quant_fp8",
        "kron",
        "masked_select",
        "masked_scatter",
        "upsample_nearest1d",
        "upsample_nearest2d",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_depthwise2d",
        "max_pool2d_with_indices",
        "mse_loss",
        "nan_to_num",
        "nll_loss_forward",
        "nll_loss2d_forward",
        "glu",
        "cummax",
        "cummin",
        "index_add",
        "index_put",
        "layout_cast",
        "cast",
        "iota",
        "gather",
        "where",
        "dropout",
        "correlation",
        "resize",
        "warp",
        "rope",
        "reduce_sum",
        "reduce_any",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "mean",
        "var",
        "std",
        "argmax",
        "argmin",
        "cumsum",
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
        if op.op == "exp":
            base = op.attrs.get("base")
            if base is not None and str(base) in {"2", "2.0"}:
                return np.exp2(x)
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
    if op.op in BITWISE_BIN_OPS:
        a_name = op.inputs[0]
        b_name = op.inputs[1]
        a = _get(env, a_name).astype(np.int64, copy=False)
        b = _get(env, b_name).astype(np.int64, copy=False)
        a, b = _align_shapes_for_elemwise_named(intent, a_name, a, b_name, b, shape_bindings)
        return BITWISE_BIN_OPS[op.op](a, b)
    if op.op in BITWISE_UNARY_OPS:
        x = _get(env, op.inputs[0]).astype(np.int64, copy=False)
        return BITWISE_UNARY_OPS[op.op](x)
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
    if op.op == "concat":
        if not op.inputs:
            raise ValueError("concat requires at least one input")
        xs = [_get(env, name) for name in op.inputs]
        axis = int(op.attrs.get("axis", 0))
        return np.concatenate(xs, axis=axis)
    if op.op == "stack":
        if not op.inputs:
            raise ValueError("stack requires at least one input")
        xs = [_get(env, name) for name in op.inputs]
        axis = int(op.attrs.get("axis", 0))
        return np.stack(xs, axis=axis)
    if op.op == "reshape":
        x = _get(env, op.inputs[0])
        shape = _shape_from_attr(op.attrs.get("shape"), shape_bindings)
        return np.reshape(x, shape)
    if op.op == "tile":
        x = _get(env, op.inputs[0])
        repeats = _normalize_repeats(op.attrs.get("repeats"))
        return np.tile(x, repeats)
    if op.op in {"repeat", "repeat_interleave"}:
        x = _get(env, op.inputs[0])
        repeats_raw = op.attrs.get("repeats")
        if repeats_raw is None:
            raise ValueError(f"{op.op} requires attrs.repeats")
        axis = op.attrs.get("axis")
        axis_int = None if axis is None else int(axis)
        repeats = _normalize_repeat_repeats(repeats_raw)
        return np.repeat(x, repeats, axis=axis_int)
    if op.op == "pad":
        x = _get(env, op.inputs[0])
        pad_width = _normalize_pad_width(op.attrs.get("pad_width"), x.ndim)
        mode = str(op.attrs.get("mode", "constant"))
        if mode == "constant":
            value = op.attrs.get("value", 0)
            return np.pad(x, pad_width, mode=mode, constant_values=value)
        return np.pad(x, pad_width, mode=mode)
    if op.op == "sort":
        x = _get(env, op.inputs[0])
        axis = int(op.attrs.get("axis", -1))
        descending = bool(op.attrs.get("descending", False))
        stable = bool(op.attrs.get("stable", False))
        try:
            out = np.sort(x, axis=axis, kind=("stable" if stable else "quicksort"))
        except TypeError:
            out = np.sort(x, axis=axis)
        if descending:
            out = np.flip(out, axis=axis)
        return out
    if op.op == "topk":
        x = _get(env, op.inputs[0])
        k = int(op.attrs.get("k"))
        axis = int(op.attrs.get("axis", -1))
        largest = bool(op.attrs.get("largest", True))
        sorted_out = bool(op.attrs.get("sorted", True))
        axis_norm = axis if axis >= 0 else x.ndim + axis
        if axis_norm < 0 or axis_norm >= x.ndim:
            raise ValueError(f"topk axis out of range: axis={axis} rank={x.ndim}")
        axis_dim = int(x.shape[axis_norm])
        if k < 0 or k > axis_dim:
            raise ValueError(f"topk k out of range: k={k}, axis_dim={axis_dim}")
        if k == 0:
            slicer = [slice(None)] * x.ndim
            slicer[axis_norm] = slice(0, 0)
            return x[tuple(slicer)]
        if largest:
            part = np.argpartition(x, axis_dim - k, axis=axis_norm)
            idx = np.take(part, np.arange(axis_dim - k, axis_dim), axis=axis_norm)
            vals = np.take_along_axis(x, idx, axis=axis_norm)
            if sorted_out:
                order = np.argsort(vals, axis=axis_norm)
                order = np.flip(order, axis=axis_norm)
                vals = np.take_along_axis(vals, order, axis=axis_norm)
            return vals
        part = np.argpartition(x, k - 1, axis=axis_norm)
        idx = np.take(part, np.arange(0, k), axis=axis_norm)
        vals = np.take_along_axis(x, idx, axis=axis_norm)
        if sorted_out:
            order = np.argsort(vals, axis=axis_norm)
            vals = np.take_along_axis(vals, order, axis=axis_norm)
        return vals
    if op.op == "unique":
        x = _get(env, op.inputs[0])
        axis = op.attrs.get("axis")
        sorted_out = bool(op.attrs.get("sorted", True))
        if axis is not None:
            # NumPy axis-unique is always sorted lexicographically.
            return np.unique(x, axis=int(axis))
        flat = np.ravel(x)
        if sorted_out:
            return np.unique(flat)
        uniq, first_idx = np.unique(flat, return_index=True)
        order = np.argsort(first_idx)
        return uniq[order]
    if op.op == "nonzero":
        x = _get(env, op.inputs[0])
        idx = np.nonzero(x)
        if not idx:
            return np.zeros((0, 0), dtype=np.int64)
        return np.stack(idx, axis=-1).astype(np.int64, copy=False)
    if op.op == "count_nonzero":
        x = _get(env, op.inputs[0])
        dims_raw = op.attrs.get("axes", op.attrs.get("dims", op.attrs.get("axis")))
        dims = None if dims_raw is None else _resolve_dims(dims_raw, x)
        keepdims = bool(op.attrs.get("keepdims", False))
        out = np.count_nonzero(x, axis=dims, keepdims=keepdims)
        return np.asarray(out, dtype=np.int64)
    if op.op == "trace":
        x = np.asarray(_get(env, op.inputs[0]))
        if x.ndim != 2:
            raise ValueError(f"trace expects rank-2 tensor, got shape={x.shape}")
        diagonal = int(op.attrs.get("diagonal", 0))
        out = np.trace(x, offset=diagonal)
        if np.issubdtype(x.dtype, np.floating):
            return np.asarray(out, dtype=x.dtype)
        return np.asarray(out, dtype=np.int64)
    if op.op == "triu":
        x = np.asarray(_get(env, op.inputs[0]))
        diagonal = int(op.attrs.get("diagonal", 0))
        return np.triu(x, k=diagonal)
    if op.op == "diag":
        x = _get(env, op.inputs[0])
        diagonal = int(op.attrs.get("diagonal", 0))
        return np.diag(x, k=diagonal)
    if op.op == "diag_embed":
        x = np.asarray(_get(env, op.inputs[0]))
        if x.ndim < 1:
            raise ValueError("diag_embed expects input rank >= 1")
        offset = int(op.attrs.get("offset", 0))
        dim1 = int(op.attrs.get("dim1", -2))
        dim2 = int(op.attrs.get("dim2", -1))
        out_rank = x.ndim + 1
        d1 = dim1 if dim1 >= 0 else out_rank + dim1
        d2 = dim2 if dim2 >= 0 else out_rank + dim2
        if d1 < 0 or d1 >= out_rank or d2 < 0 or d2 >= out_rank or d1 == d2:
            raise ValueError(f"diag_embed dim1/dim2 invalid for rank {out_rank}: dim1={dim1}, dim2={dim2}")
        n = int(x.shape[-1])
        diag_extent = n + abs(offset)
        out = np.zeros(tuple(int(v) for v in x.shape[:-1]) + (diag_extent, diag_extent), dtype=x.dtype)
        if offset >= 0:
            rows = np.arange(n, dtype=np.int64)
            cols = rows + int(offset)
        else:
            cols = np.arange(n, dtype=np.int64)
            rows = cols + int(-offset)
        out[..., rows, cols] = x
        if [d1, d2] != [out_rank - 2, out_rank - 1]:
            out = np.moveaxis(out, [out_rank - 2, out_rank - 1], [d1, d2])
        return out
    if op.op == "scatter":
        if len(op.inputs) != 3:
            raise ValueError("scatter requires 3 inputs (inp, index, src)")
        inp = np.array(_get(env, op.inputs[0]), copy=True)
        index = np.asarray(_get(env, op.inputs[1]), dtype=np.int64)
        src = np.asarray(_get(env, op.inputs[2]), dtype=inp.dtype)
        dim = int(op.attrs.get("dim", 0))
        axis = dim if dim >= 0 else inp.ndim + dim
        if axis < 0 or axis >= inp.ndim:
            raise ValueError(f"scatter dim out of range: dim={dim} rank={inp.ndim}")
        if tuple(index.shape) != tuple(src.shape):
            raise ValueError(f"scatter expects index/src shape match, got {index.shape} vs {src.shape}")
        if tuple(index.shape) != tuple(inp.shape):
            # Follow torch.scatter broadcasting semantics only partially for now:
            # enforce exact-shape kernels used by deterministic specs.
            raise ValueError(f"scatter currently expects index shape == inp shape, got {index.shape} vs {inp.shape}")
        axis_size = int(inp.shape[axis])
        idx = np.where(index < 0, index + axis_size, index)
        if np.any((idx < 0) | (idx >= axis_size)):
            raise ValueError("scatter index out of bounds")
        reduce = str(op.attrs.get("reduce", "none")).strip().lower()
        if reduce in {"", "none"}:
            np.put_along_axis(inp, idx, src, axis=axis)
            return inp

        # Build advanced indices for reduction variants.
        grid = np.indices(idx.shape, sparse=False)
        adv: list[np.ndarray] = []
        for d in range(inp.ndim):
            if d == axis:
                adv.append(idx)
            else:
                adv.append(grid[d])
        if reduce == "add":
            np.add.at(inp, tuple(adv), src)
            return inp
        if reduce == "multiply":
            # NumPy has no multiply.at; use scalar loop for correctness.
            for pos in np.ndindex(tuple(int(v) for v in idx.shape)):
                key = [int(p) for p in pos]
                key[axis] = int(idx[pos])
                inp[tuple(key)] *= src[pos]
            return inp
        raise ValueError(f"scatter unsupported reduce mode: {reduce}")
    if op.op == "select_scatter":
        if len(op.inputs) != 2:
            raise ValueError("select_scatter requires 2 inputs (inp, src)")
        inp = np.array(_get(env, op.inputs[0]), copy=True)
        src = np.asarray(_get(env, op.inputs[1]), dtype=inp.dtype)
        dim = int(op.attrs.get("dim", 0))
        axis = dim if dim >= 0 else inp.ndim + dim
        if axis < 0 or axis >= inp.ndim:
            raise ValueError(f"select_scatter dim out of range: dim={dim} rank={inp.ndim}")
        idx = int(op.attrs.get("index", 0))
        axis_size = int(inp.shape[axis])
        idx = idx if idx >= 0 else idx + axis_size
        if idx < 0 or idx >= axis_size:
            raise ValueError(f"select_scatter index out of bounds: index={idx} axis_size={axis_size}")
        expected = list(inp.shape)
        del expected[axis]
        if tuple(src.shape) != tuple(expected):
            raise ValueError(f"select_scatter src shape mismatch: expected {tuple(expected)} got {tuple(src.shape)}")
        sl = [slice(None)] * inp.ndim
        sl[axis] = idx
        inp[tuple(sl)] = src
        return inp
    if op.op == "slice_scatter":
        if len(op.inputs) != 2:
            raise ValueError("slice_scatter requires 2 inputs (inp, src)")
        inp = np.array(_get(env, op.inputs[0]), copy=True)
        src = np.asarray(_get(env, op.inputs[1]), dtype=inp.dtype)
        dim = int(op.attrs.get("dim", 0))
        axis = dim if dim >= 0 else inp.ndim + dim
        if axis < 0 or axis >= inp.ndim:
            raise ValueError(f"slice_scatter dim out of range: dim={dim} rank={inp.ndim}")
        step = int(op.attrs.get("step", 1))
        if step == 0:
            raise ValueError("slice_scatter step must be non-zero")
        axis_size = int(inp.shape[axis])
        start = int(op.attrs.get("start", 0))
        end = op.attrs.get("end")
        if end is None:
            end = axis_size
        end = int(end)
        if start < 0:
            start += axis_size
        if end < 0:
            end += axis_size
        start = max(0, min(start, axis_size))
        end = max(0, min(end, axis_size))
        sl = [slice(None)] * inp.ndim
        sl[axis] = slice(start, end, step)
        target = inp[tuple(sl)]
        if tuple(src.shape) != tuple(target.shape):
            raise ValueError(f"slice_scatter src shape mismatch: expected {target.shape} got {src.shape}")
        inp[tuple(sl)] = src
        return inp
    if op.op == "quantile":
        if len(op.inputs) != 2:
            raise ValueError("quantile requires 2 inputs (inp, q)")
        x = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        q = np.asarray(_get(env, op.inputs[1]), dtype=np.float32)
        dim = op.attrs.get("dim")
        axis = None if dim is None else int(dim)
        keepdim = bool(op.attrs.get("keepdim", False))
        method = str(op.attrs.get("interpolation", "linear"))
        try:
            out = np.quantile(x, q, axis=axis, keepdims=keepdim, method=method)
        except TypeError:
            out = np.quantile(x, q, axis=axis, keepdims=keepdim, interpolation=method)
        return np.asarray(out, dtype=np.float32)
    if op.op == "polar":
        if len(op.inputs) != 2:
            raise ValueError("polar requires 2 inputs (abs, angle)")
        abs_v = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        angle_v = np.asarray(_get(env, op.inputs[1]), dtype=np.float32)
        real = abs_v * np.cos(angle_v)
        imag = abs_v * np.sin(angle_v)
        return np.stack([real, imag], axis=-1).astype(np.float32, copy=False)
    if op.op == "scaled_dot_product_attention":
        if len(op.inputs) < 3:
            raise ValueError("scaled_dot_product_attention requires at least 3 inputs (query, key, value)")
        query = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        key = np.asarray(_get(env, op.inputs[1]), dtype=np.float32)
        value = np.asarray(_get(env, op.inputs[2]), dtype=np.float32)
        if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
            raise ValueError("scaled_dot_product_attention expects tensors with rank >= 2")
        if int(query.shape[-1]) != int(key.shape[-1]):
            raise ValueError(
                f"scaled_dot_product_attention head dim mismatch: query={query.shape[-1]} key={key.shape[-1]}"
            )
        if int(key.shape[-2]) != int(value.shape[-2]):
            raise ValueError(
                f"scaled_dot_product_attention sequence mismatch: key={key.shape[-2]} value={value.shape[-2]}"
            )
        scale = op.attrs.get("scale")
        if scale is None:
            scale_f = 1.0 / math.sqrt(max(1.0, float(query.shape[-1])))
        else:
            scale_f = float(scale)
        scores = np.matmul(query, np.swapaxes(key, -1, -2)) * np.float32(scale_f)
        if len(op.inputs) >= 4:
            attn_mask = np.asarray(_get(env, op.inputs[3]))
            if attn_mask.dtype == np.bool_:
                scores = np.where(attn_mask, scores, np.float32(-1.0e9))
            else:
                scores = scores + np.asarray(attn_mask, dtype=np.float32)
        if bool(op.attrs.get("is_causal", False)):
            q_len = int(scores.shape[-2])
            kv_len = int(scores.shape[-1])
            causal = np.triu(np.ones((q_len, kv_len), dtype=np.bool_), k=1)
            scores = np.where(causal, np.float32(-1.0e9), scores)
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        denom = np.sum(exp_scores, axis=-1, keepdims=True)
        denom = np.maximum(denom, np.finfo(np.float32).tiny)
        probs = exp_scores / denom
        out = np.matmul(probs, value)
        return np.asarray(out, dtype=np.float32)
    if op.op == "weight_norm_interface":
        if len(op.inputs) != 2:
            raise ValueError("weight_norm_interface requires 2 inputs (v, g)")
        v = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        g = np.asarray(_get(env, op.inputs[1]), dtype=np.float32)
        dim = int(op.attrs.get("dim", 0))
        axis = dim if dim >= 0 else v.ndim + dim
        if axis < 0 or axis >= v.ndim:
            raise ValueError(f"weight_norm_interface dim out of range: dim={dim} rank={v.ndim}")
        if g.ndim != 1 or int(g.shape[0]) != int(v.shape[axis]):
            raise ValueError(
                f"weight_norm_interface expects g shape [{v.shape[axis]}], got {g.shape}"
            )
        reduce_axes = tuple(i for i in range(v.ndim) if i != axis)
        eps = np.finfo(np.float32).tiny
        norm = np.sqrt(np.sum(v * v, axis=reduce_axes, keepdims=False) + eps)
        bshape = [1] * v.ndim
        bshape[axis] = int(v.shape[axis])
        g_b = np.reshape(g, bshape)
        norm_b = np.reshape(norm, bshape)
        out = (v / np.maximum(norm_b, eps)) * g_b
        return np.asarray(out, dtype=np.float32)
    if op.op == "per_token_group_quant_fp8":
        if len(op.inputs) != 1:
            raise ValueError("per_token_group_quant_fp8 requires 1 input")
        x = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        if x.ndim < 1:
            raise ValueError("per_token_group_quant_fp8 expects input rank >= 1")
        group_size = int(op.attrs.get("group_size", 0))
        if group_size <= 0:
            raise ValueError("per_token_group_quant_fp8 requires attrs.group_size > 0")
        last = int(x.shape[-1])
        if last % group_size != 0:
            raise ValueError(
                f"per_token_group_quant_fp8 last dimension must be divisible by group_size, got {last} and {group_size}"
            )
        eps = float(op.attrs.get("eps", 1.0e-10))
        scale_ue8m0 = bool(op.attrs.get("scale_ue8m0", False))
        fp8_min = np.float32(-448.0)
        fp8_max = np.float32(448.0)
        flat = np.reshape(x, (-1, last))
        groups = np.reshape(flat, (flat.shape[0], last // group_size, group_size))
        absmax = np.max(np.abs(groups), axis=-1, keepdims=True)
        scale = np.maximum(absmax, np.float32(eps)) / fp8_max
        if scale_ue8m0:
            scale = np.exp2(np.ceil(np.log2(np.maximum(np.abs(scale), np.float32(1.0e-10)))))
        q = np.clip(groups / scale, fp8_min, fp8_max)
        return np.reshape(q, x.shape).astype(np.float32, copy=False)
    if op.op == "kron":
        if len(op.inputs) != 2:
            raise ValueError("kron requires 2 inputs")
        a = np.asarray(_get(env, op.inputs[0]))
        b = np.asarray(_get(env, op.inputs[1]))
        return np.kron(a, b)
    if op.op == "masked_select":
        if len(op.inputs) != 2:
            raise ValueError("masked_select requires 2 inputs (inp, mask)")
        inp = np.asarray(_get(env, op.inputs[0]))
        mask = np.asarray(_get(env, op.inputs[1]), dtype=np.bool_)
        if inp.shape != mask.shape:
            raise ValueError(
                f"masked_select requires inp/mask shape match, got inp={inp.shape}, mask={mask.shape}"
            )
        # PyTorch semantics: flatten selected values in row-major order.
        return inp[mask].reshape(-1)
    if op.op == "masked_scatter":
        if len(op.inputs) != 3:
            raise ValueError("masked_scatter requires 3 inputs (inp, mask, source)")
        inp = np.asarray(_get(env, op.inputs[0]))
        mask = np.asarray(_get(env, op.inputs[1]), dtype=np.bool_)
        source = np.asarray(_get(env, op.inputs[2]))
        if inp.shape != mask.shape:
            raise ValueError(
                f"masked_scatter requires inp/mask shape match, got inp={inp.shape}, mask={mask.shape}"
            )
        out = np.array(inp, copy=True)
        out_flat = out.reshape(-1)
        mask_flat = mask.reshape(-1)
        src_flat = source.reshape(-1)
        required = int(mask_flat.sum())
        if int(src_flat.size) < required:
            raise ValueError(
                f"masked_scatter source too small: need {required} values, got {int(src_flat.size)}"
            )
        out_flat[mask_flat] = src_flat[:required]
        return out
    if op.op == "upsample_nearest1d":
        x = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"upsample_nearest1d expects rank-3 NCL tensor, got shape={x.shape}")
        output_size = op.attrs.get("output_size")
        if isinstance(output_size, list):
            if len(output_size) != 1:
                raise ValueError(f"upsample_nearest1d.output_size expects len=1, got {output_size}")
            out_l = int(output_size[0])
        elif isinstance(output_size, int):
            out_l = int(output_size)
        else:
            out_shape = _shape_from_tensor(intent, op.output, shape_bindings)
            if out_shape is None or len(out_shape) != 3:
                raise ValueError("upsample_nearest1d requires output_size attr or rank-3 output shape")
            out_l = int(out_shape[2])
        if out_l <= 0:
            raise ValueError(f"upsample_nearest1d output size must be > 0, got {out_l}")
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        tx = torch.from_numpy(x)
        out = F.interpolate(tx, size=out_l, mode="nearest")
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float32)
    if op.op == "upsample_nearest2d":
        x = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"upsample_nearest2d expects rank-4 NCHW tensor, got shape={x.shape}")
        output_size = op.attrs.get("output_size")
        if isinstance(output_size, list):
            if len(output_size) != 2:
                raise ValueError(f"upsample_nearest2d.output_size expects len=2, got {output_size}")
            out_h, out_w = int(output_size[0]), int(output_size[1])
        elif isinstance(output_size, int):
            out_h, out_w = int(output_size), int(output_size)
        else:
            out_shape = _shape_from_tensor(intent, op.output, shape_bindings)
            if out_shape is None or len(out_shape) != 4:
                raise ValueError("upsample_nearest2d requires output_size attr or rank-4 output shape")
            out_h, out_w = int(out_shape[2]), int(out_shape[3])
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"upsample_nearest2d output size must be > 0, got {(out_h, out_w)}")
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        tx = torch.from_numpy(x)
        out = F.interpolate(tx, size=(out_h, out_w), mode="nearest")
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float32)
    if op.op == "mse_loss":
        if len(op.inputs) != 2:
            raise ValueError("mse_loss requires 2 inputs (inp, target)")
        inp_name = op.inputs[0]
        tgt_name = op.inputs[1]
        inp = np.asarray(_get(env, inp_name))
        tgt = np.asarray(_get(env, tgt_name))
        inp, tgt = _align_shapes_for_elemwise_named(intent, inp_name, inp, tgt_name, tgt, shape_bindings)
        sq = np.square(inp - tgt)
        reduction = int(op.attrs.get("reduction", 1))
        if reduction == 0:
            return np.asarray(sq, dtype=np.float32)
        if reduction == 1:
            return np.asarray(np.mean(sq), dtype=np.float32)
        if reduction == 2:
            return np.asarray(np.sum(sq), dtype=np.float32)
        raise ValueError(f"mse_loss reduction must be 0|1|2, got {reduction}")
    if op.op == "nan_to_num":
        if len(op.inputs) != 1:
            raise ValueError("nan_to_num requires 1 input")
        x = np.asarray(_get(env, op.inputs[0]))
        nan = op.attrs.get("nan", 0.0)
        posinf = op.attrs.get("posinf", None)
        neginf = op.attrs.get("neginf", None)
        return np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    if op.op == "nll_loss_forward":
        if len(op.inputs) < 2:
            raise ValueError("nll_loss_forward requires at least 2 inputs (self, target)")
        logits = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        target = np.asarray(_get(env, op.inputs[1]), dtype=np.int64)
        if logits.ndim != 2:
            raise ValueError(f"nll_loss_forward expects logits rank=2 [N,C], got shape={logits.shape}")
        if target.ndim != 1:
            raise ValueError(f"nll_loss_forward expects target rank=1 [N], got shape={target.shape}")
        n, c = [int(v) for v in logits.shape]
        if int(target.shape[0]) != n:
            raise ValueError(f"nll_loss_forward target shape mismatch: expect ({n},), got {tuple(int(v) for v in target.shape)}")
        weight = None
        if len(op.inputs) >= 3:
            weight = np.asarray(_get(env, op.inputs[2]), dtype=np.float32).reshape(-1)
            if int(weight.shape[0]) != c:
                raise ValueError(f"nll_loss_forward weight shape mismatch: expect ({c},), got {tuple(weight.shape)}")
        reduction = int(op.attrs.get("reduction", 1))
        ignore_index = int(op.attrs.get("ignore_index", -100))

        valid = target != ignore_index
        clamped = np.clip(target, 0, max(0, c - 1))
        picked = logits[np.arange(n, dtype=np.int64), clamped]
        losses = -picked
        if weight is not None:
            losses = losses * weight[clamped]
            total_weight = float(np.sum(weight[clamped][valid]))
        else:
            total_weight = float(np.sum(valid.astype(np.float32)))
        losses = np.where(valid, losses, np.zeros_like(losses))

        if reduction == 0:
            return np.asarray(losses, dtype=np.float32)
        if reduction == 1:
            denom = max(total_weight, 1e-12)
            return np.asarray(np.sum(losses) / denom, dtype=np.float32)
        if reduction == 2:
            return np.asarray(np.sum(losses), dtype=np.float32)
        raise ValueError(f"nll_loss_forward reduction must be 0|1|2, got {reduction}")
    if op.op == "nll_loss2d_forward":
        if len(op.inputs) < 2:
            raise ValueError("nll_loss2d_forward requires at least 2 inputs (self, target)")
        logits = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        target = np.asarray(_get(env, op.inputs[1]), dtype=np.int64)
        if logits.ndim != 4:
            raise ValueError(f"nll_loss2d_forward expects logits rank=4 [N,C,H,W], got shape={logits.shape}")
        if target.ndim != 3:
            raise ValueError(f"nll_loss2d_forward expects target rank=3 [N,H,W], got shape={target.shape}")
        n, c, h, w = [int(v) for v in logits.shape]
        if tuple(int(v) for v in target.shape) != (n, h, w):
            raise ValueError(
                f"nll_loss2d_forward target shape mismatch: expect {(n, h, w)}, got {tuple(int(v) for v in target.shape)}"
            )
        weight = None
        if len(op.inputs) >= 3:
            weight = np.asarray(_get(env, op.inputs[2]), dtype=np.float32).reshape(-1)
            if int(weight.shape[0]) != c:
                raise ValueError(f"nll_loss2d_forward weight shape mismatch: expect ({c},), got {tuple(weight.shape)}")
        reduction = int(op.attrs.get("reduction", 1))
        ignore_index = int(op.attrs.get("ignore_index", -100))

        valid = target != ignore_index
        clamped = np.clip(target, 0, max(0, c - 1))
        n_idx = np.arange(n, dtype=np.int64)[:, None, None]
        h_idx = np.arange(h, dtype=np.int64)[None, :, None]
        w_idx = np.arange(w, dtype=np.int64)[None, None, :]
        picked = logits[n_idx, clamped, h_idx, w_idx]
        losses = -picked
        if weight is not None:
            losses = losses * weight[clamped]
            total_weight = float(np.sum(weight[clamped][valid]))
        else:
            total_weight = float(np.sum(valid.astype(np.float32)))
        losses = np.where(valid, losses, np.zeros_like(losses))

        if reduction == 0:
            return np.asarray(losses, dtype=np.float32)
        if reduction == 1:
            denom = max(total_weight, 1e-12)
            return np.asarray(np.sum(losses) / denom, dtype=np.float32)
        if reduction == 2:
            return np.asarray(np.sum(losses), dtype=np.float32)
        raise ValueError(f"nll_loss2d_forward reduction must be 0|1|2, got {reduction}")
    if op.op == "glu":
        x = np.asarray(_get(env, op.inputs[0]))
        axis = int(op.attrs.get("axis", -1))
        axis_norm = axis if axis >= 0 else x.ndim + axis
        if axis_norm < 0 or axis_norm >= x.ndim:
            raise ValueError(f"glu axis out of range: axis={axis} rank={x.ndim}")
        axis_extent = int(x.shape[axis_norm])
        if axis_extent % 2 != 0:
            raise ValueError(f"glu requires even extent along axis={axis_norm}, got {axis_extent}")
        lhs, rhs = np.split(x, 2, axis=axis_norm)
        return lhs * (1.0 / (1.0 + np.exp(-rhs)))
    if op.op == "cummax":
        x = np.asarray(_get(env, op.inputs[0]))
        axis = int(op.attrs.get("axis", -1))
        return np.maximum.accumulate(x, axis=axis)
    if op.op == "cummin":
        x = np.asarray(_get(env, op.inputs[0]))
        axis = int(op.attrs.get("axis", -1))
        return np.minimum.accumulate(x, axis=axis)
    if op.op == "index_add":
        if len(op.inputs) != 3:
            raise ValueError("index_add requires 3 inputs (base, index, src)")
        base = np.array(_get(env, op.inputs[0]), copy=True)
        index = np.asarray(_get(env, op.inputs[1]), dtype=np.int64).reshape(-1)
        src = np.asarray(_get(env, op.inputs[2]), dtype=base.dtype)
        axis = int(op.attrs.get("axis", 0))
        axis_norm = axis if axis >= 0 else base.ndim + axis
        if axis_norm < 0 or axis_norm >= base.ndim:
            raise ValueError(f"index_add axis out of range: axis={axis} rank={base.ndim}")
        expected_shape = list(base.shape)
        expected_shape[axis_norm] = int(index.shape[0])
        if tuple(src.shape) != tuple(expected_shape):
            raise ValueError(
                f"index_add src shape mismatch: expected {tuple(expected_shape)} got {tuple(src.shape)}"
            )
        alpha = float(op.attrs.get("alpha", 1.0))
        src = np.asarray(src * alpha, dtype=base.dtype)
        base_mv = np.moveaxis(base, axis_norm, 0)
        src_mv = np.moveaxis(src, axis_norm, 0)
        np.add.at(base_mv, index, src_mv)
        return base
    if op.op == "index_put":
        if len(op.inputs) < 3:
            raise ValueError("index_put requires at least 3 inputs (base, indices..., values)")
        base = np.array(_get(env, op.inputs[0]), copy=True)
        idx_inputs = list(op.inputs[1:-1])
        value = np.asarray(_get(env, op.inputs[-1]), dtype=base.dtype)
        if not idx_inputs:
            raise ValueError("index_put requires at least one index tensor")
        idx_arrays = [np.asarray(_get(env, name), dtype=np.int64) for name in idx_inputs]
        idx_b = [np.asarray(arr, dtype=np.int64) for arr in np.broadcast_arrays(*idx_arrays)]
        idx_shape = tuple(int(x) for x in idx_b[0].shape)
        if value.shape != idx_shape:
            value = np.broadcast_to(value, idx_shape)
        accumulate = bool(op.attrs.get("accumulate", False))
        if accumulate:
            np.add.at(base, tuple(idx_b), value)
        else:
            base[tuple(idx_b)] = value
        return base
    if op.op == "avg_pool2d":
        x = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"avg_pool2d expects rank-4 NCHW tensor, got shape={x.shape}")

        def _pair(v: Any, *, default: tuple[int, int] | None = None) -> tuple[int, int]:
            if v is None:
                if default is None:
                    raise ValueError("avg_pool2d requires non-null pair")
                return default
            if isinstance(v, int):
                return (int(v), int(v))
            if isinstance(v, list) and len(v) == 2 and all(isinstance(t, int) for t in v):
                return (int(v[0]), int(v[1]))
            raise ValueError(f"avg_pool2d pair attr must be int or list[int,int], got: {v!r}")

        kernel_h, kernel_w = _pair(op.attrs.get("kernel_size"))
        stride_h, stride_w = _pair(op.attrs.get("stride"), default=(kernel_h, kernel_w))
        pad_h, pad_w = _pair(op.attrs.get("padding"), default=(0, 0))
        ceil_mode = bool(op.attrs.get("ceil_mode", False))
        count_include_pad = bool(op.attrs.get("count_include_pad", True))

        n, c, h, w = [int(v) for v in x.shape]
        out_h_f = ((h + 2 * pad_h - kernel_h) / stride_h) + 1.0
        out_w_f = ((w + 2 * pad_w - kernel_w) / stride_w) + 1.0
        out_h = int(np.ceil(out_h_f) if ceil_mode else np.floor(out_h_f))
        out_w = int(np.ceil(out_w_f) if ceil_mode else np.floor(out_w_f))
        out_h = max(0, out_h)
        out_w = max(0, out_w)

        x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0.0)
        mask = np.pad(
            np.ones((n, c, h, w), dtype=np.float32),
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0.0,
        )
        out = np.empty((n, c, out_h, out_w), dtype=np.float32)
        for oy in range(out_h):
            iy = oy * stride_h
            for ox in range(out_w):
                ix = ox * stride_w
                window = x_pad[:, :, iy : iy + kernel_h, ix : ix + kernel_w]
                if count_include_pad:
                    denom = float(kernel_h * kernel_w)
                    out[:, :, oy, ox] = np.sum(window, axis=(2, 3)) / denom
                else:
                    m = mask[:, :, iy : iy + kernel_h, ix : ix + kernel_w]
                    denom = np.sum(m, axis=(2, 3))
                    out[:, :, oy, ox] = np.sum(window, axis=(2, 3)) / np.maximum(denom, 1.0)
        return out
    if op.op == "max_pool2d_with_indices":
        x = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"max_pool2d_with_indices expects rank-4 NCHW tensor, got shape={x.shape}")

        def _pair(v: Any, *, default: tuple[int, int] | None = None) -> tuple[int, int]:
            if v is None:
                if default is None:
                    raise ValueError("max_pool2d_with_indices requires non-null pair")
                return default
            if isinstance(v, int):
                return (int(v), int(v))
            if isinstance(v, list) and len(v) == 2 and all(isinstance(t, int) for t in v):
                return (int(v[0]), int(v[1]))
            raise ValueError(f"max_pool2d_with_indices pair attr must be int or list[int,int], got: {v!r}")

        kernel_h, kernel_w = _pair(op.attrs.get("kernel_size"))
        stride_h, stride_w = _pair(op.attrs.get("stride"), default=(kernel_h, kernel_w))
        pad_h, pad_w = _pair(op.attrs.get("padding"), default=(0, 0))
        dil_h, dil_w = _pair(op.attrs.get("dilation"), default=(1, 1))
        ceil_mode = bool(op.attrs.get("ceil_mode", False))
        select = str(op.attrs.get("select", "values")).strip().lower()

        n, c, h, w = [int(v) for v in x.shape]
        out_h_f = ((h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h) + 1.0
        out_w_f = ((w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w) + 1.0
        out_h = int(np.ceil(out_h_f) if ceil_mode else np.floor(out_h_f))
        out_w = int(np.ceil(out_w_f) if ceil_mode else np.floor(out_w_f))
        out_h = max(0, out_h)
        out_w = max(0, out_w)

        vals = np.full((n, c, out_h, out_w), -np.inf, dtype=np.float32)
        idxs = np.full((n, c, out_h, out_w), -1, dtype=np.int64)
        x_pad = np.full((n, c, h + 2 * pad_h, w + 2 * pad_w), -np.inf, dtype=np.float32)
        i_pad = np.full((n, c, h + 2 * pad_h, w + 2 * pad_w), -1, dtype=np.int64)
        x_pad[:, :, pad_h : pad_h + h, pad_w : pad_w + w] = x
        base_idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
        i_pad[:, :, pad_h : pad_h + h, pad_w : pad_w + w] = base_idx

        for oy in range(out_h):
            iy0 = oy * stride_h
            for ox in range(out_w):
                ix0 = ox * stride_w
                best_val = np.full((n, c), -np.inf, dtype=np.float32)
                best_idx = np.full((n, c), -1, dtype=np.int64)
                for ky in range(kernel_h):
                    sy = iy0 + ky * dil_h
                    if sy < 0 or sy >= x_pad.shape[2]:
                        continue
                    for kx in range(kernel_w):
                        sx = ix0 + kx * dil_w
                        if sx < 0 or sx >= x_pad.shape[3]:
                            continue
                        cand_val = x_pad[:, :, sy, sx]
                        cand_idx = i_pad[:, :, sy, sx]
                        better = cand_val > best_val
                        best_val = np.where(better, cand_val, best_val)
                        best_idx = np.where(better, cand_idx, best_idx)
                vals[:, :, oy, ox] = best_val
                idxs[:, :, oy, ox] = best_idx

        if select in {"indices", "index"}:
            return idxs
        return vals
    if op.op == "conv1d":
        if len(op.inputs) < 2:
            raise ValueError("conv1d requires at least 2 inputs (input, weight)")
        x = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        w = np.asarray(_get(env, op.inputs[1]), dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"conv1d expects input rank=3 [N,C,L], got shape={x.shape}")
        if w.ndim != 3:
            raise ValueError(f"conv1d expects weight rank=3 [C_out,C_in/groups,K], got shape={w.shape}")
        b = None
        if len(op.inputs) >= 3:
            b = np.asarray(_get(env, op.inputs[2]), dtype=np.float32).reshape(-1)

        def _norm_1d(v: Any, *, default: int) -> int:
            if v is None:
                return int(default)
            if isinstance(v, int):
                return int(v)
            if isinstance(v, list) and len(v) == 1 and isinstance(v[0], int):
                return int(v[0])
            raise ValueError(f"conv1d attr must be int or list[int] len=1, got {v!r}")

        stride = _norm_1d(op.attrs.get("stride"), default=1)
        padding = _norm_1d(op.attrs.get("padding"), default=0)
        dilation = _norm_1d(op.attrs.get("dilation"), default=1)
        groups = int(op.attrs.get("groups", 1))

        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        tx = torch.from_numpy(x)
        tw = torch.from_numpy(w)
        tb = torch.from_numpy(b) if b is not None else None
        out = F.conv1d(tx, tw, bias=tb, stride=int(stride), padding=int(padding), dilation=int(dilation), groups=groups)
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float32)
    if op.op == "conv2d":
        if len(op.inputs) < 2:
            raise ValueError("conv2d requires at least 2 inputs (input, weight)")
        x = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        w = np.asarray(_get(env, op.inputs[1]), dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"conv2d expects input rank=4 [N,C,H,W], got shape={x.shape}")
        if w.ndim != 4:
            raise ValueError(f"conv2d expects weight rank=4 [C_out,C_in/groups,KH,KW], got shape={w.shape}")
        b = None
        if len(op.inputs) >= 3:
            b = np.asarray(_get(env, op.inputs[2]), dtype=np.float32).reshape(-1)

        def _to_int_attr(v: Any, name: str) -> int:
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, str):
                if v in shape_bindings:
                    return int(shape_bindings[v])
                if v.lstrip("-").isdigit():
                    return int(v)
            raise ValueError(f"conv2d {name} must resolve to int, got {v!r}")

        def _norm_2d(v: Any, *, default: tuple[int, int], name: str) -> tuple[int, int]:
            if v is None:
                return default
            if isinstance(v, (int, np.integer, str)):
                x0 = _to_int_attr(v, name)
                return (x0, x0)
            if isinstance(v, list) and len(v) == 2:
                return (_to_int_attr(v[0], name), _to_int_attr(v[1], name))
            raise ValueError(f"conv2d {name} must be int/symbol or list[len=2], got {v!r}")

        stride = _norm_2d(op.attrs.get("stride"), default=(1, 1), name="stride")
        padding = _norm_2d(op.attrs.get("padding"), default=(0, 0), name="padding")
        dilation = _norm_2d(op.attrs.get("dilation"), default=(1, 1), name="dilation")
        groups = _to_int_attr(op.attrs.get("groups", 1), "groups")

        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        tx = torch.from_numpy(x)
        tw = torch.from_numpy(w)
        tb = torch.from_numpy(b) if b is not None else None
        out = F.conv2d(tx, tw, bias=tb, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float32)
    if op.op == "conv3d":
        if len(op.inputs) < 2:
            raise ValueError("conv3d requires at least 2 inputs (input, weight)")
        x = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        w = np.asarray(_get(env, op.inputs[1]), dtype=np.float32)
        if x.ndim != 5:
            raise ValueError(f"conv3d expects input rank=5 [N,C,D,H,W], got shape={x.shape}")
        if w.ndim != 5:
            raise ValueError(f"conv3d expects weight rank=5 [C_out,C_in/groups,KD,KH,KW], got shape={w.shape}")
        b = None
        if len(op.inputs) >= 3:
            b = np.asarray(_get(env, op.inputs[2]), dtype=np.float32).reshape(-1)

        def _norm_3d(v: Any, *, default: tuple[int, int, int]) -> tuple[int, int, int]:
            if v is None:
                return default
            if isinstance(v, int):
                return (int(v), int(v), int(v))
            if isinstance(v, list) and len(v) == 3 and all(isinstance(t, int) for t in v):
                return (int(v[0]), int(v[1]), int(v[2]))
            raise ValueError(f"conv3d attr must be int or list[int,int,int], got {v!r}")

        stride = _norm_3d(op.attrs.get("stride"), default=(1, 1, 1))
        padding = _norm_3d(op.attrs.get("padding"), default=(0, 0, 0))
        dilation = _norm_3d(op.attrs.get("dilation"), default=(1, 1, 1))
        groups = int(op.attrs.get("groups", 1))

        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        tx = torch.from_numpy(x)
        tw = torch.from_numpy(w)
        tb = torch.from_numpy(b) if b is not None else None
        out = F.conv3d(tx, tw, bias=tb, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float32)
    if op.op == "conv_depthwise2d":
        if len(op.inputs) < 2:
            raise ValueError("conv_depthwise2d requires at least 2 inputs (input, weight)")
        x = np.asarray(_get(env, op.inputs[0]), dtype=np.float32)
        w = np.asarray(_get(env, op.inputs[1]), dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"conv_depthwise2d expects input rank=4 [N,C,H,W], got shape={x.shape}")
        if w.ndim != 4:
            raise ValueError(f"conv_depthwise2d expects weight rank=4 [C_out,1,KH,KW], got shape={w.shape}")
        b = None
        if len(op.inputs) >= 3:
            b = np.asarray(_get(env, op.inputs[2]), dtype=np.float32).reshape(-1)

        def _norm_2d(v: Any, *, default: tuple[int, int]) -> tuple[int, int]:
            if v is None:
                return default
            if isinstance(v, int):
                return (int(v), int(v))
            if isinstance(v, list) and len(v) == 2 and all(isinstance(t, int) for t in v):
                return (int(v[0]), int(v[1]))
            raise ValueError(f"conv_depthwise2d attr must be int or list[int,int], got {v!r}")

        stride = _norm_2d(op.attrs.get("stride"), default=(1, 1))
        padding = _norm_2d(op.attrs.get("padding"), default=(0, 0))
        dilation = _norm_2d(op.attrs.get("dilation"), default=(1, 1))
        groups = int(op.attrs.get("groups", int(x.shape[1])))

        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        tx = torch.from_numpy(x)
        tw = torch.from_numpy(w)
        tb = torch.from_numpy(b) if b is not None else None
        out = F.conv2d(tx, tw, bias=tb, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float32)
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
    if op.op == "correlation":
        if len(op.inputs) != 3:
            raise ValueError("correlation requires 3 inputs (src0, src1, out_shift)")
        src0 = np.asarray(_get(env, op.inputs[0]), dtype=np.int8, order="C")
        src1 = np.asarray(_get(env, op.inputs[1]), dtype=np.int8, order="C")
        out_shift = int(np.asarray(_get(env, op.inputs[2])).reshape(()))
        out_shape = _shape_from_tensor(intent, op.output, shape_bindings)
        if out_shape is None or len(out_shape) != 3:
            raise ValueError(f"correlation output must be rank-3, got {out_shape}")
        out_c, H, W = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
        if src0.ndim != 3 or src1.ndim != 3:
            raise ValueError(f"correlation inputs must be rank-3, got {src0.shape} and {src1.shape}")
        if src0.shape != src1.shape:
            raise ValueError(f"correlation inputs shape mismatch: {src0.shape} vs {src1.shape}")
        in_c, h0, w0 = (int(src0.shape[0]), int(src0.shape[1]), int(src0.shape[2]))
        if (h0, w0) != (H, W):
            raise ValueError(f"correlation spatial shape mismatch: src={src0.shape[1:]} out={(H, W)}")
        out = np.zeros((out_c, H, W), dtype=np.int8)
        if out_c == 0 or in_c == 0 or H == 0 or W == 0:
            return out
        for oc in range(out_c):
            acc = np.zeros((H, W), dtype=np.int32)
            if oc == 0:
                for k in range(in_c):
                    acc += (src0[k].astype(np.int16) * src1[k].astype(np.int16)).astype(np.int32)
            else:
                # Shift src1 by oc along width: src1[..., w-oc] contributes to out[..., w].
                for k in range(in_c):
                    a = src0[k].astype(np.int16)
                    b = src1[k].astype(np.int16)
                    acc[:, oc:] += (a[:, oc:] * b[:, : W - oc]).astype(np.int32)
            out[oc] = (acc >> out_shift).astype(np.int8)
        return out
    if op.op == "resize":
        if len(op.inputs) != 1:
            raise ValueError("resize requires 1 input (src)")
        src = np.asarray(_get(env, op.inputs[0]), dtype=np.int8, order="C")
        out_shape = _shape_from_tensor(intent, op.output, shape_bindings)
        if out_shape is None or len(out_shape) != 3:
            raise ValueError(f"resize output must be rank-3, got {out_shape}")
        if src.ndim != 3:
            raise ValueError(f"resize input must be rank-3, got {src.shape}")
        C, H, W = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
        OC, OH, OW = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
        if OC != C:
            raise ValueError(f"resize channel mismatch: src C={C} out C={OC}")
        hw_fl = int(op.attrs.get("hw_fl", 7))
        factor = int(1 << hw_fl)
        # Current semantic: bilinear resize with 2x upsample (matches AI-Benchmark kernel).
        if OH != 2 * H or OW != 2 * W:
            raise ValueError(f"resize currently expects 2x upsample: src={(H, W)} out={(OH, OW)}")
        # Vectorized coordinate maps (integer fixed-point math).
        h_idx = np.arange(OH, dtype=np.int32)
        input_y = h_idx << (hw_fl - 1)
        y0 = input_y >> hw_fl
        h1 = input_y - (y0 << hw_fl)
        h0 = factor - h1
        y1 = np.minimum(y0 + 1, H - 1)

        w_idx = np.arange(OW, dtype=np.int32)
        input_x = w_idx << (hw_fl - 1)
        x0 = input_x >> hw_fl
        w1 = input_x - (x0 << hw_fl)
        w0 = factor - w1
        x1 = np.minimum(x0 + 1, W - 1)

        out = np.empty((C, OH, OW), dtype=np.int8)
        w0_i32 = w0.astype(np.int32, copy=False)
        w1_i32 = w1.astype(np.int32, copy=False)
        h0_i32 = h0.astype(np.int32, copy=False)[:, None]
        h1_i32 = h1.astype(np.int32, copy=False)[:, None]
        for c in range(C):
            s = src[c].astype(np.int16, copy=False)
            s0 = s[y0]  # (OH, W)
            s1 = s[y1]
            y0x0 = s0[:, x0]
            y0x1 = s0[:, x1]
            y1x0 = s1[:, x0]
            y1x1 = s1[:, x1]
            sum1 = ((y0x0.astype(np.int32) * w0_i32 + y0x1.astype(np.int32) * w1_i32) >> hw_fl).astype(np.int32)
            sum2 = ((y1x0.astype(np.int32) * w0_i32 + y1x1.astype(np.int32) * w1_i32) >> hw_fl).astype(np.int32)
            val = ((sum1 * h0_i32 + sum2 * h1_i32) >> hw_fl).astype(np.int32)
            out[c] = val.astype(np.int8)
        return out
    if op.op == "warp":
        if len(op.inputs) != 2:
            raise ValueError("warp requires 2 inputs (src, offset)")
        src = np.asarray(_get(env, op.inputs[0]), dtype=np.int8, order="C")
        offset = np.asarray(_get(env, op.inputs[1]), dtype=np.int16, order="C")
        out_shape = _shape_from_tensor(intent, op.output, shape_bindings)
        if out_shape is None or len(out_shape) != 3:
            raise ValueError(f"warp output must be rank-3, got {out_shape}")
        if src.ndim != 3 or offset.ndim != 2:
            raise ValueError(f"warp expects src rank-3 and offset rank-2, got {src.shape} and {offset.shape}")
        C, H, W = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
        OC, OH, OW = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
        if (OC, OH, OW) != (C, H, W):
            raise ValueError(f"warp output shape mismatch: src={(C, H, W)} out={(OC, OH, OW)}")
        if offset.shape != (H, W):
            raise ValueError(f"warp offset shape mismatch: expected {(H, W)} got {offset.shape}")
        # Match AI-Benchmark warp semantics (Q8.8 offset packed into int16).
        indvar = np.arange(W, dtype=np.int16).astype(np.int8)
        out = np.empty((C, H, W), dtype=np.int8)
        for h in range(H):
            off = offset[h].astype(np.int16, copy=False)
            offset_int = (off >> 8).astype(np.int8)
            offset_frac = ((off << 8) >> 8).astype(np.int8)
            right_i8 = (indvar.astype(np.int16) - offset_int.astype(np.int16)).astype(np.int8)
            left_i8 = (right_i8.astype(np.int16) - 1).astype(np.int8)
            right = right_i8.astype(np.int16)
            left = left_i8.astype(np.int16)
            right_ok = right >= 0
            left_ok = left >= 0
            right_idx = np.clip(right.astype(np.int64), 0, W - 1)
            left_idx = np.clip(left.astype(np.int64), 0, W - 1)
            frac_i16 = offset_frac.astype(np.int16)
            for c in range(C):
                row = src[c, h]
                rv = row[right_idx].astype(np.int8, copy=False)
                lv = row[left_idx].astype(np.int8, copy=False)
                rv = np.where(right_ok, rv, np.int8(0))
                lv = np.where(left_ok, lv, np.int8(0))
                outv = (rv.astype(np.int16) << 8) + (lv.astype(np.int16) - rv.astype(np.int16)) * frac_i16
                out[c, h] = (outv >> 8).astype(np.int8)
        return out
    if op.op == "rope":
        if len(op.inputs) != 3:
            raise ValueError("rope requires 3 inputs (input, cos, sin)")
        x = np.asarray(_get(env, op.inputs[0]), dtype=np.float32, order="C")
        cos = np.asarray(_get(env, op.inputs[1]), dtype=np.float32, order="C")
        sin = np.asarray(_get(env, op.inputs[2]), dtype=np.float32, order="C")
        if x.ndim != 4:
            raise ValueError(f"rope expects input rank-4 [SEQ,BATCH,HEAD,HEAD_DIM], got {x.shape}")
        if cos.ndim != 2 or sin.ndim != 2:
            raise ValueError(f"rope expects cos/sin rank-2 [SEQ,HEAD_DIM/2], got cos={cos.shape} sin={sin.shape}")
        SEQ, B, H, D = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
        if D % 2 != 0:
            raise ValueError("rope expects even HEAD_DIM")
        half = D // 2
        if cos.shape != (SEQ, half) or sin.shape != (SEQ, half):
            raise ValueError(f"rope cos/sin shape mismatch: expected ({SEQ},{half}) got cos={cos.shape} sin={sin.shape}")
        cos_b = cos[:, None, None, :]
        sin_b = sin[:, None, None, :]
        x1 = x[..., :half]
        x2 = x[..., half:]
        y1 = x1 * cos_b - x2 * sin_b
        y2 = x1 * sin_b + x2 * cos_b
        return np.concatenate([y1, y2], axis=-1).astype(np.float32, copy=False)
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
        combine_fn = str(op.attrs.get("combine_fn", "")).strip().lower()
        if combine_fn in {"and", "all"}:
            return np.all(x, axis=dims, keepdims=keepdims)
        return np.any(x, axis=dims, keepdims=keepdims)
    if op.op == "reduce_max":
        x = _get(env, op.inputs[0])
        dims_raw = op.attrs.get("axes", op.attrs.get("dims", op.attrs.get("axis")))
        dims = _resolve_dims(dims_raw, x)
        keepdims = bool(op.attrs.get("keepdims", False))
        return np.max(x, axis=dims, keepdims=keepdims)
    if op.op == "reduce_min":
        x = _get(env, op.inputs[0])
        dims_raw = op.attrs.get("axes", op.attrs.get("dims", op.attrs.get("axis")))
        dims = _resolve_dims(dims_raw, x)
        keepdims = bool(op.attrs.get("keepdims", False))
        return np.min(x, axis=dims, keepdims=keepdims)
    if op.op == "reduce_prod":
        x = _get(env, op.inputs[0])
        dims_raw = op.attrs.get("axes", op.attrs.get("dims", op.attrs.get("axis")))
        dims = _resolve_dims(dims_raw, x)
        keepdims = bool(op.attrs.get("keepdims", False))
        return np.prod(x, axis=dims, keepdims=keepdims)
    if op.op == "mean":
        x = _get(env, op.inputs[0])
        dims_raw = op.attrs.get("axes", op.attrs.get("dims", op.attrs.get("axis")))
        dims = _resolve_dims(dims_raw, x)
        keepdims = bool(op.attrs.get("keepdims", False))
        return np.mean(x, axis=dims, keepdims=keepdims)
    if op.op == "var":
        x = _get(env, op.inputs[0])
        dims_raw = op.attrs.get("axes", op.attrs.get("dims", op.attrs.get("axis")))
        dims = _resolve_dims(dims_raw, x)
        keepdims = bool(op.attrs.get("keepdims", False))
        return np.var(x, axis=dims, keepdims=keepdims)
    if op.op == "std":
        x = _get(env, op.inputs[0])
        dims_raw = op.attrs.get("axes", op.attrs.get("dims", op.attrs.get("axis")))
        dims = _resolve_dims(dims_raw, x)
        keepdims = bool(op.attrs.get("keepdims", False))
        correction = op.attrs.get("correction", op.attrs.get("ddof", 0))
        try:
            ddof = int(correction)
        except Exception:
            ddof = 0
        return np.std(x, axis=dims, keepdims=keepdims, ddof=ddof)
    if op.op == "argmax":
        x = _get(env, op.inputs[0])
        axis = op.attrs.get("axis", -1)
        if axis is None:
            axis = -1
        return np.argmax(x, axis=int(axis))
    if op.op == "argmin":
        x = _get(env, op.inputs[0])
        axis = op.attrs.get("axis", -1)
        if axis is None:
            axis = -1
        return np.argmin(x, axis=int(axis))
    if op.op == "cumsum":
        x = _get(env, op.inputs[0])
        axis = op.attrs.get("axis", -1)
        if axis is None:
            axis = -1
        return np.cumsum(x, axis=int(axis))
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


def _shape_from_tensor(intent: IntentFunction, tensor_name: str, bindings: Dict[str, int]) -> tuple[int, ...] | None:
    tt = intent.tensors.get(tensor_name)
    if tt is None:
        return None
    out: list[int] = []
    for d in tt.shape:
        if getattr(d, "kind", None) == "const":
            out.append(int(getattr(d, "value")))
            continue
        if getattr(d, "kind", None) == "sym":
            sym = str(getattr(d, "value"))
            if sym not in bindings:
                raise ValueError(f"unbound symbolic dim in tensor shape: {tensor_name}.{sym}")
            out.append(int(bindings[sym]))
            continue
        raise ValueError(f"invalid dim kind for {tensor_name}: {d}")
    return tuple(out)


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


def _normalize_repeats(repeats_raw: Any) -> tuple[int, ...]:
    if isinstance(repeats_raw, (int, np.integer)):
        return (int(repeats_raw),)
    if isinstance(repeats_raw, list) and repeats_raw and all(isinstance(x, (int, np.integer)) for x in repeats_raw):
        return tuple(int(x) for x in repeats_raw)
    raise ValueError(f"tile.repeats must be int or non-empty list[int], got: {type(repeats_raw).__name__}")


def _normalize_repeat_repeats(repeats_raw: Any) -> int | np.ndarray:
    if isinstance(repeats_raw, (int, np.integer)):
        return int(repeats_raw)
    if isinstance(repeats_raw, list) and repeats_raw and all(isinstance(x, (int, np.integer)) for x in repeats_raw):
        return np.asarray([int(x) for x in repeats_raw], dtype=np.int64)
    raise ValueError(f"repeat.repeats must be int or non-empty list[int], got: {type(repeats_raw).__name__}")


def _normalize_pad_width(pad_raw: Any, rank: int) -> list[tuple[int, int]]:
    if isinstance(pad_raw, dict):
        pairs = pad_raw.get("pairs")
        if isinstance(pairs, list):
            pad_raw = pairs
    if isinstance(pad_raw, list):
        if len(pad_raw) == 2 and all(isinstance(x, (int, np.integer)) for x in pad_raw):
            pair = (int(pad_raw[0]), int(pad_raw[1]))
            return [pair for _ in range(int(rank))]
        if len(pad_raw) == int(rank) and all(isinstance(x, list) and len(x) == 2 for x in pad_raw):
            out: list[tuple[int, int]] = []
            for x in pad_raw:
                if not all(isinstance(v, (int, np.integer)) for v in x):
                    raise ValueError("pad_width must contain int pairs")
                out.append((int(x[0]), int(x[1])))
            return out
    raise ValueError("pad_width must be {'pairs': [[l,r],...]} or [l,r] or [[l,r], ...]")


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
