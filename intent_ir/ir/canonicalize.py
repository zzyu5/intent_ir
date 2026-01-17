"""
IntentIR canonicalization passes.

This module provides lightweight, semantics-preserving rewrites that help:
  - reduce frontend/LLM representational drift (e.g., lt vs gt swap, relu vs max(0,x))
  - normalize common "plumbing" patterns (broadcast/cast wrappers) when safe

The primary consumer today is the paper experiment E4 (cross-frontend consistency),
but these passes are written as general utilities and are safe to apply on an
in-memory IntentFunction.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Set

from .ir_types import Dim, IntentFunction, Op, TensorLayout, TensorType

_CMP_SWAP = {"lt": "gt", "le": "ge"}

_ELEMENTWISE_BIN_OPS: Set[str] = {
    "add",
    "sub",
    "mul",
    "div",
    "max",
    "min",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "and",
    "or",
}

_BROADCAST_FOLD_CONSUMERS: Set[str] = set(_ELEMENTWISE_BIN_OPS) | {"where"}


def canonicalize_for_consistency(intent: IntentFunction) -> IntentFunction:
    """
    Canonicalize common equivalent surface forms.

    Conservative rules only; intended to be semantics-preserving.
    """
    fn: IntentFunction = copy.deepcopy(intent)

    # Pass 1: normalize comparison direction (lt/le -> gt/ge by swapping inputs).
    for op in fn.ops:
        if op.op in _CMP_SWAP and len(op.inputs) == 2:
            op.op = _CMP_SWAP[op.op]
            op.inputs = [op.inputs[1], op.inputs[0]]

    # Pass 2: fold "safe" broadcast_in_dim into elementwise consumers.
    _fold_trailing_broadcasts(fn)

    # Build a producer map after the broadcast folding.
    prod: Dict[str, Op] = {op.output: op for op in fn.ops}

    # Pass 3: relu canonicalization (max(x,0) -> relu(x)).
    for op in fn.ops:
        if op.op != "max" or len(op.inputs) != 2:
            continue
        a, b = op.inputs[0], op.inputs[1]
        a0 = _is_zero_like(fn, prod, a)
        b0 = _is_zero_like(fn, prod, b)
        if a0 and not b0:
            op.op = "relu"
            op.inputs = [b]
        elif b0 and not a0:
            op.op = "relu"
            op.inputs = [a]

    # Refresh producer map (op kinds changed).
    prod = {op.output: op for op in fn.ops}

    # Pass 4: mask multiply canonicalization (mul(x, cast(mask)) -> where(mask, x, 0)).
    _canonicalize_mask_mul_to_where(fn, prod)

    # Pass 5: semantic normalizations (safe, local, conservative).
    # For cross-frontend comparison we prefer a lowered, explicit softmax form.
    _expand_softmax_ops(fn)
    _rewrite_reduce_max_ne_zero_to_reduce_any(fn)

    # Pass 5: dead code elimination (drop unused broadcast/const/cast/identity chains).
    fn.ops = _dce_ops(fn.ops, outputs=set(fn.outputs))
    return fn


def _tensor_rank(fn: IntentFunction, name: str) -> Optional[int]:
    t = fn.tensors.get(name)
    if t is None:
        return None
    return int(len(t.shape))


def _is_scalar_like_tensor(fn: IntentFunction, name: str) -> bool:
    """
    Treat rank-0 tensors as scalars, and also accept the common LLM mistake of
    emitting a "scalar" as a length-1 vector (shape=[1]).
    """
    t = fn.tensors.get(name)
    if t is None:
        return False
    if len(t.shape) == 0:
        return True
    if len(t.shape) == 1:
        d0 = t.shape[0]
        return getattr(d0, "kind", None) == "const" and int(getattr(d0, "value", -1)) == 1
    return False


def _fold_trailing_broadcasts(fn: IntentFunction) -> None:
    if not fn.ops:
        return

    # Build use-sites for each value.
    uses: Dict[str, Set[int]] = {}
    for i, op in enumerate(fn.ops):
        for inp in op.inputs:
            uses.setdefault(inp, set()).add(i)

    # Rewrite consumers in-place; rely on DCE later to remove folded broadcasts.
    for i, op in enumerate(fn.ops):
        if op.op != "broadcast_in_dim" or len(op.inputs) != 1:
            continue
        out = op.output
        if out in set(fn.outputs):
            continue
        src = op.inputs[0]
        in_rank = _tensor_rank(fn, src)
        if in_rank is None:
            continue
        out_rank = _tensor_rank(fn, out)
        if out_rank is None:
            out_shape = op.attrs.get("out_shape")
            if isinstance(out_shape, list):
                out_rank = int(len(out_shape))
        if out_rank is None:
            continue

        bcast_dims = op.attrs.get("broadcast_dims", [])
        if not isinstance(bcast_dims, list) or any(not isinstance(x, int) for x in bcast_dims):
            continue

        # Safe fold cases:
        #  1) scalar -> any rank (broadcast dims are irrelevant)
        #  2) "numpy trailing" alignment: dims == [out-r_in, ..., out-1]
        foldable = False
        if in_rank == 0:
            foldable = True
        else:
            want = list(range(out_rank - in_rank, out_rank))
            if len(bcast_dims) == in_rank and [int(x) for x in bcast_dims] == want:
                foldable = True
        if not foldable:
            continue

        # Only fold if all consumers are elementwise-like.
        consumer_ids = uses.get(out, set())
        if not consumer_ids:
            continue
        if any(fn.ops[c].op not in _BROADCAST_FOLD_CONSUMERS for c in consumer_ids):
            continue

        for c in sorted(consumer_ids):
            cop = fn.ops[c]
            cop.inputs = [src if x == out else x for x in cop.inputs]


def _is_scalar_const_zero(fn: IntentFunction, prod: Dict[str, Op], name: str) -> bool:
    op = prod.get(name)
    if op is None or op.op != "const" or op.inputs:
        return False
    if not _is_scalar_like_tensor(fn, name):
        return False
    attrs = op.attrs or {}
    v = attrs.get("value")
    if isinstance(v, bool):
        return False
    if isinstance(v, str):
        return False
    try:
        return float(v) == 0.0
    except Exception:
        return False


def _is_zero_like(fn: IntentFunction, prod: Dict[str, Op], name: str) -> bool:
    # Direct scalar const(0).
    if _is_scalar_const_zero(fn, prod, name):
        return True

    # Broadcasted scalar const(0) (common in LLM outputs).
    op = prod.get(name)
    if op is None or op.op != "broadcast_in_dim" or len(op.inputs) != 1:
        return False
    src = op.inputs[0]
    return _is_scalar_const_zero(fn, prod, src)


def _canonicalize_mask_mul_to_where(fn: IntentFunction, prod: Dict[str, Op]) -> None:
    zero_cache: Dict[str, str] = {}
    prefix_ops: List[Op] = []

    def ensure_zero(dtype: str) -> str:
        if dtype in zero_cache:
            return zero_cache[dtype]
        # Reuse an existing const 0 if present.
        for op in fn.ops:
            if op.op != "const" or op.inputs:
                continue
            if str((op.attrs or {}).get("dtype") or "") != str(dtype):
                continue
            v = (op.attrs or {}).get("value")
            try:
                if float(v) == 0.0:
                    zero_cache[dtype] = op.output
                    return op.output
            except Exception:
                continue

        # Insert a fresh one (as a prefix op; do not mutate fn.ops while iterating).
        name = f"zero_{dtype}"
        if name in fn.tensors:
            # Ensure uniqueness if the name already exists (rare).
            i = 0
            while f"{name}_{i}" in fn.tensors:
                i += 1
            name = f"{name}_{i}"
        prefix_ops.append(Op(op="const", inputs=[], output=name, attrs={"value": 0.0, "dtype": dtype}))
        fn.tensors[name] = TensorType(dtype=str(dtype), shape=[], layout=TensorLayout(kind="row_major", params={}))
        zero_cache[dtype] = name
        return name

    def is_bool_dtype(dt: str | None) -> bool:
        return str(dt) in {"bool", "i1"}

    for op in fn.ops:
        if op.op != "mul" or len(op.inputs) != 2:
            continue
        a, b = op.inputs[0], op.inputs[1]
        cand_cast_a = prod.get(a)
        cand_cast_b = prod.get(b)

        # Pattern: mul(x, cast(mask)) or mul(cast(mask), x).
        cast_out = None
        x_name = None
        mask_name = None
        if cand_cast_a is not None and cand_cast_a.op == "cast" and len(cand_cast_a.inputs) == 1:
            cast_out = a
            x_name = b
            mask_name = cand_cast_a.inputs[0]
        elif cand_cast_b is not None and cand_cast_b.op == "cast" and len(cand_cast_b.inputs) == 1:
            cast_out = b
            x_name = a
            mask_name = cand_cast_b.inputs[0]
        else:
            continue

        # Require mask dtype to be boolean.
        mask_tt = fn.tensors.get(mask_name)
        if mask_tt is None or not is_bool_dtype(str(mask_tt.dtype)):
            continue

        # Require cast target to be a float dtype (common masking convention).
        cast_op = prod.get(cast_out)
        to = str((cast_op.attrs or {}).get("to") or "")
        if to not in {"f16", "bf16", "f32", "f64"}:
            continue

        # Determine output dtype (prefer mul output tensor dtype if available).
        out_tt = fn.tensors.get(op.output)
        x_tt = fn.tensors.get(x_name) if x_name else None
        out_dtype = str((out_tt.dtype if out_tt is not None else (x_tt.dtype if x_tt is not None else to)))
        if out_dtype not in {"f16", "bf16", "f32", "f64"}:
            continue

        zero = ensure_zero(out_dtype)
        op.op = "where"
        op.inputs = [mask_name, str(x_name), zero]
        op.attrs = {}

    if prefix_ops:
        fn.ops = list(prefix_ops) + list(fn.ops)


def _reduce_dims(op: Op) -> Optional[List[int]]:
    attrs = op.attrs or {}
    d = attrs.get("dims", None)
    if isinstance(d, list) and all(isinstance(x, int) for x in d):
        return [int(x) for x in d]
    a = attrs.get("axis", None)
    if isinstance(a, int):
        return [int(a)]
    if isinstance(a, list) and all(isinstance(x, int) for x in a):
        return [int(x) for x in a]
    return None


def _expand_softmax_ops(fn: IntentFunction) -> None:
    """
    Expand `softmax(x)` into a canonical lowered pattern:
      reduce_max(x) -> mx
      x - mx -> shifted
      exp(shifted) -> e
      reduce_sum(e) -> s
      e / s -> y

    This improves cross-frontend comparability when another frontend exposes
    auxiliary outputs like row_max/row_sum.
    """
    if not fn.ops:
        return

    taken: Set[str] = set(fn.tensors.keys()) | {op.output for op in fn.ops}

    def fresh(base: str) -> str:
        if base not in taken:
            taken.add(base)
            return base
        k = 0
        while f"{base}_{k}" in taken:
            k += 1
        name = f"{base}_{k}"
        taken.add(name)
        return name

    new_ops: List[Op] = []
    for op in fn.ops:
        if op.op != "softmax" or len(op.inputs) != 1:
            new_ops.append(op)
            continue
        x = op.inputs[0]
        y = op.output
        axis = int((op.attrs or {}).get("axis", -1))
        mx = fresh(f"{y}__mx")
        shifted = fresh(f"{y}__shift")
        e = fresh(f"{y}__exp")
        s = fresh(f"{y}__sum")
        new_ops.append(Op(op="reduce_max", inputs=[x], output=mx, attrs={"dims": [axis], "keepdims": False}))
        new_ops.append(Op(op="sub", inputs=[x, mx], output=shifted, attrs={}))
        new_ops.append(Op(op="exp", inputs=[shifted], output=e, attrs={}))
        new_ops.append(Op(op="reduce_sum", inputs=[e], output=s, attrs={"dims": [axis], "keepdims": False}))
        new_ops.append(Op(op="div", inputs=[e, s], output=y, attrs={}))
    fn.ops = new_ops


def _rewrite_reduce_max_ne_zero_to_reduce_any(fn: IntentFunction) -> None:
    """
    Normalize `any(x != 0)` patterns:

      reduce_max(x, dims=[d]) -> r
      ne(r, 0) -> out

    becomes:
      ne(x, 0) -> tmp
      reduce_any(tmp, dims=[d]) -> out

    This is a conservative rewrite used to reduce frontend drift for "any"-like kernels.
    """
    if not fn.ops:
        return

    # Use counts to ensure intermediates are not shared.
    uses: Dict[str, int] = {}
    for out in fn.outputs:
        uses[out] = uses.get(out, 0) + 1
    for op in fn.ops:
        for inp in op.inputs:
            uses[inp] = uses.get(inp, 0) + 1

    # Producer lookup (current ops) for constant detection.
    prod: Dict[str, Op] = {op.output: op for op in fn.ops}

    # Precompute rewrite sites: consumer index -> rewrite payload.
    skip: Set[int] = set()
    replace_at: Dict[int, Dict[str, object]] = {}
    for i, op0 in enumerate(fn.ops):
        if op0.op != "reduce_max" or len(op0.inputs) != 1:
            continue
        x = op0.inputs[0]
        dims0 = _reduce_dims(op0)
        r = op0.output
        if not dims0 or uses.get(r, 0) != 1:
            continue
        # Find the unique consumer of r.
        j = None
        for k in range(i + 1, len(fn.ops)):
            if r in fn.ops[k].inputs:
                j = k
                break
        if j is None:
            continue
        op1 = fn.ops[j]
        if op1.op != "ne" or len(op1.inputs) != 2 or r not in op1.inputs:
            continue
        other = op1.inputs[0] if op1.inputs[1] == r else op1.inputs[1]
        if not _is_zero_like(fn, prod, other):
            continue
        tmp = f"{op1.output}__nz"
        if tmp in fn.tensors:
            t = 0
            while f"{tmp}_{t}" in fn.tensors:
                t += 1
            tmp = f"{tmp}_{t}"
        skip.add(i)
        replace_at[int(j)] = {"x": x, "other": other, "out": op1.output, "dims": list(dims0), "tmp": tmp}

    if not replace_at:
        return

    new_ops: List[Op] = []
    for idx, op in enumerate(fn.ops):
        if idx in skip:
            continue
        payload = replace_at.get(int(idx))
        if payload is not None:
            x = str(payload["x"])
            other = str(payload["other"])
            out = str(payload["out"])
            tmp = str(payload["tmp"])
            dims = list(payload["dims"])  # type: ignore[arg-type]
            new_ops.append(Op(op="ne", inputs=[x, other], output=tmp, attrs={}))
            new_ops.append(Op(op="reduce_any", inputs=[tmp], output=out, attrs={"dims": dims}))
            continue
        new_ops.append(op)
    fn.ops = new_ops


def _dce_ops(ops: List[Op], *, outputs: Set[str]) -> List[Op]:
    if not ops:
        return []
    used: Set[str] = set(outputs)
    kept_rev: List[Op] = []
    for op in reversed(ops):
        if op.output in used:
            kept_rev.append(op)
            for inp in op.inputs:
                used.add(inp)
    return list(reversed(kept_rev))
