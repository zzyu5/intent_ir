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
from typing import Dict, List, Optional, Set, Tuple

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

    # Pass 1b: iota+broadcast canonicalization.
    # Some frontends build 2D index grids via iota(1D)+broadcast_in_dim, while
    # others emit iota directly at the final rank. Rewrite the former into the
    # latter to reduce representational drift.
    _rewrite_broadcasted_iota(fn)

    # Pass 2: fold "safe" broadcast_in_dim into elementwise consumers.
    _fold_trailing_broadcasts(fn)

    # Pass 2b: normalize matmul transpose surface forms.
    # Different frontends/LLM outputs may represent transposed matmul operands as
    # either:
    #   - an explicit transpose op feeding matmul
    #   - matmul attrs transpose_{a,b}=True
    # For cross-frontend consistency, fold the former into the latter.
    _fold_transpose_into_matmul(fn)

    # Pass 2c: treat reshape that only adjusts trailing singleton dims as a no-op
    # for consistency purposes (e.g., [M,1] -> [M]).
    _rewrite_trailing_ones_reshape_as_identity(fn)

    # Pass 2d: prefer `div(reduce_sum(x), N)` over `mul(reduce_sum(x), inv_N)`
    # when inv_N is a scalar numeric const. This reduces drift between frontends
    # that constant-fold the reciprocal.
    _rewrite_reduce_sum_mul_const_to_div(fn)

    # Pass 2d2: normalize scalar "num elements" plumbing.
    # Some frontends constant-fold element counts (e.g., num_elements=256.0),
    # while others compute them from shape symbols (e.g., cast(group_size*HW)).
    # Rewrite the latter into a canonical `const` expression so subsequent hashes
    # don't depend on whether the frontend folded the product.
    _rewrite_cast_mul_const_symbols_to_const_expr(fn)

    # Pass 2d3: normalize variance surface forms.
    # GroupNorm can compute variance as either:
    #   E[(x-mean)^2]  (centered-square form)
    # or:
    #   E[x^2] - mean^2 (mean2 form)
    # Canonicalize the centered-square form into the mean2 form to reduce
    # cross-frontend drift (TileLang frequently emits mean2 form).
    _rewrite_variance_centered_square_to_mean2(fn)

    # Pass 2e: normalize identity wrappers and +/-inf max/min sentinels.
    # Some frontends express reductions via explicit sentinel init (e.g., max(-inf, x))
    # and add identity wrappers around inputs/outputs. These are semantics-preserving
    # normalizations that reduce cross-frontend drift.
    _rewrite_max_min_with_infinity_as_identity(fn)
    _fold_identity_ops(fn)

    # Build a producer map after the broadcast folding.
    prod: Dict[str, Op] = {op.output: op for op in fn.ops}

    # Pass 2e: drop no-op casts (e.g., f32 -> f32).
    _rewrite_noop_cast_as_identity(fn, prod)

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

    # Pass 4b: affine masking canonicalization.
    # Normalize:
    #   where(mask, x, 0) + neg_inf * (1 - cast(mask))  ->  where(mask, x, neg_inf)
    # This reduces drift between frontends that implement masking as a fused
    # arithmetic expression vs a direct where.
    _canonicalize_affine_mask_to_where(fn)
    # The affine rewrite can strand now-dead arithmetic/cast chains. Drop them
    # before the next folding pass so broadcasts become foldable (e.g., a mask
    # broadcast that used to feed cast + mul/sub but now only feeds where).
    fn.ops = _dce_ops(fn.ops, outputs=set(fn.outputs))
    _fold_trailing_broadcasts(fn)

    # Pass 5: semantic normalizations (safe, local, conservative).
    # For cross-frontend comparison we prefer a lowered, explicit softmax form.
    _expand_softmax_ops(fn)
    _rewrite_reduce_max_ne_zero_to_reduce_any(fn)

    # Pass 5: dead code elimination (drop unused broadcast/const/cast/identity chains).
    fn.ops = _dce_ops(fn.ops, outputs=set(fn.outputs))

    # Pass 6: infer axis_roles for interface symbols.
    # Different frontends/LLM runs often omit or partially fill axis_roles.
    # For cross-frontend comparisons (E4) and paper metrics, we infer a stable
    # axis_roles mapping from the *interface* tensor shapes + reduced/contracted
    # axes, without relying on LLM-provided metadata.
    fn.axis_roles = _infer_axis_roles_from_interface(fn)
    return fn


def _value_uses(fn: IntentFunction) -> Dict[str, int]:
    uses: Dict[str, int] = {}
    for out in fn.outputs:
        uses[out] = uses.get(out, 0) + 1
    for op in fn.ops:
        for inp in op.inputs:
            uses[inp] = uses.get(inp, 0) + 1
    return uses


def _const_string_value(fn: IntentFunction, prod: Dict[str, Op], name: str) -> str | None:
    op = prod.get(name)
    if op is None or op.op != "const":
        return None
    v = (op.attrs or {}).get("value")
    if not isinstance(v, str) or not v.strip():
        return None
    if v.startswith("placeholder:"):
        return None
    return v.strip()


def _rewrite_cast_mul_const_symbols_to_const_expr(fn: IntentFunction) -> None:
    """
    Canonicalize scalar element-count expressions.

    Pattern:
      mul(const("A"), const("B")) -> tmp
      cast(tmp) {to="f32"}        -> out

    becomes:
      const() {value="A*B"}       -> out

    This preserves semantics as long as A and B are shape bindings (which our
    interpreter and backend already support via `eval()` in const resolution).
    """
    if not fn.ops:
        return
    prod: Dict[str, Op] = {op.output: op for op in fn.ops}
    uses = _value_uses(fn)

    for op in fn.ops:
        if op.op != "cast" or len(op.inputs) != 1:
            continue
        to = (op.attrs or {}).get("to") or (op.attrs or {}).get("dtype")
        if str(to) not in {"f32", "float32"}:
            continue
        inp = op.inputs[0]
        mul = prod.get(inp)
        if mul is None or mul.op != "mul" or len(mul.inputs) != 2:
            continue
        if uses.get(inp, 0) != 1:
            continue
        a, b = mul.inputs[0], mul.inputs[1]
        av = _const_string_value(fn, prod, a)
        bv = _const_string_value(fn, prod, b)
        if av is None or bv is None:
            continue
        # Avoid rewriting arbitrary expressions; keep it simple and stable.
        expr = f"({av})*({bv})"
        op.op = "const"
        op.inputs = []
        op.attrs = {"value": expr}


def _rewrite_variance_centered_square_to_mean2(fn: IntentFunction) -> None:
    """
    Rewrite variance computed via centered-square into mean2 form.

    centered-square:
      mean = div(reduce_sum(x), denom)
      xc   = sub(x, broadcast(mean))
      var  = div(reduce_sum(mul(xc,xc)), denom)

    mean2:
      mean  = div(reduce_sum(x), denom)
      mean2 = div(reduce_sum(mul(x,x)), denom)
      var   = sub(mean2, mul(mean,mean))
    """
    if not fn.ops:
        return

    prod: Dict[str, Op] = {op.output: op for op in fn.ops}
    uses = _value_uses(fn)
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

    # Find div(sum_sq, denom) where sum_sq comes from reduce_sum(mul(sub(x, bcast(mean)), sub(x, bcast(mean)))).
    i = 0
    while i < len(fn.ops):
        op_var = fn.ops[i]
        i += 1
        if op_var.op != "div" or len(op_var.inputs) != 2:
            continue
        sum_sq, denom = op_var.inputs
        if uses.get(sum_sq, 0) != 1:
            continue

        op_sum_sq = prod.get(sum_sq)
        if op_sum_sq is None or op_sum_sq.op != "reduce_sum" or len(op_sum_sq.inputs) != 1:
            continue
        x_sq = op_sum_sq.inputs[0]
        if uses.get(x_sq, 0) != 1:
            continue
        op_x_sq = prod.get(x_sq)
        if op_x_sq is None or op_x_sq.op != "mul" or len(op_x_sq.inputs) != 2:
            continue
        if op_x_sq.inputs[0] != op_x_sq.inputs[1]:
            continue
        x_centered = op_x_sq.inputs[0]

        op_xc = prod.get(x_centered)
        if op_xc is None or op_xc.op != "sub" or len(op_xc.inputs) != 2:
            continue
        a0, b0 = op_xc.inputs[0], op_xc.inputs[1]

        mean = None
        x0 = None

        # Case 1: explicit broadcast_in_dim(mean) (pre-folding).
        for maybe_mean_bcast, other in ((a0, b0), (b0, a0)):
            op_bcast = prod.get(maybe_mean_bcast)
            if op_bcast is None or op_bcast.op != "broadcast_in_dim" or len(op_bcast.inputs) != 1:
                continue
            mean = op_bcast.inputs[0]
            x0 = other
            break

        # Case 2: broadcast folded into elementwise (sub(x, mean)).
        if mean is None:
            def is_mean_value(v: str) -> bool:
                op_m = prod.get(v)
                if op_m is None:
                    return False
                if op_m.op == "div" and len(op_m.inputs) == 2:
                    num = op_m.inputs[0]
                    op_num = prod.get(num)
                    return op_num is not None and op_num.op == "reduce_sum" and len(op_num.inputs) == 1
                if op_m.op == "reduce_sum" and len(op_m.inputs) == 1:
                    # Some frontends encode mean via reduce_sum(scale=1/N) directly.
                    return True
                return False

            if is_mean_value(a0) and not is_mean_value(b0):
                mean, x0 = a0, b0
            elif is_mean_value(b0) and not is_mean_value(a0):
                mean, x0 = b0, a0
            else:
                continue

        # mean should be div(sum, denom2) (denom may differ; we only need mean value).
        op_mean = prod.get(str(mean)) if mean is not None else None
        if op_mean is None or op_mean.op != "div" or len(op_mean.inputs) != 2:
            # If mean is reduce_sum(scale=...), accept it as-is (already canonical enough).
            if op_mean is None or op_mean.op != "reduce_sum":
                continue

        # Rewrite x_sq to raw square: mul(x,x).
        if x0 is None:
            continue
        op_x_sq.inputs = [x0, x0]

        # sum_sq now becomes sum2 = reduce_sum(mul(x,x)) (keep dims attrs).
        # Insert mean2 and mean_sq before op_var (current index i-1).
        mean2 = fresh(f"{op_var.output}__mean2")
        mean_sq = fresh(f"{op_var.output}__mean_sq")

        fn.ops.insert(i - 1, Op(op="div", inputs=[sum_sq, denom], output=mean2, attrs={}))
        fn.ops.insert(i, Op(op="mul", inputs=[str(mean), str(mean)], output=mean_sq, attrs={}))
        i += 2  # account for inserted ops

        # Replace var = div(sum_sq, denom) with var = sub(mean2, mean_sq).
        op_var.op = "sub"
        op_var.inputs = [mean2, mean_sq]
        op_var.attrs = {}

        # Producers map changed; rebuild for subsequent rewrites.
        prod = {op.output: op for op in fn.ops}
        uses = _value_uses(fn)



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


def _interface_tensors(fn: IntentFunction) -> Tuple[List[str], List[str]]:
    produced = {op.output for op in fn.ops if op.output}
    used: List[str] = []
    for op in fn.ops:
        used.extend(list(op.inputs))
    external_inputs = [n for n in used if (n in fn.tensors and n not in produced)]
    outputs = [n for n in (fn.outputs or []) if n in fn.tensors]
    # De-dup while preserving order.
    seen: set[str] = set()
    ext: List[str] = []
    for n in external_inputs:
        if n in seen:
            continue
        seen.add(n)
        ext.append(n)
    out: List[str] = []
    for n in outputs:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return ext, out


def _sym_dims(t: TensorType | None) -> List[str]:
    if t is None:
        return []
    out: List[str] = []
    for d in list(t.shape):
        if getattr(d, "kind", None) != "sym":
            continue
        v = str(getattr(d, "value", "")).strip()
        if v:
            out.append(v)
    return out


def _sym_set(t: TensorType | None) -> Set[str]:
    return set(_sym_dims(t))


def _sym_at(t: TensorType | None, axis: int) -> Optional[str]:
    if t is None:
        return None
    shape = list(t.shape)
    if not shape:
        return None
    ax = int(axis)
    if ax < 0:
        ax = len(shape) + ax
    if ax < 0 or ax >= len(shape):
        return None
    d = shape[ax]
    if getattr(d, "kind", None) != "sym":
        return None
    v = str(getattr(d, "value", "")).strip()
    return v or None


def _tokenize_sym(sym: str) -> List[str]:
    s = str(sym).strip().upper().replace("-", "_")
    return [t for t in s.split("_") if t]


def _looks_spatial(sym: str) -> bool:
    toks = _tokenize_sym(sym)
    # Treat "CTX"/(H,W) family and common image/sequence tokens as spatial.
    for t in toks:
        if t in {"H", "W", "IH", "IW", "OH", "OW", "HW", "CTX", "SEQ", "POS"}:
            return True
    return False


def _looks_channel(sym: str) -> bool:
    toks = _tokenize_sym(sym)
    # Head dims / channels / groups / embedding dims.
    for t in toks:
        if t in {"C", "CH", "CHANNEL", "HEAD", "NUMHEAD", "DIM", "EMBED", "HIDDEN", "FEATURE", "GROUP", "GROUPS", "G"}:
            return True
    s = str(sym).upper()
    return ("NUMHEAD" in s) or ("HEAD" in s) or ("CHANNEL" in s) or ("EMBED" in s) or ("HIDDEN" in s)


def _infer_axis_roles_from_interface(fn: IntentFunction) -> Dict[str, str]:
    ext_inputs, outputs = _interface_tensors(fn)
    if not ext_inputs and not outputs:
        return dict(fn.axis_roles or {})

    # Interface symbol set (inputs + outputs only).
    iface_syms: Set[str] = set()
    output_syms: Set[str] = set()
    iface_pos: Dict[str, Set[Tuple[int, int]]] = {}

    def add_tensor(name: str, *, is_output: bool) -> None:
        t = fn.tensors.get(name)
        if t is None:
            return
        r = int(len(t.shape))
        for i, d in enumerate(list(t.shape)):
            if getattr(d, "kind", None) != "sym":
                continue
            s = str(getattr(d, "value", "")).strip()
            if not s:
                continue
            iface_syms.add(s)
            iface_pos.setdefault(s, set()).add((r, int(i)))
            if is_output:
                output_syms.add(s)

    for n in ext_inputs:
        add_tensor(n, is_output=False)
    for n in outputs:
        add_tensor(n, is_output=True)

    if not iface_syms:
        return dict(fn.axis_roles or {})

    # Reduced/contracted axes: infer from ops, but only mark as `reduction` when
    # the axis does not appear in final outputs (avoid labeling spatial dims like
    # HW in groupnorm as reduction just because it is used to compute stats).
    reduced_any: Set[str] = set()
    reduced_eliminated: Set[str] = set()

    for op in list(fn.ops):
        if op.op == "matmul" and len(op.inputs) >= 2 and op.output:
            a = fn.tensors.get(op.inputs[0])
            b = fn.tensors.get(op.inputs[1])
            o = fn.tensors.get(op.output)
            if a is not None and b is not None:
                common = _sym_set(a) & _sym_set(b)
                out_syms_op = _sym_set(o)
                if out_syms_op:
                    contracted = {s for s in common if s not in out_syms_op}
                else:
                    # If the matmul output tensor type is missing (common in LLM output),
                    # identify the contraction axis by position: batch-like dims appear at
                    # the same index in both inputs, while K-like dims appear at different
                    # indices (e.g., [B,M,K] x [B,K,N]).
                    a_syms = _sym_dims(a)
                    b_syms = _sym_dims(b)
                    a_pos: Dict[str, Set[int]] = {}
                    b_pos: Dict[str, Set[int]] = {}
                    for i, s in enumerate(a_syms):
                        a_pos.setdefault(s, set()).add(int(i))
                    for i, s in enumerate(b_syms):
                        b_pos.setdefault(s, set()).add(int(i))
                    contracted = set()
                    for s in common:
                        pa = a_pos.get(s, set())
                        pb = b_pos.get(s, set())
                        if any(int(i) != int(j) for i in pa for j in pb):
                            contracted.add(s)
                for s in contracted:
                    if s in iface_syms:
                        reduced_any.add(s)
            continue

        if (op.op.startswith("reduce_") or op.op in {"reduce_sum", "reduce_max", "reduce_min", "reduce_any", "reduce_all"}) and op.inputs and op.output:
            a = fn.tensors.get(op.inputs[0])
            o = fn.tensors.get(op.output)
            dims = (op.attrs or {}).get("dims")
            if dims is None:
                dims = (op.attrs or {}).get("axis")
                if isinstance(dims, int):
                    dims = [dims]
            if isinstance(dims, list) and all(isinstance(x, int) for x in dims) and a is not None:
                for ax in dims:
                    s = _sym_at(a, int(ax))
                    if s and s in iface_syms:
                        reduced_any.add(s)
            else:
                cand = _sym_set(a) - _sym_set(o)
                for s in cand:
                    if s in iface_syms:
                        reduced_any.add(s)
            continue

    for s in list(reduced_any):
        if s in output_syms:
            continue
        reduced_eliminated.add(s)

    # Batch/channel cues from interface shapes.
    batch_syms: Set[str] = set()
    channel_syms: Set[str] = set()

    for s, poses in iface_pos.items():
        # Only use rank>=3 positional heuristics for batch/channel.
        if any(r >= 3 and i == 0 for (r, i) in poses):
            batch_syms.add(s)
        if any(r >= 3 and i == 1 for (r, i) in poses):
            channel_syms.add(s)

    # 1D weight/bias tensors imply "channel" (feature) axes.
    for n in ext_inputs:
        t = fn.tensors.get(n)
        if t is None or len(t.shape) != 1:
            continue
        s0 = _sym_at(t, 0)
        if s0 and s0 in iface_syms and not _looks_spatial(s0):
            channel_syms.add(s0)

    # Reduced-but-preserved axes: likely feature axes (softmax/layernorm).
    for s in (reduced_any & output_syms):
        if _looks_spatial(s):
            continue
        channel_syms.add(s)

    # Name-based channel hints (e.g., HEAD_DIM, *_numhead).
    for s in list(iface_syms):
        if _looks_channel(s) and not _looks_spatial(s):
            channel_syms.add(s)

    # Final role assignment (interface symbols only).
    out: Dict[str, str] = {}
    for s in sorted(iface_syms):
        if s in reduced_eliminated:
            out[s] = "reduction"
        elif s in batch_syms:
            out[s] = "batch"
        elif s in channel_syms:
            out[s] = "channel"
        else:
            out[s] = "spatial"
    return out


def _fold_trailing_broadcasts(fn: IntentFunction) -> None:
    if not fn.ops:
        return

    prod: Dict[str, Op] = {op.output: op for op in fn.ops}
    produced: Set[str] = set(prod.keys())

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
            # Many LLM/frontends omit tensor type entries for intermediates.
            # For broadcast_in_dim, the input rank is typically the length of
            # broadcast_dims. Use that as a best-effort fallback.
            bd = op.attrs.get("broadcast_dims")
            if isinstance(bd, list) and all(isinstance(x, int) for x in bd):
                in_rank = int(len(bd))
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
        #     (also accept scalar-like shape=[1], a common LLM surface form)
        #  2) "numpy trailing" alignment: dims == [out-r_in, ..., out-1]
        #  3) prefix-aligned broadcast for non-interface intermediates:
        #     dims == [0,1,...,r_in-1]
        foldable = False
        if _is_scalar_like_tensor(fn, src):
            foldable = True
        else:
            want = list(range(out_rank - in_rank, out_rank))
            if len(bcast_dims) == in_rank and [int(x) for x in bcast_dims] == want:
                foldable = True
            else:
                want_prefix = list(range(0, int(in_rank)))
                # Only allow prefix folding for intermediates (not external inputs),
                # to avoid collapsing interface masks like row_mask/col_mask that
                # are semantically distinct 1D vectors.
                if src in produced and len(bcast_dims) == in_rank and [int(x) for x in bcast_dims] == want_prefix:
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


def _rewrite_broadcasted_iota(fn: IntentFunction) -> None:
    if not fn.ops:
        return
    prod: Dict[str, Op] = {op.output: op for op in fn.ops}
    for op in fn.ops:
        if op.op != "broadcast_in_dim" or len(op.inputs) != 1:
            continue
        src = op.inputs[0]
        src_op = prod.get(src)
        if src_op is None or src_op.op != "iota" or src_op.inputs:
            continue
        iota_shape = (src_op.attrs or {}).get("shape")
        iota_axis = (src_op.attrs or {}).get("axis")
        if not isinstance(iota_shape, list) or len(iota_shape) != 1:
            continue
        if not isinstance(iota_axis, int) or int(iota_axis) != 0:
            continue
        out_shape = (op.attrs or {}).get("out_shape")
        bcast_dims = (op.attrs or {}).get("broadcast_dims")
        if not isinstance(out_shape, list) or not out_shape:
            continue
        if not isinstance(bcast_dims, list) or len(bcast_dims) != 1 or not isinstance(bcast_dims[0], int):
            continue
        axis = int(bcast_dims[0])
        dtype = (src_op.attrs or {}).get("dtype") or "i32"
        op.op = "iota"
        op.inputs = []
        op.attrs = {"shape": list(out_shape), "axis": axis, "dtype": str(dtype)}


def _rewrite_trailing_ones_reshape_as_identity(fn: IntentFunction) -> None:
    if not fn.ops:
        return

    def tokens(shape: List[Dim]) -> List[tuple[str, str | int]]:
        return [(str(d.kind), d.value) for d in list(shape)]

    def strip_trailing_ones(ts: List[tuple[str, str | int]]) -> List[tuple[str, str | int]]:
        out = list(ts)
        while out:
            k, v = out[-1]
            if k != "const":
                break
            try:
                if int(v) != 1:
                    break
            except Exception:
                break
            out.pop()
        return out

    for op in fn.ops:
        if op.op != "reshape" or len(op.inputs) != 1:
            continue
        src = op.inputs[0]
        in_tt = fn.tensors.get(src)
        out_tt = fn.tensors.get(op.output)
        if in_tt is None or out_tt is None:
            continue
        if strip_trailing_ones(tokens(in_tt.shape)) == strip_trailing_ones(tokens(out_tt.shape)):
            op.op = "identity"
            op.attrs = {}


def _rewrite_reduce_sum_mul_const_to_div(fn: IntentFunction) -> None:
    if not fn.ops:
        return

    prod: Dict[str, Op] = {op.output: op for op in fn.ops}
    taken: Set[str] = set(fn.tensors.keys()) | set(prod.keys())
    prefix_ops: List[Op] = []

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

    def const_scalar_numeric(name: str) -> Optional[float]:
        c = prod.get(name)
        if c is None or c.op != "const" or c.inputs:
            return None
        tt = fn.tensors.get(name)
        if tt is not None and not _is_scalar_like_tensor(fn, name):
            return None
        v = (c.attrs or {}).get("value")
        if isinstance(v, bool):
            return None
        if isinstance(v, str):
            return None
        try:
            return float(v)
        except Exception:
            return None

    for op in fn.ops:
        if op.op != "mul" or len(op.inputs) != 2:
            continue
        a, b = op.inputs[0], op.inputs[1]
        pa = prod.get(a)
        pb = prod.get(b)

        sum_name = None
        inv_name = None
        inv_val = None
        if pa is not None and pa.op == "reduce_sum":
            inv_val = const_scalar_numeric(b)
            if inv_val is not None:
                sum_name, inv_name = a, b
        if sum_name is None and pb is not None and pb.op == "reduce_sum":
            inv_val = const_scalar_numeric(a)
            if inv_val is not None:
                sum_name, inv_name = b, a
        if sum_name is None or inv_name is None or inv_val is None:
            continue
        if inv_val == 0.0:
            continue

        denom_val = 1.0 / float(inv_val)
        denom_name = fresh(f"{op.output}__denom")
        prefix_ops.append(Op(op="const", inputs=[], output=denom_name, attrs={"value": float(denom_val), "dtype": "f32"}))
        fn.tensors[denom_name] = TensorType(dtype="f32", shape=[], layout=TensorLayout(kind="row_major", params={}))

        op.op = "div"
        op.inputs = [str(sum_name), denom_name]
        op.attrs = {}

    if prefix_ops:
        fn.ops = list(prefix_ops) + list(fn.ops)


def _rewrite_noop_cast_as_identity(fn: IntentFunction, prod: Dict[str, Op]) -> None:
    for op in fn.ops:
        if op.op != "cast" or len(op.inputs) != 1:
            continue
        src = op.inputs[0]
        src_tt = fn.tensors.get(src)
        if src_tt is None:
            continue
        to = (op.attrs or {}).get("to")
        if to is None:
            continue
        if str(src_tt.dtype) != str(to):
            continue
        out_tt = fn.tensors.get(op.output)
        if out_tt is not None and str(out_tt.dtype) != str(to):
            continue
        op.op = "identity"
        op.attrs = {}


def _fold_transpose_into_matmul(fn: IntentFunction) -> None:
    if not fn.ops:
        return

    # Count uses (including function outputs).
    uses: Dict[str, int] = {}
    for out in fn.outputs:
        uses[out] = uses.get(out, 0) + 1
    for op in fn.ops:
        for inp in op.inputs:
            uses[inp] = uses.get(inp, 0) + 1

    prod: Dict[str, Op] = {op.output: op for op in fn.ops}
    outputs_set = set(fn.outputs or [])

    def is_last_two_swap_perm(perm: object) -> bool:
        if not isinstance(perm, list) or len(perm) < 2:
            return False
        try:
            p = [int(x) for x in perm]
        except Exception:
            return False
        r = len(p)
        if sorted(p) != list(range(r)):
            return False
        want = list(range(r))
        want[-2], want[-1] = want[-1], want[-2]
        return p == want

    for op in fn.ops:
        if op.op != "matmul" or len(op.inputs) != 2:
            continue
        attrs = dict(op.attrs or {})
        ta = bool(attrs.get("transpose_a", False))
        tb = bool(attrs.get("transpose_b", False))

        a, b = op.inputs[0], op.inputs[1]

        pa = prod.get(a)
        if (
            pa is not None
            and pa.op == "transpose"
            and len(pa.inputs) == 1
            and uses.get(a, 0) == 1
            and a not in outputs_set
            and is_last_two_swap_perm((pa.attrs or {}).get("perm"))
        ):
            op.inputs[0] = pa.inputs[0]
            ta = not ta

        pb = prod.get(b)
        if (
            pb is not None
            and pb.op == "transpose"
            and len(pb.inputs) == 1
            and uses.get(b, 0) == 1
            and b not in outputs_set
            and is_last_two_swap_perm((pb.attrs or {}).get("perm"))
        ):
            op.inputs[1] = pb.inputs[0]
            tb = not tb

        # Canonicalize attrs: only keep transpose flags when True.
        if ta:
            attrs["transpose_a"] = True
        else:
            attrs.pop("transpose_a", None)
        if tb:
            attrs["transpose_b"] = True
        else:
            attrs.pop("transpose_b", None)
        op.attrs = attrs


def _is_scalar_const_zero(fn: IntentFunction, prod: Dict[str, Op], name: str) -> bool:
    op = prod.get(name)
    if op is None or op.op != "const" or op.inputs:
        return False
    # Many frontends/LLM outputs omit tensor type entries for intermediates.
    # For the purpose of consistency canonicalization, treat typeless consts as
    # scalar-like (rank-0).
    tt = fn.tensors.get(name)
    if tt is not None and not _is_scalar_like_tensor(fn, name):
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


def _is_const_inf(fn: IntentFunction, prod: Dict[str, Op], name: str, *, sign: int) -> bool:
    """
    Best-effort detection of +/-inf const tensors.

    Note: unlike `_is_scalar_const_zero`, we do NOT require a scalar rank here.
    Many lowings materialize +/-inf sentinels as full tensors (e.g., `mx = const(-inf)` with shape [M,N]).
    """
    op = prod.get(name)
    if op is None or op.op != "const" or op.inputs:
        return False
    v = (op.attrs or {}).get("value")
    try:
        fv = float(v)  # type: ignore[arg-type]
    except Exception:
        s = str(v).strip().lower()
        if sign < 0 and s in {"-inf", "-infinity"}:
            return True
        if sign > 0 and s in {"+inf", "inf", "infinity", "+infinity"}:
            return True
        return False
    if sign < 0:
        return fv == float("-inf")
    return fv == float("inf")


def _rewrite_max_min_with_infinity_as_identity(fn: IntentFunction) -> None:
    """
    Normalize elementwise +/-inf sentinels:

      max(x, -inf) -> identity(x)
      min(x, +inf) -> identity(x)

    This is a conservative rewrite used to reduce frontend drift in reduction
    initializations (common in CUDA/TIR lowering).
    """
    if not fn.ops:
        return
    prod: Dict[str, Op] = {op.output: op for op in fn.ops}
    for op in fn.ops:
        if op.op not in {"max", "min"} or len(op.inputs) != 2:
            continue
        a, b = op.inputs[0], op.inputs[1]
        if op.op == "max":
            a_inf = _is_const_inf(fn, prod, a, sign=-1)
            b_inf = _is_const_inf(fn, prod, b, sign=-1)
        else:
            a_inf = _is_const_inf(fn, prod, a, sign=+1)
            b_inf = _is_const_inf(fn, prod, b, sign=+1)
        if a_inf and not b_inf:
            op.op = "identity"
            op.inputs = [b]
            op.attrs = {}
        elif b_inf and not a_inf:
            op.op = "identity"
            op.inputs = [a]
            op.attrs = {}


def _fold_identity_ops(fn: IntentFunction) -> None:
    """
    Fold `identity(x)` away by rewiring its uses.

    Important: avoid turning function outputs into pure aliases of external inputs
    (e.g., `outputs=["inp"]`), which can confuse E4 structural comparisons. For
    output identities, we only eliminate them by renaming the producer of `x`
    to the output name when the producer exists and `x` is not shared.
    """
    if not fn.ops:
        return

    outputs_set = set(fn.outputs or [])

    # Use counts to ensure we only rewrite unshared values.
    uses: Dict[str, int] = {}
    for out in outputs_set:
        uses[out] = uses.get(out, 0) + 1
    for op in fn.ops:
        for inp in op.inputs:
            uses[inp] = uses.get(inp, 0) + 1

    prod: Dict[str, Op] = {op.output: op for op in fn.ops if op.output}

    # Step 1: eliminate `out = identity(x)` when:
    #   - out is a function output
    #   - x has a producer op
    #   - x is not shared (only used by this identity)
    # This keeps output names stable while removing the wrapper.
    drop_idxs: set[int] = set()
    for idx, op in enumerate(list(fn.ops)):
        if op.op != "identity" or len(op.inputs) != 1:
            continue
        out = str(op.output)
        if out not in outputs_set:
            continue
        x = str(op.inputs[0])
        p = prod.get(x)
        if p is None:
            continue
        if uses.get(x, 0) != 1:
            continue
        # Rename producer output to the function output name.
        p.output = out
        # Move tensor type if available.
        if out not in fn.tensors and x in fn.tensors:
            fn.tensors[out] = fn.tensors.pop(x)
        drop_idxs.add(int(idx))

    if drop_idxs:
        fn.ops = [op for i, op in enumerate(fn.ops) if int(i) not in drop_idxs]
        prod = {op.output: op for op in fn.ops if op.output}

    # Step 2: fold non-output identities by rewiring their uses.
    repl: Dict[str, str] = {}
    for op in fn.ops:
        if op.op == "identity" and len(op.inputs) == 1 and op.output and str(op.output) not in outputs_set:
            repl[str(op.output)] = str(op.inputs[0])

    if not repl:
        return

    def resolve(name: str) -> str:
        seen: set[str] = set()
        cur = str(name)
        while cur in repl and cur not in seen:
            seen.add(cur)
            cur = str(repl[cur])
        return cur

    # Rewrite op inputs.
    for op in fn.ops:
        if not op.inputs:
            continue
        op.inputs = [resolve(x) for x in op.inputs]

    # Drop identity ops; DCE later will clean up now-dead chains.
    fn.ops = [op for op in fn.ops if not (op.op == "identity" and len(op.inputs) == 1 and op.output in repl)]


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


def _canonicalize_affine_mask_to_where(fn: IntentFunction) -> None:
    if not fn.ops:
        return

    prod: Dict[str, Op] = {op.output: op for op in fn.ops}

    def const_scalar_numeric(name: str) -> Optional[float]:
        c = prod.get(name)
        if c is None or c.op != "const" or c.inputs:
            return None
        tt = fn.tensors.get(name)
        if tt is not None and not _is_scalar_like_tensor(fn, name):
            return None
        v = (c.attrs or {}).get("value")
        if isinstance(v, bool):
            return None
        if isinstance(v, str):
            return None
        try:
            return float(v)
        except Exception:
            return None

    def is_const_zero(name: str) -> bool:
        v = const_scalar_numeric(name)
        return v is not None and float(v) == 0.0

    def is_const_one(name: str) -> bool:
        v = const_scalar_numeric(name)
        return v is not None and float(v) == 1.0

    def is_const_neg(name: str) -> bool:
        v = const_scalar_numeric(name)
        return v is not None and float(v) < 0.0

    for op in fn.ops:
        if op.op != "add" or len(op.inputs) != 2:
            continue
        a, b = op.inputs[0], op.inputs[1]

        # Try both input orderings.
        for masked, term in [(a, b), (b, a)]:
            w = prod.get(masked)
            m = prod.get(term)
            if w is None or w.op != "where" or len(w.inputs) != 3:
                continue
            if m is None or m.op != "mul" or len(m.inputs) != 2:
                continue
            cond, x, y = w.inputs[0], w.inputs[1], w.inputs[2]
            if not is_const_zero(y):
                continue

            neg_name = None
            inv_name = None
            if is_const_neg(m.inputs[0]):
                neg_name, inv_name = m.inputs[0], m.inputs[1]
            elif is_const_neg(m.inputs[1]):
                neg_name, inv_name = m.inputs[1], m.inputs[0]
            if neg_name is None or inv_name is None:
                continue

            inv_op = prod.get(inv_name)
            if inv_op is None or inv_op.op != "sub" or len(inv_op.inputs) != 2:
                continue
            if not is_const_one(inv_op.inputs[0]):
                continue
            mask_f = inv_op.inputs[1]

            cast_op = prod.get(mask_f)
            if cast_op is None or cast_op.op != "cast" or len(cast_op.inputs) != 1:
                continue
            if cast_op.inputs[0] != cond:
                continue

            # Rewrite in-place.
            op.op = "where"
            op.inputs = [cond, x, neg_name]
            op.attrs = {}
            break


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
