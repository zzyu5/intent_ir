"""
TileLang facts extraction (MVP).

The TileLang frontend consumes real TVMScript (a `tvm.tir.PrimFunc`) produced by
TileLang and extracts:
  - anchors (copy/gemm/reduce/etc)
  - structured indexing for global memory accesses, by decoding `tl.tileop.*`
    calls (notably `tl.tileop.copy(T.region(...), T.region(...))`).

This is intentionally MVP-level: best-effort affine extraction for indices, and
predicate clauses are attached only when TileLang emits an explicit mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from frontends.common.evidence import AccessSummary, IndexExpr, Predicate, sort_accesses


@dataclass
class TileLangFacts:
    schema_version: str
    anchors: Dict[str, Any]
    accesses: List[AccessSummary] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


def extract_facts(source_text: str, *, tvm_ir_json_path: str | None = None, tvm_ir_json: str | None = None) -> TileLangFacts:
    """
    Parse TileLang TVMScript into TileLangFacts.

    The input is expected to be the output of `PrimFunc.script(show_meta=False)`
    and should contain `@T.prim_func`.
    """

    import tilelang  # noqa: PLC0415
    import tvm  # type: ignore  # noqa: PLC0415
    from tvm import tir  # noqa: PLC0415

    # 1) Get PrimFunc (prefer deterministic TVM JSON snapshot; fall back to parsing TVMScript).
    prim_func = None
    raw_version = getattr(tilelang, "__version__", None)
    if tvm_ir_json:
        try:
            prim_func = tvm.ir.load_json(str(tvm_ir_json))
        except Exception:
            prim_func = None
    if prim_func is None and tvm_ir_json_path:
        try:
            p = Path(str(tvm_ir_json_path))
            if p.is_file():
                prim_func = tvm.ir.load_json(p.read_text(encoding="utf-8"))
        except Exception:
            prim_func = None
    if prim_func is None:
        # Best-effort fallback: TVMScript parsing requires the right decorator/macro env.
        # (We expect the adapter to provide tvm_ir_json_path in normal runs.)
        from tilelang import language as T  # noqa: PLC0415

        parsed = tvm.script.from_source(str(source_text), extra_vars={"T": T})
        if isinstance(parsed, tir.PrimFunc):
            prim_func = parsed
        elif hasattr(parsed, "functions"):
            funcs = list(getattr(parsed, "functions").values())
            for f in funcs:
                if isinstance(f, tir.PrimFunc):
                    prim_func = f
                    break
    if prim_func is None:
        raise ValueError("failed to obtain TileLang PrimFunc (missing tvm_ir_json_path?)")

    anchors: Dict[str, Any] = {
        "kernel_kind_hint": "unknown",
        "has_copy": False,
        "has_dot": False,
        "has_reduce": False,
        "has_atomic": False,
    }

    accesses: List[AccessSummary] = []

    def affine(expr: Any, *, max_depth: int = 64) -> tuple[Dict[str, int], int, bool]:
        if max_depth <= 0:
            return ({}, 0, False)
        if isinstance(expr, tir.IntImm):
            return ({}, int(expr.value), True)
        if isinstance(expr, tir.Var):
            return ({str(expr.name): 1}, 0, True)
        if isinstance(expr, tir.Cast):
            return affine(expr.value, max_depth=max_depth - 1)
        if isinstance(expr, tir.Add):
            ta, ca, oka = affine(expr.a, max_depth=max_depth - 1)
            tb, cb, okb = affine(expr.b, max_depth=max_depth - 1)
            if not (oka and okb):
                return ({}, 0, False)
            out = dict(ta)
            for k, v in tb.items():
                out[k] = out.get(k, 0) + int(v)
                if out[k] == 0:
                    out.pop(k, None)
            return (out, int(ca) + int(cb), True)
        if isinstance(expr, tir.Sub):
            ta, ca, oka = affine(expr.a, max_depth=max_depth - 1)
            tb, cb, okb = affine(expr.b, max_depth=max_depth - 1)
            if not (oka and okb):
                return ({}, 0, False)
            out = dict(ta)
            for k, v in tb.items():
                out[k] = out.get(k, 0) - int(v)
                if out[k] == 0:
                    out.pop(k, None)
            return (out, int(ca) - int(cb), True)
        if isinstance(expr, tir.Mul):
            # support const * expr (either side)
            if isinstance(expr.a, tir.IntImm):
                c = int(expr.a.value)
                t, k0, ok = affine(expr.b, max_depth=max_depth - 1)
                if not ok:
                    return ({}, 0, False)
                return ({k: v * c for k, v in t.items() if v * c != 0}, k0 * c, True)
            if isinstance(expr.b, tir.IntImm):
                c = int(expr.b.value)
                t, k0, ok = affine(expr.a, max_depth=max_depth - 1)
                if not ok:
                    return ({}, 0, False)
                return ({k: v * c for k, v in t.items() if v * c != 0}, k0 * c, True)
            return ({}, 0, False)
        return ({}, 0, False)

    def to_index_expr(expr: Any) -> IndexExpr:
        terms, const, ok = affine(expr)
        if not ok:
            return IndexExpr(terms={}, const=0)
        return IndexExpr(terms=terms, const=int(const))

    def parse_region(region_call: Any) -> tuple[str, str, int, List[IndexExpr], str]:
        if not isinstance(region_call, tir.Call) or getattr(region_call.op, "name", "") != "tl.tileop.region":
            raise ValueError("expected tl.tileop.region call")
        if not region_call.args:
            raise ValueError("region missing args")
        bl = region_call.args[0]
        if not isinstance(bl, tir.BufferLoad):
            raise ValueError("region arg0 must be BufferLoad")
        buf = bl.buffer
        scope = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
        idx = [to_index_expr(i) for i in list(bl.indices)]
        return (str(buf.name), str(buf.dtype), int(len(buf.shape)), idx, str(scope))

    def parse_copy(call: tir.Call) -> None:
        nonlocal accesses
        if len(call.args) < 2:
            return
        src_region = call.args[0]
        dst_region = call.args[1]
        try:
            src_name, src_dtype, src_rank, src_idx, src_scope = parse_region(src_region)
            dst_name, dst_dtype, dst_rank, dst_idx, dst_scope = parse_region(dst_region)
        except Exception:
            return

        # Only record global memory edges for MVP: global->* are loads, *->global are stores.
        if src_scope == "global":
            accesses.append(
                AccessSummary(
                    kind="load",
                    tensor=src_name,
                    dtype=src_dtype,
                    rank=src_rank,
                    index_exprs=src_idx,
                    predicate=None,
                    address_space=src_scope,
                    meta={"tileop": "copy", "dst_scope": dst_scope},
                )
            )
        if dst_scope == "global":
            accesses.append(
                AccessSummary(
                    kind="store",
                    tensor=dst_name,
                    dtype=dst_dtype,
                    rank=dst_rank,
                    index_exprs=dst_idx,
                    predicate=None,
                    address_space=dst_scope,
                    meta={"tileop": "copy", "src_scope": src_scope},
                )
            )

    def visit(node: Any) -> None:
        nonlocal anchors
        if not isinstance(node, tir.Call):
            return
        op_name = getattr(node.op, "name", "")
        if op_name.startswith("tl.tileop."):
            anchors["kernel_kind_hint"] = "tilelang_tileop"
        if op_name == "tl.tileop.copy":
            anchors["has_copy"] = True
            parse_copy(node)
        if "gemm" in op_name:
            anchors["has_dot"] = True
        if "reduce" in op_name:
            anchors["has_reduce"] = True
        if "atomic" in op_name:
            anchors["has_atomic"] = True

    tir.stmt_functor.post_order_visit(prim_func.body, visit)

    return TileLangFacts(
        schema_version="tilelang_tir_v0.1",
        anchors=dict(anchors),
        accesses=sort_accesses(accesses),
        raw={"source_kind": "tvm_script", "tilelang_version": raw_version},
    )


__all__ = ["TileLangFacts", "extract_facts"]
