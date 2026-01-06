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
    # schedule-level hints (not part of semantic_facts golden lock):
    # - symbol_ranges: domains for canonical loop/thread symbols (r0/r1/..., pid0/1/2)
    # - tile_hints: "tile-ish" constants for guided tuning (e.g., loop extents, block sizes)
    symbol_ranges: Dict[str, Dict[str, int]] = field(default_factory=dict)
    tile_hints: List[int] = field(default_factory=list)
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
        "has_barrier": False,
        "has_async": False,
    }

    accesses: List[AccessSummary] = []
    loop_ranges: Dict[str, Dict[str, int]] = {}

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

    def _add_ix(a: IndexExpr, b: IndexExpr) -> IndexExpr:
        terms = dict(a.terms)
        for k, v in (b.terms or {}).items():
            terms[k] = terms.get(k, 0) + int(v)
            if terms[k] == 0:
                terms.pop(k, None)
        return IndexExpr(terms=terms, const=int(a.const) + int(b.const))

    def _scale_ix(ix: IndexExpr, c: int) -> IndexExpr:
        if int(c) == 0:
            return IndexExpr(terms={}, const=0)
        return IndexExpr(terms={k: int(v) * int(c) for k, v in (ix.terms or {}).items() if int(v) * int(c) != 0}, const=int(ix.const) * int(c))

    def _dim_token(d: Any) -> int | str:
        if isinstance(d, tir.IntImm):
            return int(d.value)
        if isinstance(d, tir.Var):
            return str(d.name)
        try:
            return str(d)
        except Exception:
            return "<unresolved>"

    def _render_affine(ix: IndexExpr) -> str:
        # Keep a stable, SMT-friendly string form: "8*pid0 + r0 - 1"
        parts: List[str] = []
        for k in sorted((ix.terms or {}).keys()):
            c = int(ix.terms[k])
            if c == 0:
                continue
            if c == 1:
                parts.append(str(k))
            elif c == -1:
                parts.append(f"-{k}")
            else:
                parts.append(f"{c}*{k}")
        if int(ix.const) != 0 or not parts:
            parts.append(str(int(ix.const)))
        # Normalize "a + -b" into "a - b" for readability.
        out = " + ".join(parts)
        out = out.replace("+ -", "- ")
        return out

    def _predicate_clauses(expr: Any, *, max_nodes: int = 64) -> List[str]:
        if max_nodes <= 0:
            return []
        if isinstance(expr, tir.IntImm):
            # Constant predicate: ignore (does not constrain inbounds).
            return []
        if isinstance(expr, tir.And):
            return _predicate_clauses(expr.a, max_nodes=max_nodes - 1) + _predicate_clauses(expr.b, max_nodes=max_nodes - 1)
        if isinstance(expr, (tir.LT, tir.LE, tir.GT, tir.GE, tir.EQ, tir.NE)):
            lhs = to_index_expr(expr.a)
            rhs = to_index_expr(expr.b)
            op = {
                tir.LT: "<",
                tir.LE: "<=",
                tir.GT: ">",
                tir.GE: ">=",
                tir.EQ: "==",
                tir.NE: "!=",
            }.get(type(expr))
            if not op:
                return []
            return [f"{_render_affine(lhs)} {op} {_render_affine(rhs)}"]
        # Some predicates are expressed as casts to bool.
        if isinstance(expr, tir.Cast):
            return _predicate_clauses(expr.value, max_nodes=max_nodes - 1)
        return []

    def parse_region(region_call: Any) -> tuple[str, str, int, List[IndexExpr], str, Dict[str, Any]]:
        if not isinstance(region_call, tir.Call) or getattr(region_call.op, "name", "") != "tl.tileop.region":
            raise ValueError("expected tl.tileop.region call")
        if not region_call.args:
            raise ValueError("region missing args")
        bl = region_call.args[0]
        if not isinstance(bl, tir.BufferLoad):
            raise ValueError("region arg0 must be BufferLoad")
        buf = bl.buffer
        scope = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
        nd_idx = [to_index_expr(i) for i in list(bl.indices)]
        meta: Dict[str, Any] = {
            "nd_index_exprs": [ix.to_json_dict() for ix in nd_idx],
            "shape": [_dim_token(d) for d in list(getattr(buf, "shape", []) or [])],
        }
        strides = list(getattr(buf, "strides", []) or [])
        if strides:
            meta["strides"] = [_dim_token(s) for s in strides]
        # Region signature (best-effort): region(buffer_load, kind, dim0, dim1, ...)
        # where `kind` differentiates src/dst regions and dims denote the region extents.
        try:
            if len(region_call.args) >= 2:
                kind = region_call.args[1]
                if isinstance(kind, tir.IntImm):
                    meta["region_kind"] = int(kind.value)
                ext = region_call.args[2:]
                if ext:
                    meta["region_extents"] = [_dim_token(x) for x in list(ext)]
        except Exception:
            pass

        # Linearize multi-dimensional indices into a single "flat element offset" when strides are constant.
        flat = IndexExpr(terms={}, const=0)
        unresolved = False
        if not strides or len(strides) != len(nd_idx):
            unresolved = True
        else:
            for ix, st in zip(nd_idx, strides):
                if not isinstance(st, tir.IntImm):
                    unresolved = True
                    break
                flat = _add_ix(flat, _scale_ix(ix, int(st.value)))
        if unresolved:
            meta["unresolved"] = True
            # Fallback: keep the first dimension as a best-effort witness.
            flat = nd_idx[0] if nd_idx else IndexExpr(terms={}, const=0)
        return (str(buf.name), str(buf.dtype), int(len(buf.shape)), [flat], str(scope), meta)

    def parse_copy(call: tir.Call) -> None:
        nonlocal accesses
        if len(call.args) < 2:
            return
        src_region = call.args[0]
        dst_region = call.args[1]
        try:
            src_name, src_dtype, src_rank, src_idx, src_scope, src_meta = parse_region(src_region)
            dst_name, dst_dtype, dst_rank, dst_idx, dst_scope, dst_meta = parse_region(dst_region)
        except Exception:
            return

        pred: Predicate | None = None
        try:
            # TileLang lowering commonly uses: copy(src_region, dst_region, other, mask, ...)
            # Keep the predicate clauses in a simple conjunction form when possible.
            if len(call.args) >= 4:
                clauses = _predicate_clauses(call.args[3])
                if clauses:
                    pred = Predicate(clauses=[str(c) for c in clauses])
        except Exception:
            pred = None

        # Only record global memory edges for MVP: global->* are loads, *->global are stores.
        if src_scope == "global":
            accesses.append(
                AccessSummary(
                    kind="load",
                    tensor=src_name,
                    dtype=src_dtype,
                    rank=src_rank,
                    index_exprs=src_idx,
                    predicate=pred,
                    address_space=src_scope,
                    meta={"tileop": "copy", "dst_scope": dst_scope, **(src_meta or {})},
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
                    predicate=pred,
                    address_space=dst_scope,
                    meta={"tileop": "copy", "src_scope": src_scope, **(dst_meta or {})},
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
        if "barrier" in op_name or "syncthreads" in op_name or "thread_allreduce" in op_name or "allreduce" in op_name:
            anchors["has_barrier"] = True
        if "async" in op_name:
            anchors["has_async"] = True
        # Some syncs are expressed as extern calls; conservatively scan string args.
        if op_name in {"tir.call_extern", "tir.call_packed"}:
            for a in list(getattr(node, "args", []) or []):
                try:
                    s = str(a)
                except Exception:
                    continue
                if "barrier" in s or "syncthreads" in s:
                    anchors["has_barrier"] = True
                if "async" in s:
                    anchors["has_async"] = True

    tir.stmt_functor.post_order_visit(prim_func.body, visit)

    # Collect constant loop/thread ranges (used for bounded reasoning / guided tuning).
    def visit_stmt(node: Any) -> None:
        if isinstance(node, tir.For):
            try:
                v = str(getattr(node.loop_var, "name", ""))
            except Exception:
                v = ""
            if not v:
                return
            mn = getattr(node, "min", None)
            ex = getattr(node, "extent", None)
            if isinstance(mn, tir.IntImm) and isinstance(ex, tir.IntImm):
                s = int(mn.value)
                e = s + int(ex.value)
                if e > s:
                    loop_ranges[v] = {"start": s, "end": e}

    tir.stmt_functor.post_order_visit(prim_func.body, visit_stmt)

    # Canonicalize indexing symbols to keep evidence stable across versions and avoid
    # treating loop/thread indices as data-dependent symbols downstream.
    reserved: set[str] = {"pid0", "pid1", "pid2"}
    try:
        # Preserve param-like symbols (buffers + scalar args).
        for p in list(getattr(prim_func, "params", []) or []):
            if isinstance(p, tir.Var):
                reserved.add(str(p.name))
        # Preserve shape-like symbols from buffer shapes (e.g., M/N/K).
        for _buf in getattr(prim_func, "buffer_map", {}).values():
            for d in list(getattr(_buf, "shape", []) or []):
                if isinstance(d, tir.Var):
                    reserved.add(str(d.name))
    except Exception:
        pass

    def collect_terms(ix: IndexExpr) -> set[str]:
        out: set[str] = set()
        for k in (ix.terms or {}).keys():
            if isinstance(k, str) and k:
                out.add(str(k))
        return out

    used_syms: set[str] = set()
    for a in accesses:
        for ix in a.index_exprs:
            used_syms |= collect_terms(ix)

    sym_map: Dict[str, str] = {}
    # Common program-id style symbols in TileLang TIR.
    pid_alias = {"bx": "pid0", "by": "pid1", "bz": "pid2"}
    for k, v in pid_alias.items():
        if k in used_syms:
            sym_map[k] = v

    def _fresh_r_name(i0: int) -> str:
        i = int(i0)
        while True:
            cand = f"r{i}"
            if cand not in reserved and cand not in sym_map.values():
                return cand
            i += 1

    r_i = 0
    for v in sorted(used_syms):
        if v in sym_map:
            continue
        if v in reserved:
            continue
        sym_map[v] = _fresh_r_name(r_i)
        r_i += 1

    def remap_ix(ix: IndexExpr) -> IndexExpr:
        terms = {}
        for k, c in (ix.terms or {}).items():
            kk = sym_map.get(str(k), str(k))
            terms[kk] = terms.get(kk, 0) + int(c)
            if terms[kk] == 0:
                terms.pop(kk, None)
        return IndexExpr(terms=terms, const=int(ix.const))

    remapped: List[AccessSummary] = []
    for a in accesses:
        remapped.append(
            AccessSummary(
                kind=a.kind,
                tensor=a.tensor,
                dtype=a.dtype,
                rank=int(a.rank),
                index_exprs=[remap_ix(ix) for ix in (a.index_exprs or [])],
                predicate=a.predicate,
                address_space=a.address_space,
                meta=dict(a.meta or {}),
            )
        )
    accesses = sort_accesses(remapped)

    symbol_ranges: Dict[str, Dict[str, int]] = {}
    for orig, rr in loop_ranges.items():
        if orig not in sym_map:
            continue
        canon = sym_map[orig]
        if canon not in symbol_ranges:
            symbol_ranges[canon] = {"start": int(rr.get("start", 0)), "end": int(rr.get("end", 0))}

    # Tile-ish integer hints (for Guided schedule search).
    tile_hints: set[int] = set()
    for rr in symbol_ranges.values():
        try:
            v = int(rr.get("end", 0)) - int(rr.get("start", 0))
        except Exception:
            continue
        if 2 <= v <= 2048:
            tile_hints.add(int(v))
    try:
        for _buf in getattr(prim_func, "buffer_map", {}).values():
            for d in list(getattr(_buf, "shape", []) or []):
                if isinstance(d, tir.IntImm):
                    v = int(d.value)
                    if 2 <= v <= 2048:
                        tile_hints.add(v)
    except Exception:
        pass
    filtered = [v for v in tile_hints if 2 <= int(v) <= 2048]
    preferred = [v for v in filtered if (int(v) & (int(v) - 1) == 0)]
    tile_hints_list = sorted(set(preferred if preferred else filtered))

    return TileLangFacts(
        schema_version="tilelang_tir_v0.1",
        anchors=dict(anchors),
        accesses=accesses,
        symbol_ranges=symbol_ranges,
        tile_hints=tile_hints_list,
        raw={"source_kind": "tvm_script", "tilelang_version": raw_version},
    )


__all__ = ["TileLangFacts", "extract_facts"]
