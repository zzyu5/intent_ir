"""
TileLang constraints extraction (MVP).

For PR#9 we only derive a minimal FrontendConstraints:
- needs_mask: True if any access carries predicate clauses.

Strengthening (P2/P3): also attach *computable* access witnesses (stride/penalty)
derived from CanonicalEvidence-style access summaries so downstream components
(contract/casegen/tuning) can consume richer constraints without importing heavy
TileLang/TVM machinery.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from pipeline.interfaces import FrontendConstraints, KernelDescriptor

from .facts import TileLangFacts


def _dim_token(expr: Any) -> int | str:
    try:
        # TileLang/TVM types are imported lazily to keep this module import-light.
        from tvm import tir  # type: ignore  # noqa: PLC0415

        if isinstance(expr, tir.IntImm):
            return int(expr.value)
        if isinstance(expr, tir.Var):
            return str(expr.name)
    except Exception:
        pass
    try:
        return str(expr)
    except Exception:
        return "<unresolved>"


def _extract_buffer_layout(desc: KernelDescriptor) -> Dict[str, Any]:
    """
    Best-effort extraction of dynamic/static dims + buffer stride witness.

    This is intended for contract/casegen/tuning, and does not affect the stable
    CertificateV2 semantic_facts golden locks.
    """
    tvm_path = desc.meta.get("tvm_ir_json_path")
    if not isinstance(tvm_path, str) or not tvm_path:
        return {}
    try:
        from pathlib import Path  # noqa: PLC0415

        p = Path(tvm_path)
        if not p.is_file():
            return {}
        import tvm  # type: ignore  # noqa: PLC0415
        from tvm import tir  # type: ignore  # noqa: PLC0415

        prim = tvm.ir.load_json(p.read_text(encoding="utf-8"))
        if not isinstance(prim, tir.PrimFunc):
            return {}
    except Exception:
        return {}

    buffers: Dict[str, Dict[str, Any]] = {}
    dynamic_dims: set[str] = set()
    static_ints: set[int] = set()
    for buf in prim.buffer_map.values():
        try:
            scope = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
        except Exception:
            scope = "global"
        if scope != "global":
            continue
        shape_toks = [_dim_token(d) for d in list(getattr(buf, "shape", []) or [])]
        strides_raw = list(getattr(buf, "strides", []) or [])
        stride_toks = [_dim_token(s) for s in strides_raw] if strides_raw else []
        for t in shape_toks:
            if isinstance(t, str) and t and not t.isdigit():
                dynamic_dims.add(t)
            if isinstance(t, int):
                static_ints.add(int(t))
        for t in stride_toks:
            if isinstance(t, int):
                static_ints.add(int(t))
        buffers[str(buf.name)] = {
            "dtype": str(getattr(buf, "dtype", "unknown")),
            "rank": int(len(shape_toks)),
            "shape": shape_toks,
            "strides": stride_toks,
        }

    return {
        "buffers": buffers,
        "dynamic_dims": sorted(dynamic_dims),
        "static_ints": sorted(v for v in static_ints if 0 <= int(v) <= 1_000_000),
    }


def extract_constraints(desc: KernelDescriptor, facts: TileLangFacts) -> FrontendConstraints:
    needs_mask = any(a.predicate and a.predicate.clauses for a in (facts.accesses or []))
    suggested = ["non_divisible_edge"] if needs_mask else []
    meta: Dict[str, Any] = {
        "tilelang_schema": facts.schema_version,
        "symbol_ranges": dict(getattr(facts, "symbol_ranges", {}) or {}),
        "tile_hints": list(getattr(facts, "tile_hints", []) or []),
    }
    meta.update(_extract_buffer_layout(desc))
    # Summarize predicate clauses for casegen/repair loops.
    clauses: List[str] = []
    for a in facts.accesses or []:
        if a.predicate and a.predicate.clauses:
            clauses.extend([str(c) for c in a.predicate.clauses])
    if clauses:
        meta["predicate_clauses"] = sorted(set(clauses))

    # Access witness: stride + penalties derived from CanonicalEvidence-style accesses.
    # This is intentionally best-effort and should never raise.
    try:
        from frontends.common.access_witness import build_stride_summary  # noqa: PLC0415

        shape_bindings: Dict[str, int] = {}
        try:
            cs = (desc.launch or {}).get("canonical_shapes")
            if isinstance(cs, dict):
                for k, v in cs.items():
                    if isinstance(k, str) and isinstance(v, (int, float)):
                        shape_bindings[str(k)] = int(v)
        except Exception:
            shape_bindings = {}

        evidence_like = {
            "canonical_evidence": {"accesses": [a.to_json_dict() for a in (facts.accesses or [])]},
            "schedule_hints": {"symbol_ranges": dict(getattr(facts, "symbol_ranges", {}) or {})},
        }
        summary = build_stride_summary(evidence_like, shape_bindings=shape_bindings)
        meta["access_witness"] = summary.to_json_dict()
    except Exception:
        pass

    return FrontendConstraints(needs_mask=bool(needs_mask), suggested_edge_cases=suggested, meta=meta)


__all__ = ["extract_constraints", "FrontendConstraints", "asdict"]
