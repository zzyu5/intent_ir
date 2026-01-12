"""
CUDA constraints extraction (MVP).

For consistency with other frontends, we emit a dependency-light FrontendConstraints:
- needs_mask: any predicated global access
- suggested_edge_cases: coarse hints for case generation
- meta: symbol ranges, tile hints, predicate clauses, and access witness summary
"""

from __future__ import annotations

from typing import Any, Dict, List

from pipeline.interfaces import FrontendConstraints, KernelDescriptor

from .facts import CudaFacts


def extract_constraints(desc: KernelDescriptor, facts: CudaFacts) -> FrontendConstraints:
    needs_mask = any(a.predicate and a.predicate.clauses for a in (facts.accesses or []))
    suggested = ["non_divisible_edge"] if needs_mask else []

    meta: Dict[str, Any] = {
        "symbol_ranges": dict(facts.symbol_ranges or {}),
        "tile_hints": list(facts.tile_hints or []),
        "predicate_clauses": list(facts.predicate_clauses or []),
    }

    # Access witness: stride + penalties derived from CanonicalEvidence-style accesses.
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
            "schedule_hints": {"symbol_ranges": dict(facts.symbol_ranges or {})},
        }
        summary = build_stride_summary(evidence_like, shape_bindings=shape_bindings)
        meta["access_witness"] = summary.to_json_dict()
    except Exception:
        pass

    return FrontendConstraints(needs_mask=bool(needs_mask), suggested_edge_cases=suggested, meta=meta)


__all__ = ["extract_constraints"]

