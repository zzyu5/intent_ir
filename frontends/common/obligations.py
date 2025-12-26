"""
Obligations (PR#5): executable, cross-frontend obligation evaluation.

Obligations operate on the stable CertificateV2 payload (anchors + CanonicalEvidence),
so they should not depend on frontend IR details (TTIR line numbers/op names).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, TYPE_CHECKING

from .evidence import CanonicalEvidence, IndexExpr
from .smt_o3 import check_mask_implies_inbounds

if TYPE_CHECKING:  # pragma: no cover
    from pipeline.interfaces import KernelDescriptor
    from .certificate_v2 import SemanticCertificateV2


ObligationStatus = Literal["PASS", "FAIL", "UNKNOWN"]


O1_HAS_SEMANTIC_ANCHOR = "O1_has_semantic_anchor"
O2_AFFINE_OR_STRUCTURED_INDEXING = "O2_affine_or_structured_indexing"
O3_MASK_IMPLIES_INBOUNDS = "O3_mask_implies_inbounds"
O4_SHAPE_LAYOUT_MATCH = "O4_shape_layout_match"
O5_NO_DATA_DEPENDENT_ADDRESS = "O5_no_data_dependent_address"
O6_STRUCTURED_SYNC = "O6_structured_sync"
O7_NO_ATOMICS_OR_CONTROLLED_ATOMICS = "O7_no_atomics_or_controlled_atomics"


@dataclass(frozen=True)
class ObligationResult:
    id: str
    status: ObligationStatus
    reason: str = ""
    witness: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        out = {"id": str(self.id), "status": str(self.status), "witness": dict(self.witness)}
        if self.reason:
            out["reason"] = str(self.reason)
        return out


_ARG_INDEX_RE = re.compile(r"^arg\d+$")


def _extract_canonical_evidence(cert_v2: "SemanticCertificateV2") -> Optional[CanonicalEvidence]:
    ce = (cert_v2.semantic_facts or {}).get("canonical_evidence")
    return ce if isinstance(ce, CanonicalEvidence) else None


def _extract_anchors(cert_v2: "SemanticCertificateV2") -> Dict[str, Any]:
    anchors = (cert_v2.semantic_facts or {}).get("anchors")
    return dict(anchors) if isinstance(anchors, dict) else {}


def _allowed_symbols(desc: "KernelDescriptor", cert_v2: "SemanticCertificateV2") -> Set[str]:
    allowed: Set[str] = set()
    # Standard non-negative program ids
    allowed.update({"pid0", "pid1", "pid2"})
    # Kernel args (includes scalar dims/strides/etc)
    arg_names = desc.io_spec.get("arg_names")
    if isinstance(arg_names, list):
        allowed.update(str(x) for x in arg_names if isinstance(x, str) and x)
    # Known axes from pipeline spec (useful for assumptions / readability)
    launch = desc.launch or {}
    for key in ("vary_axes",):
        xs = launch.get(key)
        if isinstance(xs, list):
            allowed.update(str(x) for x in xs if isinstance(x, str) and x)
    # Known concrete shape names
    canonical_shapes = launch.get("canonical_shapes")
    if isinstance(canonical_shapes, dict):
        allowed.update(str(k) for k in canonical_shapes.keys())
    # Range symbols from evidence meta (r0/r1/...)
    # Prefer schedule_hints domains (symbol_ranges), but keep back-compat with older meta layouts.
    sh = getattr(cert_v2, "schedule_hints", None)
    if isinstance(sh, dict):
        sr = sh.get("symbol_ranges")
        if isinstance(sr, dict):
            allowed.update(str(k) for k in sr.keys())
    ce = _extract_canonical_evidence(cert_v2)
    if ce and isinstance(ce.meta, dict):
        syms = ce.meta.get("symbols")
        if isinstance(syms, dict):
            ranges = syms.get("ranges")
            if isinstance(ranges, dict):
                allowed.update(str(k) for k in ranges.keys())
            if isinstance(ranges, list):
                allowed.update(str(k) for k in ranges if isinstance(k, str) and k)
    return allowed


def evaluate_obligations(desc: "KernelDescriptor", cert_v2: "SemanticCertificateV2") -> List[ObligationResult]:
    """
    Evaluate obligations from stable CertificateV2 evidence.
    """
    anchors = _extract_anchors(cert_v2)
    ce = _extract_canonical_evidence(cert_v2)
    accesses = list(ce.accesses) if ce is not None else []
    allowed_syms = _allowed_symbols(desc, cert_v2)

    results: List[ObligationResult] = []

    # O1: anchor present
    has_anchor = bool(anchors.get("has_dot") or anchors.get("has_reduce"))
    results.append(
        ObligationResult(
            id=O1_HAS_SEMANTIC_ANCHOR,
            status="PASS" if has_anchor else "FAIL",
            witness={"anchors": dict(anchors)},
            reason="" if has_anchor else "no semantic anchor (dot/reduce) found",
        )
    )

    # O7: no atomics
    has_atomic = bool(anchors.get("has_atomic"))
    results.append(
        ObligationResult(
            id=O7_NO_ATOMICS_OR_CONTROLLED_ATOMICS,
            status="FAIL" if has_atomic else "PASS",
            witness={"has_atomic": has_atomic},
            reason="contains atomic ops" if has_atomic else "",
        )
    )

    # O2: affine/structured indexing (per-access)
    if not accesses:
        results.append(ObligationResult(id=O2_AFFINE_OR_STRUCTURED_INDEXING, status="UNKNOWN", reason="no accesses extracted"))
    else:
        unresolved = [a for a in accesses if isinstance(a.meta, dict) and a.meta.get("unresolved")]
        # Treat as UNKNOWN only if unresolved extraction removed all meaningful terms.
        unknown_tensors: List[str] = []
        for a in unresolved:
            has_terms = any(bool(ix.terms) for ix in a.index_exprs)
            if not has_terms:
                unknown_tensors.append(str(a.tensor))
        if unknown_tensors:
            results.append(
                ObligationResult(
                    id=O2_AFFINE_OR_STRUCTURED_INDEXING,
                    status="UNKNOWN",
                    reason="some access indices are unresolved/non-affine",
                    witness={"unresolved_tensors": sorted(set(unknown_tensors))[:8]},
                )
            )
        else:
            results.append(
                ObligationResult(
                    id=O2_AFFINE_OR_STRUCTURED_INDEXING,
                    status="PASS",
                    witness={"num_accesses": len(accesses), "unresolved_accesses": len(unresolved)},
                )
            )

    # O4: shape/layout match (MVP: access tensor names are known kernel args)
    unknown_tensors = [a.tensor for a in accesses if (a.tensor not in allowed_syms and not _ARG_INDEX_RE.match(str(a.tensor)))]
    if not accesses:
        results.append(ObligationResult(id=O4_SHAPE_LAYOUT_MATCH, status="UNKNOWN", reason="no accesses extracted"))
    elif unknown_tensors:
        results.append(
            ObligationResult(
                id=O4_SHAPE_LAYOUT_MATCH,
                status="UNKNOWN",
                reason="some access tensors are not in KernelDescriptor.io_spec.arg_names",
                witness={"unknown_tensors": sorted(set(unknown_tensors))[:8]},
            )
        )
    else:
        results.append(ObligationResult(id=O4_SHAPE_LAYOUT_MATCH, status="PASS", witness={"arg_cover": True}))

    # O5: no data-dependent address (MVP: index terms stay within stable symbol set)
    bad_terms: List[str] = []
    for a in accesses:
        for ix in a.index_exprs:
            for v in (ix.terms or {}).keys():
                if v not in allowed_syms and not _ARG_INDEX_RE.match(str(v)):
                    bad_terms.append(str(v))
    if not accesses:
        results.append(ObligationResult(id=O5_NO_DATA_DEPENDENT_ADDRESS, status="UNKNOWN", reason="no accesses extracted"))
    elif bad_terms:
        results.append(
            ObligationResult(
                id=O5_NO_DATA_DEPENDENT_ADDRESS,
                status="UNKNOWN",
                reason="access index uses unknown symbol (cannot rule out data-dependent addressing yet)",
                witness={"unknown_terms": sorted(set(bad_terms))[:8]},
            )
        )
    else:
        results.append(ObligationResult(id=O5_NO_DATA_DEPENDENT_ADDRESS, status="PASS", witness={"terms_subset_ok": True}))

    # O6: structured sync (placeholder for future: barriers/async copies)
    results.append(ObligationResult(id=O6_STRUCTURED_SYNC, status="UNKNOWN", reason="not implemented"))

    # O3: mask implies inbounds (MVP: check predicate clauses are parseable and LHS is provably non-negative)
    clauses: List[str] = []
    for a in accesses:
        if a.predicate and a.predicate.clauses:
            clauses.extend(list(a.predicate.clauses))
    if not clauses:
        results.append(ObligationResult(id=O3_MASK_IMPLIES_INBOUNDS, status="UNKNOWN", reason="no predicate clauses extracted"))
    else:
        ranges = {}
        # Prefer schedule_hints for symbol domains (tile sizes / make_range endpoints are schedule-level).
        try:
            sh = getattr(cert_v2, "schedule_hints", None)
            if isinstance(sh, dict) and isinstance(sh.get("symbol_ranges"), dict):
                ranges = dict(sh.get("symbol_ranges") or {})
        except Exception:
            pass
        # Back-compat: older certificates stored domains under canonical_evidence.meta.symbols.ranges.
        if not ranges and ce is not None and isinstance(ce.meta, dict):
            syms = ce.meta.get("symbols")
            if isinstance(syms, dict) and isinstance(syms.get("ranges"), dict):
                ranges = dict(syms.get("ranges") or {})
        # Use KernelDescriptor canonical shapes as shape hints for DIM symbols.
        shape_hints = {}
        if isinstance(desc.launch, dict) and isinstance(desc.launch.get("canonical_shapes"), dict):
            for k, v in desc.launch.get("canonical_shapes").items():
                if isinstance(k, str) and isinstance(v, (int, float)):
                    shape_hints[str(k)] = int(v)

        o3 = check_mask_implies_inbounds(accesses, symbol_ranges=ranges, shape_hints=shape_hints)
        results.append(
            ObligationResult(
                id=O3_MASK_IMPLIES_INBOUNDS,
                status=o3.status,
                reason=(
                    "mask ⇒ inbounds proved"
                    if o3.status == "PASS"
                    else ("counterexample found" if o3.status == "FAIL" else "insufficient evidence/proof")
                ),
                witness=o3.to_json_dict(),
            )
        )

    # Deterministic order for reporting.
    order = [
        O1_HAS_SEMANTIC_ANCHOR,
        O2_AFFINE_OR_STRUCTURED_INDEXING,
        O3_MASK_IMPLIES_INBOUNDS,
        O4_SHAPE_LAYOUT_MATCH,
        O5_NO_DATA_DEPENDENT_ADDRESS,
        O6_STRUCTURED_SYNC,
        O7_NO_ATOMICS_OR_CONTROLLED_ATOMICS,
    ]
    rank = {oid: i for i, oid in enumerate(order)}
    return sorted(results, key=lambda r: (rank.get(r.id, 999), r.id))


__all__ = [
    "ObligationStatus",
    "ObligationResult",
    "O1_HAS_SEMANTIC_ANCHOR",
    "O2_AFFINE_OR_STRUCTURED_INDEXING",
    "O3_MASK_IMPLIES_INBOUNDS",
    "O4_SHAPE_LAYOUT_MATCH",
    "O5_NO_DATA_DEPENDENT_ADDRESS",
    "O6_STRUCTURED_SYNC",
    "O7_NO_ATOMICS_OR_CONTROLLED_ATOMICS",
    "evaluate_obligations",
]
