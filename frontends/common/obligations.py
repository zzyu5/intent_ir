"""
Obligations (PR#5): executable, cross-frontend obligation evaluation.

Obligations operate on the stable CertificateV2 payload (anchors + CanonicalEvidence),
so they should not depend on frontend IR details (TTIR line numbers/op names).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from .evidence import CanonicalEvidence, IndexExpr

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


_CMP_RE = re.compile(r"^(?P<lhs>.+?)\s*(?P<op><=|>=|==|!=|<|>)\s*(?P<rhs>.+?)\s*$")
_ARG_INDEX_RE = re.compile(r"^arg\d+$")


def _parse_cmp(clause: str) -> Optional[Tuple[str, str, str]]:
    m = _CMP_RE.match(str(clause).strip())
    if not m:
        return None
    return (m.group("lhs").strip(), m.group("op").strip(), m.group("rhs").strip())


def _parse_affine_expr(expr: str) -> Optional[IndexExpr]:
    """
    Parse a simple affine expression formatted by `frontends.triton.certificate._format_index_expr`.
    Supported tokens: var, int, +/- separators, and `c*var`.
    """
    s = str(expr).strip()
    if not s or s == "<unresolved>":
        return None
    # Normalize: turn subtraction into "+ -term"
    s = s.replace("-", "+-")
    parts = [p.strip() for p in s.split("+") if p.strip()]
    terms: Dict[str, int] = {}
    const = 0
    for p in parts:
        if "*" in p:
            a, b = p.split("*", 1)
            try:
                c = int(a.strip())
            except Exception:
                return None
            v = b.strip()
            if not v:
                return None
            terms[v] = terms.get(v, 0) + c
            continue
        if p.lstrip("-").isdigit():
            const += int(p)
            continue
        # bare var, maybe with unary '-'
        if p.startswith("-") and len(p) > 1:
            v = p[1:].strip()
            if not v:
                return None
            terms[v] = terms.get(v, 0) - 1
        else:
            v = p.strip()
            if not v:
                return None
            terms[v] = terms.get(v, 0) + 1
    terms = {k: v for k, v in terms.items() if v != 0}
    return IndexExpr(terms=terms, const=int(const))


def _prove_nonneg_affine(ix: IndexExpr, *, nonneg_vars: Set[str]) -> Optional[bool]:
    """
    Conservative proof attempt:
    - If all coefficients are >=0 and all vars are known non-negative and const >=0 -> PROVE >=0.
    - Otherwise -> UNKNOWN (None).
    """
    if ix.const < 0:
        return None
    for v, c in ix.terms.items():
        if c < 0:
            return None
        if v not in nonneg_vars:
            return None
    return True


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
    ce = _extract_canonical_evidence(cert_v2)
    if ce and isinstance(ce.meta, dict):
        syms = ce.meta.get("symbols")
        if isinstance(syms, dict):
            ranges = syms.get("ranges")
            if isinstance(ranges, dict):
                allowed.update(str(k) for k in ranges.keys())
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
        # Known non-negative variables: pid*, r*.
        nonneg_vars: Set[str] = set({"pid0", "pid1", "pid2"})
        ce_meta = ce.meta if ce is not None else {}
        if isinstance(ce_meta, dict):
            syms = ce_meta.get("symbols")
            if isinstance(syms, dict):
                ranges = syms.get("ranges")
                if isinstance(ranges, dict):
                    nonneg_vars.update(str(k) for k in ranges.keys())
        # Also treat all axis symbols as non-negative.
        nonneg_vars.update(v for v in allowed_syms if v.isidentifier())
        # Treat positional arg indices as non-negative in MVP (usually dims/strides).
        nonneg_vars.update({f"arg{i}" for i in range(0, 64)})

        first_unknown: Optional[str] = None
        for c in clauses:
            parsed = _parse_cmp(c)
            if parsed is None:
                first_unknown = f"unparseable clause: {c!r}"
                break
            lhs, op, rhs = parsed
            if op in {"<", "<="}:
                ix = _parse_affine_expr(lhs)
                if ix is None:
                    first_unknown = f"cannot parse lhs affine: {lhs!r}"
                    break
                proved = _prove_nonneg_affine(ix, nonneg_vars=nonneg_vars)
                if proved is not True:
                    first_unknown = f"cannot prove nonneg for lhs: {lhs!r}"
                    break
            elif op in {">", ">="}:
                # Treat as rhs <= lhs; try to prove rhs nonneg similarly.
                ix = _parse_affine_expr(rhs)
                if ix is None:
                    first_unknown = f"cannot parse rhs affine: {rhs!r}"
                    break
                proved = _prove_nonneg_affine(ix, nonneg_vars=nonneg_vars)
                if proved is not True:
                    first_unknown = f"cannot prove nonneg for rhs: {rhs!r}"
                    break
            else:
                # == / != are not used in MVP proof.
                continue

        if first_unknown is None:
            results.append(
                ObligationResult(
                    id=O3_MASK_IMPLIES_INBOUNDS,
                    status="PASS",
                    witness={"num_clauses": len(clauses)},
                    reason="proved non-negativity for bounded indices (MVP); full SMT witness in PR#7",
                )
            )
        else:
            results.append(
                ObligationResult(
                    id=O3_MASK_IMPLIES_INBOUNDS,
                    status="UNKNOWN",
                    reason=first_unknown,
                    witness={"example_clause": str(clauses[0]) if clauses else ""},
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
