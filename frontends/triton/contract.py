"""
Task4: Recoverability Contract evaluation based on TTIR facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .facts import TTIRConstraints, TTIRFacts
from pipeline.interfaces import KernelDescriptor
from frontends.common.certificate_v2 import SemanticCertificateV2
from frontends.common.obligations import (
    ObligationResult,
    O1_HAS_SEMANTIC_ANCHOR,
    O2_AFFINE_OR_STRUCTURED_INDEXING,
    O3_MASK_IMPLIES_INBOUNDS,
    O4_SHAPE_LAYOUT_MATCH,
    O5_NO_DATA_DEPENDENT_ADDRESS,
    O7_NO_ATOMICS_OR_CONTROLLED_ATOMICS,
 )


@dataclass
class ContractReport:
    level: Literal["FULL", "PARTIAL", "OUT_OF_SCOPE"]
    kernel_kind_hint: str | None
    reasons: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)


def evaluate_contract(facts: TTIRFacts, constraints: TTIRConstraints | None = None) -> ContractReport:
    reasons: list[str] = []
    assumptions: list[str] = []
    signals: Dict[str, Any] = {
        "op_counts": facts.op_counts,
        "num_loads": len(facts.load_sites),
        "num_stores": len(facts.store_sites),
        "num_masks": len(facts.mask_sites),
        "has_dot": facts.has_dot,
        "has_reduce": facts.has_reduce,
        "has_atomic": facts.has_atomic,
    }

    if facts.has_atomic:
        reasons.append("contains atomic operations")
        return ContractReport(level="OUT_OF_SCOPE", kernel_kind_hint=None, reasons=reasons, assumptions=assumptions, signals=signals)
    if not (facts.has_dot or facts.has_reduce):
        reasons.append("no dot or reduce anchors found")
        return ContractReport(level="OUT_OF_SCOPE", kernel_kind_hint=None, reasons=reasons, assumptions=assumptions, signals=signals)

    needs_mask = constraints.needs_mask if constraints else any(site.has_mask for site in facts.load_sites + facts.store_sites)
    level: Literal["FULL", "PARTIAL"] = "FULL"
    if facts.has_dot:
        kernel_kind = "matmul"
        if len(facts.store_sites) < 1 or len(facts.load_sites) < 2:
            reasons.append("insufficient load/store sites for matmul pattern")
            level = "PARTIAL"
        if not needs_mask:
            reasons.append("mask not detected; edge handling uncertain")
            level = "PARTIAL"
    elif facts.has_reduce:
        kernel_kind = "reduce"
        if len(facts.store_sites) < 1 or len(facts.load_sites) < 1:
            reasons.append("insufficient load/store sites for reduce pattern")
            level = "PARTIAL"
        if not needs_mask:
            reasons.append("mask not detected; edge handling uncertain")
            level = "PARTIAL"
    else:
        kernel_kind = None
        level = "PARTIAL"

    return ContractReport(level=level, kernel_kind_hint=kernel_kind, reasons=reasons, assumptions=assumptions, signals=signals)


def evaluate_contract_v2(
    desc: KernelDescriptor,
    cert_v2: SemanticCertificateV2,
    obligations: List[ObligationResult],
    *,
    constraints: TTIRConstraints | None = None,
) -> ContractReport:
    """
    PR#5: Contract evaluation driven by obligations + stable CertificateV2 evidence.
    """
    reasons: list[str] = []
    assumptions: list[str] = []
    by_id = {o.id: o for o in obligations}

    anchors = (cert_v2.semantic_facts or {}).get("anchors") if isinstance(cert_v2.semantic_facts, dict) else {}
    anchors = dict(anchors) if isinstance(anchors, dict) else {}

    kernel_kind_hint = anchors.get("kernel_kind_hint") if isinstance(anchors.get("kernel_kind_hint"), str) else None

    def status(oid: str) -> str:
        o = by_id.get(oid)
        return str(o.status) if o is not None else "UNKNOWN"

    # OUT_OF_SCOPE rules (MVP)
    if status(O1_HAS_SEMANTIC_ANCHOR) == "FAIL":
        reasons.append("O1 FAIL: no semantic anchor")
        return ContractReport(level="OUT_OF_SCOPE", kernel_kind_hint=kernel_kind_hint, reasons=reasons, assumptions=assumptions, signals={"anchors": anchors})
    if status(O7_NO_ATOMICS_OR_CONTROLLED_ATOMICS) == "FAIL":
        reasons.append("O7 FAIL: atomics present")
        return ContractReport(level="OUT_OF_SCOPE", kernel_kind_hint=kernel_kind_hint, reasons=reasons, assumptions=assumptions, signals={"anchors": anchors})
    if status(O5_NO_DATA_DEPENDENT_ADDRESS) == "FAIL":
        reasons.append("O5 FAIL: data-dependent addressing suspected")
        return ContractReport(level="OUT_OF_SCOPE", kernel_kind_hint=kernel_kind_hint, reasons=reasons, assumptions=assumptions, signals={"anchors": anchors})

    # needs_mask: from frontend constraints or extracted predicates
    needs_mask = bool(constraints.needs_mask) if constraints is not None else False
    try:
        ce = (cert_v2.semantic_facts or {}).get("canonical_evidence")
        accesses = []
        if hasattr(ce, "accesses"):
            accesses = list(getattr(ce, "accesses") or [])
        elif isinstance(ce, dict):
            accesses = list(ce.get("accesses") or [])
        if any((a.predicate and a.predicate.clauses) for a in accesses if hasattr(a, "predicate")):
            needs_mask = True
        if any(isinstance(a, dict) and isinstance((a.get("predicate") or {}).get("clauses"), list) for a in accesses):
            needs_mask = True
    except Exception:
        pass

    # assumptions when no mask/predicate is available
    if not needs_mask:
        tile_hints = []
        th = (cert_v2.schedule_hints or {}).get("tile_hints")
        if isinstance(th, list):
            tile_hints = [int(x) for x in th if isinstance(x, (int, float)) and int(x) > 1]
        tile = max(tile_hints) if tile_hints else 16
        axes = []
        if isinstance(desc.launch, dict) and isinstance(desc.launch.get("vary_axes"), list):
            axes = [str(a) for a in desc.launch.get("vary_axes") if isinstance(a, str)]
        for ax in axes:
            assumptions.append(f"{ax} % {tile} == 0")
        if axes:
            reasons.append("no mask/predicate extracted; assume divisible sizes for inbounds")
        else:
            assumptions.append(f"assume_inbounds_without_mask == true")
            reasons.append("no mask/predicate extracted; assume inbounds")

    # FULL / PARTIAL aggregation
    required = [
        O1_HAS_SEMANTIC_ANCHOR,
        O2_AFFINE_OR_STRUCTURED_INDEXING,
        O4_SHAPE_LAYOUT_MATCH,
        O5_NO_DATA_DEPENDENT_ADDRESS,
        O7_NO_ATOMICS_OR_CONTROLLED_ATOMICS,
    ]
    missing = [oid for oid in required if status(oid) != "PASS"]
    if missing:
        for oid in missing:
            reasons.append(f"{oid} {status(oid)}")
        level: Literal["FULL", "PARTIAL"] = "PARTIAL"
    else:
        level = "FULL"

    # O3 gating (mask implies inbounds)
    if needs_mask and status(O3_MASK_IMPLIES_INBOUNDS) != "PASS":
        reasons.append(f"{O3_MASK_IMPLIES_INBOUNDS} {status(O3_MASK_IMPLIES_INBOUNDS)}")
        level = "PARTIAL"

    signals: Dict[str, Any] = {
        "anchors": anchors,
        "needs_mask": bool(needs_mask),
        "obligations": [o.to_json_dict() for o in obligations],
    }
    return ContractReport(level=level, kernel_kind_hint=kernel_kind_hint, reasons=reasons, assumptions=assumptions, signals=signals)


__all__ = ["ContractReport", "evaluate_contract", "evaluate_contract_v2"]
