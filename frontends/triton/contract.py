"""
Task4: Recoverability Contract evaluation based on TTIR facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal

from .facts import TTIRConstraints, TTIRFacts


@dataclass
class ContractReport:
    level: Literal["FULL", "PARTIAL", "OUT_OF_SCOPE"]
    kernel_kind_hint: str | None
    reasons: list[str] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)


def evaluate_contract(facts: TTIRFacts, constraints: TTIRConstraints | None = None) -> ContractReport:
    reasons: list[str] = []
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
        return ContractReport(level="OUT_OF_SCOPE", kernel_kind_hint=None, reasons=reasons, signals=signals)
    if not (facts.has_dot or facts.has_reduce):
        reasons.append("no dot or reduce anchors found")
        return ContractReport(level="OUT_OF_SCOPE", kernel_kind_hint=None, reasons=reasons, signals=signals)

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

    return ContractReport(level=level, kernel_kind_hint=kernel_kind, reasons=reasons, signals=signals)


__all__ = ["ContractReport", "evaluate_contract"]
