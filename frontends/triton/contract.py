"""
Task4: Recoverability Contract evaluation based on TTIR facts.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from .facts import TTIRConstraints, TTIRFacts
from frontends.common.contract_v2 import ContractReport, evaluate_contract_v2


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


__all__ = ["ContractReport", "evaluate_contract", "evaluate_contract_v2"]
