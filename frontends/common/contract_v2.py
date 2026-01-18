"""
Contract V2: obligation-driven recoverability contract (cross-frontend).

The key rule is that Contract V2 depends only on:
  - KernelDescriptor (frontend-agnostic metadata)
  - SemanticCertificateV2 (stable semantic_facts + drift-allowed schedule_hints)
  - ObligationResult list (PASS/FAIL/UNKNOWN + witness)

It must NOT embed frontend IR details (TTIR line numbers/op names).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from pipeline.interfaces import FrontendConstraints, KernelDescriptor

from .certificate_v2 import SemanticCertificateV2
from .obligations import (
    ObligationResult,
    O1_HAS_SEMANTIC_ANCHOR,
    O2_AFFINE_OR_STRUCTURED_INDEXING,
    O3_MASK_IMPLIES_INBOUNDS,
    O4_SHAPE_LAYOUT_MATCH,
    O5_NO_DATA_DEPENDENT_ADDRESS,
    O6_STRUCTURED_SYNC,
    O7_NO_ATOMICS_OR_CONTROLLED_ATOMICS,
)


@dataclass
class ContractReport:
    level: Literal["FULL", "PARTIAL", "OUT_OF_SCOPE"]
    kernel_kind_hint: str | None
    reasons: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)


def evaluate_contract_v2(
    desc: KernelDescriptor,
    cert_v2: SemanticCertificateV2,
    obligations: List[ObligationResult],
    *,
    constraints: FrontendConstraints | None = None,
) -> ContractReport:
    """
    Evaluate contract level driven by obligations + CertificateV2 evidence.
    """
    reasons: list[str] = []
    assumptions: list[str] = []
    by_id = {o.id: o for o in obligations}

    anchors = (cert_v2.semantic_facts or {}).get("anchors") if isinstance(cert_v2.semantic_facts, dict) else {}
    anchors = dict(anchors) if isinstance(anchors, dict) else {}

    # `kernel_kind_hint` is meant to be semantic ("matmul"/"reduce"/"attention"/"copy").
    # Some frontends historically used it as a frontend identifier ("cuda_ptx",
    # "tilelang_tileop"). For the contract, prefer semantic kinds; otherwise
    # derive best-effort from anchor booleans.
    kernel_kind_hint: str | None = None
    raw_kind = anchors.get("kernel_kind_hint")
    if isinstance(raw_kind, str) and raw_kind.strip() in {"matmul", "reduce", "attention", "copy"}:
        kernel_kind_hint = raw_kind.strip()
    if kernel_kind_hint is None:
        if bool(anchors.get("has_dot")) and bool(anchors.get("has_reduce")):
            kernel_kind_hint = "attention"
        elif bool(anchors.get("has_dot")):
            kernel_kind_hint = "matmul"
        elif bool(anchors.get("has_reduce")):
            kernel_kind_hint = "reduce"
        elif bool(anchors.get("has_copy")):
            kernel_kind_hint = "copy"

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
    if status(O6_STRUCTURED_SYNC) == "FAIL":
        reasons.append("O6 FAIL: sync/barrier present")
        return ContractReport(level="OUT_OF_SCOPE", kernel_kind_hint=kernel_kind_hint, reasons=reasons, assumptions=assumptions, signals={"anchors": anchors})
    if status(O5_NO_DATA_DEPENDENT_ADDRESS) == "FAIL":
        reasons.append("O5 FAIL: data-dependent addressing suspected")
        return ContractReport(level="OUT_OF_SCOPE", kernel_kind_hint=kernel_kind_hint, reasons=reasons, assumptions=assumptions, signals={"anchors": anchors})

    # needs_mask: from frontend constraints or extracted predicates
    needs_mask = bool(getattr(constraints, "needs_mask", False)) if constraints is not None else False
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
        tile_hints: list[int] = []
        th = (cert_v2.schedule_hints or {}).get("tile_hints")
        if isinstance(th, list):
            tile_hints = [int(x) for x in th if isinstance(x, (int, float)) and int(x) > 1]
        axes: list[str] = []
        if isinstance(desc.launch, dict) and isinstance(desc.launch.get("vary_axes"), list):
            axes = [str(a) for a in desc.launch.get("vary_axes") if isinstance(a, str)]
        base_shapes = {}
        if isinstance(desc.launch, dict) and isinstance(desc.launch.get("canonical_shapes"), dict):
            base_shapes = dict(desc.launch.get("canonical_shapes") or {})
        for ax in axes:
            # Choose an axis-specific divisibility assumption that is consistent with the
            # frontend-provided base shape (avoids empty in-contract case sets).
            base_v = base_shapes.get(ax)
            tile = None
            if isinstance(base_v, (int, float)) and int(base_v) > 0:
                base_i = int(base_v)
                divisors = [h for h in tile_hints if h > 1 and base_i % int(h) == 0]
                if divisors:
                    tile = int(max(divisors))
                elif base_i > 1:
                    tile = int(base_i)
            if tile is None:
                tile = int(max(tile_hints) if tile_hints else 16)
            if int(tile) > 1:
                assumptions.append(f"{ax} % {int(tile)} == 0")
        if axes:
            reasons.append("no mask/predicate extracted; assume divisible sizes for inbounds")
        else:
            assumptions.append("assume_inbounds_without_mask == true")
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
    level: Literal["FULL", "PARTIAL"] = "FULL"
    if missing:
        for oid in missing:
            reasons.append(f"{oid} {status(oid)}")
        level = "PARTIAL"

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


__all__ = ["ContractReport", "evaluate_contract_v2"]
