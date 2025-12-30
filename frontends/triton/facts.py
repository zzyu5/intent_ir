"""
Task4: Extract TTIR facts and lightweight constraints from TTIR text.
Uses regex/line scanning, not a full MLIR parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from pipeline.interfaces import FrontendConstraints


@dataclass
class AccessSite:
    kind: Literal["load", "store"]
    line_no: int
    line: str
    has_mask: bool
    tensor_hint: str | None = None
    ptr: str | None = None
    mask: str | None = None


@dataclass
class MaskSite:
    line_no: int
    line: str
    cmp_kind: str | None = None


@dataclass
class TTIRFacts:
    op_counts: Dict[str, int]
    has_dot: bool
    has_reduce: bool
    has_atomic: bool
    has_barrier: bool
    has_async: bool
    load_sites: List[AccessSite] = field(default_factory=list)
    store_sites: List[AccessSite] = field(default_factory=list)
    mask_sites: List[MaskSite] = field(default_factory=list)
    raw_summary: Dict[str, Any] = field(default_factory=dict)


TTIRConstraints = FrontendConstraints


DOT_RE = re.compile(r"\btt\.dot\b|\bdot\b")
REDUCE_RE = re.compile(r"\breduce\b|\btt\.reduce\b")
LOAD_RE = re.compile(r"\btt\.load\b")
STORE_RE = re.compile(r"\btt\.store\b")
ATOMIC_RE = re.compile(r"\batomic\b|\btt\.atomic\b|\batomic_rmw\b")
BARRIER_RE = re.compile(r"\bgpu\.barrier\b|\bttg\.barrier\b|\bbarrier\b|\bsyncthreads\b")
ASYNC_RE = re.compile(r"\basync_copy\b|\basync_wait\b|\bttg\.async\b|\btt\.async\b")
MASK_IN_LINE_RE = re.compile(r"\bmask\b")
CMP_RE = re.compile(r"\barith\.cmpi\s+([a-z]+)|\barith\.cmpf\s+([a-z]+)")


def extract_facts(ttir: str) -> TTIRFacts:
    lines = ttir.splitlines()
    op_counts: Dict[str, int] = {}
    load_sites: List[AccessSite] = []
    store_sites: List[AccessSite] = []
    mask_sites: List[MaskSite] = []
    has_dot = False
    has_reduce = False
    has_atomic = False
    has_barrier = False
    has_async = False

    for idx, line in enumerate(lines, start=1):
        text = line.strip()
        # op counts
        for pat, name in [
            (DOT_RE, "dot"),
            (REDUCE_RE, "reduce"),
            (LOAD_RE, "load"),
            (STORE_RE, "store"),
            (ATOMIC_RE, "atomic"),
            (BARRIER_RE, "barrier"),
            (ASYNC_RE, "async"),
        ]:
            if pat.search(text):
                op_counts[name] = op_counts.get(name, 0) + 1
        if DOT_RE.search(text):
            has_dot = True
        if REDUCE_RE.search(text):
            has_reduce = True
        if ATOMIC_RE.search(text):
            has_atomic = True
        if BARRIER_RE.search(text):
            has_barrier = True
        if ASYNC_RE.search(text):
            has_async = True
        if LOAD_RE.search(text):
            operands = _extract_op_operands(text, "tt.load")
            has_mask = len(operands) >= 2
            ptr = operands[0] if operands else None
            mask = operands[1] if len(operands) >= 2 else None
            load_sites.append(
                AccessSite(
                    kind="load",
                    line_no=idx,
                    line=text,
                    has_mask=has_mask,
                    tensor_hint=(operands[0].lstrip("%") if operands else _extract_tensor_name(text)),
                    ptr=ptr,
                    mask=mask,
                )
            )
        if STORE_RE.search(text):
            operands = _extract_op_operands(text, "tt.store")
            has_mask = len(operands) >= 3
            ptr = operands[0] if operands else None
            mask = operands[2] if len(operands) >= 3 else None
            store_sites.append(
                AccessSite(
                    kind="store",
                    line_no=idx,
                    line=text,
                    has_mask=has_mask,
                    tensor_hint=(operands[0].lstrip("%") if operands else _extract_tensor_name(text)),
                    ptr=ptr,
                    mask=mask,
                )
            )
        m = CMP_RE.search(text)
        if m:
            cmp_kind = m.group(1) or m.group(2)
            mask_sites.append(MaskSite(line_no=idx, line=text, cmp_kind=cmp_kind))

    facts = TTIRFacts(
        op_counts=op_counts,
        has_dot=has_dot,
        has_reduce=has_reduce,
        has_atomic=has_atomic,
        has_barrier=has_barrier,
        has_async=has_async,
        load_sites=load_sites,
        store_sites=store_sites,
        mask_sites=mask_sites,
        raw_summary={"num_lines": len(lines)},
    )
    return facts


def extract_constraints(ttir: str, facts: TTIRFacts | None = None) -> TTIRConstraints:
    facts = facts or extract_facts(ttir)
    needs_mask = any(site.has_mask for site in facts.load_sites + facts.store_sites)
    suggested = []
    if needs_mask:
        suggested.append("non_divisible_edge")
    return TTIRConstraints(needs_mask=needs_mask, suggested_edge_cases=suggested)


def _extract_tensor_name(line: str) -> str | None:
    m = re.search(r"%([A-Za-z0-9_]+)", line)
    return m.group(1) if m else None


def _extract_op_operands(line: str, op_name: str) -> List[str]:
    """
    Very small TTIR text parser: extract SSA-ish operands after an op name.

    Examples:
      "%30 = tt.load %29, %27, %cst_0 : tensor<...>" -> ["%29","%27","%cst_0"]
      "tt.store %21, %22, %12 : tensor<...>"         -> ["%21","%22","%12"]
    """
    if op_name not in line:
        return []
    # Strip result assignment if present.
    rhs = line.split("=", 1)[1].strip() if "=" in line and line.strip().startswith("%") else line
    # Keep only the segment after op_name and before ":" (type annotation) if any.
    after = rhs.split(op_name, 1)[1].strip()
    after = after.split(" loc(", 1)[0].strip()
    after = after.split(" :", 1)[0].strip()
    after = after.split(" : ", 1)[0].strip()
    after = after.split(":", 1)[0].strip()
    if not after:
        return []
    # Split by commas; keep SSA-ish tokens (%foo) and const tokens (%cst_0).
    parts = [p.strip() for p in after.split(",")]
    out: List[str] = []
    for p in parts:
        tok = p.split()[0]
        if tok.startswith("%"):
            out.append(tok)
    return out


__all__ = [
    "TTIRFacts",
    "TTIRConstraints",
    "AccessSite",
    "MaskSite",
    "extract_facts",
    "extract_constraints",
]
