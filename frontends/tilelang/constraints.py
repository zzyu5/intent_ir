"""
TileLang constraints extraction (MVP).

For PR#9 we only derive a minimal FrontendConstraints:
- needs_mask: True if any access carries predicate clauses.
"""

from __future__ import annotations

from dataclasses import asdict

from pipeline.interfaces import FrontendConstraints

from .facts import TileLangFacts


def extract_constraints(_source_text: str, facts: TileLangFacts) -> FrontendConstraints:
    needs_mask = any(a.predicate and a.predicate.clauses for a in (facts.accesses or []))
    suggested = ["non_divisible_edge"] if needs_mask else []
    return FrontendConstraints(needs_mask=bool(needs_mask), suggested_edge_cases=suggested, meta={"tilelang_schema": facts.schema_version})


__all__ = ["extract_constraints", "FrontendConstraints", "asdict"]

