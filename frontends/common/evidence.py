"""
Canonical, frontend-agnostic evidence schema (PR#4).

This is the stable "carrier" for facts extracted from a frontend IR (TTIR/TileLang/CUDA).
CertificateV2 and downstream verification should depend on this schema rather than
raw frontend IR text/line numbers/op names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass(frozen=True)
class IndexExpr:
    """
    MVP: affine expression const + Σ coeff[var]*var.

    `terms` keys should be stable symbolic names (e.g., pid0, r0, M, N).
    """

    terms: Dict[str, int] = field(default_factory=dict)
    const: int = 0

    def sort_key(self) -> Tuple[Tuple[Tuple[str, int], ...], int]:
        return (tuple(sorted(self.terms.items())), int(self.const))

    def to_json_dict(self) -> Dict[str, Any]:
        return {"terms": dict(self.terms), "const": int(self.const)}


@dataclass(frozen=True)
class Predicate:
    """
    MVP: list of textual constraint clauses.

    Example: ["0 <= m", "m < M", "0 <= k", "k < K"].
    """

    clauses: List[str] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        return {"clauses": list(self.clauses)}


@dataclass(frozen=True)
class AccessSummary:
    kind: Literal["load", "store"]
    tensor: str
    dtype: str
    rank: int
    index_exprs: List[IndexExpr]
    predicate: Optional[Predicate] = None
    address_space: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def sort_key(self) -> Tuple[int, str, int, Tuple[Tuple[Tuple[Tuple[str, int], ...], int], ...]]:
        kind_key = 0 if self.kind == "load" else 1
        idx_key = tuple(ix.sort_key() for ix in self.index_exprs)
        return (kind_key, str(self.tensor), int(self.rank), idx_key)

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict() already converts nested dataclasses, but keep stable types.
        d["rank"] = int(self.rank)
        d["index_exprs"] = [ix.to_json_dict() for ix in self.index_exprs]
        if self.predicate is not None:
            d["predicate"] = self.predicate.to_json_dict()
        return d


@dataclass
class CanonicalEvidence:
    anchors: Dict[str, Any] = field(default_factory=dict)  # has_dot/has_reduce/...
    accesses: List[AccessSummary] = field(default_factory=list)
    schedule_hints: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def canonicalize(self) -> "CanonicalEvidence":
        self.accesses = sorted(list(self.accesses), key=lambda a: a.sort_key())
        return self

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "anchors": dict(self.anchors),
            "accesses": [a.to_json_dict() for a in self.accesses],
            "schedule_hints": dict(self.schedule_hints),
            "meta": dict(self.meta),
        }


def sort_accesses(accesses: List[AccessSummary]) -> List[AccessSummary]:
    """
    Deterministic ordering for golden tests:
      1) kind: load < store
      2) tensor name (lexicographic)
      3) rank
      4) index_exprs serialized (terms,const) sequence
    """

    return sorted(list(accesses), key=lambda a: a.sort_key())


__all__ = [
    "IndexExpr",
    "Predicate",
    "AccessSummary",
    "CanonicalEvidence",
    "sort_accesses",
]

