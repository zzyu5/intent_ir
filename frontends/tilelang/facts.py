"""
TileLang facts extraction (MVP).

For PR#9, we treat TileLang source as a structured JSON DSL that explicitly
describes anchors + accesses. This keeps the extractor deterministic and
frontend-independent, while leaving room to replace the JSON parser with a real
TileLang AST parser later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from frontends.common.evidence import AccessSummary, IndexExpr, Predicate, sort_accesses


@dataclass
class TileLangFacts:
    schema_version: str
    anchors: Dict[str, Any]
    accesses: List[AccessSummary] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


def _parse_index_expr(d: Dict[str, Any]) -> IndexExpr:
    terms = d.get("terms") or {}
    const = d.get("const", 0)
    if not isinstance(terms, dict):
        raise ValueError("index_expr.terms must be object")
    out_terms: Dict[str, int] = {}
    for k, v in terms.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, (int, float)):
            raise ValueError(f"index_expr.terms[{k}] must be int")
        out_terms[str(k)] = int(v)
    if not isinstance(const, (int, float)):
        raise ValueError("index_expr.const must be int")
    return IndexExpr(terms=out_terms, const=int(const))


def _parse_predicate(d: Dict[str, Any] | None) -> Predicate | None:
    if d is None:
        return None
    clauses = d.get("clauses")
    if clauses is None:
        return None
    if not isinstance(clauses, list):
        raise ValueError("predicate.clauses must be list")
    out = []
    for c in clauses:
        if isinstance(c, str) and c.strip():
            out.append(str(c).strip())
    return Predicate(clauses=out)


def _parse_access(d: Dict[str, Any]) -> AccessSummary:
    kind = d.get("kind")
    tensor = d.get("tensor")
    dtype = d.get("dtype", "unknown")
    rank = d.get("rank", 1)
    idxs = d.get("index_exprs") or []
    if kind not in {"load", "store"}:
        raise ValueError("access.kind must be 'load'|'store'")
    if not isinstance(tensor, str) or not tensor:
        raise ValueError("access.tensor must be non-empty string")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("access.dtype must be string")
    if not isinstance(rank, (int, float)):
        raise ValueError("access.rank must be int")
    if not isinstance(idxs, list):
        raise ValueError("access.index_exprs must be list")
    index_exprs = [_parse_index_expr(x) for x in idxs if isinstance(x, dict)]
    pred = _parse_predicate(d.get("predicate") if isinstance(d.get("predicate"), dict) else None)
    addr = d.get("address_space", "global")
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    return AccessSummary(
        kind=str(kind),
        tensor=str(tensor),
        dtype=str(dtype),
        rank=int(rank),
        index_exprs=index_exprs,
        predicate=pred,
        address_space=str(addr) if isinstance(addr, str) else "global",
        meta=dict(meta),
    )


def extract_facts(source_text: str) -> TileLangFacts:
    """
    Parse TileLang JSON DSL into TileLangFacts.

    Expected minimal schema:
      {
        "schema_version": "tilelang_dsl_v0.1",
        "anchors": {...},
        "accesses": [AccessSummary-like dicts...]
      }
    """
    data = json.loads(source_text)
    if not isinstance(data, dict):
        raise ValueError("TileLang DSL must be a JSON object")
    schema_version = str(data.get("schema_version", "tilelang_dsl_v0.1"))
    anchors = data.get("anchors") or {}
    if not isinstance(anchors, dict):
        raise ValueError("anchors must be an object")
    accesses_raw = data.get("accesses") or []
    if not isinstance(accesses_raw, list):
        raise ValueError("accesses must be a list")
    accesses = [_parse_access(a) for a in accesses_raw if isinstance(a, dict)]
    accesses = sort_accesses(accesses)
    return TileLangFacts(schema_version=schema_version, anchors=dict(anchors), accesses=accesses, raw=dict(data))


__all__ = ["TileLangFacts", "extract_facts"]

