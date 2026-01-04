"""
Frontend-agnostic access witness extraction from CanonicalEvidence.

Goal (Task6 tuning): turn "pretty" index_expr strings into *computable* signals:
  - which range symbols (rK) appear in an access
  - approximate stride (elements/bytes) for those ranges when bindings are known
  - whether an access is contiguous vs strided/irregular

This module is dependency-light and intentionally conservative. It never tries
to prove full legality; it only produces stable heuristics usable by a cost
model / tuning layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .evidence import CanonicalEvidence


_RANGE_SYM_RE = re.compile(r"^r\d+$")
_MUL_RE = re.compile(r"^mul\\((.*)\\)$")


def _split_top_level_commas(text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def _parse_mul_term(term: str) -> Optional[List[str]]:
    term = str(term).strip()
    m = _MUL_RE.match(term)
    if not m:
        return None
    inner = m.group(1).strip()
    args = _split_top_level_commas(inner)
    return [a for a in args if a]


def _extract_access_dicts_and_symbol_ranges(evidence: object | None) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    if evidence is None:
        return [], {}

    if isinstance(evidence, CanonicalEvidence):
        return [a.to_json_dict() for a in evidence.accesses], {}

    if isinstance(evidence, dict):
        # Common cases:
        # - certificate_v2 dict: {"schema_version", "semantic_facts":{...}, "schedule_hints":{...}}
        # - semantic_facts dict: {"canonical_evidence":{...}, ...}
        # - canonical_evidence dict itself.
        if "semantic_facts" in evidence and isinstance(evidence.get("semantic_facts"), dict):
            sf = evidence.get("semantic_facts") or {}
            ce = sf.get("canonical_evidence") or {}
            sh = evidence.get("schedule_hints") or {}
        elif "canonical_evidence" in evidence and isinstance(evidence.get("canonical_evidence"), (dict, CanonicalEvidence)):
            ce = evidence.get("canonical_evidence") or {}
            sh = evidence.get("schedule_hints") or {}
        elif "accesses" in evidence:
            ce = evidence
            sh = {}
        else:
            ce = {}
            sh = {}

        symbol_ranges: Dict[str, Dict[str, int]] = {}
        if isinstance(sh, dict):
            sr = sh.get("symbol_ranges")
            if isinstance(sr, dict):
                for k, v in sr.items():
                    if not isinstance(k, str) or not isinstance(v, dict):
                        continue
                    try:
                        a = int(v.get("start"))
                        b = int(v.get("end"))
                    except Exception:
                        continue
                    symbol_ranges[k] = {"start": a, "end": b}

        if isinstance(ce, CanonicalEvidence):
            accesses = [a.to_json_dict() for a in ce.accesses]
        elif isinstance(ce, dict):
            raw = ce.get("accesses") or []
            accesses = [a for a in raw if isinstance(a, dict)]
        else:
            accesses = []
        return accesses, symbol_ranges

    return [], {}


@dataclass(frozen=True)
class StrideContribution:
    coeff: int
    factors: Tuple[str, ...] = ()

    def to_json_dict(self) -> Dict[str, Any]:
        return {"coeff": int(self.coeff), "factors": list(self.factors)}


@dataclass(frozen=True)
class RangeStrideWitness:
    range_sym: str
    contributions: Tuple[StrideContribution, ...] = ()
    stride_elems: Optional[int] = None  # resolved if all factors known
    stride_bytes: Optional[int] = None
    range_len: Optional[int] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "range_sym": str(self.range_sym),
            "contributions": [c.to_json_dict() for c in self.contributions],
            "stride_elems": (int(self.stride_elems) if self.stride_elems is not None else None),
            "stride_bytes": (int(self.stride_bytes) if self.stride_bytes is not None else None),
            "range_len": (int(self.range_len) if self.range_len is not None else None),
        }


@dataclass(frozen=True)
class AccessStrideWitness:
    tensor: str
    kind: str
    dtype: str
    unresolved: bool
    index_expr: Dict[str, Any]
    predicate_clauses: Tuple[str, ...] = ()
    range_strides: Tuple[RangeStrideWitness, ...] = ()
    axis_bindings: Dict[str, str] = field(default_factory=dict)  # rK -> bound symbol name (e.g., N)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "tensor": str(self.tensor),
            "kind": str(self.kind),
            "dtype": str(self.dtype),
            "unresolved": bool(self.unresolved),
            "index_expr": dict(self.index_expr),
            "predicate_clauses": list(self.predicate_clauses),
            "range_strides": [r.to_json_dict() for r in self.range_strides],
            "axis_bindings": dict(self.axis_bindings),
        }


@dataclass
class EvidenceStrideSummary:
    accesses: List[AccessStrideWitness] = field(default_factory=list)
    tensor_penalty: Dict[str, float] = field(default_factory=dict)
    dominant_range: Optional[str] = None
    dominant_axis: Optional[str] = None
    dominant_range_len: Optional[int] = None
    has_contiguous_range: bool = False
    notes: List[str] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "accesses": [a.to_json_dict() for a in self.accesses],
            "tensor_penalty": {k: float(v) for k, v in self.tensor_penalty.items()},
            "dominant_range": self.dominant_range,
            "dominant_axis": self.dominant_axis,
            "dominant_range_len": self.dominant_range_len,
            "has_contiguous_range": bool(self.has_contiguous_range),
            "notes": list(self.notes),
        }


_PRED_LT_RE = re.compile(r"\\b(r\\d+)\\b\\s*<\\s*\\b([A-Za-z_][A-Za-z0-9_]*)\\b")
_PRED_GT_RE = re.compile(r"\\b([A-Za-z_][A-Za-z0-9_]*)\\b\\s*>\\s*\\b(r\\d+)\\b")


def _axis_bindings_from_predicate(clauses: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in clauses:
        s = str(c)
        m = _PRED_LT_RE.search(s)
        if m:
            out.setdefault(str(m.group(1)), str(m.group(2)))
            continue
        m = _PRED_GT_RE.search(s)
        if m:
            out.setdefault(str(m.group(2)), str(m.group(1)))
            continue
    return out


def _resolve_factor(f: str, bindings: Dict[str, int]) -> Optional[int]:
    s = str(f).strip()
    if not s:
        return None
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    v = bindings.get(s)
    return int(v) if isinstance(v, int) else None


def _range_len(sym: str, symbol_ranges: Dict[str, Dict[str, int]]) -> Optional[int]:
    spec = symbol_ranges.get(sym)
    if not isinstance(spec, dict):
        return None
    try:
        a = int(spec.get("start"))
        b = int(spec.get("end"))
    except Exception:
        return None
    if b <= a:
        return None
    return int(b - a)


def build_stride_summary(
    evidence: object | None,
    *,
    shape_bindings: Dict[str, int] | None = None,
    cache_line_bytes: int = 64,
    dtype_bytes_fn=None,
) -> EvidenceStrideSummary:
    """
    Produce stride witnesses + per-tensor penalties from CanonicalEvidence.

    `dtype_bytes_fn` may be supplied by the backend cost model to keep dtype sizing consistent.
    """
    shape_bindings = dict(shape_bindings or {})
    cache_line = int(cache_line_bytes or 64)

    if dtype_bytes_fn is None:
        # Local conservative default.
        def dtype_bytes_fn(dt: str) -> int:
            return 1 if dt in {"bool", "i1", "u8", "i8"} else 4

    raw_accesses, symbol_ranges = _extract_access_dicts_and_symbol_ranges(evidence)

    witnesses: List[AccessStrideWitness] = []
    tensor_penalty: Dict[str, float] = {}
    range_contig_hits: Dict[str, int] = {}
    range_any_hits: Dict[str, int] = {}
    range_axis: Dict[str, str] = {}

    for a in raw_accesses:
        tensor = str(a.get("tensor") or "")
        kind = str(a.get("kind") or "")
        dtype = str(a.get("dtype") or "f32")
        meta = a.get("meta") if isinstance(a.get("meta"), dict) else {}
        unresolved = bool(meta.get("unresolved")) if isinstance(meta, dict) else False
        idxs = a.get("index_exprs") if isinstance(a.get("index_exprs"), list) else []
        if not idxs:
            continue
        # MVP assumes a single flat index expr.
        ix = idxs[0] if isinstance(idxs[0], dict) else {}
        terms = ix.get("terms") if isinstance(ix.get("terms"), dict) else {}

        pred = a.get("predicate") if isinstance(a.get("predicate"), dict) else None
        clauses = list(pred.get("clauses") or []) if isinstance(pred, dict) else []
        axis_bindings = _axis_bindings_from_predicate([str(c) for c in clauses])
        for r, ax in axis_bindings.items():
            if _RANGE_SYM_RE.match(r) and isinstance(ax, str) and ax:
                range_axis.setdefault(r, ax)

        dt_bytes = int(dtype_bytes_fn(dtype))

        # Compute per-range strides.
        range_syms: set[str] = set()
        for var in terms.keys():
            v = str(var)
            if _RANGE_SYM_RE.match(v):
                range_syms.add(v)
                continue
            factors = _parse_mul_term(v)
            if not factors:
                continue
            for f in factors:
                if _RANGE_SYM_RE.match(str(f).strip()):
                    range_syms.add(str(f).strip())

        range_strides: List[RangeStrideWitness] = []
        for r in sorted(range_syms):
            contrib: List[StrideContribution] = []
            unknown = False
            stride_sum = 0
            for var, raw_c in terms.items():
                v = str(var)
                try:
                    c = int(raw_c)
                except Exception:
                    continue
                if c == 0:
                    continue
                if v == r:
                    contrib.append(StrideContribution(coeff=c, factors=()))
                    stride_sum += c
                    continue
                factors = _parse_mul_term(v)
                if not factors:
                    continue
                hits = [f for f in factors if str(f).strip() == r]
                if not hits:
                    continue
                # r appears multiple times => nonlinear.
                if len(hits) != 1:
                    unknown = True
                    contrib.append(StrideContribution(coeff=c, factors=tuple(sorted(str(x).strip() for x in factors))))
                    continue
                other = [str(x).strip() for x in factors if str(x).strip() != r]
                # If other ranges are present, the stride depends on them (treat unknown).
                if any(_RANGE_SYM_RE.match(x) for x in other):
                    unknown = True
                    contrib.append(StrideContribution(coeff=c, factors=tuple(sorted(other))))
                    continue
                # Try resolve other factors.
                prod = 1
                for f in other:
                    v0 = _resolve_factor(f, shape_bindings)
                    if v0 is None:
                        unknown = True
                        break
                    prod *= int(v0)
                contrib.append(StrideContribution(coeff=c, factors=tuple(sorted(other))))
                if not unknown:
                    stride_sum += c * prod

            stride_elems: Optional[int] = None
            stride_bytes: Optional[int] = None
            if not unknown:
                stride_elems = int(stride_sum)
                stride_bytes = int(abs(stride_elems) * dt_bytes)
            rl = _range_len(r, symbol_ranges)
            range_strides.append(
                RangeStrideWitness(
                    range_sym=r,
                    contributions=tuple(contrib),
                    stride_elems=stride_elems,
                    stride_bytes=stride_bytes,
                    range_len=rl,
                )
            )

            range_any_hits[r] = range_any_hits.get(r, 0) + 1
            if stride_elems is not None and abs(int(stride_elems)) == 1:
                range_contig_hits[r] = range_contig_hits.get(r, 0) + 1

        witnesses.append(
            AccessStrideWitness(
                tensor=tensor,
                kind=kind,
                dtype=dtype,
                unresolved=unresolved,
                index_expr=dict(ix),
                predicate_clauses=tuple(str(c) for c in clauses),
                range_strides=tuple(range_strides),
                axis_bindings=dict(axis_bindings),
            )
        )

        # Derive a stable penalty signal (cache-line vs contiguous) for backends.
        if unresolved or not range_strides:
            p = float(max(1, cache_line) / max(1, dt_bytes))
        else:
            # Pick the smallest resolved stride among ranges (best-case contiguous dimension),
            # but conservatively clamp to cache_line.
            strides = [rs.stride_bytes for rs in range_strides if rs.stride_bytes is not None and rs.stride_bytes > 0]
            if not strides:
                p = float(max(1, cache_line) / max(1, dt_bytes))
            else:
                stride = max(dt_bytes, min(cache_line, min(int(s) for s in strides)))
                p = float(stride) / float(dt_bytes)
        if tensor:
            tensor_penalty[tensor] = max(float(tensor_penalty.get(tensor, 1.0)), float(p))

    # Dominant range: prefer "contiguous" ranges (stride==1) since that implies vectorizable axis.
    dominant_range = None
    dominant_axis = None
    dominant_len = None
    has_contig = bool(range_contig_hits)
    if range_contig_hits:
        dominant_range = max(sorted(range_contig_hits.keys()), key=lambda k: (range_contig_hits[k], -range_any_hits.get(k, 0)))
    elif range_any_hits:
        dominant_range = max(sorted(range_any_hits.keys()), key=lambda k: range_any_hits[k])
    if dominant_range:
        dominant_axis = range_axis.get(dominant_range)
        dominant_len = _range_len(dominant_range, symbol_ranges)

    notes: List[str] = []
    if dominant_range:
        msg = f"dominant_range={dominant_range}"
        if dominant_axis:
            msg += f" axis={dominant_axis}"
        if dominant_len:
            msg += f" len={dominant_len}"
        notes.append(msg)

    return EvidenceStrideSummary(
        accesses=witnesses,
        tensor_penalty=tensor_penalty,
        dominant_range=dominant_range,
        dominant_axis=dominant_axis,
        dominant_range_len=dominant_len,
        has_contiguous_range=has_contig,
        notes=notes,
    )


__all__ = [
    "StrideContribution",
    "RangeStrideWitness",
    "AccessStrideWitness",
    "EvidenceStrideSummary",
    "build_stride_summary",
]
