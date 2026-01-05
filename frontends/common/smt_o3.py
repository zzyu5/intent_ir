"""
PR#7: O3 SMT core (mask/predicate ⇒ inbounds).

This module is frontend-agnostic: it consumes the stable CertificateV2 evidence:
- AccessSummary.index_exprs (affine IndexExpr)
- Predicate.clauses (strings, MVP grammar)

Goals:
- PASS: produce a witness explaining which clauses/bounds justify inbounds.
- FAIL: produce a concrete counterexample model (variable assignments).
- UNKNOWN: explain why we can't decide (unsupported clause / missing bound / non-affine).

We intentionally avoid a hard dependency on Z3. The MVP proof engine is:
- syntactic/affine reasoning for PASS (sound under domain assumptions)
- bounded, deterministic model search for FAIL witness
If neither applies => UNKNOWN.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

from .evidence import AccessSummary, IndexExpr, Predicate


Status = Literal["PASS", "FAIL", "UNKNOWN"]


@dataclass(frozen=True)
class CounterexampleModel:
    assignments: Dict[str, int]


@dataclass(frozen=True)
class O3DimCheck:
    dim: int
    index: IndexExpr
    upper_bound: Optional[IndexExpr]
    status: Status
    reason: str = ""
    witness: Dict[str, Any] = field(default_factory=dict)
    counterexample: Optional[CounterexampleModel] = None


@dataclass(frozen=True)
class O3AccessCheck:
    kind: str
    tensor: str
    rank: int
    status: Status
    dims: List[O3DimCheck]
    reason: str = ""


@dataclass(frozen=True)
class O3Report:
    status: Status
    total: int
    passed: int
    failed: int
    unknown: int
    access_checks: List[O3AccessCheck]

    def to_json_dict(self) -> Dict[str, Any]:
        def enc_ix(ix: IndexExpr) -> Dict[str, Any]:
            return {"terms": dict(ix.terms), "const": int(ix.const)}

        def enc_model(m: CounterexampleModel | None) -> Any:
            if m is None:
                return None
            return {"assignments": {str(k): int(v) for k, v in m.assignments.items()}}

        out = {
            "status": self.status,
            "total": int(self.total),
            "passed": int(self.passed),
            "failed": int(self.failed),
            "unknown": int(self.unknown),
            "access_checks": [],
        }
        for a in self.access_checks:
            ad = {
                "kind": str(a.kind),
                "tensor": str(a.tensor),
                "rank": int(a.rank),
                "status": str(a.status),
                "reason": str(a.reason),
                "dims": [],
            }
            for d in a.dims:
                ad["dims"].append(
                    {
                        "dim": int(d.dim),
                        "index": enc_ix(d.index),
                        "upper_bound": (enc_ix(d.upper_bound) if d.upper_bound is not None else None),
                        "status": str(d.status),
                        "reason": str(d.reason),
                        "witness": dict(d.witness),
                        "counterexample": enc_model(d.counterexample),
                    }
                )
            out["access_checks"].append(ad)
        return out


_CMP_RE = re.compile(r"^(?P<lhs>.+?)\s*(?P<op><=|>=|==|!=|<|>)\s*(?P<rhs>.+?)\s*$")
_ARG_INDEX_RE = re.compile(r"^arg\d+$")
_R_INDEX_RE = re.compile(r"^r\d+$")


def _parse_cmp(clause: str) -> Optional[Tuple[str, str, str]]:
    m = _CMP_RE.match(str(clause).strip())
    if not m:
        return None
    return (m.group("lhs").strip(), m.group("op").strip(), m.group("rhs").strip())


def _parse_affine(expr: str) -> Optional[IndexExpr]:
    """
    Parse a simple affine expression:
      - constants: "0", "-1"
      - vars: "pid0", "r0", "M", "arg4"
      - sums: "8*pid0 + r0 - 1"
    """
    s = str(expr).strip()
    if not s or s == "<unresolved>":
        return None
    # Normalize subtraction into "+ -term"
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
        # integer literal (with or without whitespace after unary '-')
        if p.lstrip("-").isdigit():
            const += int(p)
            continue
        if p.startswith("-"):
            rest = p[1:].strip()
            if rest.isdigit():
                const -= int(rest)
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


@dataclass(frozen=True)
class _Constraint:
    lhs: IndexExpr
    op: str
    rhs: IndexExpr
    raw: str


def _is_single_var(ix: IndexExpr) -> Optional[str]:
    if ix.const != 0:
        return None
    if len(ix.terms) != 1:
        return None
    (v, c), = ix.terms.items()
    return str(v) if int(c) == 1 else None


def _add_ix(a: IndexExpr, b: IndexExpr) -> IndexExpr:
    terms = dict(a.terms)
    for k, v in b.terms.items():
        terms[k] = terms.get(k, 0) + int(v)
        if terms[k] == 0:
            terms.pop(k, None)
    return IndexExpr(terms=terms, const=int(a.const) + int(b.const))


def _mul_ix(ix: IndexExpr, c: int) -> IndexExpr:
    if c == 0:
        return IndexExpr(terms={}, const=0)
    return IndexExpr(terms={k: int(v) * int(c) for k, v in ix.terms.items() if int(v) * int(c) != 0}, const=int(ix.const) * int(c))


def _substitute(ix: IndexExpr, subst: Dict[str, IndexExpr], *, max_steps: int = 64) -> IndexExpr:
    out = ix
    for _ in range(max_steps):
        changed = False
        new_terms: Dict[str, int] = {}
        acc = IndexExpr(terms={}, const=int(out.const))
        for v, c in out.terms.items():
            if v in subst:
                acc = _add_ix(acc, _mul_ix(subst[v], int(c)))
                changed = True
            else:
                new_terms[str(v)] = new_terms.get(str(v), 0) + int(c)
        new_terms = {k: v for k, v in new_terms.items() if v != 0}
        acc = _add_ix(acc, IndexExpr(terms=new_terms, const=0))
        out = acc
        if not changed:
            break
    return out


def _build_subst(constraints: List[_Constraint]) -> Dict[str, IndexExpr]:
    subst: Dict[str, IndexExpr] = {}
    for c in constraints:
        if c.op != "==":
            continue
        lv = _is_single_var(c.lhs)
        rv = _is_single_var(c.rhs)
        if lv is not None and rv is None:
            subst[lv] = c.rhs
        elif rv is not None and lv is None:
            subst[rv] = c.lhs
        elif lv is not None and rv is not None and lv != rv:
            # Canonicalize by mapping the lexicographically larger to the smaller.
            a, b = sorted([lv, rv])
            subst[b] = IndexExpr(terms={a: 1}, const=0)
    # Transitive closure: substitute inside substitutions.
    for _ in range(16):
        changed = False
        for k, v in list(subst.items()):
            nv = _substitute(v, subst)
            if nv != v:
                subst[k] = nv
                changed = True
        if not changed:
            break
    return subst


def _eval_ix(ix: IndexExpr, env: Dict[str, int]) -> int:
    v = int(ix.const)
    for name, c in ix.terms.items():
        v += int(c) * int(env.get(str(name), 0))
    return v


def _eval_constraint(c: _Constraint, env: Dict[str, int]) -> bool:
    a = _eval_ix(c.lhs, env)
    b = _eval_ix(c.rhs, env)
    if c.op == "<":
        return a < b
    if c.op == "<=":
        return a <= b
    if c.op == ">":
        return a > b
    if c.op == ">=":
        return a >= b
    if c.op == "==":
        return a == b
    if c.op == "!=":
        return a != b
    return False


def _variables_in_constraints(constraints: List[_Constraint]) -> Set[str]:
    out: Set[str] = set()
    for c in constraints:
        out.update(str(k) for k in c.lhs.terms.keys())
        out.update(str(k) for k in c.rhs.terms.keys())
    return out


def _default_domains(
    vars_: Iterable[str],
    *,
    symbol_ranges: Dict[str, Dict[str, int]] | None,
    shape_hints: Dict[str, int] | None,
    max_pid: int = 4,
) -> Tuple[Dict[str, List[int]], Dict[str, int], Dict[str, Dict[str, Any]]]:
    """
    Build deterministic finite domains for bounded search and minimal lower bounds.
    """
    domains: Dict[str, List[int]] = {}
    lower_bounds: Dict[str, int] = {}
    domain_info: Dict[str, Dict[str, Any]] = {}
    symbol_ranges = symbol_ranges or {}
    shape_hints = shape_hints or {}

    for v in sorted(set(str(x) for x in vars_)):
        if v in {"pid0", "pid1", "pid2"}:
            domains[v] = list(range(0, max_pid + 1))
            lower_bounds[v] = 0
            domain_info[v] = {"source": "pid_default", "start": 0, "end": int(max_pid) + 1}
            continue
        if v in symbol_ranges:
            s = int(symbol_ranges[v].get("start", 0))
            e = int(symbol_ranges[v].get("end", s + 1))
            size = max(0, e - s)
            if size <= 32:
                domains[v] = list(range(s, e))
                domain_info[v] = {"source": "symbol_ranges_full", "start": s, "end": e}
            else:
                # Sample endpoints + midpoints deterministically (keeps search space bounded).
                mid = (s + e) // 2
                q1 = (s + mid) // 2
                q3 = (mid + e) // 2
                cand = [s, s + 1, q1, mid, q3, e - 2, e - 1]
                cand = [x for x in cand if s <= x < e]
                domains[v] = sorted(set(cand)) or [s]
                domain_info[v] = {"source": "symbol_ranges_sampled", "start": s, "end": e}
            lower_bounds[v] = s
            continue
        if v in shape_hints:
            base = max(1, int(shape_hints[v]))
            cap = min(base, 128)
            # Adaptive: enumerate fully for small caps, else sample to keep product bounded.
            if cap <= 32:
                domains[v] = list(range(1, cap + 1))
                domain_info[v] = {"source": "shape_hints_full", "start": 1, "end": cap + 1}
            else:
                # Prioritize small + boundary values deterministically.
                head = list(range(1, 9))  # 1..8
                mid = [cap // 4, cap // 2, (3 * cap) // 4]
                tail = [cap - 3, cap - 2, cap - 1, cap]
                cand = [*head, *mid, *tail]
                cand = [x for x in cand if 1 <= x <= cap]
                domains[v] = sorted(set(cand))
                domain_info[v] = {"source": "shape_hints_sampled", "start": 1, "end": cap + 1}
            lower_bounds[v] = 1
            continue
        if _ARG_INDEX_RE.match(v):
            domains[v] = [0, 1, 2, 4, 8, 16]
            lower_bounds[v] = 0
            domain_info[v] = {"source": "arg_default"}
            continue
        if _R_INDEX_RE.match(v):
            # Loop/reduction indices are non-negative in TTIR/TIR semantics.
            domains[v] = [0, 1, 2, 3, 4, 7, 8, 15, 16, 31]
            lower_bounds[v] = 0
            domain_info[v] = {"source": "r_default", "assumption": "non_negative"}
            continue
        # Unknown symbol: keep a tiny domain (may include negative offsets).
        domains[v] = [-1, 0, 1, 2, 3, 4]
        lower_bounds[v] = -1
        domain_info[v] = {"source": "unknown_default", "assumption": "may_be_negative"}

    return domains, lower_bounds, domain_info


def _summarize_domains(domains: Dict[str, List[int]], domain_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for v in sorted(domains.keys()):
        vals = list(domains[v])
        info = dict(domain_info.get(v, {}))
        if len(vals) <= 16:
            info["values"] = vals
        else:
            info["values_head"] = vals[:8]
            info["values_tail"] = vals[-4:]
            info["min"] = int(min(vals))
            info["max"] = int(max(vals))
            info["count"] = int(len(vals))
        out[str(v)] = info
    return out


def _min_lower_bound(ix: IndexExpr, lower_bounds: Dict[str, int]) -> Optional[int]:
    """
    If all coeffs are >=0 and all vars have known lower bounds, return a proven lower bound.
    """
    lb = int(ix.const)
    for v, c in ix.terms.items():
        c = int(c)
        if c < 0:
            return None
        if str(v) not in lower_bounds:
            return None
        lb += c * int(lower_bounds[str(v)])
    return lb


def _lower_bound_assumptions(ix: IndexExpr, lower_bounds: Dict[str, int]) -> Dict[str, int]:
    """
    Return the per-variable lower bounds that a successful `_min_lower_bound` proof relies on.
    """
    out: Dict[str, int] = {}
    for v, c in ix.terms.items():
        if int(c) < 0:
            continue
        name = str(v)
        if name in lower_bounds:
            out[name] = int(lower_bounds[name])
    return out


def _find_upper_bound_clause(
    idx: IndexExpr,
    constraints: List[_Constraint],
) -> Optional[_Constraint]:
    """
    Find a constraint that directly provides an upper bound for idx:
      - idx < UB
      - idx <= UB
      - UB > idx
      - UB >= idx
    """
    for c in constraints:
        if c.op in {"<", "<="} and c.lhs == idx:
            return c
        if c.op in {">", ">="} and c.rhs == idx:
            return c
    return None


def _has_explicit_nonneg(idx: IndexExpr, constraints: List[_Constraint]) -> Optional[str]:
    zero = IndexExpr(terms={}, const=0)
    for c in constraints:
        if c.op == ">=" and c.lhs == idx and c.rhs == zero:
            return c.raw
        if c.op == "<=" and c.rhs == idx and c.lhs == zero:
            return c.raw
        if c.op == ">" and c.lhs == idx and c.rhs == IndexExpr(terms={}, const=-1):
            return c.raw
        if c.op == "<" and c.rhs == idx and c.lhs == IndexExpr(terms={}, const=-1):
            return c.raw
    return None


def _bounded_model_search(
    constraints: List[_Constraint],
    *,
    domains: Dict[str, List[int]],
    extra_cond: Optional[Tuple[IndexExpr, str, IndexExpr]] = None,
    max_models: int = 5000,
) -> Tuple[Optional[CounterexampleModel], Dict[str, Any]]:
    """
    Deterministic bounded search. Returns the first satisfying assignment plus search stats.
    """
    def _prod(ns: Iterable[int]) -> int:
        p = 1
        for n in ns:
            p *= int(n)
        return int(p)

    vars_ = sorted(domains.keys())
    value_lists = [domains[v] for v in vars_]
    space = _prod(len(vs) for vs in value_lists)
    checked = 0
    for values in itertools.product(*value_lists):
        checked += 1
        if checked > max_models:
            stats = {
                "vars": list(vars_),
                "checked": int(checked - 1),
                "max_models": int(max_models),
                "domain_space": int(space),
                "stop_reason": "hit_max_models",
                "exhausted": False,
            }
            return None, stats
        env = {vars_[i]: int(values[i]) for i in range(len(vars_))}
        ok = True
        for c in constraints:
            if not _eval_constraint(c, env):
                ok = False
                break
        if not ok:
            continue
        if extra_cond is not None:
            lhs, op, rhs = extra_cond
            cc = _Constraint(lhs=lhs, op=op, rhs=rhs, raw="(extra)")
            if not _eval_constraint(cc, env):
                continue
        stats = {
            "vars": list(vars_),
            "checked": int(checked),
            "max_models": int(max_models),
            "domain_space": int(space),
            "stop_reason": "found_model",
            "exhausted": False,
        }
        return CounterexampleModel(assignments=env), stats

    stats = {
        "vars": list(vars_),
        "checked": int(checked),
        "max_models": int(max_models),
        "domain_space": int(space),
        "stop_reason": "exhausted_domain",
        "exhausted": True,
    }
    return None, stats


def check_mask_implies_inbounds(
    accesses: Sequence[AccessSummary],
    *,
    shape_hints: Dict[str, int] | None = None,
    symbol_ranges: Dict[str, Dict[str, int]] | None = None,
    max_counterexample_search: int = 5000,
) -> O3Report:
    """
    Main entry: check O3 for all masked accesses.
    """
    access_checks: List[O3AccessCheck] = []
    total = passed = failed = unknown = 0

    for a in accesses:
        pred = a.predicate
        clauses = list(pred.clauses) if isinstance(pred, Predicate) and pred.clauses else []
        if not clauses:
            continue

        parsed: List[_Constraint] = []
        parse_error = None
        for cl in clauses:
            s = str(cl).strip()
            cmp_ = _parse_cmp(s)
            if cmp_ is None:
                parse_error = f"unsupported clause syntax: {s!r}"
                break
            lhs_s, op, rhs_s = cmp_
            lhs = _parse_affine(lhs_s)
            rhs = _parse_affine(rhs_s)
            if lhs is None or rhs is None:
                parse_error = f"non-affine clause side: {s!r}"
                break
            parsed.append(_Constraint(lhs=lhs, op=op, rhs=rhs, raw=s))
        if parse_error:
            dims = [
                O3DimCheck(dim=i, index=ix, upper_bound=None, status="UNKNOWN", reason=parse_error)
                for i, ix in enumerate(a.index_exprs)
            ]
            access_checks.append(O3AccessCheck(kind=a.kind, tensor=a.tensor, rank=a.rank, status="UNKNOWN", dims=dims, reason=parse_error))
            total += len(dims)
            unknown += len(dims)
            continue

        subst = _build_subst(parsed)
        norm_constraints = [_Constraint(lhs=_substitute(c.lhs, subst), op=c.op, rhs=_substitute(c.rhs, subst), raw=c.raw) for c in parsed]

        domains, lower_bounds, domain_info = _default_domains(
            _variables_in_constraints(norm_constraints), symbol_ranges=symbol_ranges, shape_hints=shape_hints
        )

        dim_checks: List[O3DimCheck] = []
        access_status: Status = "PASS"
        access_reason = ""
        for dim_i, ix0 in enumerate(list(a.index_exprs)):
            idx = _substitute(ix0, subst)
            ub_clause = _find_upper_bound_clause(idx, norm_constraints)
            if ub_clause is None:
                dim_checks.append(
                    O3DimCheck(
                        dim=dim_i,
                        index=idx,
                        upper_bound=None,
                        status="UNKNOWN",
                        reason="missing upper-bound clause for this access index",
                        witness={"predicate_clauses": list(clauses)[:8]},
                    )
                )
                access_status = "UNKNOWN" if access_status != "FAIL" else "FAIL"
                total += 1
                unknown += 1
                continue

            ub = ub_clause.rhs if (ub_clause.op in {"<", "<="} and ub_clause.lhs == idx) else ub_clause.lhs

            # Lower-bound proof:
            explicit = _has_explicit_nonneg(idx, norm_constraints)
            if explicit is not None:
                dim_checks.append(
                    O3DimCheck(
                        dim=dim_i,
                        index=idx,
                        upper_bound=ub,
                        status="PASS",
                        witness={"upper_bound_clause": ub_clause.raw, "lower_bound_clause": explicit},
                    )
                )
                passed += 1
                total += 1
                continue

            lb = _min_lower_bound(idx, lower_bounds)
            if lb is not None and lb >= 0:
                dim_checks.append(
                    O3DimCheck(
                        dim=dim_i,
                        index=idx,
                        upper_bound=ub,
                        status="PASS",
                        witness={
                            "upper_bound_clause": ub_clause.raw,
                            "lower_bound_proof": f"min_bound={lb}",
                            "assumptions": {"lower_bounds": _lower_bound_assumptions(idx, lower_bounds)},
                        },
                    )
                )
                passed += 1
                total += 1
                continue

            # Try to find a concrete counterexample: predicate holds but idx < 0.
            cex, stats = _bounded_model_search(
                norm_constraints,
                domains=domains,
                extra_cond=(idx, "<", IndexExpr(terms={}, const=0)),
                max_models=max_counterexample_search,
            )
            if cex is not None:
                dim_checks.append(
                    O3DimCheck(
                        dim=dim_i,
                        index=idx,
                        upper_bound=ub,
                        status="FAIL",
                        reason="found model where predicate holds but index is negative",
                        witness={
                            "upper_bound_clause": ub_clause.raw,
                            "bounded_search": {
                                "goal": "find model: predicate && (idx < 0)",
                                "domains": _summarize_domains(domains, domain_info),
                                "stats": stats,
                                "bounded": True,
                                "incomplete_note": "FAIL is sound for the found model; the search itself is bounded/incomplete for non-found cases",
                            },
                        },
                        counterexample=cex,
                    )
                )
                failed += 1
                total += 1
                access_status = "FAIL"
                access_reason = "O3 violated by counterexample"
                continue

            dim_checks.append(
                O3DimCheck(
                    dim=dim_i,
                    index=idx,
                    upper_bound=ub,
                    status="UNKNOWN",
                    reason="cannot prove index non-negative under current domain assumptions",
                    witness={
                        "upper_bound_clause": ub_clause.raw,
                        "bounded_search": {
                            "goal": "try refute: predicate && (idx < 0)",
                            "domains": _summarize_domains(domains, domain_info),
                            "stats": stats,
                            "bounded": True,
                            "incomplete_note": (
                                "No counterexample found in this bounded search; this does NOT prove safety. "
                                "Counterexamples may exist outside enumerated domains or beyond max_models."
                            ),
                        },
                        "hint": "optional: install z3-solver for stronger proofs; or increase max_counterexample_search / domain coverage",
                    },
                )
            )
            unknown += 1
            total += 1
            if access_status != "FAIL":
                access_status = "UNKNOWN"

        access_checks.append(
            O3AccessCheck(kind=a.kind, tensor=a.tensor, rank=a.rank, status=access_status, dims=dim_checks, reason=access_reason)
        )

    overall: Status
    if failed > 0:
        overall = "FAIL"
    elif unknown > 0:
        overall = "UNKNOWN"
    else:
        overall = "PASS"

    return O3Report(status=overall, total=total, passed=passed, failed=failed, unknown=unknown, access_checks=access_checks)


__all__ = ["Status", "CounterexampleModel", "O3DimCheck", "O3AccessCheck", "O3Report", "check_mask_implies_inbounds"]
