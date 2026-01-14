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


# -----------------------------
# Tiny integer expression parser
# -----------------------------


# Note: include '//' as a single token (TileLang/TVM scripts and some predicates
# use Python-style floor division).
_TOKEN_RE = re.compile(r"\s*(==|!=|<=|>=|//|[()+\-*/%]|[A-Za-z_%][A-Za-z0-9_%]*|\d+)\s*")


class _TokStream:
    def __init__(self, tokens: List[str]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def pop(self) -> str:
        if self.i >= len(self.toks):
            raise ValueError("unexpected end of tokens")
        t = self.toks[self.i]
        self.i += 1
        return t


def _tokenize(expr: str) -> List[str]:
    out = _TOKEN_RE.findall(str(expr))
    if not out:
        raise ValueError(f"failed to tokenize expression: {expr!r}")
    return out


_Node = Tuple[str, Any]  # ("num", int) | ("var", str) | ("neg", node) | ("bin", op, lhs, rhs)


def _parse_expr(ts: _TokStream) -> _Node:
    return _parse_add(ts)


def _parse_add(ts: _TokStream) -> _Node:
    node = _parse_mul(ts)
    while ts.peek() in {"+", "-"}:
        op = ts.pop()
        rhs = _parse_mul(ts)
        node = ("bin", op, node, rhs)
    return node


def _parse_mul(ts: _TokStream) -> _Node:
    node = _parse_unary(ts)
    while ts.peek() in {"*", "/", "//", "%"}:
        op = ts.pop()
        rhs = _parse_unary(ts)
        node = ("bin", op, node, rhs)
    return node


def _parse_unary(ts: _TokStream) -> _Node:
    if ts.peek() in {"+", "-"}:
        op = ts.pop()
        inner = _parse_unary(ts)
        return inner if op == "+" else ("neg", inner)
    return _parse_atom(ts)


def _parse_atom(ts: _TokStream) -> _Node:
    t = ts.pop()
    if t == "(":
        node = _parse_expr(ts)
        if ts.pop() != ")":
            raise ValueError("expected ')'")
        return node
    if t.isdigit() or (t.startswith("-") and t[1:].isdigit()):
        return ("num", int(t))
    return ("var", str(t))


def _parse_int_expr(expr: str) -> _Node:
    s = str(expr).strip()
    if not s or s == "<unresolved>":
        raise ValueError("empty/unresolved expression")
    toks = _tokenize(s)
    ts = _TokStream(toks)
    node = _parse_expr(ts)
    if ts.peek() is not None:
        raise ValueError(f"unexpected trailing token: {ts.peek()!r}")
    return node


def _parse_cmp(clause: str) -> Optional[Tuple[str, str, str]]:
    m = _CMP_RE.match(str(clause).strip())
    if not m:
        return None
    return (m.group("lhs").strip(), m.group("op").strip(), m.group("rhs").strip())


@dataclass(frozen=True)
class _Expr:
    node: _Node
    affine: Optional[IndexExpr]


@dataclass(frozen=True)
class _Constraint:
    lhs: _Expr
    op: str
    rhs: _Expr
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


def _node_to_affine(node: _Node) -> Optional[IndexExpr]:
    kind = node[0]
    if kind == "num":
        return IndexExpr(terms={}, const=int(node[1]))
    if kind == "var":
        return IndexExpr(terms={str(node[1]): 1}, const=0)
    if kind == "neg":
        inner = _node_to_affine(node[1])
        return _mul_ix(inner, -1) if inner is not None else None
    if kind == "bin":
        op, a, b = str(node[1]), node[2], node[3]
        aa = _node_to_affine(a)
        bb = _node_to_affine(b)
        if aa is None or bb is None:
            return None
        if op == "+":
            return _add_ix(aa, bb)
        if op == "-":
            return _add_ix(aa, _mul_ix(bb, -1))
        if op == "*":
            # Only support const * affine (either side).
            if not aa.terms:
                return _mul_ix(bb, int(aa.const))
            if not bb.terms:
                return _mul_ix(aa, int(bb.const))
            return None
        # Non-affine ops
        if op in {"/", "//", "%"}:
            return None
    return None


def _ix_to_node(ix: IndexExpr) -> _Node:
    """
    Convert an affine IndexExpr into a node tree. This is used to apply affine
    substitutions inside non-affine predicate clauses.
    """
    parts: List[_Node] = []
    for k in sorted((ix.terms or {}).keys()):
        c = int(ix.terms[k])
        if c == 0:
            continue
        v: _Node = ("var", str(k))
        term = v if c == 1 else ("neg", v) if c == -1 else ("bin", "*", ("num", int(c)), v)
        parts.append(term)
    if int(ix.const) != 0 or not parts:
        parts.append(("num", int(ix.const)))
    node = parts[0]
    for p in parts[1:]:
        node = ("bin", "+", node, p)
    return node


def _substitute_node(node: _Node, subst: Dict[str, _Node], *, max_depth: int = 64) -> _Node:
    if max_depth <= 0:
        return node
    kind = node[0]
    if kind == "var":
        name = str(node[1])
        return subst.get(name, node)
    if kind == "num":
        return node
    if kind == "neg":
        return ("neg", _substitute_node(node[1], subst, max_depth=max_depth - 1))
    if kind == "bin":
        return (
            "bin",
            str(node[1]),
            _substitute_node(node[2], subst, max_depth=max_depth - 1),
            _substitute_node(node[3], subst, max_depth=max_depth - 1),
        )
    return node


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
        if c.lhs.affine is None or c.rhs.affine is None:
            continue
        lv = _is_single_var(c.lhs.affine)
        rv = _is_single_var(c.rhs.affine)
        if lv is not None and rv is None:
            subst[lv] = c.rhs.affine
        elif rv is not None and lv is None:
            subst[rv] = c.lhs.affine
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


def _eval_node(node: _Node, env: Dict[str, int]) -> Optional[int]:
    kind = node[0]
    if kind == "num":
        return int(node[1])
    if kind == "var":
        return int(env.get(str(node[1]), 0))
    if kind == "neg":
        inner = _eval_node(node[1], env)
        return None if inner is None else -int(inner)
    if kind == "bin":
        op, a, b = str(node[1]), node[2], node[3]
        aa = _eval_node(a, env)
        bb = _eval_node(b, env)
        if aa is None or bb is None:
            return None
        if op == "+":
            return int(aa) + int(bb)
        if op == "-":
            return int(aa) - int(bb)
        if op == "*":
            return int(aa) * int(bb)
        if op == "/":
            if int(bb) == 0:
                return None
            return int(aa) // int(bb)
        if op == "//":
            if int(bb) == 0:
                return None
            return int(aa) // int(bb)
        if op == "%":
            if int(bb) == 0:
                return None
            return int(aa) % int(bb)
        return None
    return None


def _eval_expr(expr: _Expr, env: Dict[str, int]) -> Optional[int]:
    if expr.affine is not None:
        return _eval_ix(expr.affine, env)
    return _eval_node(expr.node, env)


def _eval_cmp(a: int, op: str, b: int) -> bool:
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    return False


def _eval_constraint(c: _Constraint, env: Dict[str, int]) -> bool:
    a = _eval_expr(c.lhs, env)
    b = _eval_expr(c.rhs, env)
    if a is None or b is None:
        return False
    return _eval_cmp(int(a), str(c.op), int(b))


def _variables_in_constraints(constraints: List[_Constraint]) -> Set[str]:
    def visit(node: _Node, acc: Set[str]) -> None:
        kind = node[0]
        if kind == "var":
            acc.add(str(node[1]))
        elif kind == "neg":
            visit(node[1], acc)
        elif kind == "bin":
            visit(node[2], acc)
            visit(node[3], acc)

    out: Set[str] = set()
    for c in constraints:
        visit(c.lhs.node, out)
        visit(c.rhs.node, out)
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
        if c.lhs.affine is not None and c.op in {"<", "<="} and c.lhs.affine == idx:
            return c
        if c.rhs.affine is not None and c.op in {">", ">="} and c.rhs.affine == idx:
            return c
    return None


def _has_explicit_nonneg(idx: IndexExpr, constraints: List[_Constraint]) -> Optional[str]:
    zero = IndexExpr(terms={}, const=0)
    for c in constraints:
        if c.lhs.affine is None or c.rhs.affine is None:
            continue
        if c.op == ">=" and c.lhs.affine == idx and c.rhs.affine == zero:
            return c.raw
        if c.op == "<=" and c.rhs.affine == idx and c.lhs.affine == zero:
            return c.raw
        if c.op == ">" and c.lhs.affine == idx and c.rhs.affine == IndexExpr(terms={}, const=-1):
            return c.raw
        if c.op == "<" and c.rhs.affine == idx and c.lhs.affine == IndexExpr(terms={}, const=-1):
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
            if not _eval_cmp(_eval_ix(lhs, env), str(op), _eval_ix(rhs, env)):
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


def _try_import_z3():
    try:
        import z3  # type: ignore

        return z3
    except Exception:
        return None


def _node_to_z3(node: _Node, z3, var_env: Dict[str, Any]):
    kind = node[0]
    if kind == "num":
        return z3.IntVal(int(node[1]))
    if kind == "var":
        name = str(node[1])
        if name not in var_env:
            var_env[name] = z3.Int(name)
        return var_env[name]
    if kind == "neg":
        return -_node_to_z3(node[1], z3, var_env)
    if kind == "bin":
        op, a, b = str(node[1]), node[2], node[3]
        za = _node_to_z3(a, z3, var_env)
        zb = _node_to_z3(b, z3, var_env)
        if op == "+":
            return za + zb
        if op == "-":
            return za - zb
        if op == "*":
            return za * zb
        if op == "/":
            return z3.IntDiv(za, zb)
        if op == "%":
            return z3.Mod(za, zb)
    return z3.IntVal(0)


def _ix_to_z3(ix: IndexExpr, z3, var_env: Dict[str, Any]):
    out = z3.IntVal(int(ix.const))
    for name, c in sorted((ix.terms or {}).items()):
        v = str(name)
        if v not in var_env:
            var_env[v] = z3.Int(v)
        out = out + z3.IntVal(int(c)) * var_env[v]
    return out


def _constraint_to_z3(c: _Constraint, z3, var_env: Dict[str, Any]):
    lhs = _ix_to_z3(c.lhs.affine, z3, var_env) if c.lhs.affine is not None else _node_to_z3(c.lhs.node, z3, var_env)
    rhs = _ix_to_z3(c.rhs.affine, z3, var_env) if c.rhs.affine is not None else _node_to_z3(c.rhs.node, z3, var_env)
    op = str(c.op)
    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs
    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    return z3.BoolVal(True)


def check_mask_implies_inbounds(
    accesses: Sequence[AccessSummary],
    *,
    shape_hints: Dict[str, int] | None = None,
    symbol_ranges: Dict[str, Dict[str, int]] | None = None,
    max_counterexample_search: int = 5000,
    solver: Literal["auto", "mvp", "z3"] = "auto",
    z3_timeout_ms: int = 2000,
) -> O3Report:
    """
    Main entry: check O3 for all masked accesses.
    """
    access_checks: List[O3AccessCheck] = []
    total = passed = failed = unknown = 0
    z3 = _try_import_z3() if solver in {"auto", "z3"} else None
    use_z3 = z3 is not None and solver in {"auto", "z3"}

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
            try:
                lhs_node = _parse_int_expr(lhs_s)
                rhs_node = _parse_int_expr(rhs_s)
            except Exception as e:
                parse_error = f"clause parse failed: {type(e).__name__}: {e} | clause={s!r}"
                break
            parsed.append(
                _Constraint(
                    lhs=_Expr(node=lhs_node, affine=_node_to_affine(lhs_node)),
                    op=op,
                    rhs=_Expr(node=rhs_node, affine=_node_to_affine(rhs_node)),
                    raw=s,
                )
            )
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
        subst_nodes = {k: _ix_to_node(v) for k, v in subst.items()}
        norm_constraints: List[_Constraint] = []
        for c in parsed:
            lhs_node = _substitute_node(c.lhs.node, subst_nodes)
            rhs_node = _substitute_node(c.rhs.node, subst_nodes)
            norm_constraints.append(
                _Constraint(
                    lhs=_Expr(node=lhs_node, affine=_node_to_affine(lhs_node)),
                    op=c.op,
                    rhs=_Expr(node=rhs_node, affine=_node_to_affine(rhs_node)),
                    raw=c.raw,
                )
            )

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

            ub_aff: Optional[IndexExpr] = None
            if ub_clause.op in {"<", "<="} and ub_clause.lhs.affine == idx:
                ub_aff = ub_clause.rhs.affine
            elif ub_clause.op in {">", ">="} and ub_clause.rhs.affine == idx:
                ub_aff = ub_clause.lhs.affine

            # Lower-bound proof:
            explicit = _has_explicit_nonneg(idx, norm_constraints)
            if explicit is not None:
                dim_checks.append(
                    O3DimCheck(
                        dim=dim_i,
                        index=idx,
                        upper_bound=ub_aff,
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
                        upper_bound=ub_aff,
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
                        upper_bound=ub_aff,
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

            # Optional stronger proof/model via Z3 (if available and enabled).
            z3_witness: Dict[str, Any] | None = None
            z3_cex: CounterexampleModel | None = None
            if use_z3:
                try:
                    var_env: Dict[str, Any] = {}
                    s_z3 = z3.Solver()
                    s_z3.set("timeout", int(z3_timeout_ms))
                    for c in norm_constraints:
                        s_z3.add(_constraint_to_z3(c, z3, var_env))
                    # Domain assumptions (sound only under these assumptions).
                    for name, v in var_env.items():
                        if name in lower_bounds:
                            s_z3.add(v >= int(lower_bounds[name]))
                        if symbol_ranges and name in symbol_ranges:
                            rr = symbol_ranges.get(name) or {}
                            try:
                                start = int(rr.get("start", 0))
                                end = int(rr.get("end", start))
                                s_z3.add(v >= start)
                                s_z3.add(v < end)
                            except Exception:
                                pass
                        if name not in lower_bounds and re.match(r"^[A-Z][A-Z0-9_]*$", name):
                            s_z3.add(v >= 1)
                    s_z3.add(_ix_to_z3(idx, z3, var_env) < 0)
                    res = s_z3.check()
                    if res == z3.unsat:
                        z3_witness = {
                            "result": "unsat",
                            "goal": "predicate && (idx < 0)",
                            "timeout_ms": int(z3_timeout_ms),
                            "domain_assumptions": {"lower_bounds": dict(lower_bounds)},
                        }
                    elif res == z3.sat:
                        m = s_z3.model()
                        assigns: Dict[str, int] = {}
                        for nm, v in var_env.items():
                            try:
                                vv = m.eval(v, model_completion=True)
                                if vv is None:
                                    continue
                                if hasattr(vv, "as_long"):
                                    assigns[str(nm)] = int(vv.as_long())
                            except Exception:
                                continue
                        z3_cex = CounterexampleModel(assignments=assigns)
                        z3_witness = {
                            "result": "sat",
                            "goal": "predicate && (idx < 0)",
                            "timeout_ms": int(z3_timeout_ms),
                            "model": {"assignments": dict(assigns)},
                            "domain_assumptions": {"lower_bounds": dict(lower_bounds)},
                        }
                    else:
                        z3_witness = {
                            "result": "unknown",
                            "goal": "predicate && (idx < 0)",
                            "timeout_ms": int(z3_timeout_ms),
                            "reason": str(getattr(s_z3, "reason_unknown", lambda: "")()),
                            "domain_assumptions": {"lower_bounds": dict(lower_bounds)},
                        }
                except Exception as e:
                    z3_witness = {"result": "error", "error": f"{type(e).__name__}: {e}", "timeout_ms": int(z3_timeout_ms)}

            if z3_witness is not None and z3_witness.get("result") == "unsat":
                dim_checks.append(
                    O3DimCheck(
                        dim=dim_i,
                        index=idx,
                        upper_bound=ub_aff,
                        status="PASS",
                        witness={"upper_bound_clause": ub_clause.raw, "z3": z3_witness},
                    )
                )
                passed += 1
                total += 1
                continue
            if z3_cex is not None:
                dim_checks.append(
                    O3DimCheck(
                        dim=dim_i,
                        index=idx,
                        upper_bound=ub_aff,
                        status="FAIL",
                        reason="z3 found model where predicate holds but index is negative",
                        witness={"upper_bound_clause": ub_clause.raw, "bounded_search": {"stats": stats}, "z3": z3_witness},
                        counterexample=z3_cex,
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
                    upper_bound=ub_aff,
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
                        **({"z3": z3_witness} if z3_witness is not None else {}),
                        "hint": (
                            "optional: enable z3 backend for stronger proofs; "
                            "or increase max_counterexample_search / domain coverage"
                        ),
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
