"""
Task4 v1.2 (minimal formal core): lightweight "SMT-ish" checks.

We intentionally avoid a hard dependency on Z3 in the repo:
- If `z3-solver` is installed, we can extend checks later.
- For now we provide a conservative, syntax-directed proof attempt that certain
  masked indices are non-negative, which is the missing half of
  `mask => 0 <= idx < Dim` given that `idx < Dim` is already witnessed by cmp.

This is not a full formal proof of memory safety; it is a small formal kernel
that can be reported as PASS/UNKNOWN/FAIL and improved incrementally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple


@dataclass(frozen=True)
class MaskConstraintCheck:
    constraint: str
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    detail: str


@dataclass(frozen=True)
class MaskInboundsSummary:
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    total: int
    passed: int
    failed: int
    unknown: int
    checks: List[MaskConstraintCheck]


# -----------------------------
# Tiny expression parser
# -----------------------------


TOKEN_RE = re.compile(r"\s*(==|!=|<=|>=|[()+\-*/%<>]|[A-Za-z_%][A-Za-z0-9_%]*|\d+)\s*")


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
    out = TOKEN_RE.findall(expr)
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
    while ts.peek() in {"*", "/", "%"}:
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
    return ("var", t)


_CMP_RE = re.compile(r"^(?P<lhs>.+?)\s*(?P<op><=|>=|==|!=|<|>)\s*(?P<rhs>.+?)\s*$")


def parse_constraint(constraint: str) -> Tuple[str, str, str]:
    m = _CMP_RE.match(constraint.strip())
    if not m:
        raise ValueError(f"unsupported constraint syntax: {constraint!r}")
    return m.group("lhs").strip(), m.group("op").strip(), m.group("rhs").strip()


# -----------------------------
# Conservative non-neg proof
# -----------------------------


ARG_RE = re.compile(r"^%?arg\d+$")
SSA_TMP_RE = re.compile(r"^%\\d+$")


def _collect_nonneg_vars(index_symbols: Dict[str, Any]) -> Set[str]:
    nonneg: Set[str] = set()
    # Canonical pid names used in witnesses
    nonneg.update({"pid0", "pid1", "pid2"})
    # Range vars
    for name, spec in (index_symbols.get("ranges") or {}).items():
        try:
            if int(spec.get("start", 0)) >= 0:
                nonneg.add(name)
        except Exception:
            pass
    return nonneg


def _prove_nonneg(node: _Node, *, nonneg_vars: Set[str]) -> Optional[bool]:
    """
    Return True if we can PROVE node >= 0, False if we can PROVE node < 0,
    None if unknown.
    """
    kind = node[0]
    if kind == "num":
        return bool(node[1] >= 0)
    if kind == "var":
        v = str(node[1])
        if v in nonneg_vars or ARG_RE.match(v) or SSA_TMP_RE.match(v):
            return True
        return None
    if kind == "neg":
        inner = node[1]
        inner_nn = _prove_nonneg(inner, nonneg_vars=nonneg_vars)
        if inner_nn is True:
            # -(x>=0) is <=0, but could be 0; can't prove negative nor non-negative.
            return None
        if inner_nn is False:
            # - (x<0) is >0
            return True
        return None
    if kind == "bin":
        op, a, b = node[1], node[2], node[3]
        an = _prove_nonneg(a, nonneg_vars=nonneg_vars)
        bn = _prove_nonneg(b, nonneg_vars=nonneg_vars)
        if op == "+":
            if an is True and bn is True:
                return True
            # If one side is provably non-negative and the other is unknown, we
            # cannot prove non-negativity but also cannot disprove; keep UNKNOWN.
            return None
        if op == "*":
            return True if (an is True and bn is True) else None
        if op == "%":
            if an is True and bn is True:
                return True
            return None
        if op == "-":
            # a - b >= 0 only if we can prove b == 0 and a >= 0 (too hard here).
            return None
        if op == "/":
            return None
        return None
    return None


# -----------------------------
# Optional Z3 backend (stronger)
# -----------------------------


def _try_import_z3():
    try:
        import z3  # type: ignore
        return z3
    except Exception:
        return None


def _to_z3(node: _Node, z3, var_env: Dict[str, Any]):
    kind = node[0]
    if kind == "num":
        return z3.IntVal(int(node[1]))
    if kind == "var":
        name = str(node[1])
        if name not in var_env:
            var_env[name] = z3.Int(name)
        return var_env[name]
    if kind == "neg":
        return -_to_z3(node[1], z3, var_env)
    if kind == "bin":
        op, a, b = node[1], node[2], node[3]
        za = _to_z3(a, z3, var_env)
        zb = _to_z3(b, z3, var_env)
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


def check_mask_constraints_inbounds(
    mask_constraints: Dict[str, List[str]],
    *,
    index_symbols: Dict[str, Any],
) -> MaskInboundsSummary:
    """
    Minimal check: for each bound-like constraint `lhs < rhs` (or <=),
    attempt to prove `lhs >= 0` under standard TTIR domain assumptions.
    """
    nonneg_vars = _collect_nonneg_vars(index_symbols)
    checks: List[MaskConstraintCheck] = []
    passed = failed = unknown = 0

    z3 = _try_import_z3()
    range_bounds = index_symbols.get("ranges") or {}
    for _, cs in mask_constraints.items():
        for c in cs:
            try:
                lhs_s, op, _rhs_s = parse_constraint(c)
            except Exception as e:
                checks.append(MaskConstraintCheck(constraint=c, status="UNKNOWN", detail=f"parse failed: {e}"))
                unknown += 1
                continue
            if op not in {"<", "<="}:
                checks.append(MaskConstraintCheck(constraint=c, status="UNKNOWN", detail=f"unsupported cmp op {op!r}"))
                unknown += 1
                continue
            try:
                lhs_node = _parse_expr(_TokStream(_tokenize(lhs_s)))
            except Exception as e:
                checks.append(MaskConstraintCheck(constraint=c, status="UNKNOWN", detail=f"lhs parse failed: {e}"))
                unknown += 1
                continue

            nn = _prove_nonneg(lhs_node, nonneg_vars=nonneg_vars)
            if nn is True:
                checks.append(MaskConstraintCheck(constraint=c, status="PASS", detail="proved lhs >= 0 (syntactic)"))
                passed += 1
                continue
            # Try Z3 if available
            if z3 is not None:
                try:
                    var_env: Dict[str, Any] = {}
                    lhs_z3 = _to_z3(lhs_node, z3, var_env)
                    s = z3.Solver()
                    for name, v in var_env.items():
                        s.add(v >= 0)
                        if name in range_bounds:
                            try:
                                end = int(range_bounds[name].get("end"))
                                s.add(v < end)
                            except Exception:
                                pass
                    res = s.check(z3.Not(lhs_z3 >= 0))
                    if res == z3.unsat:
                        checks.append(MaskConstraintCheck(constraint=c, status="PASS", detail="proved lhs >= 0 (z3)"))
                        passed += 1
                        continue
                    elif res == z3.sat:
                        checks.append(MaskConstraintCheck(constraint=c, status="FAIL", detail="z3 found lhs < 0 model"))
                        failed += 1
                        continue
                except Exception as e:
                    checks.append(MaskConstraintCheck(constraint=c, status="UNKNOWN", detail=f"z3 error: {e}"))
                    unknown += 1
                    continue
            checks.append(MaskConstraintCheck(constraint=c, status="UNKNOWN", detail="could not prove lhs >= 0"))
            unknown += 1

    total = len(checks)
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    if total == 0:
        status = "UNKNOWN"
    elif failed > 0:
        status = "FAIL"
    elif unknown > 0:
        status = "UNKNOWN"
    else:
        status = "PASS"
    return MaskInboundsSummary(status=status, total=total, passed=passed, failed=failed, unknown=unknown, checks=checks)


__all__ = ["MaskConstraintCheck", "MaskInboundsSummary", "check_mask_constraints_inbounds"]
