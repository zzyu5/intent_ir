"""
Task4 v1.2 (partial): best-effort affine expression extraction from TTIR SSA.

This is NOT a full MLIR affine analysis. It is a lightweight utility used to
build explainable "index map" witnesses in the SemanticCertificate.

Supported ops (best-effort):
- arith.constant (scalar integers)
- arith.addi/subi/muli (only const * expr)
- arith.extsi/extui
- tt.get_program_id (alias only)
- tt.make_range (alias only)
- tt.splat/broadcast/expand_dims/reshape (shape ops treated as identity)

When an expression is not provably affine, we fall back to a symbolic term.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .ttir_witness import SSAValueDef


@dataclass(frozen=True)
class AffineExpr:
    # const + sum_i coeff[var_i] * var_i
    const: int = 0
    coeff: Dict[str, int] = field(default_factory=dict)
    non_affine: bool = False

    def add(self, other: "AffineExpr") -> "AffineExpr":
        if self.non_affine or other.non_affine:
            return AffineExpr(non_affine=True)
        coeff = dict(self.coeff)
        for k, v in other.coeff.items():
            coeff[k] = coeff.get(k, 0) + v
            if coeff[k] == 0:
                coeff.pop(k, None)
        return AffineExpr(const=self.const + other.const, coeff=coeff)

    def sub(self, other: "AffineExpr") -> "AffineExpr":
        if self.non_affine or other.non_affine:
            return AffineExpr(non_affine=True)
        coeff = dict(self.coeff)
        for k, v in other.coeff.items():
            coeff[k] = coeff.get(k, 0) - v
            if coeff[k] == 0:
                coeff.pop(k, None)
        return AffineExpr(const=self.const - other.const, coeff=coeff)

    def mul_const(self, c: int) -> "AffineExpr":
        if self.non_affine:
            return self
        coeff = {k: v * c for k, v in self.coeff.items() if v * c != 0}
        return AffineExpr(const=self.const * c, coeff=coeff)

    def is_zero(self) -> bool:
        return (not self.non_affine) and self.const == 0 and not self.coeff


INT_CONST_RE = re.compile(r"\barith\.constant\s+(-?\d+)\s*:\s*i(?:32|64)\b")
INT_DENSE_CONST_RE = re.compile(r"\barith\.constant\s+dense<(-?\d+)>\s*:\s*tensor<[^>]*xi(?:32|64)>")
PID_RE = re.compile(r"\btt\.get_program_id\s+([xyz])\b")
MAKE_RANGE_START_RE = re.compile(r"\bstart\s*=\s*(-?\d+)\s*:\s*i32\b")
MAKE_RANGE_END_RE = re.compile(r"\bend\s*=\s*(-?\d+)\s*:\s*i32\b")


def build_aliases(defs: Dict[str, SSAValueDef]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for name, d in defs.items():
        m = PID_RE.search(d.line)
        if m:
            aliases[name] = f"pid_{m.group(1)}"
            continue
        if "tt.make_range" in d.line:
            sm = MAKE_RANGE_START_RE.search(d.line)
            em = MAKE_RANGE_END_RE.search(d.line)
            if sm and em:
                aliases[name] = f"range({sm.group(1)},{em.group(1)})"
                continue
    return aliases


def affine_from_ssa(
    name: str,
    defs: Dict[str, SSAValueDef],
    *,
    aliases: Optional[Dict[str, str]] = None,
    max_depth: int = 64,
    _memo: Optional[Dict[str, AffineExpr]] = None,
) -> AffineExpr:
    """
    Best-effort affine extraction for an SSA value.
    """
    aliases = aliases or {}
    memo = _memo if _memo is not None else {}
    if name in memo:
        return memo[name]

    # Treat unknown values as symbolic variables.
    def sym(v: str) -> AffineExpr:
        key = aliases.get(v, v)
        return AffineExpr(const=0, coeff={key: 1})

    # Depth guard
    if max_depth <= 0:
        out = sym(name)
        memo[name] = out
        return out

    d = defs.get(name)
    if d is None:
        out = sym(name)
        memo[name] = out
        return out

    # scalar integer constants
    m = INT_CONST_RE.search(d.line)
    if m:
        out = AffineExpr(const=int(m.group(1)), coeff={})
        memo[name] = out
        return out
    m = INT_DENSE_CONST_RE.search(d.line)
    if m:
        out = AffineExpr(const=int(m.group(1)), coeff={})
        memo[name] = out
        return out

    op = d.op
    ops = d.operands
    if op in {"arith.extsi", "arith.extui", "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape"} and ops:
        out = affine_from_ssa(ops[0], defs, aliases=aliases, max_depth=max_depth - 1, _memo=memo)
        memo[name] = out
        return out

    if op == "arith.addi" and len(ops) >= 2:
        a = affine_from_ssa(ops[0], defs, aliases=aliases, max_depth=max_depth - 1, _memo=memo)
        b = affine_from_ssa(ops[1], defs, aliases=aliases, max_depth=max_depth - 1, _memo=memo)
        out = a.add(b)
        memo[name] = out
        return out

    if op == "arith.subi" and len(ops) >= 2:
        a = affine_from_ssa(ops[0], defs, aliases=aliases, max_depth=max_depth - 1, _memo=memo)
        b = affine_from_ssa(ops[1], defs, aliases=aliases, max_depth=max_depth - 1, _memo=memo)
        out = a.sub(b)
        memo[name] = out
        return out

    if op == "arith.muli" and len(ops) >= 2:
        a = affine_from_ssa(ops[0], defs, aliases=aliases, max_depth=max_depth - 1, _memo=memo)
        b = affine_from_ssa(ops[1], defs, aliases=aliases, max_depth=max_depth - 1, _memo=memo)
        # Only support const * expr (either side).
        if not a.non_affine and not a.coeff:
            out = b.mul_const(a.const)
        elif not b.non_affine and not b.coeff:
            out = a.mul_const(b.const)
        else:
            # Common TTIR pattern for flat indexing: pid * N + r, where N is a runtime arg.
            # This is not affine in the strict sense, but we can preserve it as a stable symbolic
            # product term to avoid dropping the entire index map.
            def _monomial(e: AffineExpr) -> Optional[Tuple[int, str]]:
                if e.non_affine or e.const != 0:
                    return None
                if len(e.coeff) != 1:
                    return None
                (v, c), = e.coeff.items()
                if not isinstance(c, int) or c == 0:
                    return None
                return int(c), str(v)

            def _stable_symbol(v: str) -> bool:
                # Allow %argN (function args) but reject transient SSA temps like %17.
                return (not v.startswith("%")) or v.startswith("%arg")

            am = _monomial(a)
            bm = _monomial(b)
            if am is not None and bm is not None and _stable_symbol(am[1]) and _stable_symbol(bm[1]):
                ca, va = am
                cb, vb = bm
                out = AffineExpr(const=0, coeff={f"mul({va},{vb})": ca * cb})
            else:
                out = AffineExpr(non_affine=True)
        memo[name] = out
        return out

    # Default: symbolic term
    out = sym(name)
    memo[name] = out
    return out


def format_affine(expr: AffineExpr) -> str:
    if expr.non_affine:
        return "<non-affine>"
    parts = []
    for var, c in sorted(expr.coeff.items()):
        if c == 1:
            parts.append(f"{var}")
        elif c == -1:
            parts.append(f"-{var}")
        else:
            parts.append(f"{c}*{var}")
    if expr.const != 0 or not parts:
        parts.append(str(expr.const))
    return " + ".join(parts).replace("+ -", "- ")


def expr_to_str(
    name: str,
    defs: Dict[str, SSAValueDef],
    *,
    aliases: Optional[Dict[str, str]] = None,
    max_depth: int = 64,
    _memo: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build a symbolic expression string for an SSA value.

    Unlike `affine_from_ssa`, this does not require constant coefficients; it is
    purely a best-effort pretty-printer for address/index witnesses.
    """
    aliases = aliases or {}
    memo = _memo if _memo is not None else {}
    if name in memo:
        return memo[name]
    if max_depth <= 0:
        out = aliases.get(name, name)
        memo[name] = out
        return out
    d = defs.get(name)
    if d is None:
        out = aliases.get(name, name)
        memo[name] = out
        return out

    # scalar integer constants
    m = INT_CONST_RE.search(d.line)
    if m:
        out = m.group(1)
        memo[name] = out
        return out
    m = INT_DENSE_CONST_RE.search(d.line)
    if m:
        out = m.group(1)
        memo[name] = out
        return out

    # Common identity/shape ops
    if d.op in {"arith.extsi", "arith.extui", "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape"} and d.operands:
        out = expr_to_str(d.operands[0], defs, aliases=aliases, max_depth=max_depth - 1, _memo=memo)
        memo[name] = out
        return out

    if d.op in {"arith.addi", "arith.subi", "arith.muli"} and len(d.operands) >= 2:
        a = expr_to_str(d.operands[0], defs, aliases=aliases, max_depth=max_depth - 1, _memo=memo)
        b = expr_to_str(d.operands[1], defs, aliases=aliases, max_depth=max_depth - 1, _memo=memo)
        op = {"arith.addi": "+", "arith.subi": "-", "arith.muli": "*"}[d.op]
        out = f"({a} {op} {b})"
        memo[name] = out
        return out

    out = aliases.get(name, name)
    memo[name] = out
    return out


__all__ = ["AffineExpr", "build_aliases", "affine_from_ssa", "format_affine", "expr_to_str"]
