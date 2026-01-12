from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional


_HEX_RE = re.compile(r"^0[xX][0-9a-fA-F]+$")


def parse_int_literal(text: str) -> Optional[int]:
    s = str(text).strip()
    if not s:
        return None
    if _HEX_RE.match(s):
        try:
            return int(s, 16)
        except Exception:
            return None
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        try:
            return int(s, 10)
        except Exception:
            return None
    return None


@dataclass(frozen=True)
class AffineExpr:
    """
    Integer affine expression: const + Σ coeff[var]*var.

    Notes:
    - This is used to recover index expressions for CanonicalEvidence.IndexExpr.
    - We keep it minimal and conservative; if an operation is not provably affine,
      set ok=False and drop terms.
    """

    terms: Dict[str, int] = field(default_factory=dict)
    const: int = 0
    ok: bool = True

    @staticmethod
    def const_val(v: int) -> "AffineExpr":
        return AffineExpr(terms={}, const=int(v), ok=True)

    @staticmethod
    def sym(name: str) -> "AffineExpr":
        return AffineExpr(terms={str(name): 1}, const=0, ok=True)

    def is_const(self) -> bool:
        return bool(self.ok) and (not self.terms)

    def const_int(self) -> Optional[int]:
        return int(self.const) if self.is_const() else None

    def is_single_symbol(self) -> Optional[str]:
        if not self.ok:
            return None
        if self.const != 0:
            return None
        if len(self.terms) != 1:
            return None
        (k, c), = self.terms.items()
        return str(k) if int(c) == 1 else None

    def add(self, other: "AffineExpr") -> "AffineExpr":
        if not (self.ok and other.ok):
            return AffineExpr(ok=False)
        terms = dict(self.terms)
        for k, v in other.terms.items():
            terms[k] = terms.get(k, 0) + int(v)
            if terms[k] == 0:
                terms.pop(k, None)
        return AffineExpr(terms=terms, const=int(self.const) + int(other.const), ok=True)

    def sub(self, other: "AffineExpr") -> "AffineExpr":
        if not (self.ok and other.ok):
            return AffineExpr(ok=False)
        terms = dict(self.terms)
        for k, v in other.terms.items():
            terms[k] = terms.get(k, 0) - int(v)
            if terms[k] == 0:
                terms.pop(k, None)
        return AffineExpr(terms=terms, const=int(self.const) - int(other.const), ok=True)

    def mul_const(self, c: int) -> "AffineExpr":
        if not self.ok:
            return AffineExpr(ok=False)
        c = int(c)
        if c == 0:
            return AffineExpr.const_val(0)
        return AffineExpr(
            terms={k: int(v) * c for k, v in self.terms.items() if int(v) * c != 0},
            const=int(self.const) * c,
            ok=True,
        )


def render_affine(expr: AffineExpr) -> str:
    """
    Render an affine expression into a stable, SMT-friendly string.

    Example: {"pid0":256,"r0":1}, const=0  ->  "256*pid0 + r0"
    """

    if not expr.ok:
        return "<unresolved>"
    parts: list[str] = []
    for k in sorted(expr.terms.keys()):
        c = int(expr.terms[k])
        if c == 0:
            continue
        if c == 1:
            parts.append(str(k))
        elif c == -1:
            parts.append(f"-{k}")
        else:
            parts.append(f"{c}*{k}")
    if int(expr.const) != 0 or not parts:
        parts.append(str(int(expr.const)))
    out = " + ".join(parts)
    return out.replace("+ -", "- ")


__all__ = ["AffineExpr", "parse_int_literal", "render_affine"]

