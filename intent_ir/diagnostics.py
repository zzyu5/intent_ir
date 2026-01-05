"""
Diagnostic utilities for IntentIR validation and pipeline error reporting.

This is intentionally lightweight (no color dependencies) but supports:
  - structured diagnostics with optional location
  - suggestions/notes/hints
  - rich multi-line formatting (Clang-like)
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Literal, Optional


Level = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class IRLocation:
    op_index: Optional[int] = None
    input_index: Optional[int] = None
    output: Optional[str] = None
    tensor: Optional[str] = None


@dataclass
class Diagnostic:
    level: Level
    message: str
    location: Optional[IRLocation] = None
    suggestions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    related: List["Diagnostic"] = field(default_factory=list)


class DiagnosticEngine:
    def __init__(self) -> None:
        self.items: List[Diagnostic] = []

    def emit(self, diag: Diagnostic) -> None:
        self.items.append(diag)

    def format_rich(self, diag: Diagnostic, *, snippet: str | None = None, caret: str | None = None) -> str:
        lines: List[str] = []
        head = f"{diag.level.upper()}: {diag.message}"
        lines.append(head)
        if snippet:
            lines.append(f"  -> {snippet}")
            if caret:
                lines.append(f"     {caret}")
        for n in diag.notes:
            lines.append(f"Note: {n}")
        for s in diag.suggestions:
            lines.append(f"Hint: {s}")
        for r in diag.related:
            lines.append("")
            lines.append(self.format_rich(r))
        return "\n".join(lines)


def closest_match(name: str, candidates: Iterable[str], *, n: int = 1) -> List[str]:
    try:
        return list(difflib.get_close_matches(str(name), list(candidates), n=n, cutoff=0.6))
    except Exception:
        return []


def format_op_snippet(op: Any, *, idx: int | None = None) -> str:
    """
    Best-effort human-readable op string.
    """
    try:
        op_name = getattr(op, "op", None) or op.get("op")
        inputs = list(getattr(op, "inputs", None) or op.get("inputs") or [])
        out = getattr(op, "output", None) or op.get("output")
        prefix = f"op[{idx}]: " if isinstance(idx, int) else ""
        return f"{prefix}{op_name}({', '.join(inputs)}) -> {out}"
    except Exception:
        return f"op[{idx}]" if isinstance(idx, int) else "op"


__all__ = [
    "IRLocation",
    "Diagnostic",
    "DiagnosticEngine",
    "closest_match",
    "format_op_snippet",
]

