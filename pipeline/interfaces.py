"""
Cross-module interfaces shared by pipeline/frontends/verify/backends.

Keep this module dependency-light (no torch/triton) so it can be imported from
core logic without pulling heavy runtime requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FrontendConstraints:
    """
    Minimal frontend-derived constraints used by Task5 case generation.

    The Triton frontend currently extracts these from TTIR, but other frontends
    (CUDA C / TileLang) can populate the same structure from their own IR dumps.
    """

    needs_mask: bool = False
    suggested_edge_cases: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


__all__ = ["FrontendConstraints"]

