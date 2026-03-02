from __future__ import annotations

import os
from typing import Literal

EvidenceMode = Literal["on", "off"]


def evidence_mode() -> EvidenceMode:
    """
    Evidence mode controls how much intermediate compiler evidence we persist.

    - on  (default): keep full MLIR/LLVM shadow artifacts and pass traces.
    - off           : keep only minimal, executable/audit artifacts (contracts + PTX/ELF),
                      and avoid large intermediate dumps.
    """

    raw = os.getenv("INTENTIR_EVIDENCE_MODE", "on")
    s = str(raw or "").strip().lower()
    if not s:
        return "on"
    if s in {"0", "false", "no", "off"}:
        return "off"
    if s in {"1", "true", "yes", "on"}:
        return "on"
    # Be conservative: unknown values keep evidence enabled.
    return "on"


def evidence_enabled() -> bool:
    return evidence_mode() != "off"


__all__ = ["EvidenceMode", "evidence_mode", "evidence_enabled"]
