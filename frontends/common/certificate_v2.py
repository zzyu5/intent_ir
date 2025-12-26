"""
CertificateV2 (PR#4): stable, cross-frontend semantic certificate.

Key rule: `semantic_facts` should avoid embedding frontend IR line numbers or
op-name strings. Only CanonicalEvidence and stable summaries belong here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .evidence import CanonicalEvidence


@dataclass(frozen=True)
class ObligationResultV2:
    """
    MVP obligation payload: ID + status + witness.

    PR#5 will formalize obligation IDs and evaluation rules.
    """

    id: str
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    witness: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_json_dict(self) -> Dict[str, Any]:
        out = {"id": str(self.id), "status": str(self.status), "witness": dict(self.witness)}
        if self.reason:
            out["reason"] = str(self.reason)
        return out


@dataclass
class SemanticCertificateV2:
    """
    CertificateV2 splits stable `semantic_facts` from drift-allowed `schedule_hints`.
    """

    schema_version: str = "cert_v2.0"
    semantic_facts: Dict[str, Any] = field(default_factory=dict)
    schedule_hints: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def canonicalize(self) -> "SemanticCertificateV2":
        """
        Enforce determinism for golden tests.
        """
        ce = self.semantic_facts.get("canonical_evidence")
        if isinstance(ce, CanonicalEvidence):
            ce.canonicalize()
        return self

    def to_json_dict(self) -> Dict[str, Any]:
        def encode(obj: Any) -> Any:
            if isinstance(obj, CanonicalEvidence):
                return obj.to_json_dict()
            if isinstance(obj, ObligationResultV2):
                return obj.to_json_dict()
            if isinstance(obj, list):
                return [encode(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): encode(v) for k, v in obj.items()}
            return obj

        return {
            "schema_version": str(self.schema_version),
            "semantic_facts": encode(dict(self.semantic_facts)),
            "schedule_hints": encode(dict(self.schedule_hints)),
            "meta": encode(dict(self.meta)),
        }


__all__ = ["ObligationResultV2", "SemanticCertificateV2"]

