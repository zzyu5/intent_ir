"""
Frontend-agnostic evidence + certificate schemas.

These dataclasses define stable cross-frontend payloads used by the pipeline,
verification, and golden tests. Frontend-specific extractors (e.g., Triton TTIR)
should translate their IR facts into these schemas.
"""

from .evidence import AccessSummary, CanonicalEvidence, IndexExpr, Predicate
from .certificate_v2 import ObligationResultV2, SemanticCertificateV2

__all__ = [
    "IndexExpr",
    "Predicate",
    "AccessSummary",
    "CanonicalEvidence",
    "ObligationResultV2",
    "SemanticCertificateV2",
]

