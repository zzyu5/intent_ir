"""
Frontend-agnostic evidence + certificate schemas.

These dataclasses define stable cross-frontend payloads used by the pipeline,
verification, and golden tests. Frontend-specific extractors (e.g., Triton TTIR)
should translate their IR facts into these schemas.
"""

from .evidence import AccessSummary, CanonicalEvidence, IndexExpr, Predicate
from .access_witness import EvidenceStrideSummary, build_stride_summary
from .obligations import ObligationResult
from .certificate_v2 import SemanticCertificateV2
from .static_validate import StaticObligation, StaticValidationResult, static_validate
from .smt_o3 import O3Report, check_mask_implies_inbounds

__all__ = [
    "IndexExpr",
    "Predicate",
    "AccessSummary",
    "CanonicalEvidence",
    "EvidenceStrideSummary",
    "build_stride_summary",
    "ObligationResult",
    "SemanticCertificateV2",
    "StaticObligation",
    "StaticValidationResult",
    "static_validate",
    "O3Report",
    "check_mask_implies_inbounds",
]
