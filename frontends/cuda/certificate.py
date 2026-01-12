"""
CUDA CertificateV2 builder (MVP).

We translate CUDA PTX facts into the stable, cross-frontend CertificateV2:
- semantic_facts: anchors + canonical_evidence (access summaries)
- schedule_hints: tile hints (block dims) + symbol ranges (pid/r domains)
"""

from __future__ import annotations

from pipeline.interfaces import KernelDescriptor

from frontends.common.certificate_v2 import SemanticCertificateV2
from frontends.common.evidence import CanonicalEvidence

from .facts import CudaFacts


def build_certificate_v2(facts: CudaFacts, *, desc: KernelDescriptor | None = None) -> SemanticCertificateV2:
    anchors = dict(facts.anchors or {})
    evidence = CanonicalEvidence(
        anchors=dict(anchors),
        accesses=list(facts.accesses or []),
        schedule_hints={},
        meta={"symbols": {"ranges": sorted(_range_symbols_from_accesses(facts.accesses or []))}},
    ).canonicalize()

    cert = SemanticCertificateV2(
        schema_version="cert_v2.0",
        semantic_facts={
            "anchors": anchors,
            "io_spec": (dict(desc.io_spec) if desc is not None else {}),
            "canonical_evidence": evidence,
        },
        schedule_hints={
            "tile_hints": list(facts.tile_hints or []),
            "symbol_ranges": dict(facts.symbol_ranges or {}),
        },
        meta={"cuda_schema": facts.schema_version},
    )
    return cert.canonicalize()


def _range_symbols_from_accesses(accesses) -> list[str]:
    out: set[str] = set()
    for a in accesses:
        for ix in a.index_exprs or []:
            for v in (ix.terms or {}).keys():
                if isinstance(v, str) and v.startswith("r") and v[1:].isdigit():
                    out.add(v)
    return sorted(out)


__all__ = ["build_certificate_v2"]

