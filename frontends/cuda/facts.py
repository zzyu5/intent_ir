"""
CUDA facts extraction from PTX (MVP).

This module translates PTX text into a TileLang-like Facts object:
- anchors (has_barrier/has_atomic/has_dot/...)
- CanonicalEvidence-style AccessSummary list (global ld/st)
- schedule-level domains (symbol_ranges) and tile hints (block dims)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.interfaces import KernelDescriptor

from frontends.common.evidence import AccessSummary, sort_accesses

from .ptx import parse_ptx_kernel
from .signature import infer_runtime_io_spec


@dataclass
class CudaFacts:
    schema_version: str
    anchors: Dict[str, Any]
    accesses: List[AccessSummary] = field(default_factory=list)
    symbol_ranges: Dict[str, Dict[str, int]] = field(default_factory=dict)
    tile_hints: List[int] = field(default_factory=list)
    predicate_clauses: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


def _load_ptx_text(desc: KernelDescriptor) -> str:
    if desc.artifacts.ptx_text:
        return str(desc.artifacts.ptx_text)
    extra = getattr(desc.artifacts, "extra", None) or {}
    p = extra.get("ptx_path") if isinstance(extra, dict) else None
    if isinstance(p, str) and p:
        pp = Path(p)
        if pp.is_file():
            return pp.read_text(encoding="utf-8")
    # Fall back to a copied file in artifact_dir if present.
    ad = desc.meta.get("artifact_dir")
    if isinstance(ad, str) and ad:
        pp = Path(ad) / f"{desc.name}.ptx"
        if pp.is_file():
            return pp.read_text(encoding="utf-8")
    raise FileNotFoundError("PTX not available in KernelDescriptor artifacts/meta")


def extract_facts(desc: KernelDescriptor, *, ptx_text: Optional[str] = None) -> CudaFacts:
    ptx = str(ptx_text) if ptx_text is not None else _load_ptx_text(desc)
    entry = desc.meta.get("ptx_entry") if isinstance(getattr(desc, "meta", None), dict) else None
    kernel_entry = str(entry) if isinstance(entry, str) and entry.strip() else str(desc.name)
    io_spec = dict(desc.io_spec or {})
    # The PTX param order follows the CUDA kernel signature, which may differ
    # from the higher-level "semantic" io_spec order (TileLang exports reorder).
    # Use signature parsing to build a runtime io_spec for correct param mapping.
    try:
        io_spec = infer_runtime_io_spec(cuda_src=str(desc.source_text), kernel_name=kernel_entry, semantic_io_spec=io_spec)
    except Exception:
        io_spec = dict(desc.io_spec or {})

    parsed = parse_ptx_kernel(ptx, kernel_name=kernel_entry, io_spec=io_spec, launch=dict(desc.launch or {}))
    accesses = sort_accesses(list(parsed.accesses or []))
    return CudaFacts(
        schema_version="cuda_facts_v0.1",
        anchors=dict(parsed.anchors),
        accesses=accesses,
        symbol_ranges=dict(parsed.symbol_ranges),
        tile_hints=list(parsed.tile_hints),
        predicate_clauses=list(parsed.predicate_clauses),
        raw=dict(parsed.raw),
    )


__all__ = ["CudaFacts", "extract_facts"]
