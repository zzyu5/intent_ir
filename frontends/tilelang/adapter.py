"""
TileLang FrontendAdapter (PR#9): MVP second frontend.

This adapter treats the TileLang kernel source as a structured JSON DSL and
builds CertificateV2 directly.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from pipeline import registry
from pipeline.interfaces import KernelDescriptor

from .facts import TileLangFacts, extract_facts
from .constraints import extract_constraints
from .certificate import build_certificate_v2


def _safe_versions() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tilelang": "mvp",
    }


class TileLangAdapter:
    name = "tilelang"

    def build_descriptor(self, kernel: Any) -> KernelDescriptor:
        spec = kernel
        desc = KernelDescriptor(
            schema_version="kernel_desc_v1.0",
            name=str(getattr(spec, "name", "tilelang_kernel")),
            frontend="tilelang",
            source_kind="dsl",
            source_text=str(getattr(spec, "source_text", "")),
        )
        desc.launch = {
            "canonical_shapes": dict(getattr(spec, "canonical_shapes", {}) or {}),
            "vary_axes": list(getattr(spec, "vary_axes", []) or []),
            "exclude_axes": list(getattr(spec, "exclude_axes", []) or []),
        }
        desc.io_spec = {
            "arg_names": list(getattr(spec, "arg_names", []) or []),
            "constexpr_names": list(getattr(spec, "constexpr_names", []) or []),
        }
        desc.meta = _safe_versions()
        return desc

    def ensure_artifacts(self, desc: KernelDescriptor, _kernel: Any) -> KernelDescriptor:
        artifact_dir_raw = desc.meta.get("artifact_dir")
        if not artifact_dir_raw:
            return desc
        artifact_dir = Path(str(artifact_dir_raw))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            out_path = artifact_dir / f"{desc.name}.descriptor.json"
            out_path.write_text(json.dumps(desc.to_json_dict(), indent=2), encoding="utf-8")
            desc.meta["descriptor_path"] = str(out_path)
        except Exception:
            pass
        return desc

    def extract_facts(self, desc: KernelDescriptor) -> TileLangFacts:
        facts = extract_facts(desc.source_text)
        desc.frontend_facts = {
            "anchors": dict(facts.anchors),
            "num_accesses": int(len(facts.accesses)),
            "schema_version": str(facts.schema_version),
        }
        return facts

    def extract_constraints(self, desc: KernelDescriptor, facts: TileLangFacts):
        c = extract_constraints(desc.source_text, facts)
        desc.frontend_constraints = asdict(c)
        return c

    def build_certificate(self, desc: KernelDescriptor, facts: TileLangFacts, _constraints=None):
        return build_certificate_v2(facts, desc=desc)

    def evaluate_contract(self, _facts: Any, _constraints=None, _cert=None):
        # Contract V2 needs obligations + KernelDescriptor; evaluated by the pipeline.
        return None


registry.register(TileLangAdapter())

__all__ = ["TileLangAdapter"]

