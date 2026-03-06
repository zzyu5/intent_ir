from __future__ import annotations

from typing import Any

from pipeline.interfaces import KernelDescriptor


def build_ttir_summary(descriptor: KernelDescriptor | None) -> dict[str, Any]:
    desc = descriptor
    if desc is None:
        return {
            "schema_version": "org_ttir_summary_v1",
            "available": False,
            "reason": "descriptor_missing",
            "facts": {},
            "constraints": {},
        }
    return {
        "schema_version": "org_ttir_summary_v1",
        "available": bool(getattr(desc.artifacts, "ttir_path", None) or getattr(desc.artifacts, "ttir_text", None)),
        "facts": dict(getattr(desc, "frontend_facts", {}) or {}),
        "constraints": dict(getattr(desc, "frontend_constraints", {}) or {}),
    }


__all__ = ["build_ttir_summary"]
