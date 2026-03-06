from __future__ import annotations

from typing import Any


def extract_ptx_mechanism_facts(ptx_text: str | None, *, kernel_name: str, artifact_path: str | None = None) -> dict[str, Any]:
    text = str(ptx_text or "")
    has_text = bool(text.strip())
    has_cp_async = "cp.async" in text
    has_ldmatrix = "ldmatrix" in text
    has_shfl = "shfl.sync" in text or "shfl." in text
    return {
        "schema_version": "org_mechanism_facts_v1",
        "source": {
            "frontend": "triton",
            "kernel": str(kernel_name),
            "artifact_paths": {"ptx": (str(artifact_path) if artifact_path else None)},
        },
        "artifacts": {
            "ttir_available": False,
            "ttgir_available": False,
            "ptx_available": bool(has_text),
        },
        "mechanisms": {
            "overlap_pipeline.async_copy": {
                "present": bool(has_cp_async),
                "attrs": {"opcode": "cp.async" if has_cp_async else ""},
                "evidence_refs": [],
            },
            "special_primitive.matrix_load": {
                "present": bool(has_ldmatrix),
                "attrs": {"opcode": "ldmatrix" if has_ldmatrix else ""},
                "evidence_refs": [],
            },
            "communication.shuffle": {
                "present": bool(has_shfl),
                "attrs": {"opcode": "shfl" if has_shfl else ""},
                "evidence_refs": [],
            },
        },
        "evidence": [],
    }


__all__ = ["extract_ptx_mechanism_facts"]
