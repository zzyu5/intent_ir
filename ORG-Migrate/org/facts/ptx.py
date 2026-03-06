from __future__ import annotations

from typing import Any


def _make_evidence(*, item_id: str, kind: str, artifact_path: str | None, summary: str) -> dict[str, Any]:
    return {
        "id": str(item_id),
        "kind": str(kind),
        "path": (str(artifact_path) if artifact_path else ""),
        "summary": str(summary),
    }


def extract_ptx_mechanism_facts(ptx_text: str | None, *, kernel_name: str, artifact_path: str | None = None) -> dict[str, Any]:
    text = str(ptx_text or "")
    has_text = bool(text.strip())
    evidence: list[dict[str, Any]] = []

    def _present(marker: str, item_id: str, summary: str) -> tuple[bool, list[str]]:
        if marker in text:
            evidence.append(_make_evidence(item_id=item_id, kind="ptx_opcode", artifact_path=artifact_path, summary=summary))
            return True, [item_id]
        return False, []

    has_async, async_refs = _present("cp.async", "ptx_cp_async", "PTX contains cp.async")
    has_mma, mma_refs = _present("mma.sync", "ptx_mma_sync", "PTX contains mma.sync")
    has_ldmatrix, ldmatrix_refs = _present("ldmatrix", "ptx_ldmatrix", "PTX contains ldmatrix")
    has_shuffle, shuffle_refs = _present("shfl.sync", "ptx_shfl_sync", "PTX contains shfl.sync")
    has_block_sync, block_sync_refs = _present("bar.sync", "ptx_bar_sync", "PTX contains bar.sync")

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
            "pipeline.async_copy": {"present": bool(has_async), "attrs": {}, "evidence_refs": async_refs},
            "primitive.mma": {"present": bool(has_mma), "attrs": {}, "evidence_refs": mma_refs},
            "primitive.matrix_load": {"present": bool(has_ldmatrix), "attrs": {}, "evidence_refs": ldmatrix_refs},
            "communication.shuffle": {"present": bool(has_shuffle), "attrs": {}, "evidence_refs": shuffle_refs},
            "communication.block_sync": {"present": bool(has_block_sync), "attrs": {}, "evidence_refs": block_sync_refs},
        },
        "evidence": evidence,
    }


__all__ = ["extract_ptx_mechanism_facts"]
