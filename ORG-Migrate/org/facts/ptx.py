from __future__ import annotations

import re
from typing import Any


_REQNTID_RE = re.compile(r"\.reqntid\s+([0-9]+)(?:\s*,\s*([0-9]+))?(?:\s*,\s*([0-9]+))?")
_SHARED_RE = re.compile(r"(?:^|\s)\.(?:extern\s+)?shared\b")
_CP_ASYNC_RE = re.compile(r"\bcp\.async(?:\.[A-Za-z0-9_:]+)?\b")
_CP_ASYNC_COMMIT_RE = re.compile(r"\bcp\.async\.commit_group\b")
_CP_ASYNC_WAIT_RE = re.compile(r"\bcp\.async\.wait_group(?:\s+([0-9]+))?\b")
_MMA_SYNC_RE = re.compile(r"\bmma\.sync(?:\.aligned)?(?:\.m[0-9]+n[0-9]+k[0-9]+)?")
_WGMMA_RE = re.compile(r"\bwgmma\.[A-Za-z0-9_.]+")
_LDMATRIX_RE = re.compile(r"\bldmatrix(?:\.sync)?(?:\.aligned)?(?:\.x([124]))?")
_SHFL_RE = re.compile(r"\bshfl\.sync\.([A-Za-z0-9_]+)")
_BAR_SYNC_RE = re.compile(r"\bbar\.sync\b")


def _make_evidence(*, item_id: str, kind: str, artifact_path: str | None, summary: str) -> dict[str, Any]:
    return {
        "id": str(item_id),
        "kind": str(kind),
        "path": (str(artifact_path) if artifact_path else ""),
        "summary": str(summary),
    }


def _append_evidence(
    evidence: list[dict[str, Any]],
    *,
    item_id: str,
    kind: str,
    artifact_path: str | None,
    summary: str,
) -> list[str]:
    evidence.append(_make_evidence(item_id=item_id, kind=kind, artifact_path=artifact_path, summary=summary))
    return [str(item_id)]


def extract_ptx_mechanism_facts(ptx_text: str | None, *, kernel_name: str, artifact_path: str | None = None) -> dict[str, Any]:
    text = str(ptx_text or "")
    has_text = bool(text.strip())
    evidence: list[dict[str, Any]] = []

    reqntid_match = _REQNTID_RE.search(text)
    reqntid = []
    if reqntid_match is not None:
        reqntid = [int(x) for x in reqntid_match.groups() if x is not None]
    threads_per_block = 1
    for value in reqntid or []:
        threads_per_block *= int(value)

    shared_hits = list(_SHARED_RE.finditer(text))
    async_hits = list(_CP_ASYNC_RE.finditer(text))
    async_commit_hits = list(_CP_ASYNC_COMMIT_RE.finditer(text))
    async_wait_hits = list(_CP_ASYNC_WAIT_RE.finditer(text))
    mma_hits = list(_MMA_SYNC_RE.finditer(text))
    wgmma_hits = list(_WGMMA_RE.finditer(text))
    ldmatrix_hits = list(_LDMATRIX_RE.finditer(text))
    shfl_hits = list(_SHFL_RE.finditer(text))
    bar_sync_hits = list(_BAR_SYNC_RE.finditer(text))

    shared_refs = _append_evidence(
        evidence,
        item_id="ptx_shared_memory",
        kind="ptx_pattern",
        artifact_path=artifact_path,
        summary=f"shared-memory declarations={len(shared_hits)}",
    ) if shared_hits else []

    mapping_refs = _append_evidence(
        evidence,
        item_id="ptx_reqntid",
        kind="ptx_pattern",
        artifact_path=artifact_path,
        summary=f"reqntid={reqntid or []}",
    ) if reqntid else []

    async_refs = _append_evidence(
        evidence,
        item_id="ptx_cp_async",
        kind="ptx_pattern",
        artifact_path=artifact_path,
        summary=f"cp.async count={len(async_hits)}, commit_group={len(async_commit_hits)}, wait_group={len(async_wait_hits)}",
    ) if async_hits else []

    mma_refs = _append_evidence(
        evidence,
        item_id="ptx_mma",
        kind="ptx_pattern",
        artifact_path=artifact_path,
        summary=f"mma_sync={len(mma_hits)}, wgmma={len(wgmma_hits)}",
    ) if (mma_hits or wgmma_hits) else []

    ldmatrix_refs = _append_evidence(
        evidence,
        item_id="ptx_ldmatrix",
        kind="ptx_pattern",
        artifact_path=artifact_path,
        summary=f"ldmatrix count={len(ldmatrix_hits)}",
    ) if ldmatrix_hits else []

    shuffle_ops = sorted({str(m.group(1) or "").strip() for m in shfl_hits if str(m.group(1) or "").strip()})
    shuffle_refs = _append_evidence(
        evidence,
        item_id="ptx_shuffle",
        kind="ptx_pattern",
        artifact_path=artifact_path,
        summary=f"shuffle count={len(shfl_hits)}, ops={shuffle_ops}",
    ) if shfl_hits else []

    block_sync_refs = _append_evidence(
        evidence,
        item_id="ptx_block_sync",
        kind="ptx_pattern",
        artifact_path=artifact_path,
        summary=f"bar.sync count={len(bar_sync_hits)}",
    ) if bar_sync_hits else []

    wait_groups = sorted(
        {
            int(m.group(1))
            for m in async_wait_hits
            if m.group(1) is not None and str(m.group(1)).strip()
        }
    )
    mma_kinds = sorted({str(m.group(0) or "").strip() for m in mma_hits if str(m.group(0) or "").strip()})[:8]
    wgmma_kinds = sorted({str(m.group(0) or "").strip() for m in wgmma_hits if str(m.group(0) or "").strip()})[:8]
    ldmatrix_widths = sorted({int(m.group(1)) for m in ldmatrix_hits if m.group(1) is not None})
    has_complete_async_pipeline = bool(async_hits and async_commit_hits and async_wait_hits)
    has_complete_matrix_pipeline = bool((mma_hits or wgmma_hits) and ldmatrix_hits)
    has_complete_reduction_pattern = bool(shfl_hits and bar_sync_hits)

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
            "staging.shared_memory": {
                "present": bool(shared_hits),
                "attrs": {"declaration_count": int(len(shared_hits))},
                "evidence_refs": shared_refs,
            },
            "mapping.block_threads": {
                "present": bool(reqntid),
                "attrs": {
                    "reqntid": list(reqntid),
                    "threads_per_block": int(threads_per_block if reqntid else 0),
                    "warp_count_estimate": (int(threads_per_block // 32) if threads_per_block > 0 else 0),
                },
                "evidence_refs": mapping_refs,
            },
            "pipeline.async_copy": {
                "present": bool(async_hits),
                "attrs": {
                    "async_copy_count": int(len(async_hits)),
                    "commit_group_count": int(len(async_commit_hits)),
                    "wait_group_count": int(len(async_wait_hits)),
                    "wait_groups": list(wait_groups),
                    "complete_async_pipeline": bool(has_complete_async_pipeline),
                },
                "evidence_refs": async_refs,
            },
            "primitive.mma": {
                "present": bool(mma_hits or wgmma_hits),
                "attrs": {
                    "mma_sync_count": int(len(mma_hits)),
                    "wgmma_count": int(len(wgmma_hits)),
                    "mma_kinds": list(mma_kinds),
                    "wgmma_kinds": list(wgmma_kinds),
                    "complete_matrix_pipeline": bool(has_complete_matrix_pipeline),
                },
                "evidence_refs": mma_refs,
            },
            "primitive.matrix_load": {
                "present": bool(ldmatrix_hits),
                "attrs": {
                    "ldmatrix_count": int(len(ldmatrix_hits)),
                    "ldmatrix_widths": list(ldmatrix_widths),
                },
                "evidence_refs": ldmatrix_refs,
            },
            "communication.shuffle": {
                "present": bool(shfl_hits),
                "attrs": {
                    "shuffle_count": int(len(shfl_hits)),
                    "shuffle_ops": list(shuffle_ops),
                    "complete_reduction_pattern": bool(has_complete_reduction_pattern),
                },
                "evidence_refs": shuffle_refs,
            },
            "communication.block_sync": {
                "present": bool(bar_sync_hits),
                "attrs": {
                    "bar_sync_count": int(len(bar_sync_hits)),
                },
                "evidence_refs": block_sync_refs,
            },
        },
        "evidence": evidence,
    }


__all__ = ["extract_ptx_mechanism_facts"]
