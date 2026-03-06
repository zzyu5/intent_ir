from __future__ import annotations

import re
from typing import Any


_BLOCKED_RE = re.compile(
    r"^(#\w+)\s*=\s*#ttg\.blocked<\{sizePerThread = \[([^\]]*)\], threadsPerWarp = \[([^\]]*)\], warpsPerCTA = \[([^\]]*)\], order = \[([^\]]*)\]\}>"
)
_PROGRAM_ID_RE = re.compile(r"\btt\.get_program_id\s+([xyz])\b")
_MODULE_WARPS_RE = re.compile(r'"ttg\.num-warps"\s*=\s*([0-9]+)\s*:\s*i32')
_THREADS_PER_WARP_RE = re.compile(r'"ttg\.threads-per-warp"\s*=\s*([0-9]+)\s*:\s*i32')


def _split_ints(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw or "").split(","):
        s = str(part).strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except Exception:
            continue
    return out


def _evidence(*, item_id: str, kind: str, artifact_path: str | None, line_no: int, summary: str) -> dict[str, Any]:
    suffix = f":{int(line_no)}" if line_no > 0 else ""
    return {
        "id": str(item_id),
        "kind": str(kind),
        "path": f"{artifact_path}{suffix}" if artifact_path else suffix.lstrip(":"),
        "summary": str(summary),
    }


def extract_ttgir_mechanism_facts(ttgir_text: str, *, kernel_name: str, artifact_path: str | None = None) -> dict[str, Any]:
    text = str(ttgir_text or "")
    lines = text.splitlines()
    evidence: list[dict[str, Any]] = []
    blocked_layouts: list[dict[str, Any]] = []
    program_axes: list[str] = []
    num_warps: int | None = None
    threads_per_warp: int | None = None
    has_tile_load = False
    has_reduce = False
    has_convert_layout = False
    has_dot = False
    has_shared_like = False

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        m = _BLOCKED_RE.match(stripped)
        if m is not None:
            blocked_layouts.append(
                {
                    "symbol": str(m.group(1)),
                    "size_per_thread": _split_ints(m.group(2)),
                    "threads_per_warp_layout": _split_ints(m.group(3)),
                    "warps_per_cta": _split_ints(m.group(4)),
                    "order": _split_ints(m.group(5)),
                }
            )
            evidence.append(_evidence(item_id=f"ttgir_blocked_{len(blocked_layouts)}", kind="ttgir_layout", artifact_path=artifact_path, line_no=idx, summary="blocked layout"))
        if "shared" in stripped.lower():
            has_shared_like = True
            evidence.append(_evidence(item_id=f"ttgir_shared_{idx}", kind="ttgir_storage", artifact_path=artifact_path, line_no=idx, summary="shared/local staging indicator"))
        if "tt.load" in stripped and "tensor<" in stripped:
            has_tile_load = True
            evidence.append(_evidence(item_id=f"ttgir_load_{idx}", kind="ttgir_load", artifact_path=artifact_path, line_no=idx, summary="tile-shaped load"))
        if "\"tt.reduce\"" in stripped or "tt.reduce" in stripped:
            has_reduce = True
            evidence.append(_evidence(item_id=f"ttgir_reduce_{idx}", kind="ttgir_reduce", artifact_path=artifact_path, line_no=idx, summary="reduction op"))
        if "ttg.convert_layout" in stripped:
            has_convert_layout = True
            evidence.append(_evidence(item_id=f"ttgir_convert_layout_{idx}", kind="ttgir_layout", artifact_path=artifact_path, line_no=idx, summary="layout conversion"))
        if "tt.dot" in stripped or "dot " in stripped:
            has_dot = True
            evidence.append(_evidence(item_id=f"ttgir_dot_{idx}", kind="ttgir_dot", artifact_path=artifact_path, line_no=idx, summary="matrix primitive"))
        for axis in _PROGRAM_ID_RE.findall(stripped):
            ax = str(axis).strip()
            if ax and ax not in program_axes:
                program_axes.append(ax)
                evidence.append(_evidence(item_id=f"ttgir_pid_{ax}", kind="ttgir_mapping", artifact_path=artifact_path, line_no=idx, summary=f"program_id axis {ax}"))
        if num_warps is None:
            m = _MODULE_WARPS_RE.search(stripped)
            if m is not None:
                num_warps = int(m.group(1))
                evidence.append(_evidence(item_id="ttgir_num_warps", kind="ttgir_mapping", artifact_path=artifact_path, line_no=idx, summary=f"num warps = {num_warps}"))
        if threads_per_warp is None:
            m = _THREADS_PER_WARP_RE.search(stripped)
            if m is not None:
                threads_per_warp = int(m.group(1))
                evidence.append(_evidence(item_id="ttgir_threads_per_warp", kind="ttgir_mapping", artifact_path=artifact_path, line_no=idx, summary=f"threads per warp = {threads_per_warp}"))

    blocked_refs = [e["id"] for e in evidence if str(e["id"]).startswith("ttgir_blocked_")]
    staging_refs = [e["id"] for e in evidence if str(e["id"]).startswith("ttgir_shared_") or str(e["id"]).startswith("ttgir_load_")]
    mapping_refs = [e["id"] for e in evidence if str(e["id"]).startswith("ttgir_pid_") or str(e["id"]) in {"ttgir_num_warps", "ttgir_threads_per_warp"}]
    reduce_refs = [e["id"] for e in evidence if str(e["id"]).startswith("ttgir_reduce_")]
    layout_refs = [e["id"] for e in evidence if str(e["id"]).startswith("ttgir_convert_layout_")]
    dot_refs = [e["id"] for e in evidence if str(e["id"]).startswith("ttgir_dot_")]

    mechanisms = {
        "tiling.blocked_layout": {
            "present": bool(blocked_layouts),
            "attrs": {"layouts": list(blocked_layouts)},
            "evidence_refs": blocked_refs,
        },
        "staging.local_or_shared": {
            "present": bool(has_shared_like or has_tile_load),
            "attrs": {"shared_like": bool(has_shared_like), "tile_load": bool(has_tile_load)},
            "evidence_refs": staging_refs,
        },
        "mapping.program_axes": {
            "present": bool(program_axes),
            "attrs": {"axes": list(program_axes)},
            "evidence_refs": [x for x in mapping_refs if str(x).startswith("ttgir_pid_")],
        },
        "mapping.warp_or_cta": {
            "present": bool((num_warps is not None) or (threads_per_warp is not None) or blocked_layouts),
            "attrs": {
                "num_warps": num_warps,
                "threads_per_warp": threads_per_warp,
                "warps_per_cta": (blocked_layouts[0].get("warps_per_cta") if blocked_layouts else []),
                "threads_per_warp_layout": (blocked_layouts[0].get("threads_per_warp_layout") if blocked_layouts else []),
            },
            "evidence_refs": mapping_refs,
        },
        "communication.reduction": {
            "present": bool(has_reduce),
            "attrs": {"kind": "tt.reduce" if has_reduce else ""},
            "evidence_refs": reduce_refs,
        },
        "pipeline.stage_hint": {
            "present": False,
            "attrs": {"stage_hint": None},
            "evidence_refs": [],
        },
    }
    if str(kernel_name) == "matmul_fused_epilogue2d":
        mechanisms["primitive.mma"] = {
            "present": bool(has_dot),
            "attrs": {"dot_like": bool(has_dot)},
            "evidence_refs": dot_refs,
        }
        mechanisms["fusion.epilogue_fused_writeback"] = {
            "present": bool(has_convert_layout),
            "attrs": {"convert_layout": bool(has_convert_layout)},
            "evidence_refs": layout_refs,
        }

    return {
        "schema_version": "org_mechanism_facts_v1",
        "source": {
            "frontend": "triton",
            "kernel": str(kernel_name),
            "artifact_paths": {"ttgir": (str(artifact_path) if artifact_path else None)},
        },
        "artifacts": {
            "ttir_available": False,
            "ttgir_available": bool(text.strip()),
            "ptx_available": False,
        },
        "mechanisms": mechanisms,
        "evidence": evidence,
    }


__all__ = ["extract_ttgir_mechanism_facts"]
