from __future__ import annotations

import re
from typing import Any


_BLOCKED_RE = re.compile(
    r"^(#\w+)\s*=\s*#ttg\.blocked<\{sizePerThread = \[([^\]]*)\], threadsPerWarp = \[([^\]]*)\], warpsPerCTA = \[([^\]]*)\], order = \[([^\]]*)\]\}>"
)
_PROGRAM_ID_RE = re.compile(r"\btt\.get_program_id\s+([xyz])\b")
_MODULE_WARPS_RE = re.compile(r'"ttg\.num-warps"\s*=\s*([0-9]+)\s*:\s*i32')
_THREADS_PER_WARP_RE = re.compile(r'"ttg\.threads-per-warp"\s*=\s*([0-9]+)\s*:\s*i32')
_BLOCKED_TILE_LOAD_RE = re.compile(r"tt\.load .*tensor<\d+x\d+x!tt\.ptr<[^>]+>, #\w+>")
_SHARED_RE = re.compile(r"#ttg\.(shared|swizzled_shared|shared_memory)")


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


def _evidence_ref(*, kind: str, artifact_path: str | None, line_no: int, text: str) -> dict[str, Any]:
    suffix = f":{int(line_no)}" if line_no > 0 else ""
    return {
        "kind": str(kind),
        "path": f"{artifact_path}{suffix}" if artifact_path else suffix.lstrip(":"),
        "text": str(text).strip(),
    }


def extract_ttgir_mechanism_facts(ttgir_text: str, *, kernel_name: str, artifact_path: str | None = None) -> dict[str, Any]:
    text = str(ttgir_text or "")
    lines = text.splitlines()
    evidence: list[dict[str, Any]] = []

    blocked_layouts: list[dict[str, Any]] = []
    program_axes: list[str] = []
    stage_hint: int | None = None
    num_warps: int | None = None
    threads_per_warp: int | None = None
    has_shared = False
    has_local_staging = False

    for idx, line in enumerate(lines, start=1):
        m = _BLOCKED_RE.match(line.strip())
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
            evidence.append(_evidence_ref(kind="ttgir_line", artifact_path=artifact_path, line_no=idx, text=line))
            continue
        if _SHARED_RE.search(line):
            has_shared = True
            evidence.append(_evidence_ref(kind="ttgir_line", artifact_path=artifact_path, line_no=idx, text=line))
        for axis in _PROGRAM_ID_RE.findall(line):
            ax = str(axis).strip()
            if ax and ax not in program_axes:
                program_axes.append(ax)
                evidence.append(_evidence_ref(kind="ttgir_line", artifact_path=artifact_path, line_no=idx, text=line))
        if num_warps is None:
            m = _MODULE_WARPS_RE.search(line)
            if m is not None:
                try:
                    num_warps = int(m.group(1))
                except Exception:
                    num_warps = None
                evidence.append(_evidence_ref(kind="ttgir_line", artifact_path=artifact_path, line_no=idx, text=line))
        if threads_per_warp is None:
            m = _THREADS_PER_WARP_RE.search(line)
            if m is not None:
                try:
                    threads_per_warp = int(m.group(1))
                except Exception:
                    threads_per_warp = None
                evidence.append(_evidence_ref(kind="ttgir_line", artifact_path=artifact_path, line_no=idx, text=line))
        if not has_local_staging and _BLOCKED_TILE_LOAD_RE.search(line):
            has_local_staging = True
            evidence.append(_evidence_ref(kind="ttgir_line", artifact_path=artifact_path, line_no=idx, text=line))
        if stage_hint is None:
            m = re.search(r"num_stages\s*=\s*([0-9]+)", line)
            if m is not None:
                try:
                    stage_hint = int(m.group(1))
                except Exception:
                    stage_hint = None
                evidence.append(_evidence_ref(kind="ttgir_line", artifact_path=artifact_path, line_no=idx, text=line))

    staging_key = "staging.shared_staging" if has_shared else "staging.local_staging"
    staging_attrs = {"storage": ("shared" if has_shared else "local_blocked_tile"), "heuristic": (not has_shared)}
    warp_layout = blocked_layouts[0].get("threads_per_warp_layout") if blocked_layouts else []
    warps_per_cta = blocked_layouts[0].get("warps_per_cta") if blocked_layouts else []
    order = blocked_layouts[0].get("order") if blocked_layouts else []

    mechanisms = {
        "tiling.blocked_layout": {
            "present": bool(blocked_layouts),
            "attrs": {"layouts": list(blocked_layouts)},
            "evidence_refs": evidence[:2],
        },
        staging_key: {
            "present": bool(has_shared or has_local_staging),
            "attrs": staging_attrs,
            "evidence_refs": evidence[:3],
        },
        "parallel_mapping.program_axes": {
            "present": bool(program_axes),
            "attrs": {"axes": list(program_axes)},
            "evidence_refs": [e for e in evidence if "tt.get_program_id" in str(e.get("text") or "")][:3],
        },
        "parallel_mapping.warp_or_subgroup": {
            "present": bool((num_warps is not None) or warp_layout or warps_per_cta),
            "attrs": {
                "num_warps": num_warps,
                "threads_per_warp": threads_per_warp,
                "threads_per_warp_layout": list(warp_layout),
                "warps_per_cta": list(warps_per_cta),
                "order": list(order),
            },
            "evidence_refs": [e for e in evidence if "ttg.num-warps" in str(e.get("text") or "") or "#ttg.blocked" in str(e.get("text") or "")][:3],
        },
        "overlap_pipeline.stage_hint": {
            "present": stage_hint is not None,
            "attrs": {"stage_hint": stage_hint, "reason": ("" if stage_hint is not None else "no explicit TTGIR stage marker found")},
            "evidence_refs": [e for e in evidence if "num_stages" in str(e.get("text") or "")][:2],
        },
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
