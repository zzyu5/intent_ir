from __future__ import annotations

import re
from typing import Any


_BLOCKED_RE = re.compile(
    r"^(#\w+)\s*=\s*#ttg\.blocked<\{sizePerThread = \[([^\]]*)\], threadsPerWarp = \[([^\]]*)\], warpsPerCTA = \[([^\]]*)\], order = \[([^\]]*)\]\}>"
)
_PROGRAM_ID_RE = re.compile(r"\btt\.get_program_id\s+([xyz])\b")
_MODULE_WARPS_RE = re.compile(r'"ttg\.num-warps"\s*=\s*([0-9]+)\s*:\s*i32')
_THREADS_PER_WARP_RE = re.compile(r'"ttg\.threads-per-warp"\s*=\s*([0-9]+)\s*:\s*i32')
_SCF_FOR_RE = re.compile(r"\bscf\.for\b")
_TENSOR_2D_SHAPE_RE = re.compile(r"tensor<([0-9]+)x([0-9]+)")


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


def _tensor_2d_bytes(line: str) -> int | None:
    m = _TENSOR_2D_SHAPE_RE.search(str(line or ""))
    if m is None:
        return None
    try:
        return int(m.group(1)) * int(m.group(2)) * 4
    except Exception:
        return None


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
    has_q_resident_state = False
    has_kv_streamed_tiles = False
    has_streaming_softmax = False
    has_operand_tile_stage = False
    has_bias_epilogue = False
    q_resident_bytes_hint = 0
    kv_streamed_bytes_hint = 0
    operand_tile_bytes_hint = 0
    convert_layout_sites = 0
    inside_loop = False

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if _SCF_FOR_RE.search(stripped):
            inside_loop = True
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
            if str(kernel_name) == "flash_attention2d":
                if "%Q_ptr" in stripped and not inside_loop:
                    has_q_resident_state = True
                    q_resident_bytes_hint = max(q_resident_bytes_hint, int(_tensor_2d_bytes(stripped) or 0))
                    evidence.append(_evidence(item_id=f"ttgir_q_resident_{idx}", kind="ttgir_pattern", artifact_path=artifact_path, line_no=idx, summary="Q load before streaming loop"))
                if ("%K_ptr" in stripped or "%V_ptr" in stripped) and inside_loop:
                    has_kv_streamed_tiles = True
                    kv_streamed_bytes_hint = max(kv_streamed_bytes_hint, int(_tensor_2d_bytes(stripped) or 0))
                    evidence.append(_evidence(item_id=f"ttgir_kv_stream_{idx}", kind="ttgir_pattern", artifact_path=artifact_path, line_no=idx, summary="KV load inside streaming loop"))
            if str(kernel_name) == "matmul_fused_epilogue2d":
                if ("%A" in stripped or "%B" in stripped) and ("tt.load" in stripped):
                    has_operand_tile_stage = True
                    operand_tile_bytes_hint = max(operand_tile_bytes_hint, int(_tensor_2d_bytes(stripped) or 0))
                    evidence.append(_evidence(item_id=f"ttgir_operand_stage_{idx}", kind="ttgir_pattern", artifact_path=artifact_path, line_no=idx, summary="operand tile stage"))
        if "\"tt.reduce\"" in stripped or "tt.reduce" in stripped:
            has_reduce = True
            evidence.append(_evidence(item_id=f"ttgir_reduce_{idx}", kind="ttgir_reduce", artifact_path=artifact_path, line_no=idx, summary="reduction op"))
            if str(kernel_name) == "flash_attention2d" and inside_loop:
                has_streaming_softmax = True
                evidence.append(_evidence(item_id=f"ttgir_stream_reduce_{idx}", kind="ttgir_pattern", artifact_path=artifact_path, line_no=idx, summary="loop-carried streaming reduction"))
        if "ttg.convert_layout" in stripped:
            has_convert_layout = True
            convert_layout_sites += 1
            evidence.append(_evidence(item_id=f"ttgir_convert_layout_{idx}", kind="ttgir_layout", artifact_path=artifact_path, line_no=idx, summary="layout conversion"))
        if "tt.dot" in stripped or "dot " in stripped:
            has_dot = True
            evidence.append(_evidence(item_id=f"ttgir_dot_{idx}", kind="ttgir_dot", artifact_path=artifact_path, line_no=idx, summary="matrix primitive"))
        if str(kernel_name) == "matmul_fused_epilogue2d" and "bias" in stripped.lower():
            has_bias_epilogue = True
            evidence.append(_evidence(item_id=f"ttgir_bias_epilogue_{idx}", kind="ttgir_pattern", artifact_path=artifact_path, line_no=idx, summary="bias fused into epilogue"))
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
    q_resident_refs = [e["id"] for e in evidence if str(e["id"]).startswith("ttgir_q_resident_")]
    kv_stream_refs = [e["id"] for e in evidence if str(e["id"]).startswith("ttgir_kv_stream_")]
    stream_reduce_refs = [e["id"] for e in evidence if str(e["id"]).startswith("ttgir_stream_reduce_")]
    operand_stage_refs = [e["id"] for e in evidence if str(e["id"]).startswith("ttgir_operand_stage_")]
    bias_epilogue_refs = [e["id"] for e in evidence if str(e["id"]).startswith("ttgir_bias_epilogue_")]

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
            "attrs": {
                "kind": "tt.reduce" if has_reduce else "",
                "reduction_scope": ("warp" if (num_warps is not None and int(num_warps) <= 4) else "cta"),
            },
            "evidence_refs": reduce_refs,
        },
        "pipeline.stage_hint": {
            "present": bool(has_shared_like and inside_loop and (has_kv_streamed_tiles or has_operand_tile_stage)),
            "attrs": {
                "stage_hint": ("double_buffer_like" if has_shared_like and inside_loop else None),
                "pipeline_depth_hint": (2 if has_shared_like and inside_loop else None),
            },
            "evidence_refs": staging_refs,
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
        mechanisms["staging.operand_tile_stage"] = {
            "present": bool(has_operand_tile_stage or has_tile_load),
            "attrs": {
                "tile_load": bool(has_tile_load),
                "reuse_window": ("k_loop" if has_operand_tile_stage else ""),
                "resident_bytes_hint": int(operand_tile_bytes_hint),
            },
            "evidence_refs": operand_stage_refs or staging_refs,
        }
        mechanisms["primitive.dot_op"] = {
            "present": bool(has_dot),
            "attrs": {
                "dot_like": bool(has_dot),
                "reduction_scope": ("warp" if (num_warps is not None and int(num_warps) <= 4) else "cta"),
            },
            "evidence_refs": dot_refs,
        }
        mechanisms["fusion.bias_fused_epilogue"] = {
            "present": bool(has_bias_epilogue),
            "attrs": {"bias_seen": bool(has_bias_epilogue)},
            "evidence_refs": bias_epilogue_refs,
        }
        mechanisms["layout.output_convert"] = {
            "present": bool(has_convert_layout),
            "attrs": {"convert_layout": bool(has_convert_layout), "layout_convert_sites": int(convert_layout_sites)},
            "evidence_refs": layout_refs,
        }
    if str(kernel_name) == "flash_attention2d":
        mechanisms["staging.q_resident_state"] = {
            "present": bool(has_q_resident_state),
            "attrs": {
                "outside_loop": bool(has_q_resident_state),
                "reuse_window": ("outer_loop" if has_q_resident_state else ""),
                "resident_bytes_hint": int(q_resident_bytes_hint),
            },
            "evidence_refs": q_resident_refs,
        }
        mechanisms["staging.kv_streamed_tiles"] = {
            "present": bool(has_kv_streamed_tiles),
            "attrs": {
                "inside_loop": bool(has_kv_streamed_tiles),
                "reuse_window": ("kv_loop" if has_kv_streamed_tiles else ""),
                "resident_bytes_hint": int(kv_streamed_bytes_hint),
            },
            "evidence_refs": kv_stream_refs,
        }
        mechanisms["communication.streaming_softmax"] = {
            "present": bool(has_streaming_softmax),
            "attrs": {
                "loop_carried_reduce": bool(has_streaming_softmax),
                "reduction_scope": ("warp" if (num_warps is not None and int(num_warps) <= 4) else "cta"),
            },
            "evidence_refs": stream_reduce_refs,
        }
        mechanisms["layout.output_convert"] = {
            "present": bool(has_convert_layout),
            "attrs": {"convert_layout": bool(has_convert_layout), "layout_convert_sites": int(convert_layout_sites)},
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
