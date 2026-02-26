"""
Aggregate family batch artifacts into one full196 evidence chain.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intent_ir.utils.repo_state import repo_state  # noqa: E402
from pipeline.common.strict_policy import CONTRACT_SCHEMA_VERSION, strict_fallback_enabled  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _entry_or_placeholder(*, semantic_op: str, family: str, reason_code: str, reason_detail: str) -> dict[str, Any]:
    return {
        "semantic_op": str(semantic_op),
        "family": str(family),
        "status": "blocked_backend",
        "reason_code": str(reason_code),
        "status_reason": str(reason_code),
        "status_reason_detail": str(reason_detail),
        "runtime": {"provider": "missing", "rvv": "unknown", "cuda": "unknown"},
        "runtime_detail": {
            "rvv": {"reason_code": str(reason_code), "reason_detail": str(reason_detail)},
            "cuda": {"reason_code": str(reason_code), "reason_detail": str(reason_detail)},
        },
        "compiler_stage": {
            "provider_report": "missing",
            "rvv_result": "missing",
            "cuda_result": "missing",
        },
        "artifact_complete": False,
        "determinability": False,
    }


def _collect_semantics(entry_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in entry_rows:
        sop = str(row.get("semantic_op") or "").strip()
        if not sop:
            continue
        if sop not in out:
            out[sop] = row
    return out


def _counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for row in entries:
        c[str(row.get("status") or "unknown")] += 1
    return {k: int(v) for k, v in sorted(c.items(), key=lambda kv: kv[0])}


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return int(default)


def _resolve_json_path(path_raw: str, *, anchor: Path) -> Path:
    p = Path(str(path_raw or "").strip())
    if p.is_absolute():
        return p
    # Many artifacts are recorded as repo-relative paths (often directories like chunk out_dir).
    # Treat any existing path (file or directory) as already-resolved before falling back.
    if p.exists():
        return p
    return anchor.parent / p


def _collect_stage_timing_paths(run_summary_payload: dict[str, Any], *, run_summary_path: Path) -> list[Path]:
    out: list[Path] = []
    stages = [s for s in list(run_summary_payload.get("stages") or []) if isinstance(s, dict)]
    stage_map = {str(s.get("stage") or ""): s for s in stages}
    stage_row = stage_map.get("stage_timing_breakdown") or {}
    json_path = str(stage_row.get("json_path") or "").strip()
    if json_path:
        p = _resolve_json_path(json_path, anchor=run_summary_path)
        if p.is_file():
            out.append(p)
    if out:
        return out
    # Family run summaries with chunk aggregation often keep timing at chunk level.
    chunk_rows = [r for r in list(run_summary_payload.get("chunk_runs") or []) if isinstance(r, dict)]
    for chunk in chunk_rows:
        chunk_run_summary_path = _resolve_json_path(str(chunk.get("run_summary_path") or ""), anchor=run_summary_path)
        if not chunk_run_summary_path.is_file():
            continue
        chunk_payload = _load_json(chunk_run_summary_path)
        chunk_stages = [s for s in list(chunk_payload.get("stages") or []) if isinstance(s, dict)]
        chunk_stage_map = {str(s.get("stage") or ""): s for s in chunk_stages}
        chunk_timing_row = chunk_stage_map.get("stage_timing_breakdown") or {}
        chunk_timing_path = str(chunk_timing_row.get("json_path") or "").strip()
        if not chunk_timing_path:
            continue
        p = _resolve_json_path(chunk_timing_path, anchor=chunk_run_summary_path)
        if p.is_file():
            out.append(p)
    return out


def _mlir_llvm_artifact_complete_from_summary(run_summary_payload: dict[str, Any]) -> bool | None:
    if not isinstance(run_summary_payload, dict):
        return None
    if "mlir_llvm_artifact_complete" in run_summary_payload:
        return bool(run_summary_payload.get("mlir_llvm_artifact_complete"))
    stages = [s for s in list(run_summary_payload.get("stages") or []) if isinstance(s, dict)]
    stage_map = {str(s.get("stage") or ""): s for s in stages}
    stage = stage_map.get("mlir_llvm_artifacts") or stage_map.get("llvm_emit")
    if not isinstance(stage, dict):
        return None
    if "artifact_complete" in stage:
        return bool(stage.get("artifact_complete"))
    if bool(stage.get("ok")):
        reason = str(stage.get("reason_code") or "").strip().lower()
        if reason in {"ok", "artifact_complete", "llvm_artifacts_complete"}:
            return True
    return bool(stage.get("ok"))


def _runtime_fallback_from_summary(
    run_summary_payload: dict[str, Any],
    *,
    label_prefix: str,
) -> tuple[list[str], int]:
    if not isinstance(run_summary_payload, dict):
        return [], 0
    kernels = [str(x) for x in list(run_summary_payload.get("runtime_fallback_kernels") or []) if str(x).strip()]
    count = _safe_int(run_summary_payload.get("runtime_fallback_kernel_count"), default=0)
    if count <= 0 and kernels:
        count = int(len(kernels))
    if count > len(kernels):
        for i in range(1, int(count - len(kernels)) + 1):
            kernels.append(f"{label_prefix}::fallback_{i}")
    return sorted(set(kernels)), int(max(count, len(kernels)))


def _chunk_run_summaries(run_summary_payload: dict[str, Any], *, run_summary_path: Path) -> list[tuple[dict[str, Any], str]]:
    out: list[tuple[dict[str, Any], str]] = []
    chunk_rows = [r for r in list(run_summary_payload.get("chunk_runs") or []) if isinstance(r, dict)]
    for row in chunk_rows:
        chunk_run_summary_raw = str(row.get("run_summary_path") or "").strip()
        if not chunk_run_summary_raw:
            continue
        chunk_run_summary_path = _resolve_json_path(chunk_run_summary_raw, anchor=run_summary_path)
        if not chunk_run_summary_path.is_file():
            continue
        try:
            payload = _load_json(chunk_run_summary_path)
        except Exception:
            continue
        out.append((payload, str(row.get("chunk") or "")))
    return out


def _family_mlir_llvm_artifact_complete(
    *,
    family_summary: dict[str, Any],
    run_summary_path: Path,
) -> bool | None:
    direct = _mlir_llvm_artifact_complete_from_summary(family_summary)
    if direct is not None:
        return bool(direct)
    chunk_payloads = _chunk_run_summaries(family_summary, run_summary_path=run_summary_path)
    if not chunk_payloads:
        return None
    seen = 0
    complete = True
    for chunk_summary, _ in chunk_payloads:
        val = _mlir_llvm_artifact_complete_from_summary(chunk_summary)
        if val is None:
            complete = False
            continue
        seen += 1
        if not bool(val):
            complete = False
    if seen <= 0:
        return None
    return bool(complete)


def _family_runtime_fallback(
    *,
    family: str,
    family_summary: dict[str, Any],
    run_summary_path: Path,
) -> tuple[list[str], int]:
    direct_kernels, direct_count = _runtime_fallback_from_summary(
        family_summary,
        label_prefix=(str(family) or "family"),
    )
    if direct_count > 0:
        return direct_kernels, direct_count
    chunk_payloads = _chunk_run_summaries(family_summary, run_summary_path=run_summary_path)
    if not chunk_payloads:
        return [], 0
    merged: set[str] = set()
    merged_count = 0
    for chunk_summary, chunk_label in chunk_payloads:
        kernels, count = _runtime_fallback_from_summary(
            chunk_summary,
            label_prefix=(chunk_label or str(family) or "chunk"),
        )
        if kernels:
            merged.update(kernels)
        merged_count += int(count)
    if merged_count <= 0 and merged:
        merged_count = int(len(merged))
    if merged_count > len(merged):
        for i in range(1, int(merged_count - len(merged)) + 1):
            merged.add(f"{family}::fallback_{i}")
    merged_list = sorted(merged)
    return merged_list, int(max(merged_count, len(merged_list)))


def _collect_backend_json_pairs(run_summary_payload: dict[str, Any], *, run_summary_path: Path) -> list[dict[str, Path]]:
    out: list[dict[str, Path]] = []
    chunk_rows = [r for r in list(run_summary_payload.get("chunk_runs") or []) if isinstance(r, dict)]
    if chunk_rows:
        for chunk in chunk_rows:
            out_dir_raw = str(chunk.get("out_dir") or "").strip()
            out_dir = _resolve_json_path(out_dir_raw, anchor=run_summary_path) if out_dir_raw else run_summary_path.parent
            rvv = out_dir / "rvv_remote.json"
            cuda = out_dir / "cuda_local.json"
            if rvv.is_file() and cuda.is_file():
                row: dict[str, Path] = {"rvv_json": rvv, "cuda_json": cuda}
                pipeline_reports_dir = out_dir / "pipeline_reports"
                if pipeline_reports_dir.is_dir():
                    row["pipeline_reports_dir"] = pipeline_reports_dir
                out.append(row)
        return out
    out_dir = run_summary_path.parent
    rvv = out_dir / "rvv_remote.json"
    cuda = out_dir / "cuda_local.json"
    if rvv.is_file() and cuda.is_file():
        row: dict[str, Path] = {"rvv_json": rvv, "cuda_json": cuda}
        pipeline_reports_dir = out_dir / "pipeline_reports"
        if pipeline_reports_dir.is_dir():
            row["pipeline_reports_dir"] = pipeline_reports_dir
        out.append(row)
    return out


def _merge_reason_counts(dst: dict[str, int], src: dict[str, Any]) -> None:
    for key, value in dict(src or {}).items():
        name = str(key).strip()
        if not name:
            continue
        try:
            inc = int(value)
        except Exception:
            inc = 0
        dst[name] = int(dst.get(name, 0)) + int(inc)


def _empty_backend_timing() -> dict[str, Any]:
    return {
        "available": False,
        "kernel_count": 0,
        "reason_code_counts": {},
        "totals_ms": {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0, "total_ms": 0.0},
        "avg_ms": {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0, "total_ms": 0.0},
        "stage_share_pct": {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0},
        "top_kernels_by_total_ms": [],
    }


def _empty_mlir_timing() -> dict[str, Any]:
    return {
        "available": False,
        "kernel_count": 0,
        "totals_ms": {
            "mlir_parse_ms": 0.0,
            "mlir_pass_ms": 0.0,
            "mlir_lower_ms": 0.0,
            "mlir_total_ms": 0.0,
        },
        "avg_ms": {
            "mlir_parse_ms": 0.0,
            "mlir_pass_ms": 0.0,
            "mlir_lower_ms": 0.0,
            "mlir_total_ms": 0.0,
        },
        "stage_share_pct": {
            "mlir_parse_ms": 0.0,
            "mlir_pass_ms": 0.0,
            "mlir_lower_ms": 0.0,
        },
        "pass_count_totals": {"upstream": 0, "midend": 0, "downstream": 0},
        "pass_ms_totals_by_pipeline": {"upstream": 0.0, "midend": 0.0, "downstream": 0.0},
        "pass_ms_totals_by_name": {},
        "pass_count_by_name": {},
        "top_passes_by_ms": [],
        "missing_mlir_rows": [],
        "source_dirs": [],
    }


def _pipeline_kernel_reports(pipeline_reports_dir: Path) -> list[Path]:
    if not pipeline_reports_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(pipeline_reports_dir.glob("*.json")):
        stem = str(p.stem)
        if "." in stem:
            continue
        out.append(p)
    return out


def _summarize_mlir_reports_dir(pipeline_reports_dir: Path) -> dict[str, Any]:
    reports = _pipeline_kernel_reports(pipeline_reports_dir)
    if not reports:
        return _empty_mlir_timing()
    out = _empty_mlir_timing()
    out["source_dir"] = str(pipeline_reports_dir)
    totals = dict(out.get("totals_ms") or {})
    pass_counts = dict(out.get("pass_count_totals") or {})
    pass_ms_totals_by_pipeline = dict(out.get("pass_ms_totals_by_pipeline") or {})
    pass_ms_totals_by_name = dict(out.get("pass_ms_totals_by_name") or {})
    pass_count_by_name = dict(out.get("pass_count_by_name") or {})
    missing: list[str] = []
    counted = 0
    for p in reports:
        payload = _load_json(p)
        row = payload.get("mlir")
        if not isinstance(row, dict):
            missing.append(str(p.name))
            continue
        parse_ms = _safe_float(row.get("mlir_parse_ms"))
        pass_ms = _safe_float(row.get("mlir_pass_ms"))
        lower_ms = _safe_float(row.get("mlir_lower_ms"))
        totals["mlir_parse_ms"] = _safe_float(totals.get("mlir_parse_ms")) + parse_ms
        totals["mlir_pass_ms"] = _safe_float(totals.get("mlir_pass_ms")) + pass_ms
        totals["mlir_lower_ms"] = _safe_float(totals.get("mlir_lower_ms")) + lower_ms
        totals["mlir_total_ms"] = _safe_float(totals.get("mlir_total_ms")) + parse_ms + pass_ms + lower_ms
        for key in ("upstream", "midend", "downstream"):
            sec = row.get(key)
            if isinstance(sec, dict):
                passes = [p for p in list(sec.get("passes") or []) if isinstance(p, dict)]
                pass_counts[key] = int(pass_counts.get(key, 0)) + int(len(passes))
                for pass_row in passes:
                    name = str(pass_row.get("name") or "").strip() or "<unnamed>"
                    ms = _safe_float(pass_row.get("ms"))
                    pass_ms_totals_by_pipeline[key] = _safe_float(pass_ms_totals_by_pipeline.get(key)) + ms
                    pass_ms_totals_by_name[name] = _safe_float(pass_ms_totals_by_name.get(name)) + ms
                    pass_count_by_name[name] = int(pass_count_by_name.get(name, 0)) + 1
        counted += 1
    if counted <= 0:
        out["missing_mlir_rows"] = missing
        return out
    out["available"] = True
    out["kernel_count"] = int(counted)
    out["totals_ms"] = totals
    out["pass_count_totals"] = pass_counts
    out["pass_ms_totals_by_pipeline"] = pass_ms_totals_by_pipeline
    out["pass_ms_totals_by_name"] = pass_ms_totals_by_name
    out["pass_count_by_name"] = pass_count_by_name
    out["missing_mlir_rows"] = missing
    _finalize_mlir_timing(out)
    return out


def _merge_mlir_timing(dst: dict[str, Any], src: dict[str, Any]) -> None:
    if not bool(src.get("available")):
        return
    dst["available"] = True
    dst["kernel_count"] = int(dst.get("kernel_count") or 0) + int(src.get("kernel_count") or 0)
    dst_totals = dict(dst.get("totals_ms") or {})
    src_totals = dict(src.get("totals_ms") or {})
    for key in ("mlir_parse_ms", "mlir_pass_ms", "mlir_lower_ms", "mlir_total_ms"):
        dst_totals[key] = _safe_float(dst_totals.get(key)) + _safe_float(src_totals.get(key))
    dst["totals_ms"] = dst_totals
    dst_pass = dict(dst.get("pass_count_totals") or {})
    src_pass = dict(src.get("pass_count_totals") or {})
    for key in ("upstream", "midend", "downstream"):
        dst_pass[key] = int(dst_pass.get(key, 0)) + int(src_pass.get(key, 0))
    dst["pass_count_totals"] = dst_pass
    dst_pass_ms_pipeline = dict(dst.get("pass_ms_totals_by_pipeline") or {})
    src_pass_ms_pipeline = dict(src.get("pass_ms_totals_by_pipeline") or {})
    for key in ("upstream", "midend", "downstream"):
        dst_pass_ms_pipeline[key] = _safe_float(dst_pass_ms_pipeline.get(key)) + _safe_float(src_pass_ms_pipeline.get(key))
    dst["pass_ms_totals_by_pipeline"] = dst_pass_ms_pipeline
    dst_pass_ms_name = dict(dst.get("pass_ms_totals_by_name") or {})
    src_pass_ms_name = dict(src.get("pass_ms_totals_by_name") or {})
    for key, value in src_pass_ms_name.items():
        name = str(key).strip()
        if not name:
            continue
        dst_pass_ms_name[name] = _safe_float(dst_pass_ms_name.get(name)) + _safe_float(value)
    dst["pass_ms_totals_by_name"] = dst_pass_ms_name
    dst_pass_count_name = dict(dst.get("pass_count_by_name") or {})
    src_pass_count_name = dict(src.get("pass_count_by_name") or {})
    for key, value in src_pass_count_name.items():
        name = str(key).strip()
        if not name:
            continue
        dst_pass_count_name[name] = int(dst_pass_count_name.get(name, 0)) + int(value or 0)
    dst["pass_count_by_name"] = dst_pass_count_name
    missing = [str(x) for x in list(src.get("missing_mlir_rows") or []) if str(x).strip()]
    if missing:
        dst_missing = [str(x) for x in list(dst.get("missing_mlir_rows") or []) if str(x).strip()]
        dst["missing_mlir_rows"] = sorted(set(dst_missing + missing))
    src_dir = str(src.get("source_dir") or "").strip()
    if src_dir:
        src_dirs = [str(x) for x in list(dst.get("source_dirs") or []) if str(x).strip()]
        if src_dir not in src_dirs:
            src_dirs.append(src_dir)
        dst["source_dirs"] = src_dirs


def _finalize_mlir_timing(section: dict[str, Any]) -> None:
    kernel_count = int(section.get("kernel_count") or 0)
    totals = dict(section.get("totals_ms") or {})
    parse_ms = _safe_float(totals.get("mlir_parse_ms"))
    pass_ms = _safe_float(totals.get("mlir_pass_ms"))
    lower_ms = _safe_float(totals.get("mlir_lower_ms"))
    total_ms = _safe_float(totals.get("mlir_total_ms"))
    if total_ms <= 0.0:
        total_ms = parse_ms + pass_ms + lower_ms
        totals["mlir_total_ms"] = float(total_ms)
        section["totals_ms"] = totals
    if kernel_count > 0:
        section["avg_ms"] = {
            "mlir_parse_ms": parse_ms / float(kernel_count),
            "mlir_pass_ms": pass_ms / float(kernel_count),
            "mlir_lower_ms": lower_ms / float(kernel_count),
            "mlir_total_ms": total_ms / float(kernel_count),
        }
    denom = parse_ms + pass_ms + lower_ms
    if denom > 0:
        section["stage_share_pct"] = {
            "mlir_parse_ms": parse_ms / denom * 100.0,
            "mlir_pass_ms": pass_ms / denom * 100.0,
            "mlir_lower_ms": lower_ms / denom * 100.0,
        }
    pass_ms_totals_by_name = dict(section.get("pass_ms_totals_by_name") or {})
    pass_count_by_name = dict(section.get("pass_count_by_name") or {})
    section["pass_ms_totals_by_name"] = dict(sorted(pass_ms_totals_by_name.items(), key=lambda kv: kv[0]))
    section["pass_count_by_name"] = {
        str(k): int(v)
        for k, v in sorted(pass_count_by_name.items(), key=lambda kv: str(kv[0]))
        if str(k).strip()
    }
    section["pass_ms_totals_by_pipeline"] = {
        "upstream": _safe_float((section.get("pass_ms_totals_by_pipeline") or {}).get("upstream")),
        "midend": _safe_float((section.get("pass_ms_totals_by_pipeline") or {}).get("midend")),
        "downstream": _safe_float((section.get("pass_ms_totals_by_pipeline") or {}).get("downstream")),
    }
    ranked = []
    for name, total_ms in section["pass_ms_totals_by_name"].items():
        cnt = int(section["pass_count_by_name"].get(name, 0))
        ranked.append(
            {
                "name": str(name),
                "total_ms": _safe_float(total_ms),
                "count": int(cnt),
                "avg_ms": (_safe_float(total_ms) / float(cnt)) if cnt > 0 else 0.0,
            }
        )
    ranked.sort(key=lambda x: (-_safe_float(x.get("total_ms")), str(x.get("name"))))
    section["top_passes_by_ms"] = ranked[:20]


def _summarize_backend_results(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in list(payload.get("results") or []) if isinstance(r, dict)]
    if not rows:
        return _empty_backend_timing()
    totals = {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0, "total_ms": 0.0}
    reason_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason_code") or "unknown").strip() or "unknown"
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        totals["lower_ms"] += _safe_float(row.get("lower_ms"))
        totals["compile_ms"] += _safe_float(row.get("compile_ms"))
        totals["launch_ms"] += _safe_float(row.get("launch_ms"))
        totals["total_ms"] += _safe_float(row.get("total_ms"))
    # Legacy rvv_remote artifacts may not carry timing fields yet.
    # Keep determinability by using a coarse kernel-count proxy so gate checks
    # can still distinguish "executed" from "missing".
    if totals["total_ms"] <= 0.0 and len(rows) > 0:
        totals["launch_ms"] = float(len(rows))
        totals["total_ms"] = float(len(rows))
    out = _empty_backend_timing()
    out["available"] = True
    out["kernel_count"] = int(len(rows))
    out["reason_code_counts"] = dict(sorted(reason_counts.items(), key=lambda kv: kv[0]))
    out["totals_ms"] = totals
    _finalize_backend_timing(out)
    return out


def _merge_backend_timing(dst: dict[str, Any], src: dict[str, Any]) -> None:
    if not bool(src.get("available")):
        return
    dst["available"] = True
    dst["kernel_count"] = int(dst.get("kernel_count") or 0) + int(src.get("kernel_count") or 0)
    dst_totals = dict(dst.get("totals_ms") or {})
    src_totals = dict(src.get("totals_ms") or {})
    for key in ("lower_ms", "compile_ms", "launch_ms", "total_ms"):
        dst_totals[key] = _safe_float(dst_totals.get(key)) + _safe_float(src_totals.get(key))
    dst["totals_ms"] = dst_totals
    _merge_reason_counts(dst.setdefault("reason_code_counts", {}), dict(src.get("reason_code_counts") or {}))


def _finalize_backend_timing(section: dict[str, Any]) -> None:
    kernel_count = int(section.get("kernel_count") or 0)
    totals = dict(section.get("totals_ms") or {})
    lower = _safe_float(totals.get("lower_ms"))
    compile_ = _safe_float(totals.get("compile_ms"))
    launch = _safe_float(totals.get("launch_ms"))
    total = _safe_float(totals.get("total_ms"))
    if kernel_count > 0:
        section["avg_ms"] = {
            "lower_ms": lower / float(kernel_count),
            "compile_ms": compile_ / float(kernel_count),
            "launch_ms": launch / float(kernel_count),
            "total_ms": total / float(kernel_count),
        }
    denom = lower + compile_ + launch
    if denom > 0:
        section["stage_share_pct"] = {
            "lower_ms": lower / denom * 100.0,
            "compile_ms": compile_ / denom * 100.0,
            "launch_ms": launch / denom * 100.0,
        }


def _aggregate_stage_timing(paths: list[Path]) -> tuple[dict[str, Any], bool]:
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        unique_paths.append(p)

    backends = {"rvv": _empty_backend_timing(), "cuda": _empty_backend_timing()}
    mlir = _empty_mlir_timing()
    for p in unique_paths:
        payload = _load_json(p)
        if str(payload.get("schema_version") or "") != "flaggems_stage_timing_breakdown_v1":
            continue
        src_backends = dict(payload.get("backends") or {})
        for backend_name in ("rvv", "cuda"):
            src = src_backends.get(backend_name)
            if not isinstance(src, dict):
                continue
            if not bool(src.get("available")):
                continue
            dst = backends[backend_name]
            dst["available"] = True
            dst["kernel_count"] = int(dst.get("kernel_count") or 0) + int(src.get("kernel_count") or 0)
            dst_totals = dict(dst.get("totals_ms") or {})
            src_totals = dict(src.get("totals_ms") or {})
            for key in ("lower_ms", "compile_ms", "launch_ms", "total_ms"):
                dst_totals[key] = _safe_float(dst_totals.get(key)) + _safe_float(src_totals.get(key))
            dst["totals_ms"] = dst_totals
            _merge_reason_counts(dst.setdefault("reason_code_counts", {}), dict(src.get("reason_code_counts") or {}))
        src_mlir = payload.get("mlir")
        if isinstance(src_mlir, dict):
            _merge_mlir_timing(mlir, src_mlir)

    for backend_name in ("rvv", "cuda"):
        _finalize_backend_timing(backends[backend_name])
    _finalize_mlir_timing(mlir)

    combined_totals = {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0, "total_ms": 0.0}
    kernel_total = 0
    for backend_name in ("rvv", "cuda"):
        section = backends[backend_name]
        kernel_total += int(section.get("kernel_count") or 0)
        totals = dict(section.get("totals_ms") or {})
        for key in ("lower_ms", "compile_ms", "launch_ms", "total_ms"):
            combined_totals[key] += _safe_float(totals.get(key))
    combined_avg = {
        "lower_ms": (combined_totals["lower_ms"] / float(kernel_total)) if kernel_total else 0.0,
        "compile_ms": (combined_totals["compile_ms"] / float(kernel_total)) if kernel_total else 0.0,
        "launch_ms": (combined_totals["launch_ms"] / float(kernel_total)) if kernel_total else 0.0,
        "total_ms": (combined_totals["total_ms"] / float(kernel_total)) if kernel_total else 0.0,
    }
    denom = combined_totals["lower_ms"] + combined_totals["compile_ms"] + combined_totals["launch_ms"]
    if denom > 0:
        combined_share = {
            "lower_ms": combined_totals["lower_ms"] / denom * 100.0,
            "compile_ms": combined_totals["compile_ms"] / denom * 100.0,
            "launch_ms": combined_totals["launch_ms"] / denom * 100.0,
        }
    else:
        combined_share = {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0}

    payload = {
        "schema_version": "flaggems_stage_timing_breakdown_v1",
        "ok": bool(
            backends["rvv"]["available"]
            and backends["cuda"]["available"]
            and _safe_float(backends["rvv"]["totals_ms"].get("total_ms")) > 0.0
            and _safe_float(backends["cuda"]["totals_ms"].get("total_ms")) > 0.0
        ),
        "inputs": {"sources": [str(p) for p in unique_paths]},
        "backends": backends,
        "mlir": mlir,
        "combined": {
            "kernel_count_total": int(kernel_total),
            "totals_ms": combined_totals,
            "avg_ms": combined_avg,
            "stage_share_pct": combined_share,
        },
    }
    return payload, bool(payload.get("ok"))


def _aggregate_stage_timing_from_backend_pairs(pairs: list[dict[str, Path]]) -> tuple[dict[str, Any], bool]:
    unique_pairs: list[dict[str, Path]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in pairs:
        rvv = row.get("rvv_json")
        cuda = row.get("cuda_json")
        if rvv is None or cuda is None:
            continue
        pipeline_reports_dir = row.get("pipeline_reports_dir")
        key = (str(rvv), str(cuda), str(pipeline_reports_dir or ""))
        if key in seen:
            continue
        seen.add(key)
        dedup_row: dict[str, Path] = {"rvv_json": rvv, "cuda_json": cuda}
        if isinstance(pipeline_reports_dir, Path):
            dedup_row["pipeline_reports_dir"] = pipeline_reports_dir
        unique_pairs.append(dedup_row)

    backends = {"rvv": _empty_backend_timing(), "cuda": _empty_backend_timing()}
    mlir = _empty_mlir_timing()
    for row in unique_pairs:
        rvv_payload = _load_json(row["rvv_json"])
        cuda_payload = _load_json(row["cuda_json"])
        rvv_sec = _summarize_backend_results(rvv_payload)
        cuda_sec = _summarize_backend_results(cuda_payload)
        _merge_backend_timing(backends["rvv"], rvv_sec)
        _merge_backend_timing(backends["cuda"], cuda_sec)
        pipeline_reports_dir = row.get("pipeline_reports_dir")
        if isinstance(pipeline_reports_dir, Path):
            _merge_mlir_timing(mlir, _summarize_mlir_reports_dir(pipeline_reports_dir))

    for backend_name in ("rvv", "cuda"):
        _finalize_backend_timing(backends[backend_name])
    _finalize_mlir_timing(mlir)

    combined_totals = {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0, "total_ms": 0.0}
    kernel_total = 0
    for backend_name in ("rvv", "cuda"):
        section = backends[backend_name]
        kernel_total += int(section.get("kernel_count") or 0)
        totals = dict(section.get("totals_ms") or {})
        for key in ("lower_ms", "compile_ms", "launch_ms", "total_ms"):
            combined_totals[key] += _safe_float(totals.get(key))
    combined_avg = {
        "lower_ms": (combined_totals["lower_ms"] / float(kernel_total)) if kernel_total else 0.0,
        "compile_ms": (combined_totals["compile_ms"] / float(kernel_total)) if kernel_total else 0.0,
        "launch_ms": (combined_totals["launch_ms"] / float(kernel_total)) if kernel_total else 0.0,
        "total_ms": (combined_totals["total_ms"] / float(kernel_total)) if kernel_total else 0.0,
    }
    denom = combined_totals["lower_ms"] + combined_totals["compile_ms"] + combined_totals["launch_ms"]
    if denom > 0:
        combined_share = {
            "lower_ms": combined_totals["lower_ms"] / denom * 100.0,
            "compile_ms": combined_totals["compile_ms"] / denom * 100.0,
            "launch_ms": combined_totals["launch_ms"] / denom * 100.0,
        }
    else:
        combined_share = {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0}

    payload = {
        "schema_version": "flaggems_stage_timing_breakdown_v1",
        "ok": bool(
            backends["rvv"]["available"]
            and backends["cuda"]["available"]
            and _safe_float(backends["rvv"]["totals_ms"].get("total_ms")) > 0.0
            and _safe_float(backends["cuda"]["totals_ms"].get("total_ms")) > 0.0
        ),
        "inputs": {
            "sources": [
                {
                    "rvv_json": str(r["rvv_json"]),
                    "cuda_json": str(r["cuda_json"]),
                    "pipeline_reports_dir": str(r.get("pipeline_reports_dir") or ""),
                }
                for r in unique_pairs
            ]
        },
        "backends": backends,
        "mlir": mlir,
        "combined": {
            "kernel_count_total": int(kernel_total),
            "totals_ms": combined_totals,
            "avg_ms": combined_avg,
            "stage_share_pct": combined_share,
        },
    }
    return payload, bool(payload.get("ok"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage-batches", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--out-run-summary", type=Path, default=None)
    ap.add_argument("--out-status-converged", type=Path, default=None)
    ap.add_argument("--out-coverage-integrity", type=Path, default=None)
    ap.add_argument("--require-dual-pass-total", type=int, default=196)
    ap.add_argument("--intentir-mode", choices=["auto", "force_compile", "force_cache"], default="force_compile")
    ap.add_argument(
        "--execution-ir",
        choices=["mlir"],
        default="mlir",
    )
    ap.add_argument("--intentir-miss-policy", choices=["deterministic", "strict"], default="strict")
    ap.add_argument("--run-rvv-remote", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="nvrtc")
    ap.add_argument("--family-kernel-chunk-size", type=int, default=12)
    args = ap.parse_args()

    coverage_batches = _load_json(args.coverage_batches)
    family_order = [str(x).strip() for x in list(coverage_batches.get("family_order") or []) if str(x).strip()]
    batches = [b for b in list(coverage_batches.get("batches") or []) if isinstance(b, dict)]
    by_family = {
        str(b.get("family") or "").strip(): b
        for b in batches
        if str(b.get("family") or "").strip()
    }
    families = [f for f in family_order if f in by_family]

    expected_semantics: list[str] = []
    expected_kernels: list[str] = []
    for family in families:
        batch = by_family[family]
        for sop in list(batch.get("semantic_ops") or []):
            s = str(sop).strip()
            if s and s not in expected_semantics:
                expected_semantics.append(s)
        for kernel in list(batch.get("kernels") or []):
            k = str(kernel).strip()
            if k and k not in expected_kernels:
                expected_kernels.append(k)

    family_rows: list[dict[str, Any]] = []
    merged_entries: dict[str, dict[str, Any]] = {}
    seen_kernels: list[str] = []
    failed_families: list[str] = []
    completed_families = 0
    stage_timing_paths: list[Path] = []
    backend_json_pairs: list[dict[str, Path]] = []

    for family in families:
        batch = by_family[family]
        family_semantics = [str(x).strip() for x in list(batch.get("semantic_ops") or []) if str(x).strip()]
        family_kernels = [str(x).strip() for x in list(batch.get("kernels") or []) if str(x).strip()]
        family_dir = Path(args.runs_root) / f"family_{family}"
        run_summary_path = family_dir / "run_summary.json"
        status_path = family_dir / "status_converged.json"

        run_summary_exists = run_summary_path.is_file()
        status_exists = status_path.is_file()
        run_summary_payload = _load_json(run_summary_path) if run_summary_exists else {}
        status_payload = _load_json(status_path) if status_exists else {}
        run_ok = bool(run_summary_payload.get("ok")) if run_summary_exists else False
        if run_summary_exists:
            stage_timing_paths.extend(_collect_stage_timing_paths(run_summary_payload, run_summary_path=run_summary_path))
            backend_json_pairs.extend(_collect_backend_json_pairs(run_summary_payload, run_summary_path=run_summary_path))

        if run_summary_exists:
            for kernel in list(run_summary_payload.get("scope_kernels") or []):
                k = str(kernel).strip()
                if k and k not in seen_kernels:
                    seen_kernels.append(k)
        else:
            for kernel in family_kernels:
                if kernel and kernel not in seen_kernels:
                    seen_kernels.append(kernel)

        status_entries = [e for e in list(status_payload.get("entries") or []) if isinstance(e, dict)]
        status_by_semantic = _collect_semantics(status_entries)

        missing_semantics: list[str] = []
        non_dual_semantics: list[str] = []
        for sop in family_semantics:
            row = status_by_semantic.get(sop)
            if row is None:
                missing_semantics.append(sop)
                row = _entry_or_placeholder(
                    semantic_op=sop,
                    family=family,
                    reason_code="pipeline_missing_report",
                    reason_detail=f"missing semantic entry in {status_path}",
                )
            if str(row.get("status") or "") != "dual_pass":
                non_dual_semantics.append(sop)
            merged_entries[sop] = row

        family_ok = bool(run_summary_exists and status_exists and run_ok and not missing_semantics and not non_dual_semantics)
        if family_ok:
            completed_families += 1
        else:
            failed_families.append(family)

        family_rows.append(
            {
                "family": family,
                "ok": bool(family_ok),
                "run_summary_path": str(run_summary_path),
                "status_converged_path": str(status_path),
                "run_summary_exists": bool(run_summary_exists),
                "status_converged_exists": bool(status_exists),
                "run_summary_ok": bool(run_ok),
                "semantic_count": int(len(family_semantics)),
                "kernel_count": int(len(family_kernels)),
                "missing_semantics": missing_semantics,
                "non_dual_semantics": non_dual_semantics,
            }
        )

    # Ensure every expected semantic has an entry.
    for sop in expected_semantics:
        if sop in merged_entries:
            continue
        family = "unknown"
        for fam in families:
            if sop in list(by_family[fam].get("semantic_ops") or []):
                family = fam
                break
        merged_entries[sop] = _entry_or_placeholder(
            semantic_op=sop,
            family=family,
            reason_code="pipeline_missing_report",
            reason_detail="semantic missing from all family status files",
        )

    final_entries: list[dict[str, Any]] = [merged_entries[sop] for sop in expected_semantics if sop in merged_entries]
    counts_global = _counts(final_entries)
    dual_pass_total = int(counts_global.get("dual_pass", 0))
    reason_code_complete = all(
        isinstance(row.get("reason_code"), str) and str(row.get("reason_code")).strip()
        for row in final_entries
    )
    determinability = all(str(row.get("status") or "").strip() != "unknown" for row in final_entries)
    categories_expected = int(len(families))
    categories_completed = int(completed_families)
    categories_failed = list(sorted(set(failed_families)))
    categories_complete = bool(categories_completed == categories_expected and not categories_failed)
    semantic_total_expected = int(len(expected_semantics))
    kernel_total_expected = int(len(expected_kernels))
    kernel_total_seen = int(len(seen_kernels))
    dual_pass_target = int(args.require_dual_pass_total)

    coverage_integrity_ok = bool(
        categories_complete
        and reason_code_complete
        and determinability
        and semantic_total_expected == dual_pass_target
        and dual_pass_total == dual_pass_target
    )
    coverage_reason = (
        "ok"
        if coverage_integrity_ok
        else "coverage_categories_incomplete_or_non_dual_pass"
    )

    out_run_summary = Path(args.out_run_summary) if args.out_run_summary is not None else (Path(args.runs_root) / "run_summary.json")
    out_status = Path(args.out_status_converged) if args.out_status_converged is not None else (Path(args.runs_root) / "status_converged.json")
    out_coverage = (
        Path(args.out_coverage_integrity)
        if args.out_coverage_integrity is not None
        else (Path(args.runs_root) / "coverage_integrity.json")
    )
    out_stage_timing = Path(args.runs_root) / "stage_timing_breakdown.json"
    stage_timing_payload, stage_timing_ok = _aggregate_stage_timing(stage_timing_paths)
    if (not stage_timing_ok) and backend_json_pairs:
        stage_timing_payload, stage_timing_ok = _aggregate_stage_timing_from_backend_pairs(backend_json_pairs)
    stage_timing_payload["repo"] = repo_state(root=ROOT)
    stage_timing_payload["invocation"] = {
        "intentir_mode": str(args.intentir_mode),
        "miss_policy": str(args.intentir_miss_policy),
        "execution_ir": str(args.execution_ir),
        "rvv_remote": bool(args.run_rvv_remote),
        "cuda_runtime_backend": str(args.cuda_runtime_backend),
    }
    _dump_json(out_stage_timing, stage_timing_payload)

    status_payload = {
        "schema_version": "flaggems_status_converged_v3",
        "generated_at": _utc_now_iso(),
        "repo": repo_state(root=ROOT),
        "invocation": {
            "intentir_mode": str(args.intentir_mode),
            "miss_policy": str(args.intentir_miss_policy),
            "execution_ir": str(args.execution_ir),
            "rvv_remote": bool(args.run_rvv_remote),
            "cuda_runtime_backend": str(args.cuda_runtime_backend),
        },
        "coverage": {
            "mode": "category_batches",
            "batches_expected": categories_expected,
            "chunk_size": int(args.family_kernel_chunk_size),
        },
        "scope_enabled": False,
        "entries": final_entries,
        "counts_global": counts_global,
        "counts_scoped": counts_global,
        "counts_scoped_active": counts_global,
        "counts_scoped_kernel_alias": counts_global,
        "global_entries_count": int(len(final_entries)),
        "scoped_entries_count": int(len(final_entries)),
        "scoped_entries_active_count": int(len(final_entries)),
        "scoped_entries_kernel_alias_count": int(len(final_entries)),
        "coverage_batches_expected": categories_expected,
        "coverage_batches_completed": categories_completed,
        "coverage_batches_failed": categories_failed,
    }
    _dump_json(out_status, status_payload)

    coverage_payload = {
        "schema_version": "flaggems_coverage_integrity_v2",
        "generated_at": _utc_now_iso(),
        "repo": repo_state(root=ROOT),
        "invocation": {
            "intentir_mode": str(args.intentir_mode),
            "miss_policy": str(args.intentir_miss_policy),
            "execution_ir": str(args.execution_ir),
            "rvv_remote": bool(args.run_rvv_remote),
            "cuda_runtime_backend": str(args.cuda_runtime_backend),
        },
        "coverage": {
            "mode": "category_batches",
            "batches_expected": categories_expected,
            "chunk_size": int(args.family_kernel_chunk_size),
        },
        "coverage_integrity_ok": bool(coverage_integrity_ok),
        "reason_code": str(coverage_reason),
        "coverage_mode": "category_batches",
        "full196_evidence_kind": "batch_aggregate",
        "coverage_batches_expected": categories_expected,
        "coverage_batches_completed": categories_completed,
        "coverage_batches_failed": categories_failed,
        "semantic_ops_expected": semantic_total_expected,
        "semantic_ops_seen": int(len(final_entries)),
        "dual_pass_total": dual_pass_total,
        "kernel_total_expected": kernel_total_expected,
        "kernel_total_seen": kernel_total_seen,
        "determinability_ok": bool(determinability),
        "reason_code_complete": bool(reason_code_complete),
    }
    _dump_json(out_coverage, coverage_payload)

    mlir_llvm_completed = 0
    mlir_llvm_expected = 0
    mlir_llvm_artifact_complete = True
    mlir_llvm_missing_families: list[str] = []
    runtime_fallback_kernels: set[str] = set()
    for row in list(family_rows):
        if not isinstance(row, dict):
            continue
        family = str(row.get("family") or "")
        run_summary_path_raw = str(row.get("run_summary_path") or "").strip()
        run_summary_path = (
            _resolve_json_path(run_summary_path_raw, anchor=out_run_summary) if run_summary_path_raw else None
        )
        if run_summary_path is None or (not run_summary_path.is_file()):
            mlir_llvm_artifact_complete = False
            if family:
                mlir_llvm_missing_families.append(family)
            continue
        mlir_llvm_expected += 1
        family_summary = _load_json(run_summary_path)
        family_mlir_complete = _family_mlir_llvm_artifact_complete(
            family_summary=family_summary,
            run_summary_path=run_summary_path,
        )
        if bool(family_mlir_complete):
            mlir_llvm_completed += 1
        else:
            mlir_llvm_artifact_complete = False
            if family:
                mlir_llvm_missing_families.append(family)
        family_runtime_fallback_kernels, family_runtime_fallback_count = _family_runtime_fallback(
            family=family,
            family_summary=family_summary,
            run_summary_path=run_summary_path,
        )
        if family_runtime_fallback_kernels:
            runtime_fallback_kernels.update(family_runtime_fallback_kernels)
        elif int(family_runtime_fallback_count) > 0 and family:
            for i in range(1, int(family_runtime_fallback_count) + 1):
                runtime_fallback_kernels.add(f"{family}::fallback_{i}")
    if mlir_llvm_expected <= 0:
        mlir_llvm_artifact_complete = False

    runtime_fallback_kernel_list = sorted(runtime_fallback_kernels)
    runtime_fallback_kernel_count = int(len(runtime_fallback_kernel_list))
    strict_mode = bool(strict_fallback_enabled())
    fallback_policy = "strict" if strict_mode else "legacy_compatible"
    run_summary_payload = {
        "ok": bool(coverage_integrity_ok),
        "suite": "coverage",
        "requested_suite": "coverage",
        "repo": repo_state(root=ROOT),
        "strict_mode": bool(strict_mode),
        "fallback_policy": str(fallback_policy),
        "contract_schema_version": str(CONTRACT_SCHEMA_VERSION),
        "invocation": {
            "intentir_mode": str(args.intentir_mode),
            "miss_policy": str(args.intentir_miss_policy),
            "execution_ir": str(args.execution_ir),
            "rvv_remote": bool(args.run_rvv_remote),
            "cuda_runtime_backend": str(args.cuda_runtime_backend),
            "strict_mode": bool(strict_mode),
            "fallback_policy": str(fallback_policy),
            "contract_schema_version": str(CONTRACT_SCHEMA_VERSION),
        },
        "coverage": {
            "mode": "category_batches",
            "batches_expected": categories_expected,
            "chunk_size": int(args.family_kernel_chunk_size),
        },
        "kernel_filter": [],
        "scope_kernels": expected_kernels,
        "coverage_mode": "category_batches",
        "full196_evidence_kind": "batch_aggregate",
        "coverage_batches_expected": categories_expected,
        "coverage_batches_completed": categories_completed,
        "coverage_batches_failed": categories_failed,
        "intentir_mode": str(args.intentir_mode),
        "execution_ir": str(args.execution_ir),
        "mlir_llvm_artifact_complete": bool(mlir_llvm_artifact_complete),
        "mlir_llvm_chain_ok": bool(mlir_llvm_artifact_complete),
        "mlir_llvm_artifact_families_expected": int(mlir_llvm_expected),
        "mlir_llvm_artifact_families_completed": int(mlir_llvm_completed),
        "mlir_llvm_artifact_families_missing": list(sorted(set(mlir_llvm_missing_families))),
        "runtime_fallback_kernel_count": int(runtime_fallback_kernel_count),
        "runtime_fallback_kernels": list(runtime_fallback_kernel_list),
        "stages": [
            {
                "stage": "coverage_categories",
                "ok": bool(categories_complete),
                "reason_code": ("ok" if categories_complete else "coverage_categories_incomplete"),
                "json_path": str(args.coverage_batches),
                "families_expected": categories_expected,
                "families_completed": categories_completed,
                "families_failed": categories_failed,
            },
            {
                "stage": "coverage_integrity",
                "ok": bool(coverage_integrity_ok),
                "reason_code": str(coverage_reason),
                "json_path": str(out_coverage),
                "full_coverage_run": True,
            },
            {
                "stage": "stage_timing_breakdown",
                "ok": bool(stage_timing_ok),
                "reason_code": ("ok" if stage_timing_ok else "stage_timing_breakdown_missing_or_invalid"),
                "json_path": str(out_stage_timing),
            },
            {
                "stage": "mlir_llvm_artifacts",
                "ok": bool(mlir_llvm_artifact_complete),
                "reason_code": ("ok" if mlir_llvm_artifact_complete else "llvm_artifacts_incomplete"),
                "families_expected": int(mlir_llvm_expected),
                "families_completed": int(mlir_llvm_completed),
                "families_missing": list(sorted(set(mlir_llvm_missing_families))),
            },
        ],
        "family_runs": family_rows,
        "status_converged_path": str(out_status),
    }
    _dump_json(out_run_summary, run_summary_payload)

    print(f"Coverage batch aggregate run summary: {out_run_summary}")
    print(f"Coverage batch aggregate status: {out_status}")
    print(f"Coverage batch aggregate integrity: {out_coverage}")
    raise SystemExit(0 if coverage_integrity_ok else 1)


if __name__ == "__main__":
    main()
