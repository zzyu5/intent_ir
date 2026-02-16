"""
Aggregate backend stage timings into a single breakdown artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def _summarize_backend(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in list(payload.get("results") or []) if isinstance(r, dict)]
    if not rows:
        return {
            "available": False,
            "kernel_count": 0,
            "reason_code_counts": {},
            "totals_ms": {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0, "total_ms": 0.0},
            "avg_ms": {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0, "total_ms": 0.0},
            "stage_share_pct": {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0},
            "top_kernels_by_total_ms": [],
        }

    totals = {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0, "total_ms": 0.0}
    reason_counts: dict[str, int] = {}
    ranked: list[dict[str, Any]] = []
    for row in rows:
        reason = str(row.get("reason_code") or "unknown").strip() or "unknown"
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        lower_ms = _safe_float(row.get("lower_ms"))
        compile_ms = _safe_float(row.get("compile_ms"))
        launch_ms = _safe_float(row.get("launch_ms"))
        total_ms = _safe_float(row.get("total_ms"))
        totals["lower_ms"] += lower_ms
        totals["compile_ms"] += compile_ms
        totals["launch_ms"] += launch_ms
        totals["total_ms"] += total_ms
        ranked.append(
            {
                "kernel": str(row.get("kernel") or ""),
                "reason_code": reason,
                "lower_ms": lower_ms,
                "compile_ms": compile_ms,
                "launch_ms": launch_ms,
                "total_ms": total_ms,
            }
        )

    kernel_count = len(rows)
    avg = {
        "lower_ms": totals["lower_ms"] / float(kernel_count),
        "compile_ms": totals["compile_ms"] / float(kernel_count),
        "launch_ms": totals["launch_ms"] / float(kernel_count),
        "total_ms": totals["total_ms"] / float(kernel_count),
    }
    stage_denom = totals["lower_ms"] + totals["compile_ms"] + totals["launch_ms"]
    if stage_denom <= 0.0:
        stage_share = {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0}
    else:
        stage_share = {
            "lower_ms": totals["lower_ms"] / stage_denom * 100.0,
            "compile_ms": totals["compile_ms"] / stage_denom * 100.0,
            "launch_ms": totals["launch_ms"] / stage_denom * 100.0,
        }
    ranked.sort(key=lambda x: float(x.get("total_ms", 0.0)), reverse=True)

    return {
        "available": True,
        "kernel_count": int(kernel_count),
        "reason_code_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[0])),
        "totals_ms": totals,
        "avg_ms": avg,
        "stage_share_pct": stage_share,
        "top_kernels_by_total_ms": ranked[:10],
    }


def _combine(backends: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0, "total_ms": 0.0}
    kernel_count = 0
    for section in backends.values():
        totals_sec = dict(section.get("totals_ms") or {})
        totals["lower_ms"] += _safe_float(totals_sec.get("lower_ms"))
        totals["compile_ms"] += _safe_float(totals_sec.get("compile_ms"))
        totals["launch_ms"] += _safe_float(totals_sec.get("launch_ms"))
        totals["total_ms"] += _safe_float(totals_sec.get("total_ms"))
        kernel_count += int(section.get("kernel_count") or 0)
    avg = {
        "lower_ms": (totals["lower_ms"] / float(kernel_count)) if kernel_count else 0.0,
        "compile_ms": (totals["compile_ms"] / float(kernel_count)) if kernel_count else 0.0,
        "launch_ms": (totals["launch_ms"] / float(kernel_count)) if kernel_count else 0.0,
        "total_ms": (totals["total_ms"] / float(kernel_count)) if kernel_count else 0.0,
    }
    stage_denom = totals["lower_ms"] + totals["compile_ms"] + totals["launch_ms"]
    if stage_denom <= 0.0:
        share = {"lower_ms": 0.0, "compile_ms": 0.0, "launch_ms": 0.0}
    else:
        share = {
            "lower_ms": totals["lower_ms"] / stage_denom * 100.0,
            "compile_ms": totals["compile_ms"] / stage_denom * 100.0,
            "launch_ms": totals["launch_ms"] / stage_denom * 100.0,
        }
    return {
        "kernel_count_total": int(kernel_count),
        "totals_ms": totals,
        "avg_ms": avg,
        "stage_share_pct": share,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvv-json", type=Path, required=True)
    ap.add_argument("--cuda-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rvv_payload = _load_json(args.rvv_json)
    cuda_payload = _load_json(args.cuda_json)
    backends = {
        "rvv": _summarize_backend(rvv_payload),
        "cuda": _summarize_backend(cuda_payload),
    }
    payload = {
        "schema_version": "flaggems_stage_timing_breakdown_v1",
        "ok": bool(backends["rvv"]["available"] and backends["cuda"]["available"]),
        "inputs": {"rvv_json": str(args.rvv_json), "cuda_json": str(args.cuda_json)},
        "backends": backends,
        "combined": _combine(backends),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Stage timing breakdown written: {args.out}")
    raise SystemExit(0 if bool(payload.get("ok")) else 1)


if __name__ == "__main__":
    main()
