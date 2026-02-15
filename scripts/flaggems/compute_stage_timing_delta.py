"""
Compute stage timing deltas between current and baseline backend smoke reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [r for r in list(payload.get("results") or []) if isinstance(r, dict)]
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        kernel = str(row.get("kernel") or "").strip()
        if not kernel:
            continue
        out[kernel] = row
    return out


def _delta(current: float, baseline: float) -> dict[str, float]:
    d = float(current) - float(baseline)
    pct = (d / float(baseline) * 100.0) if float(baseline) != 0.0 else 0.0
    return {"delta_ms": d, "delta_pct": pct}


def _backend_delta(current_payload: dict[str, Any], baseline_payload: dict[str, Any]) -> dict[str, Any]:
    cur = _result_map(current_payload)
    base = _result_map(baseline_payload)
    kernels = sorted(set(cur.keys()) & set(base.keys()))
    missing_baseline = sorted(set(cur.keys()) - set(base.keys()))
    baseline_available = bool(base)
    rows: list[dict[str, Any]] = []
    for kernel in kernels:
        c = cur[kernel]
        b = base[kernel]
        row = {"kernel": kernel}
        for key in ("lower_ms", "compile_ms", "launch_ms", "total_ms"):
            try:
                row[key] = {
                    "current_ms": float(c.get(key, 0.0)),
                    "baseline_ms": float(b.get(key, 0.0)),
                    **_delta(float(c.get(key, 0.0)), float(b.get(key, 0.0))),
                }
            except Exception:
                row[key] = {"current_ms": 0.0, "baseline_ms": 0.0, "delta_ms": 0.0, "delta_pct": 0.0}
        rows.append(row)
    return {
        "baseline_available": baseline_available,
        "matched_kernels": len(kernels),
        "missing_baseline_kernels": missing_baseline,
        "compare_enabled": bool(baseline_available and len(kernels) > 0),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-rvv", type=Path, required=True)
    ap.add_argument("--current-cuda", type=Path, required=True)
    ap.add_argument("--baseline-rvv", type=Path, default=None)
    ap.add_argument("--baseline-cuda", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    current_rvv = _load_json(args.current_rvv)
    current_cuda = _load_json(args.current_cuda)
    baseline_rvv = _load_json(args.baseline_rvv)
    baseline_cuda = _load_json(args.baseline_cuda)

    payload = {
        "ok": True,
        "schema_version": "flaggems_timing_delta_v2",
        "current": {
            "rvv": str(args.current_rvv),
            "cuda": str(args.current_cuda),
        },
        "baseline": {
            "rvv": str(args.baseline_rvv) if args.baseline_rvv else "",
            "cuda": str(args.baseline_cuda) if args.baseline_cuda else "",
        },
        "rvv": _backend_delta(current_rvv, baseline_rvv),
        "cuda": _backend_delta(current_cuda, baseline_cuda),
    }
    payload["summary"] = {
        "baseline_compare_ready": bool(payload["rvv"]["compare_enabled"] or payload["cuda"]["compare_enabled"]),
        "matched_kernels_total": int(payload["rvv"]["matched_kernels"]) + int(payload["cuda"]["matched_kernels"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Stage timing delta report written: {args.out}")


if __name__ == "__main__":
    main()
