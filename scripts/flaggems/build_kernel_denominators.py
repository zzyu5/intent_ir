#!/usr/bin/env python3
"""Build a consolidated kernel denominator report.

This is a lightweight, evidence-driven snapshot that answers:
- semantic ops denominator (196) vs executable kernel denominator (159)
- which semantic ops are missing from gpu_perf and why
- which Triton-native coverage kernels are missing from gpu_perf

The report is intended to live under workflow state so it can be committed and
reviewed like other workflow truth artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intent_ir.utils.repo_state import repo_state  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _find_latest(root: Path, rel_glob: str) -> Path | None:
    try:
        paths = [p for p in root.glob(rel_glob) if p.is_file()]
    except Exception:
        paths = []
    if not paths:
        return None
    paths.sort(key=lambda p: float(p.stat().st_mtime), reverse=True)
    return paths[0]


def _coverage_denominators(coverage_batches: dict[str, Any]) -> dict[str, Any]:
    batches = [b for b in list(coverage_batches.get("batches") or []) if isinstance(b, dict)]
    semantic_ops_total = 0
    kernels_total = 0
    families: list[str] = []
    for b in batches:
        semantic_ops_total += len(list(b.get("semantic_ops") or []))
        kernels_total += len(list(b.get("kernels") or []))
        fam = str(b.get("family") or "").strip()
        if fam:
            families.append(fam)
    family_order = [str(x) for x in list(coverage_batches.get("family_order") or []) if str(x).strip()]
    return {
        "schema_version": str(coverage_batches.get("schema_version") or ""),
        "generated_at": str(coverage_batches.get("generated_at") or ""),
        "families": sorted(set(families)),
        "families_count": int(len(set(families))),
        "family_order": list(family_order),
        "semantic_ops_total": int(semantic_ops_total),
        "kernels_total": int(kernels_total),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coverage-batches",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "coverage_batches.json"),
        help="Path to workflow coverage_batches.json (default: workflow/flaggems/state/coverage_batches.json)",
    )
    ap.add_argument(
        "--perf-missing-report",
        type=Path,
        default=None,
        help="Path to perf_missing_kernel_report_v1.json (default: auto-discover latest under artifacts/validation_rounds)",
    )
    ap.add_argument(
        "--triton-native-missing-report",
        type=Path,
        default=None,
        help="Path to perf_missing_kernel_report_v3_triton_native.json (default: auto-discover latest under artifacts/validation_rounds)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "kernel_denominators.json"),
        help="Output JSON path (default: workflow/flaggems/state/kernel_denominators.json)",
    )
    args = ap.parse_args()

    cov_path = Path(args.coverage_batches)
    if not cov_path.is_file():
        raise SystemExit(f"missing coverage_batches.json: {cov_path}")
    cov = _load_json(cov_path)
    cov_den = _coverage_denominators(cov)

    perf_path = Path(args.perf_missing_report) if args.perf_missing_report is not None else None
    if perf_path is None:
        perf_path = _find_latest(ROOT, "artifacts/validation_rounds/*/perf_missing_kernel_report_v1.json")
    perf_payload: dict[str, Any] | None = None
    if perf_path is not None and perf_path.is_file():
        perf_payload = _load_json(perf_path)

    triton_path = (
        Path(args.triton_native_missing_report) if args.triton_native_missing_report is not None else None
    )
    if triton_path is None:
        triton_path = _find_latest(ROOT, "artifacts/validation_rounds/*/perf_missing_kernel_report_v3_triton_native.json")
    triton_payload: dict[str, Any] | None = None
    if triton_path is not None and triton_path.is_file():
        triton_payload = _load_json(triton_path)

    out: dict[str, Any] = {
        "schema_version": "intentir_kernel_denominators_v1",
        "generated_at_utc": _utc_now_iso(),
        "repo": repo_state(root=ROOT),
        "sources": {
            "coverage_batches": str(cov_path.relative_to(ROOT)),
            "perf_missing_kernel_report_v1": (
                str(perf_path.relative_to(ROOT)) if perf_path is not None else ""
            ),
            "perf_missing_kernel_report_v3_triton_native": (
                str(triton_path.relative_to(ROOT)) if triton_path is not None else ""
            ),
        },
        "coverage_batches": cov_den,
        "perf_missing_semantic_ops_in_gpu_perf": {},
        "triton_native_perf_gap": {},
        "notes": {
            "semantic_ops_total": "Semantic-op denominator (196) is not equal to executable kernel specs denominator (159).",
            "gpu_perf_denominator": "gpu_perf graph is based on executable kernel specs (coverage_batches.kernels).",
        },
    }

    if perf_payload is not None:
        counts = dict(perf_payload.get("counts") or {})
        out["perf_missing_semantic_ops_in_gpu_perf"] = {
            "schema_version": str(perf_payload.get("schema_version") or ""),
            "generated_at_utc": str(perf_payload.get("generated_at_utc") or ""),
            "counts": {
                "semantic_ops_total": counts.get("semantic_ops_total"),
                "gpu_perf_entry_count": counts.get("gpu_perf_entry_count"),
                "gpu_perf_unique_kernel_count": counts.get("gpu_perf_unique_kernel_count"),
                "missing_semantic_ops_in_perf_count": counts.get("missing_semantic_ops_in_perf_count"),
            },
            "registry_counts": dict(perf_payload.get("registry_counts") or {}),
            "missing_reason_counts": dict(perf_payload.get("missing_reason_counts") or {}),
            "missing_semantic_ops": list(perf_payload.get("missing_semantic_ops") or []),
            "gpu_perf_policy": dict(perf_payload.get("gpu_perf_policy") or {}),
            "sources": dict(perf_payload.get("sources") or {}),
        }
    else:
        out["perf_missing_semantic_ops_in_gpu_perf"] = {
            "error": "perf_missing_kernel_report_v1.json not found (pass --perf-missing-report or generate under artifacts/validation_rounds)",
        }

    if triton_payload is not None:
        out["triton_native_perf_gap"] = {
            "generated_at": str(triton_payload.get("generated_at") or ""),
            "triton_native_expected_kernels": triton_payload.get("triton_native_expected_kernels"),
            "triton_native_pass_count": triton_payload.get("triton_native_pass_count"),
            "triton_native_fail_count": triton_payload.get("triton_native_fail_count"),
            "triton_native_missing_in_gpu_perf_count": triton_payload.get("triton_native_missing_in_gpu_perf_count"),
            "triton_native_missing_in_gpu_perf": list(triton_payload.get("triton_native_missing_in_gpu_perf") or []),
            "sources": dict(triton_payload.get("sources") or {}),
            "notes": dict(triton_payload.get("notes") or {}),
        }
    else:
        out["triton_native_perf_gap"] = {
            "error": "perf_missing_kernel_report_v3_triton_native.json not found (pass --triton-native-missing-report or generate under artifacts/validation_rounds)",
        }

    out_path = _dump_json(Path(args.out), out)
    rel = str(out_path.relative_to(ROOT))
    print(rel)


if __name__ == "__main__":
    main()
