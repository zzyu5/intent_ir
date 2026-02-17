"""
Aggregate family batch artifacts into one full196 evidence chain.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage-batches", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--out-run-summary", type=Path, default=None)
    ap.add_argument("--out-status-converged", type=Path, default=None)
    ap.add_argument("--out-coverage-integrity", type=Path, default=None)
    ap.add_argument("--require-dual-pass-total", type=int, default=196)
    ap.add_argument("--intentir-mode", choices=["auto", "force_compile", "force_cache"], default="force_compile")
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

    status_payload = {
        "schema_version": "flaggems_status_converged_v3",
        "generated_at": _utc_now_iso(),
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

    run_summary_payload = {
        "ok": bool(coverage_integrity_ok),
        "suite": "coverage",
        "requested_suite": "coverage",
        "kernel_filter": [],
        "scope_kernels": expected_kernels,
        "coverage_mode": "category_batches",
        "full196_evidence_kind": "batch_aggregate",
        "coverage_batches_expected": categories_expected,
        "coverage_batches_completed": categories_completed,
        "coverage_batches_failed": categories_failed,
        "intentir_mode": str(args.intentir_mode),
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
