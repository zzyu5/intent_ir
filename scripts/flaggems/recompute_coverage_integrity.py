"""
Recompute coverage integrity from matrix artifacts.

This script is meant for post-rewrite periods where prior dual_pass results
must be revalidated against fresh evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _required_stage_ok(run_summary: dict[str, Any], stage: str) -> bool:
    stages = [s for s in list(run_summary.get("stages") or []) if isinstance(s, dict)]
    for row in stages:
        if str(row.get("stage") or "") != stage:
            continue
        return bool(row.get("ok"))
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"))
    ap.add_argument("--run-summary", type=Path, required=True)
    ap.add_argument("--status-converged", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.registry.is_file():
        raise FileNotFoundError(f"registry not found: {args.registry}")
    if not args.run_summary.is_file():
        raise FileNotFoundError(f"run summary not found: {args.run_summary}")
    if not args.status_converged.is_file():
        raise FileNotFoundError(f"status converged not found: {args.status_converged}")

    registry = _load_json(args.registry)
    run_summary = _load_json(args.run_summary)
    status = _load_json(args.status_converged)

    entries = [e for e in list(status.get("entries") or []) if isinstance(e, dict)]
    semantic_total = len([e for e in list(registry.get("entries") or []) if isinstance(e, dict)])
    dual_pass = sum(1 for e in entries if str(e.get("status") or "") == "dual_pass")
    unknown_reason = sum(1 for e in entries if str(e.get("reason_code") or "").strip() in {"", "unknown"})
    artifact_missing = sum(1 for e in entries if not bool((e.get("runtime") or {}).get("provider", {}).get("exists")))

    required_stage_status = {
        "pipeline": _required_stage_ok(run_summary, "pipeline"),
        "rvv_local": _required_stage_ok(run_summary, "rvv_local"),
        "cuda_local": _required_stage_ok(run_summary, "cuda_local"),
        "converge_status": _required_stage_ok(run_summary, "converge_status"),
    }
    stage_all_ok = all(required_stage_status.values())

    payload = {
        "schema_version": "flaggems_coverage_integrity_v1",
        "registry_path": str(args.registry),
        "run_summary_path": str(args.run_summary),
        "status_converged_path": str(args.status_converged),
        "semantic_ops_total": int(semantic_total),
        "status_entries_total": len(entries),
        "dual_pass_entries": int(dual_pass),
        "unknown_reason_code_entries": int(unknown_reason),
        "provider_artifact_missing_entries": int(artifact_missing),
        "required_stage_status": required_stage_status,
        "required_stage_all_ok": bool(stage_all_ok),
        "determinability_ok": bool(unknown_reason == 0),
        "artifact_completeness_ok": bool(artifact_missing == 0),
        "coverage_integrity_ok": bool(
            stage_all_ok
            and dual_pass == semantic_total
            and unknown_reason == 0
            and artifact_missing == 0
        ),
    }
    payload["evidence_hash"] = _sha1_text(
        json.dumps(
            {
                "run_summary": run_summary,
                "status_converged": status,
                "registry_counts": registry.get("counts"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )

    out = args.out
    if out is None:
        out = Path(args.run_summary).parent / "coverage_integrity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Coverage integrity report written: {out}")
    raise SystemExit(0 if bool(payload.get("coverage_integrity_ok")) else 1)


if __name__ == "__main__":
    main()
