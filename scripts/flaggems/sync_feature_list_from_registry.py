"""
Sync workflow feature list from the frozen FlagGems registry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.providers.flaggems.registry import DEFAULT_REGISTRY_PATH
from pipeline.triton.providers.flaggems.specs import coverage_flaggems_kernel_specs
from pipeline.triton.providers.flaggems.workflow import (
    append_metrics_history,
    build_feature_list_payload,
    dump_json,
    freeze_baseline_snapshot,
    load_json,
    utc_now_iso,
)


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    ap.add_argument("--feature-out", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "feature_list.json"))
    ap.add_argument("--baselines-dir", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "baselines"))
    ap.add_argument(
        "--metrics-history",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "metrics_history.jsonl"),
    )
    ap.add_argument(
        "--roadmap",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "roadmap.json"),
    )
    ap.add_argument(
        "--status-converged",
        type=Path,
        default=(ROOT / "artifacts" / "flaggems_matrix" / "status_converged.json"),
    )
    ap.add_argument(
        "--freeze-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write a timestamped baseline snapshot under workflow/flaggems/state/baselines.",
    )
    args = ap.parse_args()

    registry_payload = load_json(args.registry)
    feature_payload = build_feature_list_payload(
        registry_payload=registry_payload,
        source_registry_path=_to_repo_rel(args.registry),
    )
    feature_path = dump_json(args.feature_out, feature_payload)
    print(f"Feature list synced: {feature_path}")

    summary = dict(feature_payload.get("summary") or {})
    print(
        json.dumps(
            {
                "semantic_ops": summary.get("semantic_ops"),
                "by_status": summary.get("by_status"),
                "by_family": summary.get("by_family"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if bool(args.freeze_baseline):
        coverage_specs = [str(s.name) for s in coverage_flaggems_kernel_specs(flaggems_opset="deterministic_forward", backend_target="rvv")]
        status_path = Path(_to_repo_rel(args.status_converged)) if args.status_converged.is_file() else None
        baseline_path = freeze_baseline_snapshot(
            baselines_dir=args.baselines_dir,
            registry_payload=registry_payload,
            coverage_specs=coverage_specs,
            status_converged_path=status_path,
        )
        print(f"Baseline snapshot frozen: {baseline_path}")

    summary = dict(feature_payload.get("summary") or {})
    append_metrics_history(
        metrics_history_path=args.metrics_history,
        entry={
            "ts": utc_now_iso(),
            "source_registry_path": _to_repo_rel(args.registry),
            "semantic_ops": int(summary.get("semantic_ops") or 0),
            "by_status": dict(summary.get("by_status") or {}),
            "by_family": dict(summary.get("by_family") or {}),
        },
    )
    print(f"Metrics history appended: {args.metrics_history}")

    if not args.roadmap.is_file():
        roadmap = {
            "schema_version": "flaggems_roadmap_v1",
            "generated_at": utc_now_iso(),
            "milestones": [
                {"id": "M1", "target": "blocked_ir<=50 + e2e_specs>=75 + workflow hard gate", "status": "in_progress"},
                {"id": "M2", "target": "blocked_ir<=35 + wave-a batch leaves blocked_ir", "status": "pending"},
                {"id": "M3", "target": "blocked_ir<=20 + reason_code standardized", "status": "pending"},
                {"id": "M4", "target": "blocked_ir==0", "status": "pending"},
                {"id": "M5", "target": "blocked_backend<=15 + cuda determinism>=95%", "status": "pending"},
            ],
            "notes": "Update milestone status as phases converge; do not delete history.",
        }
        dump_json(args.roadmap, roadmap)
        print(f"Roadmap initialized: {args.roadmap}")


if __name__ == "__main__":
    main()
