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

from pipeline.triton.flaggems_registry import DEFAULT_REGISTRY_PATH
from pipeline.triton.flaggems_specs import coverage_flaggems_kernel_specs
from pipeline.triton.flaggems_workflow import (
    build_feature_list_payload,
    dump_json,
    freeze_baseline_snapshot,
    load_json,
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


if __name__ == "__main__":
    main()
