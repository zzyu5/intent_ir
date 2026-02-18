"""
Sync mixed-track feature list from FlagGems registry + workflow task templates.

Track model:
- coverage: registry-derived semantic-op items
- ir_arch: IntentIR abstraction quality tasks
- backend_compiler: backend compilerization tasks
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


def _load_manual_tasks(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return []
    out: list[dict[str, object]] = []
    for task in tasks:
        if isinstance(task, dict):
            out.append(dict(task))
    return out


def _ensure_default_mlir_tasks(tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    has_mlir = any(str(t.get("track") or "") == "mlir_migration" for t in tasks)
    if has_mlir:
        return tasks
    defaults = [
        {
            "id": "mlir_migration::module_bridge_bootstrap",
            "track": "mlir_migration",
            "task_type": "compilerization",
            "status": "pending",
            "passes": False,
            "priority": 10,
            "gate_profile": "mlir_migration",
            "reason_code": "pending",
            "next_action": "implement_intent_mlir_module",
            "acceptance": [
                "intent_to_mlir_roundtrip_passes",
                "mlir_pass_pipeline_upstream_midend_available",
            ],
            "depends_on": [],
            "evidence_paths": [],
            "description": "Bootstrap IntentIR <-> MLIR dual-track bridge and pass orchestration.",
        }
    ]
    return [*tasks, *defaults]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    ap.add_argument("--feature-out", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "feature_list.json"))
    ap.add_argument(
        "--task-templates",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "task_templates.json"),
        help="Manual non-coverage tasks merged into feature list.",
    )
    ap.add_argument("--baselines-dir", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "baselines"))
    ap.add_argument(
        "--metrics-history",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "metrics_history.jsonl"),
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
    manual_tasks = _ensure_default_mlir_tasks(_load_manual_tasks(args.task_templates))
    feature_payload = build_feature_list_payload(
        registry_payload=registry_payload,
        source_registry_path=_to_repo_rel(args.registry),
        manual_tasks=manual_tasks,
    )
    feature_path = dump_json(args.feature_out, feature_payload)
    print(f"Feature list synced (mixed): {feature_path}")

    summary = dict(feature_payload.get("summary") or {})
    print(
        json.dumps(
            {
                "semantic_ops": summary.get("semantic_ops"),
                "by_status": summary.get("by_status"),
                "by_family": summary.get("by_family"),
                "tasks_total": summary.get("tasks_total"),
                "by_track": summary.get("by_track"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if bool(args.freeze_baseline):
        from pipeline.triton.providers.flaggems.specs import (  # noqa: PLC0415
            coverage_flaggems_kernel_specs,
        )

        coverage_specs = [
            str(s.name)
            for s in coverage_flaggems_kernel_specs(
                flaggems_opset="deterministic_forward",
                backend_target="rvv",
            )
        ]
        status_path = Path(_to_repo_rel(args.status_converged)) if args.status_converged.is_file() else None
        baseline_path = freeze_baseline_snapshot(
            baselines_dir=args.baselines_dir,
            registry_payload=registry_payload,
            coverage_specs=coverage_specs,
            status_converged_path=status_path,
        )
        print(f"Baseline snapshot frozen: {baseline_path}")

    append_metrics_history(
        metrics_history_path=args.metrics_history,
        entry={
            "ts": utc_now_iso(),
            "source_registry_path": _to_repo_rel(args.registry),
            "semantic_ops": int(summary.get("semantic_ops") or 0),
            "by_status": dict(summary.get("by_status") or {}),
            "by_family": dict(summary.get("by_family") or {}),
            "by_track": dict(summary.get("by_track") or {}),
            "tasks_total": int(summary.get("tasks_total") or 0),
        },
    )
    print(f"Metrics history appended: {args.metrics_history}")


if __name__ == "__main__":
    main()
