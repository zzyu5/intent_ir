from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_sync_feature_list_mixed_merges_manual_tracks(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    feature_out = tmp_path / "feature_list.json"
    templates = tmp_path / "task_templates.json"
    metrics = tmp_path / "metrics.jsonl"
    baselines = tmp_path / "baselines"
    registry.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "semantic_op": "add",
                        "family": "elementwise_broadcast",
                        "status": "dual_pass",
                        "status_reason": "runtime_dual_backend_pass",
                        "e2e_spec": "add2d",
                        "intent_ops": ["add"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    templates.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_task_templates_v1",
                "tasks": [
                    {
                        "id": "ir_arch::x",
                        "track": "ir_arch",
                        "status": "pending",
                        "passes": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/sync_feature_list_mixed.py",
            "--registry",
            str(registry),
            "--feature-out",
            str(feature_out),
            "--task-templates",
            str(templates),
            "--metrics-history",
            str(metrics),
            "--baselines-dir",
            str(baselines),
            "--no-freeze-baseline",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(feature_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "flaggems_feature_list_v2"
    assert payload["summary"]["tasks_total"] == 2
    assert payload["summary"]["by_track"]["coverage"] == 1
    assert payload["summary"]["by_track"]["ir_arch"] == 1


def test_build_workflow_state_writes_current_and_context(tmp_path: Path) -> None:
    feature = tmp_path / "feature_list.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    current = tmp_path / "current_status.json"
    context = tmp_path / "session_context.json"
    feature.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_feature_list_v2",
                "summary": {
                    "semantic_ops": 1,
                    "by_status": {"dual_pass": 1},
                    "by_family": {"elementwise_broadcast": 1},
                    "tasks_total": 2,
                    "by_track": {"coverage": 1, "ir_arch": 1},
                },
                "features": [
                    {"id": "flaggems::add", "track": "coverage", "status": "dual_pass", "passes": True},
                    {"id": "ir_arch::primitive", "track": "ir_arch", "status": "pending", "passes": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    progress.write_text(
        json.dumps(
            {
                "summary": "x",
                "run_ok": False,
                "run_summary_path": "a.json",
                "status_converged_path": "b.json",
                "next_focus": "focus-x",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: focus-y\n", encoding="utf-8")
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/build_workflow_state.py",
            "--feature-list",
            str(feature),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--current-status-out",
            str(current),
            "--session-context-out",
            str(context),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    status_payload = json.loads(current.read_text(encoding="utf-8"))
    context_payload = json.loads(context.read_text(encoding="utf-8"))
    assert status_payload["schema_version"] == "flaggems_current_status_v1"
    assert status_payload["mode"] == "mixed_development"
    assert "coverage_integrity_phase" in status_payload
    assert "mapping_quality" in status_payload
    assert status_payload["script_governance"]["catalog_path"].endswith("scripts/CATALOG.json")
    assert context_payload["schema_version"] == "flaggems_session_context_v1"
    assert context_payload["next_focus"] == "focus-y"
    assert context_payload["must_read_scripts_catalog"].endswith("scripts/CATALOG.json")
    assert "ir_arch" in list(context_payload.get("active_lanes") or [])
