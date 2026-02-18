from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_plan_next_batch_and_start_session(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    feature = tmp_path / "feature_list.json"
    progress = tmp_path / "progress.jsonl"
    active = tmp_path / "active_batch.json"
    current_status = tmp_path / "current_status.json"
    session_context = tmp_path / "session_context.json"

    registry.write_text(
        json.dumps(
            {
                "entries": [
                    {"semantic_op": "a", "family": "elementwise_broadcast", "status": "blocked_ir", "status_reason": "no_intentir_mapping", "e2e_spec": None, "intent_ops": []},
                    {"semantic_op": "b", "family": "elementwise_broadcast", "status": "dual_pass", "status_reason": "ok", "e2e_spec": "b2d", "intent_ops": ["add"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    feature.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_feature_list_v2",
                "source_registry_path": str(registry),
                "summary": {
                    "semantic_ops": 2,
                    "by_status": {"blocked_ir": 1, "dual_pass": 1},
                    "by_family": {"elementwise_broadcast": 2},
                    "tasks_total": 2,
                    "by_track": {"coverage": 2},
                },
                "features": [
                    {"semantic_op": "a", "status": "blocked_ir", "reason_code": "no_intentir_mapping", "track": "coverage", "passes": False},
                    {"semantic_op": "b", "status": "dual_pass", "reason_code": "ok", "track": "coverage", "passes": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    progress.write_text("", encoding="utf-8")
    current_status.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_current_status_v1",
                "mode": "mixed_development",
            }
        ),
        encoding="utf-8",
    )
    session_context.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_session_context_v1",
                "read_order": ["current_status", "active_batch"],
                "next_focus": "focus-a",
            }
        ),
        encoding="utf-8",
    )

    p1 = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/plan_next_batch.py",
            "--registry",
            str(registry),
            "--feature-list",
            str(feature),
            "--lane",
            "coverage",
            "--active-batch",
            str(active),
            "--batch-size",
            "1",
            "--no-strict-sync",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p1.returncode == 0
    payload = json.loads(active.read_text(encoding="utf-8"))
    assert payload["items"][0]["semantic_op"] == "a"

    p2 = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/start_session.py",
            "--lane",
            "coverage",
            "--active-batch",
            str(active),
            "--current-status",
            str(current_status),
            "--session-context",
            str(session_context),
            "--print-json",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p2.returncode == 0
    out = json.loads(p2.stdout)
    assert int(out["summary"]["selected_items"]) == 1


def test_end_session_requires_run_and_status_artifacts(tmp_path: Path) -> None:
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/end_session.py",
            "--summary",
            "x",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0


def test_start_session_allows_empty_batch_when_opted_out(tmp_path: Path) -> None:
    active = tmp_path / "active_batch.json"
    current_status = tmp_path / "current_status.json"
    session_context = tmp_path / "session_context.json"
    active.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_active_batch_v2",
                "generated_at": "2026-02-15T00:00:00+00:00",
                "branch": "flaggems",
                "lane": "coverage",
                "batch_size": 10,
                "items": [],
            }
        ),
        encoding="utf-8",
    )
    current_status.write_text(
        json.dumps({"schema_version": "flaggems_current_status_v1", "mode": "maintenance"}),
        encoding="utf-8",
    )
    session_context.write_text(
        json.dumps({"schema_version": "flaggems_session_context_v1", "read_order": [], "next_focus": ""}),
        encoding="utf-8",
    )
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/start_session.py",
            "--lane",
            "coverage",
            "--active-batch",
            str(active),
            "--current-status",
            str(current_status),
            "--session-context",
            str(session_context),
            "--no-require-non-empty",
            "--print-json",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0
    payload = json.loads(p.stdout)
    assert int(payload["summary"]["selected_items"]) == 0


def test_plan_next_batch_workflow_lane_selects_pending_task(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    feature = tmp_path / "feature_list.json"
    active = tmp_path / "active_batch_workflow.json"

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
    feature.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_feature_list_v2",
                "source_registry_path": str(registry),
                "summary": {
                    "semantic_ops": 1,
                    "by_status": {"dual_pass": 1},
                    "by_family": {"elementwise_broadcast": 1},
                    "tasks_total": 2,
                    "by_track": {"coverage": 1, "workflow": 1},
                },
                "features": [
                    {"id": "flaggems::add", "semantic_op": "add", "status": "dual_pass", "track": "coverage", "passes": True},
                    {
                        "id": "workflow::cleanup_v1_archive",
                        "semantic_op": "",
                        "status": "pending",
                        "track": "workflow",
                        "passes": False,
                        "priority": 10,
                        "depends_on": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/plan_next_batch.py",
            "--registry",
            str(registry),
            "--feature-list",
            str(feature),
            "--lane",
            "workflow",
            "--active-batch",
            str(active),
            "--batch-size",
            "2",
            "--no-strict-sync",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(active.read_text(encoding="utf-8"))
    assert payload["lane"] == "workflow"
    assert len(payload["items"]) == 1
    assert payload["items"][0]["id"] == "workflow::cleanup_v1_archive"
