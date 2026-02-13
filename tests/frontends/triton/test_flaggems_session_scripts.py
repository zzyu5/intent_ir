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
                "schema_version": "flaggems_feature_list_v1",
                "source_registry_path": str(registry),
                "summary": {"semantic_ops": 2, "by_status": {"blocked_ir": 1, "dual_pass": 1}, "by_family": {"elementwise_broadcast": 2}},
                "features": [
                    {"semantic_op": "a", "status": "blocked_ir", "reason_code": "no_intentir_mapping"},
                    {"semantic_op": "b", "status": "dual_pass", "reason_code": "ok"},
                ],
            }
        ),
        encoding="utf-8",
    )
    progress.write_text("", encoding="utf-8")

    p1 = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/plan_next_batch.py",
            "--registry",
            str(registry),
            "--feature-list",
            str(feature),
            "--progress-log",
            str(progress),
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
            "--active-batch",
            str(active),
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
