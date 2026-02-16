from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _head_commit() -> str:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    return str(p.stdout or "").strip()


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
    assert "full196_validated_commit" in status_payload
    assert "full196_commits_since_validated" in status_payload
    assert "full196_validated_mode" in status_payload
    assert "full196_validated_scope" in status_payload
    assert "full196_validated_with_rvv_remote" in status_payload
    assert status_payload["script_governance"]["catalog_path"].endswith("scripts/CATALOG.json")
    assert context_payload["schema_version"] == "flaggems_session_context_v1"
    assert context_payload["next_focus"] == "focus-y"
    assert context_payload["must_read_scripts_catalog"].endswith("scripts/CATALOG.json")
    assert "ir_arch" in list(context_payload.get("active_lanes") or [])


def test_build_workflow_state_activates_ir_arch_lane_when_mapping_quality_breaches(tmp_path: Path) -> None:
    feature = tmp_path / "feature_list.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    current = tmp_path / "current_status.json"
    context = tmp_path / "session_context.json"
    # Coverage-only list with no pending tasks, but mapping complexity intentionally high.
    feature.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_feature_list_v2",
                "summary": {
                    "semantic_ops": 4,
                    "by_status": {"dual_pass": 4},
                    "by_family": {"elementwise_broadcast": 4},
                    "tasks_total": 4,
                    "by_track": {"coverage": 4},
                },
                "features": [
                    {"id": "flaggems::a", "track": "coverage", "family": "elementwise_broadcast", "status": "dual_pass", "passes": True, "intent_ops": ["add"]},
                    {"id": "flaggems::b", "track": "coverage", "family": "elementwise_broadcast", "status": "dual_pass", "passes": True, "intent_ops": ["sub"]},
                    {"id": "flaggems::c", "track": "coverage", "family": "elementwise_broadcast", "status": "dual_pass", "passes": True, "intent_ops": ["mul"]},
                    {"id": "flaggems::d", "track": "coverage", "family": "elementwise_broadcast", "status": "dual_pass", "passes": True, "intent_ops": ["div"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    progress.write_text("", encoding="utf-8")
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: none\n", encoding="utf-8")
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
    assert "ir_arch" in list(status_payload.get("active_lanes") or [])
    assert "ir_arch" in list(context_payload.get("active_lanes") or [])
    assert "ir_arch" in dict(status_payload.get("next_focus_by_lane") or {})


def test_build_workflow_state_prefers_latest_full196_run_over_latest_partial(tmp_path: Path) -> None:
    feature = tmp_path / "feature_list.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    current = tmp_path / "current_status.json"
    context = tmp_path / "session_context.json"
    full_run = tmp_path / "full196_run_summary.json"
    partial_run = tmp_path / "partial_run_summary.json"
    full_cov = tmp_path / "coverage_integrity_full196.json"
    partial_status = tmp_path / "status_partial.json"
    head_commit = _head_commit()

    feature.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_feature_list_v2",
                "summary": {
                    "semantic_ops": 1,
                    "by_status": {"dual_pass": 1},
                    "by_family": {"elementwise_broadcast": 1},
                    "tasks_total": 1,
                    "by_track": {"coverage": 1},
                },
                "features": [
                    {"id": "flaggems::add", "track": "coverage", "status": "dual_pass", "passes": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    full_cov.write_text(
        json.dumps({"coverage_integrity_ok": True}),
        encoding="utf-8",
    )
    full_run.write_text(
        json.dumps(
            {
                "ok": True,
                "suite": "coverage",
                "kernel_filter": [],
                "scope_kernels": ["add2d"],
                "stages": [
                    {
                        "stage": "coverage_integrity",
                        "ok": True,
                        "json_path": str(full_cov),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    partial_run.write_text(
        json.dumps(
            {
                "ok": True,
                "suite": "coverage",
                "kernel_filter": ["add2d"],
                "scope_kernels": ["add2d"],
                "stages": [],
            }
        ),
        encoding="utf-8",
    )
    partial_status.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        (
            json.dumps(
                {
                    "summary": "full196",
                    "run_ok": True,
                    "commit": head_commit,
                    "run_summary_path": str(full_run),
                    "status_converged_path": str(partial_status),
                }
            )
            + "\n"
            + json.dumps({"summary": "selected8", "run_ok": True, "run_summary_path": str(partial_run), "status_converged_path": str(partial_status)})
            + "\n"
        ),
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: full196 verify\n", encoding="utf-8")

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
    assert status_payload["full196_last_run"].endswith("full196_run_summary.json")
    assert status_payload["latest_artifacts"]["run_summary"].endswith("partial_run_summary.json")
    assert status_payload["coverage_integrity_phase"] == "recomputed_ok"
    assert status_payload["full196_last_ok"] is True
    assert status_payload["full196_validated_commit"] == head_commit
    assert status_payload["full196_commits_since_validated"] == 0
    assert status_payload["full196_validated_scope"] == "coverage_158_kernels_to_196_semantics"
    assert status_payload["full196_validated_mode"] == ""
    assert status_payload["full196_validated_with_rvv_remote"] is False


def test_build_workflow_state_accepts_full196_with_resolved_kernel_filter(tmp_path: Path) -> None:
    feature = tmp_path / "feature_list.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    current = tmp_path / "current_status.json"
    context = tmp_path / "session_context.json"
    full_run = tmp_path / "full196_run_summary.json"
    full_cov = tmp_path / "coverage_integrity_full196.json"
    status_json = tmp_path / "status_full196.json"
    head_commit = _head_commit()

    feature.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_feature_list_v2",
                "summary": {
                    "semantic_ops": 196,
                    "by_status": {"dual_pass": 196},
                    "by_family": {"elementwise_broadcast": 125},
                    "tasks_total": 196,
                    "by_track": {"coverage": 196},
                },
                "features": [
                    {"id": "flaggems::abs", "track": "coverage", "status": "dual_pass", "passes": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    full_cov.write_text(json.dumps({"coverage_integrity_ok": False}), encoding="utf-8")
    full_run.write_text(
        json.dumps(
            {
                "ok": False,
                "suite": "coverage",
                "kernel_filter": ["abs2d", "add2d"],
                "scope_kernels": ["abs2d", "add2d"],
                "stages": [
                    {
                        "stage": "coverage_integrity",
                        "ok": False,
                        "json_path": str(full_cov),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    status_json.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps(
            {
                "summary": "full196",
                "run_ok": False,
                "commit": head_commit,
                "run_summary_path": str(full_run),
                "status_converged_path": str(status_json),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: full196 retry\n", encoding="utf-8")

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
    assert status_payload["full196_last_run"].endswith("full196_run_summary.json")
    assert status_payload["coverage_integrity_phase"] == "recomputed_failed"
    assert status_payload["full196_last_ok"] is False
    assert status_payload["full196_validated_commit"] == head_commit
    assert status_payload["full196_commits_since_validated"] == 0


def test_build_workflow_state_prefers_coverage_integrity_over_run_ok(tmp_path: Path) -> None:
    feature = tmp_path / "feature_list.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    current = tmp_path / "current_status.json"
    context = tmp_path / "session_context.json"
    full_run = tmp_path / "full196_run_summary.json"
    full_cov = tmp_path / "coverage_integrity_full196.json"
    status_json = tmp_path / "status_full196.json"
    head_commit = _head_commit()

    feature.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_feature_list_v2",
                "summary": {
                    "semantic_ops": 196,
                    "by_status": {"dual_pass": 196},
                    "by_family": {"elementwise_broadcast": 125},
                    "tasks_total": 196,
                    "by_track": {"coverage": 196},
                },
                "features": [
                    {"id": "flaggems::abs", "track": "coverage", "status": "dual_pass", "passes": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    full_cov.write_text(json.dumps({"coverage_integrity_ok": True}), encoding="utf-8")
    full_run.write_text(
        json.dumps(
            {
                "ok": False,
                "suite": "coverage",
                "kernel_filter": ["abs2d", "add2d"],
                "scope_kernels": ["abs2d", "add2d"],
                "stages": [
                    {
                        "stage": "coverage_integrity",
                        "ok": False,
                        "json_path": str(full_cov),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    status_json.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps(
            {
                "summary": "full196",
                "run_ok": False,
                "commit": head_commit,
                "run_summary_path": str(full_run),
                "status_converged_path": str(status_json),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: full196 follow-up\n", encoding="utf-8")

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
    assert status_payload["coverage_integrity_phase"] == "recomputed_ok"
    assert status_payload["full196_last_ok"] is True
    assert status_payload["full196_validated_commit"] == head_commit
    assert status_payload["full196_commits_since_validated"] == 0
