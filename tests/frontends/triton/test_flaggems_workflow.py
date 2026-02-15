from __future__ import annotations

import json
from pathlib import Path

from pipeline.triton.flaggems_workflow import (
    append_progress_log,
    append_metrics_history,
    build_active_batch_payload,
    build_current_status_payload,
    build_session_context_payload,
    build_feature_list_payload,
    freeze_baseline_snapshot,
    load_json,
    load_progress_tail,
    select_next_batch,
    summarize_registry,
    validate_feature_list_sync,
    write_handoff,
)


def _sample_registry_payload() -> dict[str, object]:
    return {
        "entries": [
            {
                "semantic_op": "add",
                "family": "elementwise_broadcast",
                "status": "dual_pass",
                "status_reason": "ok",
                "e2e_spec": "add2d",
                "intent_ops": ["add"],
            },
            {
                "semantic_op": "acos",
                "family": "elementwise_broadcast",
                "status": "blocked_ir",
                "status_reason": "no_intentir_mapping",
                "e2e_spec": None,
                "intent_ops": [],
            },
            {
                "semantic_op": "argmax",
                "family": "reduction",
                "status": "blocked_backend",
                "status_reason": "missing_e2e_spec",
                "e2e_spec": None,
                "intent_ops": ["reduce_max"],
            },
            {
                "semantic_op": "cumsum",
                "family": "reduction",
                "status": "blocked_backend",
                "status_reason": "backend_missing_ops",
                "e2e_spec": "cumsum2d",
                "intent_ops": ["cumsum"],
            },
        ]
    }


def test_summarize_registry_counts() -> None:
    summary = summarize_registry(_sample_registry_payload())
    assert summary["semantic_ops"] == 4
    assert summary["by_status"] == {"dual_pass": 1, "blocked_ir": 1, "blocked_backend": 2}
    assert summary["by_family"] == {"elementwise_broadcast": 2, "reduction": 2}


def test_feature_list_payload_and_batch_priority() -> None:
    payload = build_feature_list_payload(
        registry_payload=_sample_registry_payload(),
        source_registry_path="pipeline/triton/flaggems_registry.json",
    )
    features = list(payload["features"])
    by_op = {str(f["semantic_op"]): f for f in features}
    assert by_op["add"]["passes"] is True
    assert by_op["acos"]["next_action"] == "semantic_mapping"
    assert by_op["argmax"]["next_action"] == "add_e2e_spec"
    assert by_op["cumsum"]["next_action"] == "backend_lowering"
    assert by_op["acos"]["track"] == "coverage"
    assert payload["summary"]["tasks_total"] == 4

    batch = select_next_batch(feature_payload=payload, batch_size=3, lane="coverage")
    assert [str(x["semantic_op"]) for x in batch] == ["acos", "argmax", "cumsum"]
    ok, errs = validate_feature_list_sync(
        feature_payload=payload,
        registry_payload=_sample_registry_payload(),
        expected_source_registry_path="pipeline/triton/flaggems_registry.json",
    )
    assert ok is True
    assert errs == []

    mismatch_payload = dict(payload)
    mismatch_payload["source_registry_path"] = "wrong/path.json"
    ok2, errs2 = validate_feature_list_sync(
        feature_payload=mismatch_payload,
        registry_payload=_sample_registry_payload(),
        expected_source_registry_path="pipeline/triton/flaggems_registry.json",
    )
    assert ok2 is False
    assert errs2

    active = build_active_batch_payload(
        batch=batch,
        branch="flaggems",
        batch_size=3,
        lane="coverage",
        feature_list_path="workflow/flaggems/state/feature_list.json",
        status_snapshot_path="workflow/flaggems/state/current_status.json",
        session_context_path="workflow/flaggems/state/session_context.json",
    )
    assert active["schema_version"] == "flaggems_active_batch_v2"
    assert [str(x["semantic_op"]) for x in active["items"]] == ["acos", "argmax", "cumsum"]
    assert active["lane"] == "coverage"


def test_freeze_baseline_snapshot_writes_latest_and_stamped(tmp_path: Path) -> None:
    baselines_dir = tmp_path / "baselines"
    status_converged = tmp_path / "status_converged.json"
    status_converged.write_text("{}", encoding="utf-8")
    stamped = freeze_baseline_snapshot(
        baselines_dir=baselines_dir,
        registry_payload=_sample_registry_payload(),
        coverage_specs=["add2d", "acos2d"],
        status_converged_path=status_converged,
    )
    assert stamped.is_file()
    latest = baselines_dir / "registry_baseline_latest.json"
    assert latest.is_file()
    latest_payload = load_json(latest)
    metrics = dict(latest_payload["metrics"])
    assert metrics["semantic_ops"] == 4
    assert metrics["dual_pass"] == 1
    assert metrics["blocked_ir"] == 1
    assert metrics["blocked_backend"] == 2
    assert metrics["coverage_specs"] == 2
    assert latest_payload["status_converged_path"] == str(status_converged)


def test_progress_log_append_and_handoff_write(tmp_path: Path) -> None:
    progress_log = tmp_path / "progress_log.jsonl"
    append_progress_log(progress_log_path=progress_log, entry={"commit": "a", "summary": "batch-a"})
    append_progress_log(progress_log_path=progress_log, entry={"commit": "b", "summary": "batch-b"})
    lines = progress_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["commit"] == "a"
    assert json.loads(lines[1])["commit"] == "b"

    handoff = tmp_path / "handoff.md"
    write_handoff(handoff_path=handoff, content="# Handoff\n- done\n")
    assert handoff.read_text(encoding="utf-8") == "# Handoff\n- done\n"

    metrics = tmp_path / "metrics_history.jsonl"
    append_metrics_history(metrics_history_path=metrics, entry={"semantic_ops": 4})
    lines = metrics.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["semantic_ops"] == 4
    tail = load_progress_tail(progress_log_path=progress_log, lines=1)
    assert tail and tail[0]["commit"] == "b"


def test_current_status_and_session_context_builders() -> None:
    feature_payload = build_feature_list_payload(
        registry_payload=_sample_registry_payload(),
        source_registry_path="pipeline/triton/flaggems_registry.json",
        manual_tasks=[
            {
                "id": "ir_arch::primitive",
                "track": "ir_arch",
                "status": "pending",
                "passes": False,
            }
        ],
    )
    current = build_current_status_payload(
        branch="flaggems",
        head_commit="abc123",
        feature_payload=feature_payload,
        latest_run_summary_path="artifacts/run_summary.json",
        latest_status_converged_path="artifacts/status_converged.json",
        lane_batch_paths={"coverage": "a.json", "ir_arch": "b.json", "backend_compiler": "c.json"},
    )
    assert current["schema_version"] == "flaggems_current_status_v1"
    assert current["mode"] == "mixed_development"
    assert current["lanes"]["ir_arch"]["pending"] == 1
    assert "mapping_quality" in current
    assert "coverage_integrity_phase" in current
    assert current["script_governance"]["catalog_path"] == "scripts/CATALOG.json"

    ctx = build_session_context_payload(
        git_log_short="a\nb",
        progress_tail=[{"summary": "x"}],
        next_focus="run primitive guard",
        known_risks=["risk-a"],
    )
    assert ctx["schema_version"] == "flaggems_session_context_v1"
    assert ctx["next_focus"] == "run primitive guard"
    assert ctx["must_read_scripts_catalog"] == "scripts/CATALOG.json"


def test_select_next_batch_respects_dependencies_for_non_coverage() -> None:
    feature_payload = {
        "features": [
            {
                "id": "backend_compiler::cuda_pipeline_modularization",
                "track": "backend_compiler",
                "status": "pending",
                "passes": False,
                "priority": 10,
                "depends_on": [],
            },
            {
                "id": "backend_compiler::rvv_pipeline_modularization",
                "track": "backend_compiler",
                "status": "pending",
                "passes": False,
                "priority": 20,
                "depends_on": [],
            },
            {
                "id": "backend_compiler::stage_timing_unification",
                "track": "backend_compiler",
                "status": "pending",
                "passes": False,
                "priority": 30,
                "depends_on": [
                    "backend_compiler::cuda_pipeline_modularization",
                    "backend_compiler::rvv_pipeline_modularization",
                ],
            },
        ]
    }
    batch = select_next_batch(feature_payload=feature_payload, batch_size=10, lane="backend_compiler")
    assert [str(x["id"]) for x in batch] == [
        "backend_compiler::cuda_pipeline_modularization",
        "backend_compiler::rvv_pipeline_modularization",
    ]
