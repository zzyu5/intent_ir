from __future__ import annotations

import json
from pathlib import Path

from pipeline.triton.flaggems_workflow import (
    append_progress_log,
    build_feature_list_payload,
    freeze_baseline_snapshot,
    load_json,
    select_next_batch,
    summarize_registry,
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

    batch = select_next_batch(feature_payload=payload, batch_size=3)
    assert [str(x["semantic_op"]) for x in batch] == ["acos", "argmax", "cumsum"]


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
