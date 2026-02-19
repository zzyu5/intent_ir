from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pipeline.triton.providers.flaggems.workflow import build_feature_list_payload


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


def _run_ci_gate(tmp_path: Path, *, reason_code: str) -> subprocess.CompletedProcess[str]:
    registry = tmp_path / "registry.json"
    feature = tmp_path / "feature_list.json"
    active = tmp_path / "active_batch.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    current_status = tmp_path / "current_status.json"
    out = tmp_path / "ci_gate.json"
    head = _head_commit()

    registry_payload = {
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
    registry.write_text(json.dumps(registry_payload), encoding="utf-8")
    feature_payload = build_feature_list_payload(
        registry_payload=registry_payload,
        source_registry_path=str(registry),
    )
    feature.write_text(json.dumps(feature_payload), encoding="utf-8")

    active.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_active_batch_v2",
                "lane": "coverage",
                "items": [],
            }
        ),
        encoding="utf-8",
    )
    run_summary.write_text(json.dumps({"ok": True}), encoding="utf-8")
    status_converged.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "semantic_op": "add",
                        "status": "dual_pass",
                        "reason_code": reason_code,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    progress.write_text(
        json.dumps(
            {
                "run_summary_path": str(run_summary),
                "status_converged_path": str(status_converged),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: nightly gate\n", encoding="utf-8")
    current_status.write_text(
        json.dumps(
            {
                "full196_validated_commit": head,
                "full196_last_ok": True,
                "full196_commits_since_validated": 0,
                "coverage_mode": "category_batches",
                "full196_evidence_kind": "batch_aggregate",
                "coverage_batches_expected": 7,
                "coverage_batches_completed": 7,
                "coverage_batches_failed": [],
            }
        ),
        encoding="utf-8",
    )

    return subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/ci_gate.py",
            "--registry",
            str(registry),
            "--feature-list",
            str(feature),
            "--active-batch-coverage",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--current-status",
            str(current_status),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def test_ci_gate_passes_with_complete_reason_codes(tmp_path: Path) -> None:
    p = _run_ci_gate(tmp_path, reason_code="runtime_dual_backend_pass")
    assert p.returncode == 0, p.stderr


def test_ci_gate_fails_with_unknown_reason_code(tmp_path: Path) -> None:
    p = _run_ci_gate(tmp_path, reason_code="")
    assert p.returncode != 0


def _run_ci_gate_backend_budget(
    tmp_path: Path,
    *,
    max_total_regression_pct: float,
    min_regression_delta_ms: float,
    max_regression_ratio: float,
) -> subprocess.CompletedProcess[str]:
    registry = tmp_path / "registry.json"
    feature = tmp_path / "feature_list.json"
    active_cov = tmp_path / "active_batch_cov.json"
    active_backend = tmp_path / "active_batch_backend.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    current_status = tmp_path / "current_status.json"
    out = tmp_path / "ci_gate_backend.json"
    rvv_json = tmp_path / "rvv_local.json"
    cuda_json = tmp_path / "cuda_local.json"
    timing_delta_json = tmp_path / "timing_delta.json"
    stage_timing_json = tmp_path / "stage_timing_breakdown.json"
    head = _head_commit()

    registry_payload = {
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
    registry.write_text(json.dumps(registry_payload), encoding="utf-8")
    feature_payload = build_feature_list_payload(
        registry_payload=registry_payload,
        source_registry_path=str(registry),
    )
    feature.write_text(json.dumps(feature_payload), encoding="utf-8")

    active_cov.write_text(
        json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "coverage", "items": []}),
        encoding="utf-8",
    )
    active_backend.write_text(
        json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "backend_compiler", "items": [{}]}),
        encoding="utf-8",
    )
    rvv_json.write_text(
        json.dumps({"results": [{"kernel": "add2d", "reason_code": "ok", "lower_ms": 1.0, "compile_ms": 2.0, "launch_ms": 3.0, "total_ms": 6.0}]}),
        encoding="utf-8",
    )
    cuda_json.write_text(
        json.dumps({"results": [{"kernel": "add2d", "reason_code": "ok", "lower_ms": 1.5, "compile_ms": 2.5, "launch_ms": 3.5, "total_ms": 7.5}]}),
        encoding="utf-8",
    )
    timing_delta_json.write_text(
        json.dumps(
            {
                "rvv": {
                    "compare_enabled": True,
                    "rows": [
                        {"kernel": "add2d", "total_ms": {"delta_pct": 20.0, "delta_ms": 120.0}},
                    ],
                },
                "cuda": {
                    "compare_enabled": True,
                    "rows": [
                        {"kernel": "add2d", "total_ms": {"delta_pct": 1.0, "delta_ms": 8.0}},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    stage_timing_json.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_stage_timing_breakdown_v1",
                "backends": {
                    "rvv": {
                        "available": True,
                        "kernel_count": 1,
                        "totals_ms": {"lower_ms": 1.0, "compile_ms": 2.0, "launch_ms": 3.0, "total_ms": 6.0},
                        "avg_ms": {"lower_ms": 1.0, "compile_ms": 2.0, "launch_ms": 3.0, "total_ms": 6.0},
                        "stage_share_pct": {"lower_ms": 16.0, "compile_ms": 33.0, "launch_ms": 50.0},
                    },
                    "cuda": {
                        "available": True,
                        "kernel_count": 1,
                        "totals_ms": {"lower_ms": 1.5, "compile_ms": 2.5, "launch_ms": 3.5, "total_ms": 7.5},
                        "avg_ms": {"lower_ms": 1.5, "compile_ms": 2.5, "launch_ms": 3.5, "total_ms": 7.5},
                        "stage_share_pct": {"lower_ms": 20.0, "compile_ms": 33.0, "launch_ms": 47.0},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {
                "ok": True,
                "stages": [
                    {"stage": "rvv_local", "ok": True, "json_path": str(rvv_json)},
                    {"stage": "cuda_local", "ok": True, "json_path": str(cuda_json)},
                    {"stage": "stage_timing_breakdown", "ok": True, "json_path": str(stage_timing_json)},
                    {"stage": "timing_delta", "ok": True, "json_path": str(timing_delta_json)},
                ],
            }
        ),
        encoding="utf-8",
    )
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps(
            {
                "run_summary_path": str(run_summary),
                "status_converged_path": str(status_converged),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: backend nightly\n", encoding="utf-8")
    current_status.write_text(
        json.dumps(
            {
                "full196_validated_commit": head,
                "full196_last_ok": True,
                "full196_commits_since_validated": 0,
                "coverage_mode": "category_batches",
                "full196_evidence_kind": "batch_aggregate",
                "coverage_batches_expected": 7,
                "coverage_batches_completed": 7,
                "coverage_batches_failed": [],
            }
        ),
        encoding="utf-8",
    )

    return subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/ci_gate.py",
            "--registry",
            str(registry),
            "--feature-list",
            str(feature),
            "--active-batch-coverage",
            str(active_cov),
            "--active-batch-backend-compiler",
            str(active_backend),
            "--profiles",
            "backend_compiler",
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--current-status",
            str(current_status),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--max-total-regression-pct",
            str(max_total_regression_pct),
            "--min-regression-delta-ms",
            str(min_regression_delta_ms),
            "--max-regression-ratio",
            str(max_regression_ratio),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def test_ci_gate_backend_budget_fails_when_ratio_exceeds_threshold(tmp_path: Path) -> None:
    p = _run_ci_gate_backend_budget(
        tmp_path,
        max_total_regression_pct=8.0,
        min_regression_delta_ms=50.0,
        max_regression_ratio=0.0,
    )
    assert p.returncode != 0


def test_ci_gate_backend_budget_passes_with_lenient_ratio(tmp_path: Path) -> None:
    p = _run_ci_gate_backend_budget(
        tmp_path,
        max_total_regression_pct=8.0,
        min_regression_delta_ms=50.0,
        max_regression_ratio=1.0,
    )
    assert p.returncode == 0, p.stderr


def test_ci_gate_fails_when_full196_freshness_is_stale(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    feature = tmp_path / "feature_list.json"
    active = tmp_path / "active_batch.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    current_status = tmp_path / "current_status.json"
    out = tmp_path / "ci_gate_stale.json"

    registry_payload = {
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
    registry.write_text(json.dumps(registry_payload), encoding="utf-8")
    feature_payload = build_feature_list_payload(
        registry_payload=registry_payload,
        source_registry_path=str(registry),
    )
    feature.write_text(json.dumps(feature_payload), encoding="utf-8")
    active.write_text(json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "coverage", "items": []}), encoding="utf-8")
    run_summary.write_text(json.dumps({"ok": True}), encoding="utf-8")
    status_converged.write_text(
        json.dumps({"entries": [{"semantic_op": "add", "status": "dual_pass", "reason_code": "runtime_dual_backend_pass"}]}),
        encoding="utf-8",
    )
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: refresh full196\n", encoding="utf-8")
    current_status.write_text(
        json.dumps(
            {
                "full196_validated_commit": "deadbeef",
                "full196_last_ok": True,
                "full196_commits_since_validated": 3,
            }
        ),
        encoding="utf-8",
    )

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/ci_gate.py",
            "--registry",
            str(registry),
            "--feature-list",
            str(feature),
            "--active-batch-coverage",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--current-status",
            str(current_status),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0


def _run_ci_gate_mlir(
    tmp_path: Path,
    *,
    mlir_commit: str,
    validated_execution_ir: str = "mlir",
) -> subprocess.CompletedProcess[str]:
    registry = tmp_path / "registry.json"
    feature = tmp_path / "feature_list.json"
    active_cov = tmp_path / "active_batch_cov.json"
    active_mlir = tmp_path / "active_batch_mlir.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    current_status = tmp_path / "current_status.json"
    out = tmp_path / "ci_gate_mlir.json"
    head = _head_commit()

    registry_payload = {
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
    registry.write_text(json.dumps(registry_payload), encoding="utf-8")
    feature_payload = build_feature_list_payload(
        registry_payload=registry_payload,
        source_registry_path=str(registry),
    )
    feature.write_text(json.dumps(feature_payload), encoding="utf-8")
    active_cov.write_text(
        json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "coverage", "items": []}),
        encoding="utf-8",
    )
    active_mlir.write_text(
        json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "mlir_migration", "items": [{}]}),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {
                "ok": True,
                "suite": "coverage",
                "full_coverage_run": True,
                "full196_evidence_kind": "batch_aggregate",
            }
        ),
        encoding="utf-8",
    )
    status_converged.write_text(
        json.dumps({"entries": [{"semantic_op": "add", "status": "dual_pass", "reason_code": "runtime_dual_backend_pass"}]}),
        encoding="utf-8",
    )
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: mlir freshness\n", encoding="utf-8")
    current_status.write_text(
        json.dumps(
            {
                "full196_validated_commit": head,
                "full196_last_ok": True,
                "full196_validated_execution_ir": "intent",
                "full196_commits_since_validated": 0,
                "coverage_mode": "category_batches",
                "full196_evidence_kind": "batch_aggregate",
                "coverage_batches_expected": 7,
                "coverage_batches_completed": 7,
                "coverage_batches_failed": [],
                "mlir_full196_validated_commit": str(mlir_commit),
            }
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "scripts/flaggems/ci_gate.py",
        "--registry",
        str(registry),
        "--feature-list",
        str(feature),
        "--active-batch-coverage",
        str(active_cov),
        "--active-batch-mlir-migration",
        str(active_mlir),
        "--profiles",
        "mlir_migration",
        "--run-summary",
        str(run_summary),
        "--status-converged",
        str(status_converged),
        "--current-status",
        str(current_status),
        "--progress-log",
        str(progress),
        "--handoff",
        str(handoff),
        "--out",
        str(out),
    ]
    if validated_execution_ir:
        payload = json.loads(current_status.read_text(encoding="utf-8"))
        payload["full196_validated_execution_ir"] = str(validated_execution_ir)
        current_status.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def test_ci_gate_mlir_profile_passes_when_mlir_fresh_on_head(tmp_path: Path) -> None:
    p = _run_ci_gate_mlir(tmp_path, mlir_commit=_head_commit(), validated_execution_ir="mlir")
    assert p.returncode == 0, p.stderr


def test_ci_gate_mlir_profile_fails_when_mlir_commit_stale(tmp_path: Path) -> None:
    p = _run_ci_gate_mlir(tmp_path, mlir_commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", validated_execution_ir="mlir")
    assert p.returncode != 0
