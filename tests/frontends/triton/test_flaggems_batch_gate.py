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


def _run_gate(
    tmp_path: Path,
    *,
    include_next_focus: bool,
    status: str = "dual_pass",
    require_active_dual_pass: bool = True,
) -> subprocess.CompletedProcess[str]:
    active = tmp_path / "active_batch.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    out = tmp_path / "batch_gate.json"

    active.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_active_batch_v2",
                "lane": "coverage",
                "items": [{"semantic_op": "angle"}],
            }
        ),
        encoding="utf-8",
    )
    run_summary.write_text(json.dumps({"ok": True}), encoding="utf-8")
    status_converged.write_text(
        json.dumps({"entries": [{"semantic_op": "angle", "status": str(status), "reason_code": "runtime_dual_backend_pass"}]}),
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
    handoff_body = "# FlagGems Session Handoff\n"
    if include_next_focus:
        handoff_body += "- Next Focus: wave-a\n"
    handoff.write_text(handoff_body, encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/flaggems/check_batch_gate.py",
        "--active-batch",
        str(active),
        "--run-summary",
        str(run_summary),
        "--status-converged",
        str(status_converged),
        "--progress-log",
        str(progress),
        "--handoff",
        str(handoff),
        "--out",
        str(out),
    ]
    if not require_active_dual_pass:
        cmd.append("--no-require-active-dual-pass")
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def test_check_batch_gate_passes_with_complete_artifacts(tmp_path: Path) -> None:
    p = _run_gate(tmp_path, include_next_focus=True)
    assert p.returncode == 0


def test_check_batch_gate_fails_without_next_focus(tmp_path: Path) -> None:
    p = _run_gate(tmp_path, include_next_focus=False)
    assert p.returncode != 0


def test_check_batch_gate_fails_when_active_not_dual_pass_by_default(tmp_path: Path) -> None:
    p = _run_gate(tmp_path, include_next_focus=True, status="blocked_backend")
    assert p.returncode != 0


def test_check_batch_gate_allows_non_dual_pass_when_opted_out(tmp_path: Path) -> None:
    p = _run_gate(
        tmp_path,
        include_next_focus=True,
        status="blocked_backend",
        require_active_dual_pass=False,
    )
    assert p.returncode == 0


def test_check_batch_gate_uses_scoped_entries_when_scope_enabled(tmp_path: Path) -> None:
    active = tmp_path / "active_batch.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    out = tmp_path / "batch_gate.json"

    active.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_active_batch_v2",
                "lane": "coverage",
                "items": [{"semantic_op": "angle"}],
            }
        ),
        encoding="utf-8",
    )
    run_summary.write_text(json.dumps({"ok": True}), encoding="utf-8")
    status_converged.write_text(
        json.dumps(
            {
                "scope_enabled": True,
                "entries": [
                    {"semantic_op": "angle", "status": "blocked_ir", "reason_code": "no_intentir_mapping"},
                    {"semantic_op": "diag", "status": "dual_pass", "reason_code": "runtime_dual_backend_pass"},
                ],
                "scoped_entries_active": [
                    {"semantic_op": "angle", "status": "dual_pass", "reason_code": "runtime_dual_backend_pass"},
                ],
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
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: wave-a\n", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
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
    assert p.returncode == 0, p.stderr


def test_check_batch_gate_fails_when_active_op_missing_in_scope(tmp_path: Path) -> None:
    active = tmp_path / "active_batch.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    out = tmp_path / "batch_gate.json"

    active.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_active_batch_v2",
                "lane": "coverage",
                "items": [{"semantic_op": "angle"}],
            }
        ),
        encoding="utf-8",
    )
    run_summary.write_text(json.dumps({"ok": True}), encoding="utf-8")
    status_converged.write_text(
        json.dumps(
            {
                "scope_enabled": True,
                "entries": [
                    {"semantic_op": "angle", "status": "dual_pass", "reason_code": "runtime_dual_backend_pass"},
                ],
                "scoped_entries": [
                    {"semantic_op": "diag", "status": "blocked_backend", "reason_code": "diff_fail"},
                ],
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
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: wave-a\n", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
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


def test_check_batch_gate_allows_empty_coverage_batch_without_status_entries(tmp_path: Path) -> None:
    active = tmp_path / "active_batch.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    out = tmp_path / "batch_gate.json"

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
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: none\n", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
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
    assert p.returncode == 0, p.stderr


def test_check_batch_gate_fails_when_coverage_not_fresh_on_head(tmp_path: Path) -> None:
    active = tmp_path / "active_batch.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    current_status = tmp_path / "current_status.json"
    out = tmp_path / "batch_gate.json"

    active.write_text(
        json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "coverage", "items": []}),
        encoding="utf-8",
    )
    run_summary.write_text(json.dumps({"ok": True}), encoding="utf-8")
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: rerun full196\n", encoding="utf-8")
    current_status.write_text(
        json.dumps(
            {
                "full196_validated_commit": "deadbeef",
                "full196_last_ok": True,
                "full196_commits_since_validated": 2,
            }
        ),
        encoding="utf-8",
    )

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--current-status",
            str(current_status),
            "--require-coverage-fresh-on-head",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0


def test_check_batch_gate_mlir_profile_requires_mlir_fresh_on_head(tmp_path: Path) -> None:
    active = tmp_path / "active_batch_mlir.json"
    run_summary = tmp_path / "run_summary_mlir.json"
    status_converged = tmp_path / "status_converged_mlir.json"
    progress = tmp_path / "progress_log_mlir.jsonl"
    handoff = tmp_path / "handoff_mlir.md"
    current_status = tmp_path / "current_status_mlir.json"
    out = tmp_path / "batch_gate_mlir.json"
    head = _head_commit()

    active.write_text(
        json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "mlir_migration", "items": [{"id": "mlir::cutover"}]}),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {"ok": True, "execution_ir": "mlir", "stages": [{"stage": "mlir_llvm_artifacts", "ok": True, "artifact_complete": True}]}
        ),
        encoding="utf-8",
    )
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: mlir cutover\n", encoding="utf-8")
    current_status.write_text(
        json.dumps(
            {
                "mlir_full196_validated_commit": head,
                "full196_last_ok": True,
                "full196_validated_execution_ir": "mlir",
                "mlir_toolchain_ok": True,
                "mlir_cutover_level": "phase3_ready",
            }
        ),
        encoding="utf-8",
    )

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--profile",
            "mlir_migration",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--current-status",
            str(current_status),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr


def test_check_batch_gate_mlir_profile_fails_when_validated_execution_ir_not_mlir(tmp_path: Path) -> None:
    active = tmp_path / "active_batch_mlir.json"
    run_summary = tmp_path / "run_summary_mlir.json"
    status_converged = tmp_path / "status_converged_mlir.json"
    progress = tmp_path / "progress_log_mlir.jsonl"
    handoff = tmp_path / "handoff_mlir.md"
    current_status = tmp_path / "current_status_mlir.json"
    out = tmp_path / "batch_gate_mlir.json"
    head = _head_commit()

    active.write_text(
        json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "mlir_migration", "items": [{"id": "mlir::cutover"}]}),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {"ok": True, "execution_ir": "mlir", "stages": [{"stage": "mlir_llvm_artifacts", "ok": True, "artifact_complete": True}]}
        ),
        encoding="utf-8",
    )
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: mlir cutover\n", encoding="utf-8")
    current_status.write_text(
        json.dumps(
            {
                "mlir_full196_validated_commit": head,
                "full196_last_ok": True,
                "full196_validated_execution_ir": "intent",
                "mlir_toolchain_ok": True,
                "mlir_cutover_level": "phase3_ready",
            }
        ),
        encoding="utf-8",
    )

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--profile",
            "mlir_migration",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--current-status",
            str(current_status),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0


def test_check_batch_gate_mlir_profile_fails_when_toolchain_not_ready(tmp_path: Path) -> None:
    active = tmp_path / "active_batch_mlir.json"
    run_summary = tmp_path / "run_summary_mlir.json"
    status_converged = tmp_path / "status_converged_mlir.json"
    progress = tmp_path / "progress_log_mlir.jsonl"
    handoff = tmp_path / "handoff_mlir.md"
    current_status = tmp_path / "current_status_mlir.json"
    out = tmp_path / "batch_gate_mlir.json"
    head = _head_commit()

    active.write_text(
        json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "mlir_migration", "items": [{"id": "mlir::cutover"}]}),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {"ok": True, "execution_ir": "mlir", "stages": [{"stage": "mlir_llvm_artifacts", "ok": True, "artifact_complete": True}]}
        ),
        encoding="utf-8",
    )
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: mlir cutover\n", encoding="utf-8")
    current_status.write_text(
        json.dumps(
            {
                "mlir_full196_validated_commit": head,
                "full196_last_ok": True,
                "full196_validated_execution_ir": "mlir",
                "mlir_toolchain_ok": False,
                "mlir_cutover_level": "blocked_toolchain",
            }
        ),
        encoding="utf-8",
    )

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--profile",
            "mlir_migration",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--current-status",
            str(current_status),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0


def test_check_batch_gate_requires_all_categories_complete_when_enabled(tmp_path: Path) -> None:
    active = tmp_path / "active_batch.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    out = tmp_path / "batch_gate.json"

    active.write_text(
        json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "coverage", "items": []}),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {
                "ok": True,
                "coverage_mode": "category_batches",
                "full196_evidence_kind": "batch_aggregate",
                "coverage_batches_expected": 7,
                "coverage_batches_completed": 6,
                "coverage_batches_failed": ["attention_sequence"],
            }
        ),
        encoding="utf-8",
    )
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: rerun categories\n", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--require-all-categories-complete",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0


def test_check_batch_gate_category_completion_passes_when_full_aggregate_done(tmp_path: Path) -> None:
    active = tmp_path / "active_batch.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    out = tmp_path / "batch_gate.json"

    active.write_text(
        json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "coverage", "items": []}),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {
                "ok": True,
                "coverage_mode": "category_batches",
                "full196_evidence_kind": "batch_aggregate",
                "coverage_batches_expected": 7,
                "coverage_batches_completed": 7,
                "coverage_batches_failed": [],
            }
        ),
        encoding="utf-8",
    )
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: none\n", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--require-all-categories-complete",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr


def test_check_batch_gate_passes_when_coverage_fresh_on_head(tmp_path: Path) -> None:
    active = tmp_path / "active_batch.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    current_status = tmp_path / "current_status.json"
    out = tmp_path / "batch_gate.json"
    head = _head_commit()

    active.write_text(
        json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "coverage", "items": []}),
        encoding="utf-8",
    )
    run_summary.write_text(json.dumps({"ok": True}), encoding="utf-8")
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: run lane\n", encoding="utf-8")
    current_status.write_text(
        json.dumps(
            {
                "full196_validated_commit": head,
                "full196_last_ok": True,
                "full196_commits_since_validated": 0,
            }
        ),
        encoding="utf-8",
    )

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--current-status",
            str(current_status),
            "--require-coverage-fresh-on-head",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr


def test_check_batch_gate_ir_arch_passes_when_mapping_quality_within_thresholds(tmp_path: Path) -> None:
    active = tmp_path / "active_batch_ir_arch.json"
    run_summary = tmp_path / "run_summary_ir_arch.json"
    status_converged = tmp_path / "status_converged_ir_arch.json"
    progress = tmp_path / "progress_ir_arch.jsonl"
    handoff = tmp_path / "handoff_ir_arch.md"
    out = tmp_path / "batch_gate_ir_arch.json"
    mapping = tmp_path / "mapping_complexity_report.json"

    active.write_text(json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "ir_arch", "items": [{}]}), encoding="utf-8")
    mapping.write_text(
        json.dumps(
            {
                "composition_required_single_intent_ratio": 0.05,
                "global_unique_single_primitive_ratio": 0.30,
            }
        ),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {
                "ok": True,
                "stages": [
                    {"stage": "primitive_reuse", "ok": True},
                    {"stage": "macro_composition", "ok": True},
                    {"stage": "mapping_complexity", "ok": True, "json_path": str(mapping)},
                    {"stage": "mapping_tests", "ok": True},
                    {"stage": "intentir_semantics", "ok": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: ir cleanup\n", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--profile",
            "ir_arch",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
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
    assert p.returncode == 0, p.stderr


def test_check_batch_gate_ir_arch_fails_when_mapping_quality_exceeds_thresholds(tmp_path: Path) -> None:
    active = tmp_path / "active_batch_ir_arch.json"
    run_summary = tmp_path / "run_summary_ir_arch.json"
    status_converged = tmp_path / "status_converged_ir_arch.json"
    progress = tmp_path / "progress_ir_arch.jsonl"
    handoff = tmp_path / "handoff_ir_arch.md"
    out = tmp_path / "batch_gate_ir_arch.json"
    mapping = tmp_path / "mapping_complexity_report.json"

    active.write_text(json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "ir_arch", "items": [{}]}), encoding="utf-8")
    mapping.write_text(
        json.dumps(
            {
                "composition_required_single_intent_ratio": 0.25,
                "global_unique_single_primitive_ratio": 0.65,
            }
        ),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {
                "ok": True,
                "stages": [
                    {"stage": "primitive_reuse", "ok": True},
                    {"stage": "macro_composition", "ok": True},
                    {"stage": "mapping_complexity", "ok": True, "json_path": str(mapping)},
                    {"stage": "mapping_tests", "ok": True},
                    {"stage": "intentir_semantics", "ok": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: ir cleanup\n", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--profile",
            "ir_arch",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
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


def test_check_batch_gate_backend_profile_allows_non_timing_required_stages(tmp_path: Path) -> None:
    active = tmp_path / "active_batch_backend.json"
    run_summary = tmp_path / "run_summary_backend.json"
    status_converged = tmp_path / "status_converged_backend.json"
    progress = tmp_path / "progress_log_backend.jsonl"
    handoff = tmp_path / "handoff_backend.md"
    out = tmp_path / "batch_gate_backend.json"
    rvv_json = tmp_path / "rvv_local.json"
    cuda_json = tmp_path / "cuda_local.json"
    schedule_profiles = tmp_path / "schedule_profiles.json"
    stage_timing_json = tmp_path / "stage_timing_breakdown.json"

    active.write_text(json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "backend_compiler", "items": [{}]}), encoding="utf-8")
    rvv_json.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "kernel": "add2d",
                        "reason_code": "ok",
                        "lower_ms": 1.0,
                        "compile_ms": 2.0,
                        "launch_ms": 3.0,
                        "total_ms": 6.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cuda_json.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "kernel": "add2d",
                        "reason_code": "ok",
                        "lower_ms": 1.5,
                        "compile_ms": 2.5,
                        "launch_ms": 3.5,
                        "total_ms": 7.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    schedule_profiles.write_text(json.dumps({"ok": True}), encoding="utf-8")
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
                    {"stage": "schedule_profiles", "ok": True, "json_path": str(schedule_profiles)},
                ],
            }
        ),
        encoding="utf-8",
    )
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: wave3\n", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--profile",
            "backend_compiler",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--require-stage",
            "rvv_local",
            "--require-stage",
            "cuda_local",
            "--require-stage",
            "schedule_profiles",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr


def test_check_batch_gate_backend_profile_fails_with_codegen_fallback_flag(tmp_path: Path) -> None:
    active = tmp_path / "active_batch_backend.json"
    run_summary = tmp_path / "run_summary_backend.json"
    status_converged = tmp_path / "status_converged_backend.json"
    progress = tmp_path / "progress_log_backend.jsonl"
    handoff = tmp_path / "handoff_backend.md"
    out = tmp_path / "batch_gate_backend.json"
    rvv_json = tmp_path / "rvv_local.json"
    cuda_json = tmp_path / "cuda_local.json"

    active.write_text(json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "backend_compiler", "items": [{}]}), encoding="utf-8")
    rvv_json.write_text(
        json.dumps({"results": [{"kernel": "add2d", "reason_code": "ok", "lower_ms": 1.0, "compile_ms": 2.0, "launch_ms": 3.0, "total_ms": 6.0}]}),
        encoding="utf-8",
    )
    cuda_json.write_text(
        json.dumps({"results": [{"kernel": "add2d", "reason_code": "ok", "lower_ms": 1.5, "compile_ms": 2.5, "launch_ms": 3.5, "total_ms": 7.5}]}),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {
                "ok": True,
                "stages": [
                    {"stage": "rvv_local", "ok": True, "json_path": str(rvv_json)},
                    {
                        "stage": "cuda_local",
                        "ok": True,
                        "json_path": str(cuda_json),
                        "cmd": ["python", "scripts/cuda_backend_smoke.py", "--codegen-mode", "py"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: wave5\n", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--profile",
            "backend_compiler",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
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


def test_check_batch_gate_backend_profile_fails_on_timing_regression_budget(tmp_path: Path) -> None:
    active = tmp_path / "active_batch_backend.json"
    run_summary = tmp_path / "run_summary_backend.json"
    status_converged = tmp_path / "status_converged_backend.json"
    progress = tmp_path / "progress_log_backend.jsonl"
    handoff = tmp_path / "handoff_backend.md"
    out = tmp_path / "batch_gate_backend.json"
    rvv_json = tmp_path / "rvv_local.json"
    cuda_json = tmp_path / "cuda_local.json"
    timing_delta_json = tmp_path / "timing_delta.json"
    stage_timing_json = tmp_path / "stage_timing_breakdown.json"

    active.write_text(json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "backend_compiler", "items": [{}]}), encoding="utf-8")
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
                        {"kernel": "add2d", "total_ms": {"delta_pct": 12.0, "delta_ms": 120.0}},
                    ],
                },
                "cuda": {
                    "compare_enabled": True,
                    "rows": [
                        {"kernel": "add2d", "total_ms": {"delta_pct": 1.0, "delta_ms": 10.0}},
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
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: wave5\n", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--profile",
            "backend_compiler",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--max-total-regression-pct",
            "8",
            "--min-regression-delta-ms",
            "0",
            "--max-regression-ratio",
            "0.0",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0


def test_check_batch_gate_backend_profile_allows_single_outlier_by_default_ratio(tmp_path: Path) -> None:
    active = tmp_path / "active_batch_backend.json"
    run_summary = tmp_path / "run_summary_backend.json"
    status_converged = tmp_path / "status_converged_backend.json"
    progress = tmp_path / "progress_log_backend.jsonl"
    handoff = tmp_path / "handoff_backend.md"
    out = tmp_path / "batch_gate_backend.json"
    rvv_json = tmp_path / "rvv_local.json"
    cuda_json = tmp_path / "cuda_local.json"
    timing_delta_json = tmp_path / "timing_delta.json"
    stage_timing_json = tmp_path / "stage_timing_breakdown.json"

    active.write_text(json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "backend_compiler", "items": [{}]}), encoding="utf-8")
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
                        {"kernel": "mul2d", "total_ms": {"delta_pct": 0.5, "delta_ms": 5.0}},
                    ],
                },
                "cuda": {
                    "compare_enabled": True,
                    "rows": [
                        {"kernel": "add2d", "total_ms": {"delta_pct": 2.0, "delta_ms": 20.0}},
                        {"kernel": "mul2d", "total_ms": {"delta_pct": 1.0, "delta_ms": 10.0}},
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
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: wave5\n", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--profile",
            "backend_compiler",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
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
    assert p.returncode == 0, p.stderr


def test_check_batch_gate_gpu_perf_profile_passes(tmp_path: Path) -> None:
    active = tmp_path / "active_batch_gpu_perf.json"
    run_summary = tmp_path / "run_summary_gpu_perf.json"
    status_converged = tmp_path / "status_converged_gpu_perf.json"
    progress = tmp_path / "progress_log_gpu_perf.jsonl"
    handoff = tmp_path / "handoff_gpu_perf.md"
    current_status = tmp_path / "current_status_gpu_perf.json"
    gpu_perf = tmp_path / "gpu_perf_graph.json"
    out = tmp_path / "batch_gate_gpu_perf.json"
    head = _head_commit()

    active.write_text(json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "coverage", "items": []}), encoding="utf-8")
    gpu_perf.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_gpu_perf_graph_v1",
                "mode": "graph_only",
                "coverage_batches_expected": 7,
                "coverage_batches_completed": 7,
                "coverage_batches_failed": [],
                "devices": [{"gpu_name": "GPU0", "ok": True}],
                "entries": [
                    {
                        "kernel": "add2d",
                        "family": "elementwise_broadcast",
                        "count_in_denominator": True,
                        "ratio": 0.91,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {
                "ok": True,
                "stages": [{"stage": "gpu_perf_graph", "ok": True, "json_path": str(gpu_perf)}],
            }
        ),
        encoding="utf-8",
    )
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: perf\n", encoding="utf-8")
    current_status.write_text(
        json.dumps(
            {
                "gpu_perf_phase": "recomputed_ok",
                "gpu_perf_validated_commit": head,
                "gpu_perf_commits_since_validated": 0,
                "gpu_perf_categories_expected": 7,
                "gpu_perf_categories_completed": 7,
                "gpu_perf_categories_failed": [],
            }
        ),
        encoding="utf-8",
    )

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--profile",
            "gpu_perf",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--current-status",
            str(current_status),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr


def test_check_batch_gate_gpu_perf_profile_fails_when_ratio_below_threshold(tmp_path: Path) -> None:
    active = tmp_path / "active_batch_gpu_perf.json"
    run_summary = tmp_path / "run_summary_gpu_perf.json"
    status_converged = tmp_path / "status_converged_gpu_perf.json"
    progress = tmp_path / "progress_log_gpu_perf.jsonl"
    handoff = tmp_path / "handoff_gpu_perf.md"
    current_status = tmp_path / "current_status_gpu_perf.json"
    gpu_perf = tmp_path / "gpu_perf_graph.json"
    out = tmp_path / "batch_gate_gpu_perf.json"
    head = _head_commit()

    active.write_text(json.dumps({"schema_version": "flaggems_active_batch_v2", "lane": "coverage", "items": []}), encoding="utf-8")
    gpu_perf.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_gpu_perf_graph_v1",
                "mode": "graph_only",
                "coverage_batches_expected": 7,
                "coverage_batches_completed": 7,
                "coverage_batches_failed": [],
                "devices": [{"gpu_name": "GPU0", "ok": False}],
                "entries": [
                    {
                        "kernel": "add2d",
                        "family": "elementwise_broadcast",
                        "count_in_denominator": True,
                        "ratio": 0.73,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {
                "ok": False,
                "stages": [{"stage": "gpu_perf_graph", "ok": False, "json_path": str(gpu_perf)}],
            }
        ),
        encoding="utf-8",
    )
    status_converged.write_text(json.dumps({"entries": []}), encoding="utf-8")
    progress.write_text(
        json.dumps({"run_summary_path": str(run_summary), "status_converged_path": str(status_converged)}) + "\n",
        encoding="utf-8",
    )
    handoff.write_text("# FlagGems Session Handoff\n- Next Focus: perf\n", encoding="utf-8")
    current_status.write_text(
        json.dumps(
            {
                "gpu_perf_phase": "recomputed_failed",
                "gpu_perf_validated_commit": head,
                "gpu_perf_commits_since_validated": 0,
                "gpu_perf_categories_expected": 7,
                "gpu_perf_categories_completed": 7,
                "gpu_perf_categories_failed": [],
            }
        ),
        encoding="utf-8",
    )

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--profile",
            "gpu_perf",
            "--active-batch",
            str(active),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status_converged),
            "--progress-log",
            str(progress),
            "--handoff",
            str(handoff),
            "--current-status",
            str(current_status),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
