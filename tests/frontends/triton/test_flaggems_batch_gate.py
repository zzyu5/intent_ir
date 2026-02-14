from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


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
                "schema_version": "flaggems_active_batch_v1",
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
                "schema_version": "flaggems_active_batch_v1",
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
                "schema_version": "flaggems_active_batch_v1",
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
