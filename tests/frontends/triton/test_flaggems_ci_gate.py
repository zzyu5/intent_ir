from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pipeline.triton.providers.flaggems.workflow import build_feature_list_payload


ROOT = Path(__file__).resolve().parents[3]


def _run_ci_gate(tmp_path: Path, *, reason_code: str) -> subprocess.CompletedProcess[str]:
    registry = tmp_path / "registry.json"
    feature = tmp_path / "feature_list.json"
    active = tmp_path / "active_batch.json"
    run_summary = tmp_path / "run_summary.json"
    status_converged = tmp_path / "status_converged.json"
    progress = tmp_path / "progress_log.jsonl"
    handoff = tmp_path / "handoff.md"
    out = tmp_path / "ci_gate.json"

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

    return subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/ci_gate.py",
            "--registry",
            str(registry),
            "--feature-list",
            str(feature),
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


def test_ci_gate_passes_with_complete_reason_codes(tmp_path: Path) -> None:
    p = _run_ci_gate(tmp_path, reason_code="runtime_dual_backend_pass")
    assert p.returncode == 0, p.stderr


def test_ci_gate_fails_with_unknown_reason_code(tmp_path: Path) -> None:
    p = _run_ci_gate(tmp_path, reason_code="")
    assert p.returncode != 0
