from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _write_common_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    registry = tmp_path / "registry.json"
    run_summary = tmp_path / "run_summary.json"
    status = tmp_path / "status_converged.json"
    registry.write_text(
        json.dumps(
            {
                "entries": [
                    {"semantic_op": "add", "status": "dual_pass"},
                    {"semantic_op": "mul", "status": "dual_pass"},
                ]
            }
        ),
        encoding="utf-8",
    )
    run_summary.write_text(
        json.dumps(
            {
                "ok": True,
                "stages": [
                    {"stage": "pipeline", "ok": True},
                    {"stage": "rvv_local", "ok": True},
                    {"stage": "cuda_local", "ok": True},
                    {"stage": "converge_status", "ok": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    status.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "semantic_op": "add",
                        "status": "dual_pass",
                        "reason_code": "runtime_dual_backend_pass",
                        "runtime": {"provider": {"exists": True}},
                    },
                    {
                        "semantic_op": "mul",
                        "status": "dual_pass",
                        "reason_code": "runtime_dual_backend_pass",
                        "runtime": {"provider": {"exists": True}},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return registry, run_summary, status


def test_recompute_coverage_integrity_passes_on_complete_evidence(tmp_path: Path) -> None:
    registry, run_summary, status = _write_common_files(tmp_path)
    out = tmp_path / "coverage_integrity.json"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/recompute_coverage_integrity.py",
            "--registry",
            str(registry),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["coverage_integrity_ok"] is True
    assert payload["semantic_ops_total"] == 2
    assert payload["dual_pass_entries"] == 2


def test_recompute_coverage_integrity_fails_when_stage_missing(tmp_path: Path) -> None:
    registry, run_summary, status = _write_common_files(tmp_path)
    run_summary.write_text(json.dumps({"ok": False, "stages": [{"stage": "pipeline", "ok": False}]}), encoding="utf-8")
    out = tmp_path / "coverage_integrity.json"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/recompute_coverage_integrity.py",
            "--registry",
            str(registry),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["coverage_integrity_ok"] is False


def test_recompute_coverage_integrity_accepts_stage_aliases(tmp_path: Path) -> None:
    registry, run_summary, status = _write_common_files(tmp_path)
    run_summary.write_text(
        json.dumps(
            {
                "ok": True,
                "stages": [
                    {"stage": "provider_report_precheck", "ok": True},
                    {"stage": "rvv_local", "ok": True},
                    {"stage": "cuda_local", "ok": True},
                    {"stage": "converge", "ok": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "coverage_integrity.json"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/recompute_coverage_integrity.py",
            "--registry",
            str(registry),
            "--run-summary",
            str(run_summary),
            "--status-converged",
            str(status),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["coverage_integrity_ok"] is True
    assert payload["required_stage_status"]["pipeline"] is True
    assert payload["required_stage_status"]["converge_status"] is True
