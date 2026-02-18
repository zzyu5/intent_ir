from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _run_nightly(tmp_path: Path, *extra: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    out_root = tmp_path / "daily"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/nightly_maintenance.py",
            "--dry-run",
            "--out-root",
            str(out_root),
            "--date-tag",
            "20990101",
            "--run-name",
            "ut",
            *extra,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    summary = json.loads((out_root / "20990101" / "ut" / "nightly_maintenance_summary.json").read_text(encoding="utf-8"))
    return p, summary


def test_nightly_maintenance_dry_run_writes_summary(tmp_path: Path) -> None:
    p, summary = _run_nightly(tmp_path)
    assert p.returncode == 0, p.stderr
    assert summary["mode"] == "dry-run"
    matrix_cmd = list(summary["commands"]["matrix"])
    assert "scripts/intentir.py" in matrix_cmd
    assert "suite" in matrix_cmd
    assert "flaggems-full196" in matrix_cmd
    assert "--intentir-miss-policy" in matrix_cmd
    assert "--run-rvv-remote" in matrix_cmd
    assert "--rvv-use-key" in matrix_cmd
    assert "--allow-cuda-skip" in matrix_cmd
    assert "--cases-limit" in matrix_cmd


def test_nightly_maintenance_toggle_flags(tmp_path: Path) -> None:
    p, summary = _run_nightly(
        tmp_path,
        "--no-run-rvv-remote",
        "--no-skip-rvv-local",
        "--no-rvv-use-key",
        "--no-allow-cuda-skip",
        "--write-registry",
    )
    assert p.returncode == 0, p.stderr
    matrix_cmd = list(summary["commands"]["matrix"])
    assert "--no-run-rvv-remote" in matrix_cmd
    assert "--no-skip-rvv-local" in matrix_cmd
    assert "--no-rvv-use-key" in matrix_cmd
    assert "--no-allow-cuda-skip" in matrix_cmd
    assert "--write-registry" in matrix_cmd


def test_nightly_maintenance_single_run_mode_uses_matrix_script(tmp_path: Path) -> None:
    p, summary = _run_nightly(tmp_path, "--coverage-mode", "single_run")
    assert p.returncode == 0, p.stderr
    matrix_cmd = list(summary["commands"]["matrix"])
    assert "scripts/intentir.py" in matrix_cmd
    assert "suite" in matrix_cmd
    assert "--suite" in matrix_cmd
    assert "flaggems-coverage-single" in matrix_cmd


def test_nightly_maintenance_rejects_intentir_mode_for_original_path(tmp_path: Path) -> None:
    out_root = tmp_path / "daily"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/nightly_maintenance.py",
            "--dry-run",
            "--out-root",
            str(out_root),
            "--date-tag",
            "20990101",
            "--run-name",
            "ut_invalid",
            "--flaggems-path",
            "original",
            "--intentir-mode",
            "force_compile",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
