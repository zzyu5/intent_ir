from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/intentir.py", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def test_intentir_suite_flaggems_full196_dry_run() -> None:
    p = _run(
        "suite",
        "--suite",
        "flaggems-full196",
        "--dry-run",
        "--family",
        "reduction",
        "--no-run-rvv-remote",
    )
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "scripts/flaggems/build_coverage_batches.py" in out
    assert "scripts/flaggems/run_coverage_batches.py" in out
    assert "--family reduction" in out
    assert "--no-run-rvv-remote" in out


def test_intentir_kernel_triton_dry_run() -> None:
    p = _run(
        "kernel",
        "--frontend",
        "triton",
        "--kernel",
        "tanh2d",
        "--dry-run",
        "--no-run-rvv-remote",
    )
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "scripts/flaggems/run_multibackend_matrix.py" in out
    assert "--kernel tanh2d" in out
    assert "--suite smoke" in out


def test_intentir_env_smoke() -> None:
    p = _run("env")
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "repo_root:" in out
    assert "python:" in out
