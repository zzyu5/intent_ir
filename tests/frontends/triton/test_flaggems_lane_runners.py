from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_run_ir_arch_batch_dry_run(tmp_path: Path) -> None:
    out_dir = tmp_path / "ir_arch"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/run_ir_arch_batch.py",
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    run_summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    status = json.loads((out_dir / "status_converged.json").read_text(encoding="utf-8"))
    stage_names = [row["stage"] for row in run_summary["stages"]]
    assert stage_names == ["primitive_reuse", "macro_composition", "mapping_tests", "intentir_semantics"]
    assert status["lane"] == "ir_arch"


def test_run_backend_compiler_batch_dry_run(tmp_path: Path) -> None:
    out_dir = tmp_path / "backend"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/run_backend_compiler_batch.py",
            "--out-dir",
            str(out_dir),
            "--dry-run",
            "--kernel",
            "add2d",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads((out_dir / "backend_compiler_batch_summary.json").read_text(encoding="utf-8"))
    assert payload["lane"] == "backend_compiler"
    assert "scripts/flaggems/run_multibackend_matrix.py" in payload["cmd"]
