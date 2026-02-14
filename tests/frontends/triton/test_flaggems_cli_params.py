from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _help_text(script_rel: str) -> str:
    p = subprocess.run(
        [sys.executable, script_rel, "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0
    return f"{p.stdout}\n{p.stderr}"


def test_flaggems_pipeline_cli_uses_new_path_mode_flags() -> None:
    text = _help_text("scripts/triton/flaggems_full_pipeline_verify.py")
    assert "--flaggems-path" in text
    assert "--intentir-mode" in text
    assert "--fallback-policy" in text
    assert "--seed-cache-dir" in text
    assert "--use-llm" not in text
    assert "--no-use-llm" not in text
    assert "--use-intent-ir" not in text
    assert "--intentir-seed-policy" not in text


def test_flaggems_matrix_cli_exposes_cuda_stage_timeout_flags() -> None:
    text = _help_text("scripts/flaggems/run_multibackend_matrix.py")
    assert "--cuda-timeout-sec" in text
    assert "--cuda-compile-timeout-sec" in text
    assert "--cuda-launch-timeout-sec" in text


def test_generic_full_pipeline_cli_hides_flaggems_only_flags() -> None:
    for script_rel in ("scripts/full_pipeline_verify.py", "scripts/triton/full_pipeline_verify.py"):
        text = _help_text(script_rel)
        assert "--use-llm" not in text
        assert "--no-use-llm" not in text
        assert "--use-intent-ir" not in text
        assert "--intentir-seed-policy" not in text


def test_flaggems_pipeline_cli_rejects_invalid_path_mode_combo() -> None:
    p = subprocess.run(
        [
            sys.executable,
            "scripts/triton/flaggems_full_pipeline_verify.py",
            "--list",
            "--flaggems-path",
            "original",
            "--intentir-mode",
            "force_cache",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
    combined = f"{p.stdout}\n{p.stderr}"
    assert "--intentir-mode is only valid when --flaggems-path=intentir" in combined
