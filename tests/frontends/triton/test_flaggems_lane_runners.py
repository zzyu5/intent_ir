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
    cmd = list(payload["cmd"])
    assert "--cuda-runtime-backend" in cmd
    assert "--cuda-codegen-mode" not in cmd
    assert "--cuda-codegen-strict" not in cmd
    assert "--cuda-cpp-engine" not in cmd
    assert "--cuda-cpp-engine-strict" not in cmd


def test_run_backend_compiler_batch_uses_kernel_manifest_when_kernel_not_given(tmp_path: Path) -> None:
    out_dir = tmp_path / "backend_manifest"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"kernels": ["diag2d", "topk2d"]}), encoding="utf-8")
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/run_backend_compiler_batch.py",
            "--out-dir",
            str(out_dir),
            "--dry-run",
            "--kernel-manifest",
            str(manifest),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads((out_dir / "backend_compiler_batch_summary.json").read_text(encoding="utf-8"))
    cmd = list(payload["cmd"])
    # Order in command preserves input order; both manifest kernels should be forwarded.
    assert cmd.count("--kernel") == 2
    assert "diag2d" in cmd
    assert "topk2d" in cmd


def test_run_backend_compiler_batch_supports_manifest_chunking(tmp_path: Path) -> None:
    out_dir = tmp_path / "backend_chunk"
    manifest = tmp_path / "manifest_chunk.json"
    manifest.write_text(
        json.dumps({"kernels": ["abs2d", "add2d", "clamp2d", "diag2d", "topk2d"]}),
        encoding="utf-8",
    )
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/run_backend_compiler_batch.py",
            "--out-dir",
            str(out_dir),
            "--dry-run",
            "--kernel-manifest",
            str(manifest),
            "--chunk-size",
            "2",
            "--chunk-index",
            "1",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads((out_dir / "backend_compiler_batch_summary.json").read_text(encoding="utf-8"))
    assert payload["kernel_count_requested"] == 5
    assert payload["kernel_count_selected"] == 2
    assert payload["kernels_selected"] == ["clamp2d", "diag2d"]
    assert payload["chunk"]["enabled"] is True
    cmd = list(payload["cmd"])
    assert cmd.count("--kernel") == 2
    assert "clamp2d" in cmd
    assert "diag2d" in cmd


def test_run_backend_compiler_batch_rejects_unknown_kernel(tmp_path: Path) -> None:
    out_dir = tmp_path / "backend_unknown"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/run_backend_compiler_batch.py",
            "--out-dir",
            str(out_dir),
            "--dry-run",
            "--kernel",
            "not_a_real_kernel",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
    assert "unknown kernel(s)" in str(p.stderr)
