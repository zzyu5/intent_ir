from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    script_path = ROOT / "scripts" / "flaggems" / "cleanup_artifacts.py"
    spec = importlib.util.spec_from_file_location("cleanup_artifacts", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_build_cleanup_plan_keeps_baseline_and_purges_patterns(tmp_path: Path) -> None:
    mod = _load_module()
    artifacts = tmp_path / "artifacts"
    full_run = artifacts / "flaggems_matrix" / "daily" / "20260226" / "full196_head_refresh_v23_strict_o3"
    gpu_run = artifacts / "flaggems_matrix" / "daily" / "20260226" / "gpu_perf_head_refresh_v14_strict_policy_refresh4"
    wave_run = artifacts / "flaggems_matrix" / "daily" / "20260226" / "mlir_contract_wave_v8_strict" / "matrix_wave64_v1"
    toolchains = artifacts / "toolchains"
    _touch(full_run / "run_summary.json", "{}")
    _touch(gpu_run / "run_summary.json", "{}")
    _touch(wave_run / "run_summary.json", "{}")
    _touch(wave_run / "_triton_cache" / "x.bin")
    _touch(toolchains / "mlir-current" / "README", "ok")
    _touch(artifacts / "tmp" / "foo.txt")
    _touch(artifacts / "random_dir" / "foo.txt")

    current_status = {
        "full196_last_run": str(full_run / "run_summary.json"),
        "gpu_perf_last_run": str(gpu_run / "run_summary.json"),
        "latest_artifacts": {"run_summary": str(wave_run / "run_summary.json")},
    }
    policy = {
        "schema_version": "flaggems_artifact_retention_policy_v1",
        "mode": "minimal_baseline",
        "keep_runs": ["full196_validated", "gpu_perf_validated", "latest_mlir_wave"],
        "keep_dirs": [str(toolchains)],
        "require_distinct_latest_mlir_wave": True,
        "purge_patterns": ["_triton_cache", "_triton_dump", "tmp*", "_tmp*", "torch_extensions/*"],
    }
    plan = mod.build_cleanup_plan(
        artifacts_root=artifacts,
        policy=policy,
        current_status=current_status,
        progress_rows=[],
        purge_toolchains=False,
    )

    preserve_roots = set(plan.get("preserve_roots") or [])
    assert str(full_run) in preserve_roots
    assert str(gpu_run) in preserve_roots
    assert str(wave_run) in preserve_roots
    assert str(toolchains) in preserve_roots

    delete_rows = list(plan.get("delete_candidates") or [])
    delete_paths = {str(row.get("path") or "") for row in delete_rows}
    assert str(artifacts / "tmp") in delete_paths
    assert str(artifacts / "random_dir") in delete_paths
    assert str(wave_run / "_triton_cache") in delete_paths
    assert str(full_run) not in delete_paths
    assert str(gpu_run) not in delete_paths
    assert str(toolchains) not in delete_paths
    assert list(plan.get("strict_errors") or []) == []


def test_cleanup_artifacts_execute_removes_unkept_paths(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    reports_root = tmp_path / "reports"
    full_run = artifacts / "flaggems_matrix" / "daily" / "20260226" / "full196_head_refresh_v23_strict_o3"
    gpu_run = artifacts / "flaggems_matrix" / "daily" / "20260226" / "gpu_perf_head_refresh_v14_strict_policy_refresh4"
    wave_run = artifacts / "flaggems_matrix" / "daily" / "20260226" / "mlir_contract_wave_v8_strict" / "matrix_wave64_v1"
    _touch(full_run / "run_summary.json", "{}")
    _touch(gpu_run / "run_summary.json", "{}")
    _touch(wave_run / "run_summary.json", "{}")
    _touch(wave_run / "_triton_dump" / "x.ttir")
    _touch(artifacts / "tmp" / "foo.txt")

    policy_path = tmp_path / "policy.json"
    status_path = tmp_path / "current_status.json"
    progress_path = tmp_path / "progress_log.jsonl"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_artifact_retention_policy_v1",
                "mode": "minimal_baseline",
                "keep_runs": ["full196_validated", "gpu_perf_validated", "latest_mlir_wave"],
                "keep_dirs": [],
                "require_distinct_latest_mlir_wave": True,
                "purge_patterns": ["_triton_cache", "_triton_dump", "tmp*", "_tmp*", "torch_extensions/*"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps(
            {
                "full196_last_run": str(full_run / "run_summary.json"),
                "gpu_perf_last_run": str(gpu_run / "run_summary.json"),
                "latest_artifacts": {"run_summary": str(wave_run / "run_summary.json")},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    progress_path.write_text("", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/cleanup_artifacts.py",
            "--policy",
            str(policy_path),
            "--current-status",
            str(status_path),
            "--progress-log",
            str(progress_path),
            "--artifacts-root",
            str(artifacts),
            "--reports-root",
            str(reports_root),
            "--execute",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr

    assert (full_run / "run_summary.json").is_file()
    assert (gpu_run / "run_summary.json").is_file()
    assert (wave_run / "run_summary.json").is_file()
    assert not (wave_run / "_triton_dump").exists()
    assert not (artifacts / "tmp").exists()

    date_dirs = sorted([d for d in reports_root.iterdir() if d.is_dir()])
    assert date_dirs
    latest = date_dirs[-1]
    assert (latest / "plan.json").is_file()
    assert (latest / "deleted.jsonl").is_file()
    assert (latest / "summary.json").is_file()


def test_cleanup_artifacts_fails_when_latest_mlir_wave_falls_back_to_gpu_run(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    reports_root = tmp_path / "reports"
    full_run = artifacts / "flaggems_matrix" / "daily" / "20260226" / "full196_head_refresh_v23_strict_o3"
    gpu_run = artifacts / "flaggems_matrix" / "daily" / "20260226" / "gpu_perf_head_refresh_v14_strict_policy_refresh4"
    _touch(full_run / "run_summary.json", "{}")
    _touch(gpu_run / "run_summary.json", "{}")

    policy_path = tmp_path / "policy.json"
    status_path = tmp_path / "current_status.json"
    progress_path = tmp_path / "progress_log.jsonl"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_artifact_retention_policy_v1",
                "mode": "minimal_baseline",
                "keep_runs": ["full196_validated", "gpu_perf_validated", "latest_mlir_wave"],
                "keep_dirs": [],
                "require_distinct_latest_mlir_wave": True,
                "purge_patterns": ["_triton_cache", "_triton_dump", "tmp*", "_tmp*", "torch_extensions/*"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps(
            {
                "full196_last_run": str(full_run / "run_summary.json"),
                "gpu_perf_last_run": str(gpu_run / "run_summary.json"),
                "latest_artifacts": {"run_summary": str(gpu_run / "run_summary.json")},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    progress_path.write_text("", encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/cleanup_artifacts.py",
            "--policy",
            str(policy_path),
            "--current-status",
            str(status_path),
            "--progress-log",
            str(progress_path),
            "--artifacts-root",
            str(artifacts),
            "--reports-root",
            str(reports_root),
            "--execute",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
    msg = f"{p.stdout}\n{p.stderr}"
    assert "strict validation failed" in msg
    assert "latest_mlir_wave" in msg
