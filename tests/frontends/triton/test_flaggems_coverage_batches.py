from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_build_coverage_batches_from_repo_registry(tmp_path: Path) -> None:
    out = tmp_path / "coverage_batches.json"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/build_coverage_batches.py",
            "--registry",
            "pipeline/triton/flaggems_registry.json",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "flaggems_coverage_batches_v1"
    assert payload["family_order"] == [
        "elementwise_broadcast",
        "reduction",
        "norm_activation",
        "index_scatter_gather",
        "matmul_linear",
        "conv_pool_interp",
        "attention_sequence",
    ]
    assert payload["summary"]["semantic_ops_total"] == 196
    assert payload["summary"]["kernels_total"] == 158


def test_aggregate_coverage_batches_requires_all_families(tmp_path: Path) -> None:
    coverage_batches = tmp_path / "coverage_batches.json"
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    coverage_batches.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_coverage_batches_v1",
                "family_order": ["f1", "f2"],
                "batches": [
                    {"family": "f1", "semantic_ops": ["op1"], "kernels": ["k1"], "semantic_count": 1, "kernel_count": 1},
                    {"family": "f2", "semantic_ops": ["op2"], "kernels": ["k2"], "semantic_count": 1, "kernel_count": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    f1 = runs_root / "family_f1"
    f1.mkdir(parents=True, exist_ok=True)
    (f1 / "run_summary.json").write_text(
        json.dumps({"ok": True, "scope_kernels": ["k1"]}),
        encoding="utf-8",
    )
    (f1 / "status_converged.json").write_text(
        json.dumps({"entries": [{"semantic_op": "op1", "status": "dual_pass", "reason_code": "ok"}]}),
        encoding="utf-8",
    )
    # family_f2 intentionally missing -> aggregate should fail.
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/aggregate_coverage_batches.py",
            "--coverage-batches",
            str(coverage_batches),
            "--runs-root",
            str(runs_root),
            "--require-dual-pass-total",
            "2",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
    run_summary = json.loads((runs_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["ok"] is False
    assert run_summary["coverage_batches_completed"] == 1
    assert run_summary["coverage_batches_failed"] == ["f2"]


def test_aggregate_coverage_batches_passes_on_complete_two_family_fixture(tmp_path: Path) -> None:
    coverage_batches = tmp_path / "coverage_batches.json"
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    coverage_batches.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_coverage_batches_v1",
                "family_order": ["f1", "f2"],
                "batches": [
                    {"family": "f1", "semantic_ops": ["op1"], "kernels": ["k1"], "semantic_count": 1, "kernel_count": 1},
                    {"family": "f2", "semantic_ops": ["op2"], "kernels": ["k2"], "semantic_count": 1, "kernel_count": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    for family, op, kernel in [("f1", "op1", "k1"), ("f2", "op2", "k2")]:
        d = runs_root / f"family_{family}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "run_summary.json").write_text(
            json.dumps({"ok": True, "scope_kernels": [kernel]}),
            encoding="utf-8",
        )
        (d / "status_converged.json").write_text(
            json.dumps({"entries": [{"semantic_op": op, "status": "dual_pass", "reason_code": "ok"}]}),
            encoding="utf-8",
        )

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/aggregate_coverage_batches.py",
            "--coverage-batches",
            str(coverage_batches),
            "--runs-root",
            str(runs_root),
            "--require-dual-pass-total",
            "2",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    run_summary = json.loads((runs_root / "run_summary.json").read_text(encoding="utf-8"))
    status_converged = json.loads((runs_root / "status_converged.json").read_text(encoding="utf-8"))
    coverage_integrity = json.loads((runs_root / "coverage_integrity.json").read_text(encoding="utf-8"))
    assert run_summary["ok"] is True
    assert run_summary["full196_evidence_kind"] == "batch_aggregate"
    assert run_summary["coverage_mode"] == "category_batches"
    assert run_summary["coverage_batches_completed"] == 2
    assert status_converged["counts_global"]["dual_pass"] == 2
    assert coverage_integrity["coverage_integrity_ok"] is True


def test_run_coverage_batches_dry_run_uses_family_pipeline_dirs(tmp_path: Path) -> None:
    coverage_batches = tmp_path / "coverage_batches.json"
    out_root = tmp_path / "runs"
    coverage_batches.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_coverage_batches_v1",
                "family_order": ["f1", "f2"],
                "batches": [
                    {
                        "family": "f1",
                        "semantic_ops": ["op1"],
                        "kernels": ["k1"],
                        "semantic_count": 1,
                        "kernel_count": 1,
                    },
                    {
                        "family": "f2",
                        "semantic_ops": ["op2"],
                        "kernels": ["k2"],
                        "semantic_count": 1,
                        "kernel_count": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/run_coverage_batches.py",
            "--coverage-batches",
            str(coverage_batches),
            "--out-root",
            str(out_root),
            "--dry-run",
            "--no-resume",
            "--no-run-rvv-remote",
            "--allow-cuda-skip",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr

    runs_payload = json.loads((out_root / "coverage_batch_runs.json").read_text(encoding="utf-8"))
    assert runs_payload["ok"] is True
    seed_cache_dir = str(out_root / "seed_cache")
    rows = {str(row["family"]): row for row in runs_payload["families"]}
    assert set(rows) == {"f1", "f2"}
    for family in ("f1", "f2"):
        row = rows[family]
        expected_pipeline_dir = str(out_root / f"family_{family}" / "pipeline_reports")
        assert row["chunk_count"] == 1
        chunks = list(row["chunks"])
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk["pipeline_out_dir"] == expected_pipeline_dir
        assert chunk["seed_cache_dir"] == seed_cache_dir
        cmd = list(chunk["cmd"])
        assert "--pipeline-out-dir" in cmd
        assert "--seed-cache-dir" in cmd
        assert "--skip-rvv-local" in cmd
        assert expected_pipeline_dir in cmd
        assert seed_cache_dir in cmd

    # Dry-run intentionally skips aggregation.
    assert not (out_root / "run_summary.json").exists()


def test_run_coverage_batches_dry_run_supports_family_chunking(tmp_path: Path) -> None:
    coverage_batches = tmp_path / "coverage_batches.json"
    out_root = tmp_path / "runs"
    coverage_batches.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_coverage_batches_v1",
                "family_order": ["f1"],
                "batches": [
                    {
                        "family": "f1",
                        "semantic_ops": ["op1"],
                        "kernels": ["k1", "k2", "k3"],
                        "semantic_count": 1,
                        "kernel_count": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/run_coverage_batches.py",
            "--coverage-batches",
            str(coverage_batches),
            "--out-root",
            str(out_root),
            "--family-kernel-chunk-size",
            "2",
            "--dry-run",
            "--no-resume",
            "--run-rvv-remote",
            "--skip-rvv-local",
            "--allow-cuda-skip",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    runs_payload = json.loads((out_root / "coverage_batch_runs.json").read_text(encoding="utf-8"))
    assert runs_payload["ok"] is True
    assert len(runs_payload["families"]) == 1
    row = runs_payload["families"][0]
    assert row["chunk_enabled"] is True
    assert row["chunk_count"] == 2
    chunks = list(row["chunks"])
    assert len(chunks) == 2
    chunk0_cmd = list(chunks[0]["cmd"])
    chunk1_cmd = list(chunks[1]["cmd"])
    assert "--skip-rvv-local" in chunk0_cmd
    assert "--run-rvv-remote" in chunk0_cmd
    assert chunk0_cmd.count("--kernel") == 2
    assert chunk1_cmd.count("--kernel") == 1
