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
    seed_cache_dir = tmp_path / "seed_cache_shared"
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
            "--seed-cache-dir",
            str(seed_cache_dir),
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
    seed_cache_dir_s = str(seed_cache_dir)
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
        assert chunk["seed_cache_dir"] == seed_cache_dir_s
        cmd = list(chunk["cmd"])
        assert "--pipeline-out-dir" in cmd
        assert "--seed-cache-dir" in cmd
        assert "--skip-rvv-local" in cmd
        assert expected_pipeline_dir in cmd
        assert seed_cache_dir_s in cmd

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


def test_materialize_family_outputs_scopes_single_chunk_semantics(tmp_path: Path) -> None:
    from scripts.flaggems.run_coverage_batches import _materialize_family_outputs

    family_out = tmp_path / "family_f1"
    family_out.mkdir(parents=True, exist_ok=True)
    chunk_status = family_out / "status_converged.json"
    chunk_run_summary = family_out / "run_summary.json"

    chunk_run_summary.write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    chunk_status.write_text(
        json.dumps(
            {
                "entries": [
                    {"semantic_op": "op1", "status": "dual_pass", "reason_code": "ok"},
                    {"semantic_op": "op2", "status": "blocked_backend", "reason_code": "lowering_missing_op"},
                ]
            }
        ),
        encoding="utf-8",
    )
    chunk_rows = [
        {
            "chunk": "chunk_001",
            "ok": True,
            "rc": 0,
            "out_dir": str(family_out),
            "run_summary_path": str(chunk_run_summary),
            "status_converged_path": str(chunk_status),
            "kernel_count": 1,
            "kernels": ["k1"],
        }
    ]
    family_ok, run_summary_path, status_path = _materialize_family_outputs(
        family="f1",
        semantics=["op1"],
        kernels=["k1"],
        family_out=family_out,
        chunk_rows=chunk_rows,
    )
    assert family_ok is True
    assert run_summary_path == family_out / "run_summary.json"
    assert status_path == family_out / "status_converged.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    entries = list(status_payload["entries"])
    assert len(entries) == 1
    assert entries[0]["semantic_op"] == "op1"
    assert entries[0]["status"] == "dual_pass"
    assert status_payload["counts_global"] == {"dual_pass": 1}


def test_materialize_family_outputs_prefers_complete_chunk_evidence(tmp_path: Path) -> None:
    from scripts.flaggems.run_coverage_batches import _materialize_family_outputs

    family_out = tmp_path / "family_norm"
    family_out.mkdir(parents=True, exist_ok=True)

    chunk1_status = family_out / "chunk1_status.json"
    chunk1_run = family_out / "chunk1_run.json"
    chunk2_status = family_out / "chunk2_status.json"
    chunk2_run = family_out / "chunk2_run.json"

    chunk1_run.write_text(json.dumps({"ok": False}), encoding="utf-8")
    chunk2_run.write_text(json.dumps({"ok": True}), encoding="utf-8")

    chunk1_status.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "semantic_op": "tanh",
                        "status": "blocked_backend",
                        "reason_code": "diff_fail",
                        "artifact_complete": True,
                        "determinability": True,
                        "in_scope_kernel_alias": True,
                        "compiler_stage": {"provider_report": "present"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    chunk2_status.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "semantic_op": "tanh",
                        "status": "blocked_backend",
                        "reason_code": "provider_report_missing",
                        "artifact_complete": False,
                        "determinability": True,
                        "in_scope_kernel_alias": False,
                        "compiler_stage": {"provider_report": "missing"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    chunk_rows = [
        {
            "chunk": "chunk_001",
            "ok": False,
            "rc": 1,
            "out_dir": str(family_out),
            "run_summary_path": str(chunk1_run),
            "status_converged_path": str(chunk1_status),
            "kernel_count": 1,
            "kernels": ["tanh2d"],
        },
        {
            "chunk": "chunk_002",
            "ok": True,
            "rc": 0,
            "out_dir": str(family_out),
            "run_summary_path": str(chunk2_run),
            "status_converged_path": str(chunk2_status),
            "kernel_count": 1,
            "kernels": ["vector_norm2d"],
        },
    ]

    family_ok, _, status_path = _materialize_family_outputs(
        family="norm_activation",
        semantics=["tanh"],
        kernels=["tanh2d", "vector_norm2d"],
        family_out=family_out,
        chunk_rows=chunk_rows,
    )
    assert family_ok is False
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    entry = status_payload["entries"][0]
    assert entry["semantic_op"] == "tanh"
    assert entry["reason_code"] == "diff_fail"
    assert entry["artifact_complete"] is True


def test_compute_stage_timing_breakdown_includes_mlir_metrics(tmp_path: Path) -> None:
    rvv_json = tmp_path / "rvv.json"
    cuda_json = tmp_path / "cuda.json"
    reports = tmp_path / "pipeline_reports"
    out = tmp_path / "stage_timing_breakdown.json"
    rvv_json.write_text(
        json.dumps({"results": [{"kernel": "k1", "reason_code": "ok", "lower_ms": 1.0, "compile_ms": 2.0, "launch_ms": 3.0, "total_ms": 6.0}]}),
        encoding="utf-8",
    )
    cuda_json.write_text(
        json.dumps({"results": [{"kernel": "k1", "reason_code": "ok", "lower_ms": 1.5, "compile_ms": 2.5, "launch_ms": 3.5, "total_ms": 7.5}]}),
        encoding="utf-8",
    )
    reports.mkdir(parents=True, exist_ok=True)
    for kernel, parse_ms, pass_ms, lower_ms in [("k1", 1.0, 10.0, 4.0), ("k2", 2.0, 11.0, 5.0)]:
        (reports / f"{kernel}.json").write_text(
            json.dumps(
                {
                    "mlir": {
                        "mlir_parse_ms": parse_ms,
                        "mlir_pass_ms": pass_ms,
                        "mlir_lower_ms": lower_ms,
                        "upstream": {"passes": [{"name": "p0"}]},
                        "midend": {"passes": [{"name": "p1"}, {"name": "p2"}]},
                        "downstream": {"passes": [{"name": "p3"}]},
                    }
                }
            ),
            encoding="utf-8",
        )
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/compute_stage_timing_breakdown.py",
            "--rvv-json",
            str(rvv_json),
            "--cuda-json",
            str(cuda_json),
            "--pipeline-reports-dir",
            str(reports),
            "--kernel",
            "k1",
            "--kernel",
            "k2",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "flaggems_stage_timing_breakdown_v1"
    assert payload["mlir"]["available"] is True
    assert payload["mlir"]["kernel_count"] == 2
    assert payload["mlir"]["totals_ms"]["mlir_parse_ms"] == 3.0
    assert payload["mlir"]["totals_ms"]["mlir_pass_ms"] == 21.0
    assert payload["mlir"]["totals_ms"]["mlir_lower_ms"] == 9.0
    assert payload["mlir"]["totals_ms"]["mlir_total_ms"] == 33.0


def test_aggregate_coverage_batches_merges_mlir_stage_timing(tmp_path: Path) -> None:
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
    for family, op, kernel, parse_ms, pass_ms, lower_ms in [
        ("f1", "op1", "k1", 1.0, 10.0, 4.0),
        ("f2", "op2", "k2", 2.0, 11.0, 5.0),
    ]:
        d = runs_root / f"family_{family}"
        d.mkdir(parents=True, exist_ok=True)
        stage_json = d / "stage_timing_breakdown.json"
        stage_json.write_text(
            json.dumps(
                {
                    "schema_version": "flaggems_stage_timing_breakdown_v1",
                    "backends": {
                        "rvv": {"available": True, "kernel_count": 1, "totals_ms": {"lower_ms": 1.0, "compile_ms": 2.0, "launch_ms": 3.0, "total_ms": 6.0}},
                        "cuda": {"available": True, "kernel_count": 1, "totals_ms": {"lower_ms": 1.5, "compile_ms": 2.5, "launch_ms": 3.5, "total_ms": 7.5}},
                    },
                    "mlir": {
                        "available": True,
                        "kernel_count": 1,
                        "totals_ms": {
                            "mlir_parse_ms": parse_ms,
                            "mlir_pass_ms": pass_ms,
                            "mlir_lower_ms": lower_ms,
                            "mlir_total_ms": parse_ms + pass_ms + lower_ms,
                        },
                        "pass_count_totals": {"upstream": 1, "midend": 2, "downstream": 1},
                        "missing_mlir_rows": [],
                        "source_dir": str(d / "pipeline_reports"),
                    },
                }
            ),
            encoding="utf-8",
        )
        (d / "run_summary.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "scope_kernels": [kernel],
                    "stages": [{"stage": "stage_timing_breakdown", "ok": True, "json_path": str(stage_json)}],
                }
            ),
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
    stage_payload = json.loads((runs_root / "stage_timing_breakdown.json").read_text(encoding="utf-8"))
    assert stage_payload["mlir"]["available"] is True
    assert stage_payload["mlir"]["kernel_count"] == 2
    assert stage_payload["mlir"]["totals_ms"]["mlir_parse_ms"] == 3.0
    assert stage_payload["mlir"]["totals_ms"]["mlir_pass_ms"] == 21.0
    assert stage_payload["mlir"]["totals_ms"]["mlir_lower_ms"] == 9.0
    assert stage_payload["mlir"]["totals_ms"]["mlir_total_ms"] == 33.0
