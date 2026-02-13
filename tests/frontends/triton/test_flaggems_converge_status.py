from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _write_minimal_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "semantic_op": "angle",
                        "intent_ops": ["angle"],
                        "e2e_spec": "angle2d",
                    },
                    {
                        "semantic_op": "diag",
                        "intent_ops": ["diag"],
                        "e2e_spec": "diag2d",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_runtime_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "results": [
                    {"kernel": "angle2d", "ok": True},
                    {"kernel": "diag2d", "ok": True},
                ]
            }
        ),
        encoding="utf-8",
    )


def test_converge_status_outputs_scoped_and_global_counts(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    provider_dir = tmp_path / "provider"
    rvv_json = tmp_path / "rvv.json"
    cuda_json = tmp_path / "cuda.json"
    out = tmp_path / "status_converged.json"
    provider_dir.mkdir(parents=True, exist_ok=True)

    _write_minimal_registry(registry)
    _write_runtime_summary(rvv_json)
    _write_runtime_summary(cuda_json)
    (provider_dir / "angle2d.json").write_text(json.dumps({"diff": {"ok": True}}), encoding="utf-8")
    (provider_dir / "diag2d.json").write_text(json.dumps({"diff": {"ok": True}}), encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/converge_status.py",
            "--registry",
            str(registry),
            "--provider-report-dir",
            str(provider_dir),
            "--rvv-json",
            str(rvv_json),
            "--cuda-json",
            str(cuda_json),
            "--scope-kernels",
            "angle2d",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["counts_global"]["dual_pass"] == 2
    assert payload["counts_scoped"]["dual_pass"] == 1
    assert payload["scope_enabled"] is True
    assert payload["scope_kernels"] == ["angle2d"]
    assert payload["scope_semantic_ops"] == []
    assert payload["scoped_entries_count"] == 1
    assert payload["global_entries_count"] == 2
    scoped_ops = [str(e.get("semantic_op")) for e in payload["scoped_entries"]]
    assert scoped_ops == ["angle"]


def test_converge_status_scope_by_semantic_op(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    provider_dir = tmp_path / "provider"
    rvv_json = tmp_path / "rvv.json"
    cuda_json = tmp_path / "cuda.json"
    out = tmp_path / "status_converged.json"
    provider_dir.mkdir(parents=True, exist_ok=True)

    _write_minimal_registry(registry)
    _write_runtime_summary(rvv_json)
    _write_runtime_summary(cuda_json)
    (provider_dir / "angle2d.json").write_text(json.dumps({"diff": {"ok": True}}), encoding="utf-8")
    (provider_dir / "diag2d.json").write_text(json.dumps({"diff": {"ok": True}}), encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/converge_status.py",
            "--registry",
            str(registry),
            "--provider-report-dir",
            str(provider_dir),
            "--rvv-json",
            str(rvv_json),
            "--cuda-json",
            str(cuda_json),
            "--scope-semantic-ops",
            "diag",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["counts_scoped"]["dual_pass"] == 1
    scoped_ops = [str(e.get("semantic_op")) for e in payload["scoped_entries"]]
    assert scoped_ops == ["diag"]


def test_converge_status_propagates_lowering_missing_op_for_runtime_fail(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    provider_dir = tmp_path / "provider"
    rvv_json = tmp_path / "rvv.json"
    cuda_json = tmp_path / "cuda.json"
    out = tmp_path / "status_converged.json"
    provider_dir.mkdir(parents=True, exist_ok=True)

    _write_minimal_registry(registry)
    rvv_json.write_text(json.dumps({"results": [{"kernel": "angle2d", "ok": True}]}), encoding="utf-8")
    cuda_json.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "kernel": "angle2d",
                        "ok": False,
                        "stderr": "CudaLoweringError: unsupported op in lowering",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (provider_dir / "angle2d.json").write_text(json.dumps({"diff": {"ok": True}}), encoding="utf-8")

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/converge_status.py",
            "--registry",
            str(registry),
            "--provider-report-dir",
            str(provider_dir),
            "--rvv-json",
            str(rvv_json),
            "--cuda-json",
            str(cuda_json),
            "--scope-kernels",
            "angle2d",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    scoped_entries = list(payload.get("scoped_entries") or [])
    assert len(scoped_entries) == 1
    entry = scoped_entries[0]
    assert entry["status"] == "rvv_only"
    assert entry["reason_code"] == "lowering_missing_op"
