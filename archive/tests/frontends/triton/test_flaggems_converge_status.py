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
            "--scope-mode",
            "kernel_alias",
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
    assert payload["counts_scoped_kernel_alias"]["dual_pass"] == 1
    assert payload["counts_scoped_active"] == {}
    assert payload["scope_enabled"] is True
    assert payload["scope_mode"] == "kernel_alias"
    assert payload["scope_kernels"] == ["angle2d"]
    assert payload["scope_semantic_ops"] == []
    assert payload["scoped_entries_count"] == 1
    assert payload["scoped_entries_active_count"] == 0
    assert payload["scoped_entries_kernel_alias_count"] == 1
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
    assert payload["counts_scoped_active"]["dual_pass"] == 1
    assert payload["counts_scoped_kernel_alias"]["dual_pass"] == 1
    scoped_ops = [str(e.get("semantic_op")) for e in payload["scoped_entries"]]
    assert scoped_ops == ["diag"]


def test_converge_status_scope_mode_both_emits_active_and_kernel_alias(tmp_path: Path) -> None:
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
            "--scope-mode",
            "both",
            "--scope-kernels",
            "angle2d",
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
    assert payload["scope_mode"] == "both"
    # Legacy scoped fields stay aligned to active_only semantics.
    assert payload["counts_scoped"]["dual_pass"] == 1
    assert payload["counts_scoped_active"]["dual_pass"] == 1
    assert payload["counts_scoped_kernel_alias"]["dual_pass"] == 2
    assert [str(e.get("semantic_op")) for e in payload["scoped_entries"]] == ["diag"]
    assert sorted(str(e.get("semantic_op")) for e in payload["scoped_entries_kernel_alias"]) == ["angle", "diag"]


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
            "--scope-mode",
            "kernel_alias",
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


def test_converge_status_uses_backend_reason_code_field_when_present(tmp_path: Path) -> None:
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
                        "reason_code": "env_unavailable",
                        "error": {"type": "RuntimeError", "message": "nvrtc_unavailable"},
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
            "--scope-semantic-ops",
            "angle",
            "--scope-mode",
            "active_only",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    scoped_entries = list(payload.get("scoped_entries_active") or [])
    assert len(scoped_entries) == 1
    entry = scoped_entries[0]
    assert entry["status"] == "rvv_only"
    assert entry["reason_code"] == "env_unavailable"
    assert entry["runtime_detail"]["cuda"]["reason_code"] == "env_unavailable"


def test_converge_status_write_registry_scoped_updates_only_in_scope(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    provider_dir = tmp_path / "provider"
    rvv_json = tmp_path / "rvv.json"
    cuda_json = tmp_path / "cuda.json"
    out = tmp_path / "status_converged.json"
    provider_dir.mkdir(parents=True, exist_ok=True)

    registry.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "semantic_op": "angle",
                        "intent_ops": ["angle"],
                        "e2e_spec": "angle2d",
                        "status": "blocked_backend",
                        "reason_code": "runtime_backend_fail",
                    },
                    {
                        "semantic_op": "diag",
                        "intent_ops": ["diag"],
                        "e2e_spec": "diag2d",
                        "status": "dual_pass",
                        "reason_code": "runtime_dual_backend_pass",
                    },
                ],
                "counts": {"by_status": {"blocked_backend": 1, "dual_pass": 1}},
            }
        ),
        encoding="utf-8",
    )
    rvv_json.write_text(json.dumps({"results": [{"kernel": "angle2d", "ok": True}]}), encoding="utf-8")
    cuda_json.write_text(json.dumps({"results": [{"kernel": "angle2d", "ok": True}]}), encoding="utf-8")
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
            "--scope-semantic-ops",
            "angle",
            "--scope-mode",
            "active_only",
            "--write-registry",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr

    reg_after = json.loads(registry.read_text(encoding="utf-8"))
    entries = list(reg_after.get("entries") or [])
    assert entries[0]["semantic_op"] == "angle"
    assert entries[0]["status"] == "dual_pass"
    assert entries[1]["semantic_op"] == "diag"
    assert entries[1]["status"] == "dual_pass"
    assert reg_after["counts"]["by_status"]["dual_pass"] == 2
