from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _write_common_inputs(tmp_path: Path, *, fallback_origin: str) -> tuple[Path, Path, Path, Path, Path]:
    registry = tmp_path / "registry.json"
    provider_dir = tmp_path / "provider_reports"
    provider_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "status_converged.json"
    rvv_json = tmp_path / "rvv.json"
    cuda_json = tmp_path / "cuda.json"

    registry.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "semantic_op": "add",
                        "e2e_spec": "add2d",
                        "intent_ops": ["add"],
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    contract_path = provider_dir / "add2d.intentir.intentdialect.downstream_rvv_llvm.contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": "intent_mlir_backend_contract_v2",
                "backend": "rvv",
                "artifacts": {"rvv_kernel_src_origin": str(fallback_origin)},
                "executable": {"invocation": {}},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (provider_dir / "add2d.json").write_text(
        json.dumps(
            {
                "diff": {"ok": True},
                "mlir": {"downstream_rvv_llvm_contract_path": str(contract_path)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rvv_json.write_text(json.dumps({"results": [{"kernel": "add2d", "ok": True}]}), encoding="utf-8")
    cuda_json.write_text(json.dumps({"results": [{"kernel": "add2d", "ok": True}]}), encoding="utf-8")
    return registry, provider_dir, rvv_json, cuda_json, out


def _run_converge(
    *,
    registry: Path,
    provider_dir: Path,
    rvv_json: Path,
    cuda_json: Path,
    out: Path,
) -> dict:
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
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def test_converge_status_forbids_compat_cpp_codegen_from_dual_pass(tmp_path: Path) -> None:
    registry, provider_dir, rvv_json, cuda_json, out = _write_common_inputs(
        tmp_path,
        fallback_origin="compat_cpp_codegen",
    )
    payload = _run_converge(
        registry=registry,
        provider_dir=provider_dir,
        rvv_json=rvv_json,
        cuda_json=cuda_json,
        out=out,
    )

    row = dict((payload.get("entries") or [])[0])
    assert row["status"] == "blocked_backend"
    assert row["reason_code"] == "runtime_fallback_forbidden"
    assert bool(row.get("runtime_fallback")) is True
    assert bool(row.get("runtime_fallback_forbidden")) is True
    assert "compat_cpp_codegen" in str(row.get("runtime_fallback_detail") or "")
    assert payload["counts_global"]["blocked_backend"] == 1
    assert payload["runtime_fallback_forbidden_kernel_count"] == 1


def test_converge_status_allows_dual_pass_when_no_forbidden_fallback(tmp_path: Path) -> None:
    registry, provider_dir, rvv_json, cuda_json, out = _write_common_inputs(
        tmp_path,
        fallback_origin="llvm_llc",
    )
    payload = _run_converge(
        registry=registry,
        provider_dir=provider_dir,
        rvv_json=rvv_json,
        cuda_json=cuda_json,
        out=out,
    )

    row = dict((payload.get("entries") or [])[0])
    assert row["status"] == "dual_pass"
    assert row["reason_code"] == "runtime_dual_backend_pass"
    assert bool(row.get("runtime_fallback")) is False
    assert bool(row.get("runtime_fallback_forbidden")) is False
    assert payload["counts_global"]["dual_pass"] == 1
    assert payload["runtime_fallback_forbidden_kernel_count"] == 0
