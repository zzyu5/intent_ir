from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from intent_ir.ir import IntentFunction
from intent_ir.mlir import detect_mlir_toolchain, run_pipeline, to_intent, to_mlir
from intent_ir.mlir.passes.emit_cuda_contract import build_cuda_contract
from intent_ir.mlir.passes.emit_rvv_contract import build_rvv_contract

ROOT = Path(__file__).resolve().parents[2]


def _sample_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "add2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "z": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "add", "inputs": ["x", "y"], "output": "z", "attrs": {}},
            ],
            "outputs": ["z"],
        }
    )


def test_mlir_roundtrip_from_intent() -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    back = to_intent(module)
    assert back.name == intent.name
    assert back.outputs == intent.outputs
    assert [op.op for op in back.ops] == [op.op for op in intent.ops]


def test_mlir_pass_pipeline_runs(tmp_path: Path) -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    out_mod, trace = run_pipeline(module, "upstream", out_dir=tmp_path)
    assert isinstance(out_mod.module_text, str) and out_mod.module_text
    assert bool(trace.get("ok")) is True
    trace_path = tmp_path / "pass_trace.json"
    assert trace_path.is_file()


def test_mlir_midend_pipeline_includes_stats_and_macro_expand(tmp_path: Path) -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    _, trace = run_pipeline(module, "midend", out_dir=tmp_path)
    assert bool(trace.get("ok")) is True
    names = [str(p.get("name") or "") for p in list(trace.get("passes") or [])]
    assert "python:expand_macros_intent" in names
    assert "python:canonicalize_intent" in names
    assert "python:cse_like" in names
    assert isinstance(trace.get("input_stats"), dict)
    assert isinstance(trace.get("output_stats"), dict)


def test_mlir_toolchain_probe_schema() -> None:
    payload = detect_mlir_toolchain()
    assert payload["schema_version"] == "intent_mlir_toolchain_probe_v1"
    assert "tools" in payload
    assert "mlir-opt" in payload["tools"]
    assert "llvm-as" in payload["tools"]
    assert "opt" in payload["tools"]


def test_mlir_optional_external_passes_do_not_fail_without_toolchain(tmp_path: Path) -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    _, trace = run_pipeline(module, "downstream_cuda_llvm", backend="cuda", out_dir=tmp_path)
    assert bool(trace.get("ok")) is True
    names = [str(p.get("name") or "") for p in list(trace.get("passes") or [])]
    assert "mlir-opt?:canonicalize" in names
    assert "mlir-translate?:mlir-to-llvmir" in names


def test_mlir_cuda_contract_emitter_builds_contract() -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    contract = build_cuda_contract(module)
    payload = contract.to_json_dict()
    assert payload["backend"] == "cuda"
    assert payload["kernel_name"] == "add2d"
    assert payload["schema_version"] == "intent_mlir_backend_contract_v1"
    assert isinstance(payload.get("intent_json"), dict)


def test_mlir_rvv_contract_emitter_builds_contract() -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    contract = build_rvv_contract(module)
    payload = contract.to_json_dict()
    assert payload["backend"] == "rvv"
    assert payload["kernel_name"] == "add2d"
    assert payload["schema_version"] == "intent_mlir_backend_contract_v1"
    assert isinstance(payload.get("intent_json"), dict)


def test_intentir_cli_mlir_check(tmp_path: Path) -> None:
    intent = _sample_intent()
    intent_path = tmp_path / "intent.json"
    out = tmp_path / "check.json"
    intent_path.write_text(json.dumps(intent.to_json_dict(), indent=2), encoding="utf-8")
    p = subprocess.run(
        [
            sys.executable,
            "scripts/intentir.py",
            "mlir",
            "check",
            "--intent-json",
            str(intent_path),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr or p.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert bool(payload.get("roundtrip_ok")) is True
