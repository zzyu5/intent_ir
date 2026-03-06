from __future__ import annotations

from pathlib import Path

import pytest

from intent_ir.ir import IntentFunction
from intent_ir.mlir import detect_mlir_toolchain, run_pipeline, to_mlir


def test_downstream_cuda_std_cpp_llvm_falls_back_without_plugin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    toolchain = detect_mlir_toolchain()
    assert bool(toolchain.get("ok")) is True
    for k in ("mlir-opt", "mlir-translate", "llvm-as", "opt"):
        assert bool(((toolchain.get("tools") or {}).get(k) or {}).get("available")) is True

    monkeypatch.setenv("INTENTIR_COMPILER_STACK", "cpp_plugin")
    monkeypatch.setenv("INTENTIR_COMPILER_CPP_WAVE", "wave3")
    monkeypatch.delenv("INTENTIR_MLIR_PASS_PLUGIN", raising=False)

    intent = IntentFunction.from_json_dict(
        {
            "name": "add2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "z": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "add", "inputs": ["x", "y"], "output": "z", "attrs": {}}],
            "outputs": ["z"],
        }
    )
    mod = to_mlir(intent)
    mod.meta = dict(mod.meta or {})
    mod.meta["shape_bindings"] = {"M": 4, "N": 8}

    out, trace = run_pipeline(
        mod,
        "downstream_cuda_std_cpp_llvm",
        backend="cuda",
        out_dir=tmp_path,
        fail_on_error=True,
    )
    assert bool(trace.get("ok")) is True
    assert "source_filename = \"LLVMDialectModule\"" in str(out.module_text or "")
    first = dict((trace.get("passes") or [])[0] or {})
    assert str(first.get("kind") or "") == "python"
    assert "python_fallback:intentir_mlir_plugin_unavailable" in str(first.get("detail") or "")
