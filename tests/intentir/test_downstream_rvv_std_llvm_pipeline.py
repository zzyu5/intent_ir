from __future__ import annotations

from pathlib import Path

from intent_ir.ir import IntentFunction
from intent_ir.mlir import detect_mlir_toolchain, run_pipeline, to_mlir


def _sample_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
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


def test_downstream_rvv_std_llvm_emits_riscv_triple(tmp_path: Path) -> None:
    toolchain = detect_mlir_toolchain()
    assert bool(toolchain.get("ok")) is True
    for k in ("mlir-opt", "mlir-translate", "llvm-as", "opt"):
        assert bool(((toolchain.get("tools") or {}).get(k) or {}).get("available")) is True

    mod = to_mlir(_sample_intent())
    out, trace = run_pipeline(mod, "downstream_rvv_std_llvm", backend="rvv", out_dir=tmp_path, fail_on_error=True)
    assert bool(trace.get("ok")) is True
    text = str(out.module_text or "")
    assert "target triple = \"riscv64-unknown-linux-gnu\"" in text
    assert str((out.meta or {}).get("llvm_dialect_origin") or "") in {"", "mlir_translate"}

