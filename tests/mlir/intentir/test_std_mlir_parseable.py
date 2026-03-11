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


def test_std_mlir_parseable_by_mlir_opt(tmp_path: Path) -> None:
    toolchain = detect_mlir_toolchain()
    assert bool(((toolchain.get("tools") or {}).get("mlir-opt") or {}).get("available")) is True

    mod = to_mlir(_sample_intent())
    assert str(mod.dialect_version) == "std_mlir_v1"
    out, trace = run_pipeline(mod, "upstream_std", out_dir=tmp_path, fail_on_error=True)
    assert isinstance(out.module_text, str) and out.module_text.strip()
    assert bool(trace.get("ok")) is True

