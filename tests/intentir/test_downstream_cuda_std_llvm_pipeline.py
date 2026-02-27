from __future__ import annotations

from pathlib import Path

import pytest

from intent_ir.ir import IntentFunction
from intent_ir.mlir import detect_mlir_toolchain, run_pipeline, to_mlir


def _intent(payload: dict) -> IntentFunction:
    return IntentFunction.from_json_dict(payload)


CASES = [
    (
        "add2d",
        _intent(
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
        ),
        {"M": 4, "N": 8},
    ),
    (
        "abs2d",
        _intent(
            {
                "name": "abs2d",
                "tensors": {
                    "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "abs", "inputs": ["A"], "output": "Out", "attrs": {}}],
                "outputs": ["Out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "floor2d",
        _intent(
            {
                "name": "floor2d",
                "tensors": {
                    "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "floor", "inputs": ["inp"], "output": "out", "attrs": {}}],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "clamp2d",
        _intent(
            {
                "name": "clamp2d",
                "tensors": {
                    "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "mini": {"dtype": "f32", "shape": [], "layout": "row_major"},
                    "maxi": {"dtype": "f32", "shape": [], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "x_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "clamped_min": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [
                    {"op": "cast", "inputs": ["x"], "output": "x_f32", "attrs": {"to": "f32"}},
                    {"op": "max", "inputs": ["mini", "x_f32"], "output": "clamped_min", "attrs": {}},
                    {"op": "min", "inputs": ["clamped_min", "maxi"], "output": "out", "attrs": {}},
                ],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "add_bias2d",
        _intent(
            {
                "name": "add_bias2d",
                "tensors": {
                    "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "bias": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "add", "inputs": ["inp", "bias"], "output": "out", "attrs": {}}],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
]


@pytest.mark.parametrize("name,intent,bindings", CASES, ids=[c[0] for c in CASES])
def test_downstream_cuda_std_llvm_emits_nvptx_triple(
    name: str,
    intent: IntentFunction,
    bindings: dict[str, int],
    tmp_path: Path,
    monkeypatch,
) -> None:
    toolchain = detect_mlir_toolchain()
    assert bool(toolchain.get("ok")) is True
    for k in ("mlir-opt", "mlir-translate", "llvm-as", "opt"):
        assert bool(((toolchain.get("tools") or {}).get(k) or {}).get("available")) is True

    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")

    mod = to_mlir(intent)
    mod.meta = dict(mod.meta or {})
    mod.meta["shape_bindings"] = dict(bindings)

    out, trace = run_pipeline(
        mod,
        "downstream_cuda_std_llvm",
        backend="cuda",
        out_dir=tmp_path,
        fail_on_error=True,
    )
    assert bool(trace.get("ok")) is True

    text = str(out.module_text or "")
    assert "target triple = \"nvptx64-nvidia-cuda\"" in text
    assert f"@{name}" in text
    assert str((out.meta or {}).get("llvm_dialect_origin") or "") in {"", "mlir_translate"}
