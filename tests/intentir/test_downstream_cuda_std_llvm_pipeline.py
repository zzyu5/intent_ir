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
    (
        "sigmoid2d",
        _intent(
            {
                "name": "sigmoid2d",
                "tensors": {
                    "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "output": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [
                    {"op": "const", "inputs": [], "output": "one_const", "attrs": {"value": 1}},
                    {"op": "const", "inputs": [], "output": "neg_one_const", "attrs": {"value": -1}},
                    {"op": "mul", "inputs": ["x", "neg_one_const"], "output": "neg_x", "attrs": {}},
                    {"op": "exp", "inputs": ["neg_x"], "output": "exp_neg_x", "attrs": {}},
                    {"op": "add", "inputs": ["one_const", "exp_neg_x"], "output": "denominator", "attrs": {}},
                    {"op": "div", "inputs": ["one_const", "denominator"], "output": "output", "attrs": {}},
                ],
                "outputs": ["output"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "tanh2d",
        _intent(
            {
                "name": "tanh2d",
                "tensors": {
                    "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [
                    {"op": "const", "inputs": [], "output": "one_const", "attrs": {"value": 1}},
                    {"op": "const", "inputs": [], "output": "two_const", "attrs": {"value": 2}},
                    {"op": "mul", "inputs": ["two_const", "x"], "output": "two_x", "attrs": {}},
                    {"op": "exp", "inputs": ["two_x"], "output": "exp_two_x", "attrs": {}},
                    {"op": "sub", "inputs": ["exp_two_x", "one_const"], "output": "numer", "attrs": {}},
                    {"op": "add", "inputs": ["exp_two_x", "one_const"], "output": "denom", "attrs": {}},
                    {"op": "div", "inputs": ["numer", "denom"], "output": "out", "attrs": {}},
                ],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "sqrt2d",
        _intent(
            {
                "name": "sqrt2d",
                "tensors": {
                    "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "sqrt", "inputs": ["A"], "output": "out", "attrs": {}}],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "log2d",
        _intent(
            {
                "name": "log2d",
                "tensors": {
                    "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "log", "inputs": ["inp"], "output": "out", "attrs": {}}],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "rsqrt2d",
        _intent(
            {
                "name": "rsqrt2d",
                "tensors": {
                    "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "A_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [
                    {"op": "cast", "inputs": ["A"], "output": "A_f32", "attrs": {"to": "f32"}},
                    {"op": "rsqrt", "inputs": ["A_f32"], "output": "out", "attrs": {}},
                ],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "erf2d",
        _intent(
            {
                "name": "erf2d",
                "tensors": {
                    "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "erf", "inputs": ["x"], "output": "out", "attrs": {}}],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "exp22d",
        _intent(
            {
                "name": "exp22d",
                "tensors": {
                    "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [
                    {"op": "const", "inputs": [], "output": "ln2", "attrs": {"value": 0.6931471805599453}},
                    {"op": "mul", "inputs": ["A", "ln2"], "output": "scaled", "attrs": {}},
                    {"op": "exp", "inputs": ["scaled"], "output": "out", "attrs": {}},
                ],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "eq2d",
        _intent(
            {
                "name": "eq2d",
                "tensors": {
                    "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "eq", "inputs": ["x", "y"], "output": "out", "attrs": {}}],
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
