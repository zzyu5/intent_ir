from __future__ import annotations

from backends.spmd_rvv.codegen import cpp_driver as rvv_compiler
from backends.spmd_rvv.opset import SPMD_RVV_SUPPORTED_OPS
from intent_ir.ir import IntentFunction


def _scatter_intent(op_name: str, fn_name: str) -> IntentFunction:
    if op_name == "scatter":
        return IntentFunction.from_json_dict(
            {
                "name": fn_name,
                "tensors": {
                    "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "index": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                    "src": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "scatter", "inputs": ["inp", "index", "src"], "output": "out", "attrs": {"dim": 1}}],
                "outputs": ["out"],
                "parallel_axes": ["M", "N"],
                "schedule": {"tile_n": 64, "parallel_axes": ["M", "N"]},
            }
        )
    if op_name == "select_scatter":
        return IntentFunction.from_json_dict(
            {
                "name": fn_name,
                "tensors": {
                    "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "src": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "select_scatter", "inputs": ["inp", "src"], "output": "out", "attrs": {"dim": 1, "index": 0}}],
                "outputs": ["out"],
                "parallel_axes": ["M", "N"],
                "schedule": {"tile_n": 64, "parallel_axes": ["M", "N"]},
            }
        )
    if op_name == "slice_scatter":
        return IntentFunction.from_json_dict(
            {
                "name": fn_name,
                "tensors": {
                    "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "src": {"dtype": "f32", "shape": ["M", "L"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [
                    {
                        "op": "slice_scatter",
                        "inputs": ["inp", "src"],
                        "output": "out",
                        "attrs": {"dim": 1, "start": 0, "end": 4, "step": 1},
                    }
                ],
                "outputs": ["out"],
                "parallel_axes": ["M", "N"],
                "schedule": {"tile_n": 64, "parallel_axes": ["M", "N"]},
            }
        )
    raise AssertionError(f"unsupported op_name: {op_name}")


def test_rvv_opset_contains_scatter_family() -> None:
    assert "scatter" in SPMD_RVV_SUPPORTED_OPS
    assert "select_scatter" in SPMD_RVV_SUPPORTED_OPS
    assert "slice_scatter" in SPMD_RVV_SUPPORTED_OPS


def test_rvv_lowering_preflight_accepts_scatter_family(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_cpp_lower(
        intent: IntentFunction,
        *,
        shape_bindings,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        mode: str = "verify",
        build_type: str = "Release",
    ) -> str:
        del shape_bindings, atol, rtol, mode, build_type
        calls.append(str(intent.name))
        return "ok"

    monkeypatch.setattr(rvv_compiler, "lower_intent_to_c_with_files_cpp", _fake_cpp_lower)

    c0 = rvv_compiler.lower_intent_to_c_with_files(
        _scatter_intent("scatter", "rvv_scatter"),
        shape_bindings={"M": 4, "N": 8},
    )
    c1 = rvv_compiler.lower_intent_to_c_with_files(
        _scatter_intent("select_scatter", "rvv_select_scatter"),
        shape_bindings={"M": 4, "N": 8},
    )
    c2 = rvv_compiler.lower_intent_to_c_with_files(
        _scatter_intent("slice_scatter", "rvv_slice_scatter"),
        shape_bindings={"M": 4, "N": 8, "L": 4},
    )

    assert c0 == "ok"
    assert c1 == "ok"
    assert c2 == "ok"
    assert calls == ["rvv_scatter", "rvv_select_scatter", "rvv_slice_scatter"]
