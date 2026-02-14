from __future__ import annotations

from backends.spmd_rvv.codegen import intentir_to_c
from backends.spmd_rvv.opset import SPMD_RVV_SUPPORTED_OPS
from intent_ir.ir import IntentFunction


def _unary_intent(op_name: str, fn_name: str) -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": fn_name,
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": op_name, "inputs": ["A"], "output": "Out"}],
            "outputs": ["Out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 64, "parallel_axes": ["M", "N"]},
        }
    )


def test_rvv_opset_contains_cos_and_erf() -> None:
    assert "cos" in SPMD_RVV_SUPPORTED_OPS
    assert "erf" in SPMD_RVV_SUPPORTED_OPS


def test_rvv_lowering_preflight_accepts_cos_and_erf(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_cpp_lower(
        intent: IntentFunction,
        *,
        shape_bindings,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        mode: str = "verify",
    ) -> str:
        del shape_bindings, atol, rtol, mode
        calls.append(str(intent.name))
        return "ok"

    monkeypatch.setattr(intentir_to_c, "lower_intent_to_c_with_files_cpp", _fake_cpp_lower)

    c0 = intentir_to_c.lower_intent_to_c_with_files(
        _unary_intent("cos", "rvv_cos"),
        shape_bindings={"M": 4, "N": 8},
    )
    c1 = intentir_to_c.lower_intent_to_c_with_files(
        _unary_intent("erf", "rvv_erf"),
        shape_bindings={"M": 4, "N": 8},
    )

    assert c0 == "ok"
    assert c1 == "ok"
    assert calls == ["rvv_cos", "rvv_erf"]
