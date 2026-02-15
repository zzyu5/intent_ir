from __future__ import annotations

import pytest

from backends.spmd_rvv.codegen import cpp_driver as rvv_compiler
from intent_ir.ir import IntentFunction


def _add_intent(name: str = "rvv_add_shim") -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": name,
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "add", "inputs": ["A", "B"], "output": "C", "attrs": {}}],
            "outputs": ["C"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )


def test_rvv_cpp_driver_entry_calls_cpp_lowerer(monkeypatch: pytest.MonkeyPatch) -> None:
    cpp_calls: list[str] = []

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
        cpp_calls.append(str(intent.name))
        return "ok"

    monkeypatch.setattr(rvv_compiler, "lower_intent_to_c_with_files_cpp", _fake_cpp_lower)

    got = rvv_compiler.lower_intent_to_c_with_files(_add_intent(), shape_bindings={"M": 2, "N": 2})
    assert got == "ok"
    assert cpp_calls == ["rvv_add_shim"]


def test_rvv_cpp_driver_entry_bubbles_cpp_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rvv_compiler,
        "lower_intent_to_c_with_files_cpp",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile_timeout: fail")),
    )

    with pytest.raises(RuntimeError, match="compile_timeout"):
        rvv_compiler.lower_intent_to_c_with_files(_add_intent("rvv_add_fail"), shape_bindings={"M": 2, "N": 2})
