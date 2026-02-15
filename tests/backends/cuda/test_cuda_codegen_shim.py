from __future__ import annotations

import pytest

from backends.cuda.codegen import cpp_driver as cuda_compiler
from intent_ir.ir import IntentFunction


def _add_intent(name: str = "cuda_add_shim") -> IntentFunction:
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


def test_cuda_cpp_driver_entry_calls_cpp_lowerer(monkeypatch: pytest.MonkeyPatch) -> None:
    lowered_calls: list[tuple[str, dict[str, int]]] = []
    monkeypatch.setattr(
        "backends.cuda.codegen.cpp_driver.lower_intent_to_cuda_kernel_cpp",
        lambda intent_payload, *, bindings: (
            lowered_calls.append((str(intent_payload.name), dict(bindings)))
            or {
                "kernel_name": str(intent_payload.name),
                "cuda_src": "__global__ void k() {}",
                "io_spec": {},
                "launch": {"grid": [1, 1, 1], "block": [1, 1, 1], "shared_mem": 0},
                "output_names": ["C"],
                "bindings": dict(bindings),
            }
        ),
    )

    lowered = cuda_compiler.lower_intent_to_cuda_kernel(_add_intent(), shape_bindings={"M": 2, "N": 2})
    assert lowered.kernel_name == "cuda_add_shim"
    assert lowered_calls == [("cuda_add_shim", {"M": 2, "N": 2})]


def test_cuda_cpp_driver_entry_wraps_cpp_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backends.cuda.codegen.cpp_driver.lower_intent_to_cuda_kernel_cpp",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile_timeout: fail")),
    )

    with pytest.raises(cuda_compiler.CudaLoweringError, match="compile_timeout"):
        cuda_compiler.lower_intent_to_cuda_kernel(_add_intent("cuda_add_fail"), shape_bindings={"M": 2, "N": 2})
