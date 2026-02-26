"""
Compatibility stubs for removed CUDA C/C++ codegen.

Strict MLIR hard-cut no longer supports IntentIR->CUDA C++ codegen entrypoints.
Execution must flow through MLIR backend contracts with PTX executables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _removed() -> RuntimeError:
    return RuntimeError(
        "CUDA C/C++ compatibility codegen has been removed from strict hard-cut path; "
        "use MLIR backend contracts with executable.format in {cuda_ptx, ptx}"
    )


@dataclass(frozen=True)
class CudaLoweredKernel:
    kernel_name: str
    cuda_src: str
    io_spec: dict[str, Any]
    launch: Any
    output_names: list[str]
    bindings: dict[str, Any]


def ensure_cpp_codegen_ext_loaded(*, verbose: bool = False) -> Any:  # noqa: ARG001
    raise _removed()


def ensure_cpp_codegen_built(*, build_type: str = "Release") -> str:  # noqa: ARG001
    raise _removed()


def lower_intent_to_cuda_kernel_cpp(
    intent_or_json: Any,  # noqa: ARG001
    *,
    bindings: Mapping[str, Any],
    build_type: str = "Release",  # noqa: ARG001
) -> dict[str, Any]:
    raise _removed()


def lower_intent_json_to_cuda_kernel_cpp(
    intent_json: Mapping[str, Any],  # noqa: ARG001
    *,
    bindings: Mapping[str, Any],
    build_type: str = "Release",  # noqa: ARG001
) -> dict[str, Any]:
    raise _removed()


def lower_intent_to_cuda_kernel(
    intent_or_json: Any,  # noqa: ARG001
    *,
    shape_bindings: Mapping[str, Any] | None = None,  # noqa: ARG001
    build_type: str = "Release",  # noqa: ARG001
) -> CudaLoweredKernel:
    raise _removed()


def lower_intent_json_to_cuda_kernel(
    intent_json: Mapping[str, Any],  # noqa: ARG001
    *,
    shape_bindings: Mapping[str, Any] | None = None,  # noqa: ARG001
    build_type: str = "Release",  # noqa: ARG001
) -> CudaLoweredKernel:
    raise _removed()


__all__ = [
    "CudaLoweredKernel",
    "ensure_cpp_codegen_built",
    "ensure_cpp_codegen_ext_loaded",
    "lower_intent_to_cuda_kernel",
    "lower_intent_to_cuda_kernel_cpp",
    "lower_intent_json_to_cuda_kernel",
    "lower_intent_json_to_cuda_kernel_cpp",
]
