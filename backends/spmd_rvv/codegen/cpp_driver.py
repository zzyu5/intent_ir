"""
Compatibility stubs for removed RVV C/C++ codegen.

The strict MLIR hard-cut path executes RVV contracts via prebuilt ELF or remote
LLVM compile and no longer supports C-source generation.
"""

from __future__ import annotations

from typing import Any


def _removed() -> RuntimeError:
    return RuntimeError(
        "RVV C/C++ compatibility codegen has been removed from strict hard-cut path; "
        "use MLIR backend contracts with prebuilt_elf or remote_llvm execution"
    )


def ensure_cpp_codegen_built(*, build_type: str = "Release") -> str:  # noqa: ARG001
    raise _removed()


def lower_intent_to_c_with_files_cpp(
    intent_or_json: Any,  # noqa: ARG001
    *,
    shape_bindings: dict[str, Any] | None = None,  # noqa: ARG001
    atol: float = 1e-3,  # noqa: ARG001
    rtol: float = 1e-3,  # noqa: ARG001
    mode: str = "verify",  # noqa: ARG001
) -> str:
    raise _removed()


def lower_intent_json_to_c_with_files_cpp(
    intent_json: dict[str, Any],  # noqa: ARG001
    *,
    shape_bindings: dict[str, Any] | None = None,  # noqa: ARG001
    atol: float = 1e-3,  # noqa: ARG001
    rtol: float = 1e-3,  # noqa: ARG001
    mode: str = "verify",  # noqa: ARG001
) -> str:
    raise _removed()


def lower_intent_to_c_with_files(
    intent_or_json: Any,  # noqa: ARG001
    *,
    shape_bindings: dict[str, Any] | None = None,  # noqa: ARG001
    atol: float = 1e-3,  # noqa: ARG001
    rtol: float = 1e-3,  # noqa: ARG001
    mode: str = "verify",  # noqa: ARG001
) -> str:
    raise _removed()


__all__ = [
    "ensure_cpp_codegen_built",
    "lower_intent_to_c_with_files_cpp",
    "lower_intent_json_to_c_with_files_cpp",
    "lower_intent_to_c_with_files",
]
