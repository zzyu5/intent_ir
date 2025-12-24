"""
IntentIR ops -> standalone C program (Task6 backend).

Implementation is provided by the C++ host tool (`backend_spmd_rvv/cpp_codegen`).
This module is the stable Python entrypoint used by runners.
"""

from __future__ import annotations

from typing import Any, Mapping

from intent_ir.ir_types import IntentFunction

from .cpp_driver import lower_intent_to_c_with_files_cpp


def lower_intent_to_c_with_files(
    intent: IntentFunction,
    *,
    shape_bindings: Mapping[str, Any],
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> str:
    return lower_intent_to_c_with_files_cpp(intent, shape_bindings=shape_bindings, atol=atol, rtol=rtol)


__all__ = ["lower_intent_to_c_with_files"]

