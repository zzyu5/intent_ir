"""
IntentIR ops -> standalone C program (Task6 backend).

Implementation is provided by the C++ host tool (`backends/spmd_rvv/cpp_codegen`).
This module is the stable Python entrypoint used by runners.
"""

from __future__ import annotations

from typing import Any, Mapping

from intent_ir.ir import IntentFunction
from intent_ir.ops import EXPERIMENTAL_OPS, MACRO_OPS

from .cpp_driver import lower_intent_to_c_with_files_cpp
from ..opset import SPMD_RVV_SUPPORTED_OPS


def _preflight_supported_ops(intent: IntentFunction) -> None:
    ops = {op.op for op in intent.ops}
    unsupported = sorted([op for op in ops if op not in SPMD_RVV_SUPPORTED_OPS])
    if not unsupported:
        return
    hints: list[str] = []
    macro_hits = [op for op in unsupported if op in MACRO_OPS]
    if macro_hits:
        hints.append(f"macro ops must be expanded before RVV lowering: {macro_hits}")
    exp_hits = [op for op in unsupported if op in EXPERIMENTAL_OPS]
    if exp_hits:
        hints.append(f"experimental/out-of-scope ops (no RVV lowering yet): {exp_hits}")
    if not hints:
        hints.append("supported ops list is in backends/spmd_rvv/opset.py")
    msg = "SPMD+RVV backend does not support ops: " + ", ".join(unsupported) + "\n" + "\n".join(f"Hint: {h}" for h in hints)
    raise ValueError(msg)


def _run_pipeline_compat_check(intent: IntentFunction, *, shape_bindings: Mapping[str, Any] | None = None) -> None:
    """
    Keep legacy RVV codegen entry wired to the staged RVV pipeline driver.
    """
    from ..pipeline.driver import run_rvv_pipeline  # noqa: PLC0415

    result = run_rvv_pipeline(intent, shape_bindings=shape_bindings)
    if bool(result.ok):
        return
    reason = str(getattr(result, "reason_code", "") or "pipeline_failed")
    detail = str(getattr(result, "reason_detail", "") or "")
    msg = f"rvv pipeline compatibility stage failed: {reason}"
    if detail:
        msg = f"{msg} ({detail})"
    raise ValueError(msg)


def lower_intent_to_c_with_files(
    intent: IntentFunction,
    *,
    shape_bindings: Mapping[str, Any],
    atol: float = 1e-3,
    rtol: float = 1e-3,
    mode: str = "verify",
) -> str:
    _run_pipeline_compat_check(intent, shape_bindings=shape_bindings)
    _preflight_supported_ops(intent)
    return lower_intent_to_c_with_files_cpp(intent, shape_bindings=shape_bindings, atol=atol, rtol=rtol, mode=str(mode))


__all__ = ["lower_intent_to_c_with_files"]
