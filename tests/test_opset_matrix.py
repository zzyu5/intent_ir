from __future__ import annotations

from backends.spmd_rvv.opset import SPMD_RVV_SUPPORTED_OPS
from intent_ir.ir.ir_types import SUPPORTED_OPS as IR_SUPPORTED_OPS
from intent_ir.macros.macro_lowering.registry import supports_macro
from intent_ir.ops import CORE_OPS, EXPERIMENTAL_OPS, MACRO_OPS, SUPPORTED_OPS
from verify.interpreter import INTERPRETER_SUPPORTED_OPS


def test_opset_is_single_source_of_truth() -> None:
    assert SUPPORTED_OPS == IR_SUPPORTED_OPS


def test_core_ops_supported_by_interpreter() -> None:
    missing = sorted(CORE_OPS - INTERPRETER_SUPPORTED_OPS)
    assert not missing, f"interpreter missing core ops: {missing}"


def test_core_ops_supported_by_spmd_rvv_backend() -> None:
    missing = sorted(CORE_OPS - SPMD_RVV_SUPPORTED_OPS)
    assert not missing, f"SPMD+RVV backend missing core ops: {missing}"


def test_macro_ops_have_lowering() -> None:
    missing = sorted([op for op in MACRO_OPS if not supports_macro(op)])
    assert not missing, f"missing macro lowering implementations: {missing}"


def test_op_tiers_are_disjoint() -> None:
    assert CORE_OPS.isdisjoint(EXPERIMENTAL_OPS)
    assert CORE_OPS.isdisjoint(MACRO_OPS)
    assert EXPERIMENTAL_OPS.isdisjoint(MACRO_OPS)
