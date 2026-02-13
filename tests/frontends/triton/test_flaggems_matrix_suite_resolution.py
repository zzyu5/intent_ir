from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_matrix_module():
    script_path = ROOT / "scripts" / "flaggems" / "run_multibackend_matrix.py"
    spec = importlib.util.spec_from_file_location("run_multibackend_matrix", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_suite_auto_promotes_to_coverage_for_non_smoke_kernel() -> None:
    mod = _load_matrix_module()
    suite, kernels = mod._resolve_suite_and_kernel_filter(
        requested_suite="smoke",
        requested_kernels=["eye2d"],
        flaggems_opset="deterministic_forward",
        backend_target="rvv",
    )
    assert suite == "coverage"
    assert kernels == ["eye2d"]


def test_smoke_suite_keeps_smoke_for_smoke_kernel() -> None:
    mod = _load_matrix_module()
    suite, kernels = mod._resolve_suite_and_kernel_filter(
        requested_suite="smoke",
        requested_kernels=["add2d"],
        flaggems_opset="deterministic_forward",
        backend_target="rvv",
    )
    assert suite == "smoke"
    assert kernels == ["add2d"]


def test_unknown_kernel_rejected_with_clear_error() -> None:
    mod = _load_matrix_module()
    with pytest.raises(SystemExit) as exc:
        mod._resolve_suite_and_kernel_filter(
            requested_suite="coverage",
            requested_kernels=["not_a_real_kernel"],
            flaggems_opset="deterministic_forward",
            backend_target="rvv",
        )
    assert "unknown kernel(s)" in str(exc.value)
