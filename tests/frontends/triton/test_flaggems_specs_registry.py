from __future__ import annotations

from pipeline.triton.flaggems_specs import coverage_flaggems_kernel_specs, default_flaggems_kernel_specs


def test_flaggems_specs_include_registry_metadata() -> None:
    specs = default_flaggems_kernel_specs(flaggems_opset="deterministic_forward", backend_target="rvv")
    assert specs
    for spec in specs:
        assert getattr(spec, "provider", None) == "flaggems"
        assert isinstance(getattr(spec, "source_op", None), str)
        assert isinstance(getattr(spec, "capability_state", None), str)


def test_flaggems_coverage_specs_include_expanded_semantics() -> None:
    specs = coverage_flaggems_kernel_specs(flaggems_opset="deterministic_forward", backend_target="rvv")
    names = {str(s.name) for s in specs}
    assert "relu2d" in names
    assert "exp2d" in names
    assert "where2d" in names
    assert "row_sum" in names
    assert "row_max" in names
    assert "clamp2d" in names
