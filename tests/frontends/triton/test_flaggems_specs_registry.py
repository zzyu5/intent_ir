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
    assert "exp22d" in names
    assert "acos2d" in names
    assert "atan2d" in names
    assert "cat2d" in names
    assert "arange1d" in names
    assert "div2d" in names
    assert "eq2d" in names
    assert "ne2d" in names
    assert "gt2d" in names
    assert "ge2d" in names
    assert "lt2d" in names
    assert "le2d" in names
    assert "neg2d" in names
    assert "ceil2d" in names
    assert "reciprocal2d" in names
    assert "sqrt2d" in names
    assert "silu2d" in names
    assert "tanh2d" in names
    assert "logical_and2d" in names
    assert "logical_or2d" in names
    assert "logical_not2d" in names
    assert "logical_xor2d" in names
    assert "mm2d" in names
    assert "bmm3d" in names
    assert "addmm2d" in names
    assert "baddbmm3d" in names
    assert "dot1d" in names
    assert "vdot1d" in names
    assert "mv2d" in names
    assert "addmv2d" in names
    assert "sub2d" in names
    assert "mul2d" in names
    assert "sigmoid2d" in names
    assert "abs2d" in names
    assert "rsqrt2d" in names
    assert "where2d" in names
    assert "row_sum" in names
    assert "row_mean" in names
    assert "row_all" in names
    assert "row_max" in names
    assert "allclose2d" in names
    assert "isclose2d" in names
    assert "isfinite2d" in names
    assert "isinf2d" in names
    assert "isnan2d" in names
    assert "masked_fill2d" in names
    assert "threshold2d" in names
    assert "full2d" in names
    assert "maximum2d" in names
    assert "minimum2d" in names
    assert "identity2d" in names
    assert "cast2d" in names
    assert "gather2d" in names
    assert "index_select2d" in names
    assert "clamp2d" in names
    assert "lerp2d" in names
    assert "batch_norm2d" in names
    assert "rms_norm2d" in names
