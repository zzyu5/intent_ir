from __future__ import annotations

from pipeline.triton.flaggems_specs import default_flaggems_kernel_specs


def test_flaggems_specs_include_registry_metadata() -> None:
    specs = default_flaggems_kernel_specs(flaggems_opset="deterministic_forward", backend_target="rvv")
    assert specs
    for spec in specs:
        assert getattr(spec, "provider", None) == "flaggems"
        assert isinstance(getattr(spec, "source_op", None), str)
        assert isinstance(getattr(spec, "capability_state", None), str)
