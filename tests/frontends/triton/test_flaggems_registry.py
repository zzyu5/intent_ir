from __future__ import annotations

from pipeline.triton.flaggems_registry import (
    STATUS_VALUES,
    build_registry,
    list_supported_e2e_specs,
    normalize_semantic_name,
)


def test_normalize_semantic_name_variants() -> None:
    assert normalize_semantic_name("add_") == "add"
    assert normalize_semantic_name("add_out") == "add"
    assert normalize_semantic_name("any_dim") == "any"
    assert normalize_semantic_name("any_dims") == "any"
    assert normalize_semantic_name("_upsample_bicubic2d_aa") == "upsample_bicubic2d_aa"


def test_build_registry_resolves_all_statuses() -> None:
    payload = build_registry(
        all_ops=[
            "add",
            "any",
            "any_dim",
            "group_norm",
            "softmax",
            "upsample_bicubic2d_aa",
            "randn",  # filtered out in deterministic_forward opset
        ],
        flaggems_commit="a" * 40,
    )
    entries = payload["entries"]
    assert entries
    assert all(str(e["status"]) in STATUS_VALUES for e in entries)
    assert all(isinstance(e.get("mapping_kind"), str) for e in entries)
    assert all(isinstance(e.get("intent_pattern_id"), str) for e in entries)
    # deterministic_forward filter should drop random ops.
    semantic_names = {str(e["semantic_op"]) for e in entries}
    assert "randn" not in semantic_names
    assert "add" in semantic_names
    assert "any" in semantic_names


def test_list_supported_e2e_specs_uses_registry_entries() -> None:
    payload = build_registry(
        all_ops=[
            "add",
            "sub",
            "mul",
            "div",
            "eq",
            "ne",
            "gt",
            "ge",
            "lt",
            "le",
            "neg",
            "ceil",
            "reciprocal",
            "sqrt",
            "exp2",
            "silu",
            "tanh",
            "sigmoid",
            "logical_and",
            "logical_or",
            "logical_not",
            "logical_xor",
            "mm",
            "bmm",
            "addmm",
            "baddbmm",
            "dot",
            "vdot",
            "mv",
            "addmv",
            "abs",
            "rsqrt",
            "full",
            "ones",
            "zeros",
            "maximum",
            "minimum",
            "copy",
            "contiguous",
            "resolve_conj",
            "resolve_neg",
            "to_copy",
            "softmax",
            "group_norm",
            "batch_norm",
            "rms_norm",
            "rms_norm_forward",
            "lerp_scalar",
            "lerp_tensor",
            "relu",
            "sum",
            "mean",
            "all",
            "allclose",
            "isclose",
            "isfinite",
            "isinf",
            "isnan",
            "masked_fill",
            "max",
            "where_self",
            "index",
            "index_select",
            "threshold",
            "exp",
            "clamp",
        ],
        flaggems_commit="b" * 40,
    )
    specs = list_supported_e2e_specs(payload)
    assert "add2d" in specs
    assert "sub2d" in specs
    assert "mul2d" in specs
    assert "div2d" in specs
    assert "eq2d" in specs
    assert "ne2d" in specs
    assert "gt2d" in specs
    assert "ge2d" in specs
    assert "lt2d" in specs
    assert "le2d" in specs
    assert "neg2d" in specs
    assert "ceil2d" in specs
    assert "reciprocal2d" in specs
    assert "sqrt2d" in specs
    assert "exp22d" in specs
    assert "silu2d" in specs
    assert "tanh2d" in specs
    assert "logical_and2d" in specs
    assert "logical_or2d" in specs
    assert "logical_not2d" in specs
    assert "logical_xor2d" in specs
    assert "mm2d" in specs
    assert "bmm3d" in specs
    assert "addmm2d" in specs
    assert "baddbmm3d" in specs
    assert "dot1d" in specs
    assert "vdot1d" in specs
    assert "mv2d" in specs
    assert "addmv2d" in specs
    assert "abs2d" in specs
    assert "rsqrt2d" in specs
    assert "full2d" in specs
    assert "maximum2d" in specs
    assert "minimum2d" in specs
    assert "identity2d" in specs
    assert "cast2d" in specs
    assert "softmax_inner" in specs
    assert "relu2d" in specs
    assert "sigmoid2d" in specs
    assert "where2d" in specs
    assert "index_select2d" in specs
    assert "row_sum" in specs
    assert "row_mean" in specs
    assert "row_all" in specs
    assert "row_max" in specs
    assert "exp2d" in specs
    assert "allclose2d" in specs
    assert "isclose2d" in specs
    assert "isfinite2d" in specs
    assert "isinf2d" in specs
    assert "isnan2d" in specs
    assert "masked_fill2d" in specs
    assert "threshold2d" in specs
    assert "clamp2d" in specs
    assert "lerp2d" in specs
    assert "batch_norm2d" in specs
    assert "rms_norm2d" in specs


def test_build_registry_requires_flaggems_commit() -> None:
    try:
        build_registry(all_ops=["add"])
    except ValueError as e:
        assert "flaggems_commit" in str(e)
    else:
        raise AssertionError("expected ValueError for missing flaggems_commit")
