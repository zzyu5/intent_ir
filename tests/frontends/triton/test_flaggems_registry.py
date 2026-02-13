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
        ]
    )
    entries = payload["entries"]
    assert entries
    assert all(str(e["status"]) in STATUS_VALUES for e in entries)
    # deterministic_forward filter should drop random ops.
    semantic_names = {str(e["semantic_op"]) for e in entries}
    assert "randn" not in semantic_names
    assert "add" in semantic_names
    assert "any" in semantic_names


def test_list_supported_e2e_specs_uses_registry_entries() -> None:
    payload = build_registry(all_ops=["add", "softmax", "group_norm", "relu", "sum", "max", "where_self", "exp", "clamp"])
    specs = list_supported_e2e_specs(payload)
    assert "add2d" in specs
    assert "softmax_inner" in specs
    assert "relu2d" in specs
    assert "row_sum" in specs
    assert "row_max" in specs
    assert "exp2d" in specs
    assert "clamp2d" in specs
