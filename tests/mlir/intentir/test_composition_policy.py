from __future__ import annotations

from intent_ir.ops.composition_policy import (
    composition_required,
    evaluate_complex_family_ratio,
    single_intent_ratio_target,
)


def test_composition_required_flags_macro_and_selected_complex_semantics() -> None:
    assert composition_required(
        semantic_op="softmax",
        family="norm_activation",
        mapping_kind="macro_template",
    )
    assert composition_required(
        semantic_op="where_scalar_self",
        family="index_scatter_gather",
        mapping_kind="index_template",
    )
    assert not composition_required(
        semantic_op="gather",
        family="index_scatter_gather",
        mapping_kind="index_template",
    )
    assert not composition_required(
        semantic_op="add",
        family="elementwise_broadcast",
        mapping_kind="binary_template",
    )


def test_ratio_targets_and_evaluation() -> None:
    assert single_intent_ratio_target("m1") == 0.40
    assert single_intent_ratio_target("m2") == 0.10
    ok = evaluate_complex_family_ratio(ratio=0.08, stage="m2")
    bad = evaluate_complex_family_ratio(ratio=0.35, stage="m2")
    assert ok["ok"] is True
    assert bad["ok"] is False
