from __future__ import annotations

from pipeline.triton.flaggems_semantic_rules import resolve_semantic_mapping


def test_semantic_rule_templates_for_high_yield_ops() -> None:
    assert resolve_semantic_mapping("sub").intent_ops == ("sub",)
    assert resolve_semantic_mapping("mul").intent_ops == ("mul",)
    assert resolve_semantic_mapping("max_dim").intent_ops == ("reduce_max",)
    assert resolve_semantic_mapping("sum_dim").intent_ops == ("reduce_sum",)
    assert resolve_semantic_mapping("where_scalar_self").intent_ops == ("where",)
    assert resolve_semantic_mapping("where_scalar_other").intent_ops == ("where",)
    assert resolve_semantic_mapping("gather").intent_ops == ("gather",)


def test_semantic_rule_composite_aliases() -> None:
    assert resolve_semantic_mapping("all").intent_pattern_id == "reduce.all_via_not_any"
    assert resolve_semantic_mapping("eq").intent_ops == ("ne", "not")
    assert resolve_semantic_mapping("equal").intent_ops == ("ne", "not")
    assert resolve_semantic_mapping("maximum").intent_ops == ("max",)
    assert resolve_semantic_mapping("minimum").intent_ops == ("min",)
