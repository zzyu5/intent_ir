from __future__ import annotations

from pipeline.triton.flaggems_semantic_rules import resolve_semantic_mapping


def test_semantic_rule_templates_for_high_yield_ops() -> None:
    assert resolve_semantic_mapping("sub").intent_ops == ("sub",)
    assert resolve_semantic_mapping("mul").intent_ops == ("mul",)
    assert resolve_semantic_mapping("max").intent_ops == ("reduce_max",)
    assert resolve_semantic_mapping("max_dim").intent_ops == ("reduce_max",)
    assert resolve_semantic_mapping("sum_dim").intent_ops == ("reduce_sum",)
    assert resolve_semantic_mapping("mean").intent_ops == ("reduce_sum", "div")
    assert resolve_semantic_mapping("all").intent_ops == ("not", "reduce_any", "not")
    assert resolve_semantic_mapping("where_scalar_self").intent_ops == ("where",)
    assert resolve_semantic_mapping("where_scalar_other").intent_ops == ("where",)
    assert resolve_semantic_mapping("gather").intent_ops == ("gather",)
    assert resolve_semantic_mapping("mm").intent_ops == ("matmul",)
    assert resolve_semantic_mapping("addmm").intent_ops == ("matmul", "add")


def test_semantic_rule_composite_aliases() -> None:
    assert resolve_semantic_mapping("all").intent_pattern_id == "reduce.all_via_not_any"
    assert resolve_semantic_mapping("eq").intent_ops == ("ne", "not")
    assert resolve_semantic_mapping("equal").intent_ops == ("ne", "not")
    assert resolve_semantic_mapping("maximum").intent_ops == ("max",)
    assert resolve_semantic_mapping("minimum").intent_ops == ("min",)
    assert resolve_semantic_mapping("ge_scalar").intent_ops == ("ge",)
    assert resolve_semantic_mapping("logical_and").intent_ops == ("and",)
    assert resolve_semantic_mapping("logical_not").intent_ops == ("not",)
    assert resolve_semantic_mapping("logical_or").intent_ops == ("or",)
    assert resolve_semantic_mapping("clamp_min").intent_ops == ("max",)
    assert resolve_semantic_mapping("zeros_like").intent_ops == ("const", "broadcast_in_dim")
    assert resolve_semantic_mapping("full_like").intent_ops == ("const", "broadcast_in_dim")
    assert resolve_semantic_mapping("pow_scalar").intent_ops == ("pow",)
    assert resolve_semantic_mapping("lerp_scalar").intent_ops == ("sub", "mul", "add")
    assert resolve_semantic_mapping("logical_xor").intent_ops == ("or", "and", "not", "and")
    assert resolve_semantic_mapping("log_softmax").intent_ops == ("softmax", "log")
    assert resolve_semantic_mapping("exp2").intent_ops == ("const", "mul", "exp")
    assert resolve_semantic_mapping("sigmoid").intent_ops == ("const", "mul", "exp", "add", "div")
    assert resolve_semantic_mapping("softplus").intent_ops == ("exp", "add", "log")
