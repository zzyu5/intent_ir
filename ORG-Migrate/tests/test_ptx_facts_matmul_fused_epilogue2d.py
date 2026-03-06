from __future__ import annotations

from org.facts.ptx import extract_ptx_mechanism_facts


def test_ptx_facts_matmul_fused_epilogue2d() -> None:
    ptx = "mma.sync.aligned.m16n8k8; ldmatrix.sync.aligned.x4; cp.async.cg.shared.global;"
    facts = extract_ptx_mechanism_facts(ptx, kernel_name="matmul_fused_epilogue2d", artifact_path="matmul.ptx")
    mechanisms = dict(facts.get("mechanisms") or {})
    assert mechanisms["primitive.mma"]["present"] is True
    assert mechanisms["primitive.matrix_load"]["present"] is True
    assert mechanisms["pipeline.async_copy"]["present"] is True
