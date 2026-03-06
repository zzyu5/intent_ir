from __future__ import annotations

from org.facts.ptx import extract_ptx_mechanism_facts


def test_ptx_facts_matmul_fused_epilogue2d() -> None:
    ptx = """
    .extern .shared .align 16 .b8 global_smem[];
    .reqntid 128
    mma.sync.aligned.m16n8k8;
    mma.sync.aligned.m16n8k8;
    ldmatrix.sync.aligned.x4;
    cp.async.cg.shared.global;
    cp.async.commit_group;
    bar.sync 0;
    """.strip()
    facts = extract_ptx_mechanism_facts(ptx, kernel_name="matmul_fused_epilogue2d", artifact_path="matmul.ptx")
    mechanisms = dict(facts.get("mechanisms") or {})
    assert mechanisms["staging.shared_memory"]["present"] is True
    assert mechanisms["mapping.block_threads"]["attrs"]["warp_count_estimate"] == 4
    assert mechanisms["primitive.mma"]["attrs"]["mma_sync_count"] == 2
    assert "mma.sync.aligned.m16n8k8" in mechanisms["primitive.mma"]["attrs"]["mma_kinds"]
    assert mechanisms["primitive.mma"]["attrs"]["complete_matrix_pipeline"] is True
    assert mechanisms["primitive.matrix_load"]["attrs"]["ldmatrix_count"] == 1
    assert mechanisms["primitive.matrix_load"]["attrs"]["ldmatrix_widths"] == [4]
    assert mechanisms["pipeline.async_copy"]["attrs"]["commit_group_count"] == 1
    assert mechanisms["pipeline.async_copy"]["attrs"]["complete_async_pipeline"] is False
    assert mechanisms["communication.block_sync"]["attrs"]["bar_sync_count"] == 1
