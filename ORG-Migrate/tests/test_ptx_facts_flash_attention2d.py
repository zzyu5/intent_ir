from __future__ import annotations

from org.facts.ptx import extract_ptx_mechanism_facts


def test_ptx_facts_flash_attention2d() -> None:
    ptx = """
    .extern .shared .align 16 .b8 global_smem[];
    .reqntid 128
    cp.async.cg.shared.global;
    cp.async.commit_group;
    cp.async.wait_group 0;
    shfl.sync.bfly;
    shfl.sync.down;
    bar.sync 0;
    """.strip()
    facts = extract_ptx_mechanism_facts(ptx, kernel_name="flash_attention2d", artifact_path="flash.ptx")
    mechanisms = dict(facts.get("mechanisms") or {})
    assert mechanisms["staging.shared_memory"]["present"] is True
    assert mechanisms["mapping.block_threads"]["attrs"]["reqntid"] == [128]
    assert mechanisms["mapping.block_threads"]["attrs"]["threads_per_block"] == 128
    assert mechanisms["pipeline.async_copy"]["attrs"]["async_copy_count"] == 3
    assert mechanisms["pipeline.async_copy"]["attrs"]["commit_group_count"] == 1
    assert mechanisms["pipeline.async_copy"]["attrs"]["wait_groups"] == [0]
    assert mechanisms["communication.shuffle"]["attrs"]["shuffle_count"] == 2
    assert mechanisms["communication.shuffle"]["attrs"]["shuffle_ops"] == ["bfly", "down"]
    assert mechanisms["communication.block_sync"]["attrs"]["bar_sync_count"] == 1
