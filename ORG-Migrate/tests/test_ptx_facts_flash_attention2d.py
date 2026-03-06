from __future__ import annotations

from org.facts.ptx import extract_ptx_mechanism_facts


def test_ptx_facts_flash_attention2d() -> None:
    ptx = "cp.async.cg.shared.global; shfl.sync.bfly; bar.sync 0;"
    facts = extract_ptx_mechanism_facts(ptx, kernel_name="flash_attention2d", artifact_path="flash.ptx")
    mechanisms = dict(facts.get("mechanisms") or {})
    assert mechanisms["pipeline.async_copy"]["present"] is True
    assert mechanisms["communication.shuffle"]["present"] is True
    assert mechanisms["communication.block_sync"]["present"] is True
