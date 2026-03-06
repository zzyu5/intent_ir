from __future__ import annotations

import json
from pathlib import Path

from org.facts.source_oracle import extract_source_oracle_facts


def test_extract_source_oracle_facts_infers_unique_non_target_arch(tmp_path: Path) -> None:
    db_path = tmp_path / "cuda.jsonl"
    entries = [
        {
            "schema_version": "intentir_tuning_db_entry_v1",
            "backend": "cuda",
            "kernel": "matmul_fused_epilogue2d",
            "arch": "sm90",
            "kernel_kind": "matmul_mma_tf32_v1",
            "bindings": {"MMA_BM": 32, "MMA_BN": 32, "MMA_BK": 32, "MMA_ASYNC_COPY": 1},
            "compiler_stack": "python",
        },
        {
            "schema_version": "intentir_tuning_db_entry_v1",
            "backend": "cuda",
            "kernel": "matmul_fused_epilogue2d",
            "arch": "sm120",
            "kernel_kind": "matmul_mma_tf32_v1",
            "bindings": {"MMA_BM": 32, "MMA_BN": 32, "MMA_BK": 32, "MMA_ASYNC_COPY": 1},
            "compiler_stack": "python",
        },
    ]
    db_path.write_text("\n".join(json.dumps(x) for x in entries) + "\n", encoding="utf-8")

    facts = extract_source_oracle_facts(
        kernel="matmul_fused_epilogue2d",
        source_arch="",
        target_arch="sm120",
        shape_bindings={"M": 32, "N": 32, "K": 32},
        compiler_stack="python",
        db_path=str(db_path),
    )

    oracle = dict(facts.get("oracle") or {})
    assert facts["available"] is True
    assert oracle["arch"] == "sm90"
    assert oracle["kernel_kind"] == "matmul_mma_tf32_v1"
    assert dict(oracle["bindings"])["MMA_ASYNC_COPY"] == 1


def test_extract_source_oracle_facts_skips_ambiguous_source_arch(tmp_path: Path) -> None:
    db_path = tmp_path / "cuda.jsonl"
    entries = [
        {
            "schema_version": "intentir_tuning_db_entry_v1",
            "backend": "cuda",
            "kernel": "flash_attention2d",
            "arch": "sm90",
            "kernel_kind": "attn2d_causal_softmax_v6",
            "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6},
            "compiler_stack": "python",
        },
        {
            "schema_version": "intentir_tuning_db_entry_v1",
            "backend": "cuda",
            "kernel": "flash_attention2d",
            "arch": "sm89",
            "kernel_kind": "attn2d_causal_softmax_v6",
            "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6},
            "compiler_stack": "python",
        },
        {
            "schema_version": "intentir_tuning_db_entry_v1",
            "backend": "cuda",
            "kernel": "flash_attention2d",
            "arch": "sm120",
            "kernel_kind": "attn2d_causal_softmax_v6",
            "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6},
            "compiler_stack": "python",
        },
    ]
    db_path.write_text("\n".join(json.dumps(x) for x in entries) + "\n", encoding="utf-8")

    facts = extract_source_oracle_facts(
        kernel="flash_attention2d",
        source_arch="",
        target_arch="sm120",
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        compiler_stack="python",
        db_path=str(db_path),
    )

    oracle = dict(facts.get("oracle") or {})
    assert facts["available"] is True
    assert oracle["arch"] == "sm90"
    assert oracle["kernel_kind"] == "attn2d_causal_softmax_v6"
