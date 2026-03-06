from __future__ import annotations

from org.schema import validate_org_doc


def test_org_schema_flash_attention2d_minimal_ok() -> None:
    doc = validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "flash_attention2d",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
                "artifacts": {"ttgir_path": "flash.ttgir"},
            },
            "goals": [
                {
                    "id": "g0",
                    "tag": "resident_working_set",
                    "summary": "keep q/state close to compute",
                    "scope": "kv_loop",
                    "tensors": ["Q", "Out"],
                    "evidence_refs": ["e0"],
                }
            ],
            "mechanisms": [
                {
                    "id": "m0",
                    "tag": "kv_tile_stage",
                    "category": "staging",
                    "supports_goals": ["g0"],
                    "attrs": {"storage": "local"},
                    "dims": ["tile_kv"],
                    "evidence_refs": ["e0"],
                }
            ],
            "dims": [
                {
                    "name": "tile_kv",
                    "role": "kv_tile",
                    "candidates": [16, 32, 64],
                    "constraints": ["tile_kv <= KV_CTX"],
                    "evidence_refs": ["e0"],
                }
            ],
            "source_oracle": {
                "kernel_kind": "attn2d_causal_softmax_v6",
                "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "flash.ttgir:1", "summary": "blocked layout"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )
    assert doc.kernel == "flash_attention2d"
    assert doc.goals[0].tag == "resident_working_set"
    assert doc.mechanisms[0].category == "staging"
    assert doc.dims[0].name == "tile_kv"
    assert doc.source_oracle.kernel_kind == "attn2d_causal_softmax_v6"
