from __future__ import annotations

from org.schema import validate_org_doc


def test_org_schema_matmul_fused_epilogue2d_requires_fusion_mechanism() -> None:
    doc = validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "matmul_fused_epilogue2d",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 32, "N": 32, "K": 32},
                "artifacts": {"ttgir_path": "matmul.ttgir"},
            },
            "goals": [
                {
                    "id": "g0",
                    "tag": "fused_epilogue_avoid_writeback",
                    "summary": "keep epilogue fused before store",
                    "scope": "epilogue",
                    "tensors": ["bias", "row_mask", "col_mask"],
                    "evidence_refs": ["e0"],
                }
            ],
            "mechanisms": [
                {
                    "id": "m0",
                    "tag": "epilogue_fused_writeback",
                    "category": "fusion",
                    "supports_goals": ["g0"],
                    "attrs": {"convert_layout": True},
                    "dims": ["tile_m", "tile_n", "tile_k"],
                    "evidence_refs": ["e0"],
                }
            ],
            "dims": [
                {"name": "tile_m", "role": "m_tile", "candidates": [32, 64], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "tile_n", "role": "n_tile", "candidates": [16, 32], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "tile_k", "role": "k_tile", "candidates": [16, 32], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "matmul_mma_tf32_v1",
                "bindings": {"MMA_BM": 32, "MMA_BN": 32, "MMA_BK": 32},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "matmul.ttgir:1", "summary": "epilogue fused"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )
    assert doc.kernel == "matmul_fused_epilogue2d"
    assert doc.mechanisms[0].category == "fusion"
    assert doc.goals[0].tag == "fused_epilogue_avoid_writeback"
