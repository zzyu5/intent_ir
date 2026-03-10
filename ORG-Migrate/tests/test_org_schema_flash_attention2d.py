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


def test_org_schema_flash_attention2d_allows_descriptive_dim_without_candidates() -> None:
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
                {"id": "g0", "tag": "resident_working_set", "summary": "keep state resident", "scope": "kv_loop", "tensors": ["Q"], "evidence_refs": ["e0"]}
            ],
            "mechanisms": [
                {"id": "m0", "tag": "kv_tile_stage", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["tile_kv", "pipeline_stages"], "evidence_refs": ["e0"]}
            ],
            "dims": [
                {"name": "tile_kv", "role": "kv_tile", "candidates": [32, 64], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "pipeline_stages", "role": "pipeline_depth", "constraints": ["stage count inferred later"], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "attn2d_causal_softmax_v6",
                "bindings": {"ATTN_BLOCK_KV": 64},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "flash.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )
    assert doc.dims[1].name == "pipeline_stages"
    assert doc.dims[1].candidates == []
    assert doc.dims[1].range == {}


def test_org_schema_flash_attention2d_accepts_topological_rationale_graph() -> None:
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
                {"id": "g0", "tag": "resident_working_set", "summary": "keep q resident", "scope": "kv_loop", "tensors": ["Q"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "online reduce", "scope": "softmax", "tensors": ["scores"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "q_resident_state", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["resident_bytes"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "online_softmax_reduce", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["resident_bytes"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "resident_bytes", "role": "resident_budget", "candidates": [256], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "tensors": [
                {"id": "t0", "name": "Q", "role": "query_state", "evidence_refs": ["e0"]},
                {"id": "t1", "name": "scores", "role": "softmax_scores", "evidence_refs": ["e0"]},
            ],
            "tensor_lifetimes": [
                {
                    "id": "lt0",
                    "tensor": "t0",
                    "region": "kv_loop",
                    "storage": "register",
                    "start": "load_q",
                    "end": "dot",
                    "producer_mechanisms": ["m0"],
                    "consumer_mechanisms": ["m1"],
                    "supports_goals": ["g0"],
                    "dims": ["resident_bytes"],
                    "bytes_hint": 256,
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt1",
                    "tensor": "t1",
                    "region": "softmax",
                    "storage": "register",
                    "start": "dot",
                    "end": "softmax_reduce",
                    "producer_mechanisms": ["m1"],
                    "consumer_mechanisms": [],
                    "supports_goals": ["g1"],
                    "dims": [],
                    "evidence_refs": ["e0"],
                },
            ],
            "dataflow_edges": [
                {"id": "df0", "src": "lt0", "dst": "lt1", "tensor": "t1", "kind": "reduce", "order": 0, "mechanisms": ["m1"], "evidence_refs": ["e0"]},
            ],
            "mechanism_topology": [
                {"id": "mt0", "src": "m0", "dst": "m1", "relation": "feeds", "tensors": ["t0", "t1"], "lifetimes": ["lt0", "lt1"], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "attn2d_causal_softmax_v6",
                "bindings": {"ATTN_BLOCK_KV": 64},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "flash.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )
    assert doc.tensor_lifetimes[0].tensor == "t0"
    assert doc.dataflow_edges[0].dst == "lt1"
    assert doc.mechanism_topology[0].relation == "feeds"
