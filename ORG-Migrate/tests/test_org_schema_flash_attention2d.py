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
                {"id": "g0", "tag": "resident_working_set", "summary": "keep q and out resident through the kv loop", "scope": "kv_loop", "tensors": ["Q", "Out"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "carry max and sum state across streamed tiles", "scope": "softmax", "tensors": ["max_state", "sum_state"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "q_resident_state", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["resident_bytes"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "kv_streamed_tiles", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["tile_kv", "resident_bytes"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "online_softmax_reduce", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["score_warps"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "resident_bytes", "role": "resident_budget", "candidates": [33024], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "tile_kv", "role": "kv_tile", "candidates": [32, 64], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "score_warps", "role": "score_reduce", "candidates": [4, 6], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "tensors": [
                {"id": "t0", "name": "Q", "role": "query_state", "evidence_refs": ["e0"]},
                {"id": "t1", "name": "K", "role": "key_tile", "evidence_refs": ["e0"]},
                {"id": "t2", "name": "V", "role": "value_tile", "evidence_refs": ["e0"]},
                {"id": "t3", "name": "max_state", "role": "softmax_max", "evidence_refs": ["e0"]},
                {"id": "t4", "name": "sum_state", "role": "softmax_sum", "evidence_refs": ["e0"]},
                {"id": "t5", "name": "Out", "role": "output_accumulator", "evidence_refs": ["e0"]},
            ],
            "tensor_lifetimes": [
                {
                    "id": "lt0",
                    "tensor": "t0",
                    "region": "kv_loop",
                    "storage": "register",
                    "start": "load_q",
                    "end": "kv_loop_exit",
                    "producer_mechanisms": ["m0"],
                    "consumer_mechanisms": ["m2"],
                    "supports_goals": ["g0"],
                    "dims": ["resident_bytes"],
                    "bytes_hint": 256,
                    "reuse_window": "kv_loop",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt1",
                    "tensor": "t1",
                    "region": "kv_loop",
                    "storage": "shared",
                    "start": "load_k_tile",
                    "end": "softmax_update",
                    "producer_mechanisms": ["m1"],
                    "consumer_mechanisms": ["m2"],
                    "supports_goals": ["g1"],
                    "dims": ["tile_kv", "resident_bytes"],
                    "bytes_hint": 16384,
                    "reuse_window": "cta_tile",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt2",
                    "tensor": "t2",
                    "region": "kv_loop",
                    "storage": "shared",
                    "start": "load_v_tile",
                    "end": "softmax_update",
                    "producer_mechanisms": ["m1"],
                    "consumer_mechanisms": ["m2"],
                    "supports_goals": ["g1"],
                    "dims": ["tile_kv", "resident_bytes"],
                    "bytes_hint": 16384,
                    "reuse_window": "cta_tile",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt3",
                    "tensor": "t3",
                    "region": "kv_loop",
                    "storage": "register",
                    "start": "reduce_max",
                    "end": "softmax_update",
                    "producer_mechanisms": ["m2"],
                    "consumer_mechanisms": ["m2"],
                    "supports_goals": ["g1"],
                    "dims": ["score_warps"],
                    "bytes_hint": 4,
                    "reuse_window": "kv_loop",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt4",
                    "tensor": "t4",
                    "region": "kv_loop",
                    "storage": "register",
                    "start": "reduce_sum",
                    "end": "normalize",
                    "producer_mechanisms": ["m2"],
                    "consumer_mechanisms": ["m2"],
                    "supports_goals": ["g1"],
                    "dims": ["score_warps"],
                    "bytes_hint": 4,
                    "reuse_window": "kv_loop",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt5",
                    "tensor": "t5",
                    "region": "kv_loop",
                    "storage": "register",
                    "start": "softmax_update",
                    "end": "store",
                    "producer_mechanisms": ["m2"],
                    "consumer_mechanisms": [],
                    "supports_goals": ["g0", "g1"],
                    "dims": ["resident_bytes"],
                    "bytes_hint": 256,
                    "reuse_window": "kv_loop",
                    "evidence_refs": ["e0"],
                },
            ],
            "dataflow_edges": [
                {"id": "df0", "src": "lt0", "dst": "lt3", "tensor": "t3", "kind": "reduce", "order": 0, "mechanisms": ["m2"], "evidence_refs": ["e0"]},
                {"id": "df1", "src": "lt1", "dst": "lt3", "tensor": "t3", "kind": "stage", "order": 1, "mechanisms": ["m1", "m2"], "evidence_refs": ["e0"]},
                {"id": "df2", "src": "lt4", "dst": "lt5", "tensor": "t5", "kind": "update", "order": 2, "mechanisms": ["m2"], "evidence_refs": ["e0"]},
            ],
            "mechanism_topology": [
                {"id": "mt0", "src": "m0", "dst": "m2", "relation": "feeds", "tensors": ["t0", "t3", "t4", "t5"], "lifetimes": ["lt0", "lt3", "lt4", "lt5"], "evidence_refs": ["e0"]},
                {"id": "mt1", "src": "m1", "dst": "m2", "relation": "feeds", "tensors": ["t1", "t2", "t3", "t4", "t5"], "lifetimes": ["lt1", "lt2", "lt3", "lt4", "lt5"], "evidence_refs": ["e0"]},
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
    assert doc.tensor_lifetimes[2].storage == "shared"
    assert doc.dataflow_edges[0].dst == "lt3"
    assert doc.mechanism_topology[0].relation == "feeds"
