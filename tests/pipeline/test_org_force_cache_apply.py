from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("triton")
pytest.importorskip("torch")

from intent_ir.ir import IntentFunction
from pipeline.interfaces import KernelArtifactBundle, KernelDescriptor
import pipeline.triton.org_bridge as org_bridge
from pipeline.triton.core import _run_org_plugin
from pipeline.triton.org_bridge import load_org_attr


def _dummy_intent(name: str) -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": str(name),
            "tensors": {"x": {"dtype": "f32", "shape": [1], "layout": "row_major"}, "Out": {"dtype": "f32", "shape": [1], "layout": "row_major"}},
            "ops": [{"op": "identity", "inputs": ["x"], "output": "Out"}],
            "outputs": ["Out"],
        }
    )


def _dummy_desc(*, kernel: str, ttgir_path: Path | None = None, ptx_path: Path | None = None) -> KernelDescriptor:
    desc = KernelDescriptor(schema_version="kernel_desc_v1.0", name=str(kernel), frontend="triton")
    desc.source_text = "def kernel(): pass"
    desc.artifacts = KernelArtifactBundle(ttgir_path=(str(ttgir_path) if ttgir_path is not None else None))
    if ptx_path is not None:
        desc.artifacts.extra["ptx_path"] = str(ptx_path)
    return desc


def _seed_payload(*, kernel: str) -> dict[str, object]:
    if kernel == "flash_attention2d":
        return {
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
                {"id": "g0", "tag": "resident_working_set", "summary": "keep Q and output accumulator resident through the kv loop", "scope": "kv_loop", "tensors": ["Q", "Out"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "carry online max/sum state across streamed tiles", "scope": "softmax", "tensors": ["max_state", "sum_state"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "avoid a materialized score matrix", "scope": "softmax", "tensors": ["max_state", "sum_state"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "pipeline the next streamed K/V tile", "scope": "kv_loop", "tensors": ["K", "V"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "q_resident_state", "category": "staging", "supports_goals": ["g0"], "attrs": {"communication_scope": "cta"}, "dims": ["resident_bytes"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "kv_streamed_tiles", "category": "staging", "supports_goals": ["g0", "g3"], "attrs": {"communication_scope": "kv_loop"}, "dims": ["tile_kv", "resident_bytes", "pipeline_stages"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "online_softmax_reduce", "category": "communication", "supports_goals": ["g1", "g2"], "attrs": {"communication_scope": "warp"}, "dims": ["score_warps"], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "prefetch_pipeline", "category": "pipeline", "supports_goals": ["g3"], "attrs": {"pipeline_depth": 2}, "dims": ["pipeline_stages"], "evidence_refs": ["e0"]},
                {"id": "m4", "tag": "output_layout_convert", "category": "fusion", "supports_goals": ["g0"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "tile_kv", "role": "kv_tile", "candidates": [32, 64], "constraints": ["tile_kv <= KV_CTX"], "evidence_refs": ["e0"]},
                {"name": "score_warps", "role": "score_reduce", "candidates": [4, 6], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "pipeline_stages", "role": "pipeline_depth", "candidates": [2], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "resident_bytes", "role": "resident_budget", "candidates": [33024], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "tensors": [
                {"id": "t0", "name": "Q", "role": "query_state", "shape_refs": ["HEAD_DIM"], "evidence_refs": ["e0"]},
                {"id": "t1", "name": "K", "role": "key_tile", "shape_refs": ["tile_kv", "HEAD_DIM"], "evidence_refs": ["e0"]},
                {"id": "t2", "name": "V", "role": "value_tile", "shape_refs": ["tile_kv", "HEAD_DIM"], "evidence_refs": ["e0"]},
                {"id": "t3", "name": "max_state", "role": "softmax_max", "evidence_refs": ["e0"]},
                {"id": "t4", "name": "sum_state", "role": "softmax_sum", "evidence_refs": ["e0"]},
                {"id": "t5", "name": "Out", "role": "output_accumulator", "shape_refs": ["HEAD_DIM"], "evidence_refs": ["e0"]},
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
                    "consumer_mechanisms": ["m2", "m3"],
                    "supports_goals": ["g0", "g3"],
                    "dims": ["tile_kv", "resident_bytes", "pipeline_stages"],
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
                    "consumer_mechanisms": ["m2", "m3"],
                    "supports_goals": ["g0", "g3"],
                    "dims": ["tile_kv", "resident_bytes", "pipeline_stages"],
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
                    "consumer_mechanisms": ["m2", "m4"],
                    "supports_goals": ["g1", "g2"],
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
                    "consumer_mechanisms": ["m2", "m4"],
                    "supports_goals": ["g1", "g2"],
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
                    "consumer_mechanisms": ["m4"],
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
                {"id": "df2", "src": "lt2", "dst": "lt5", "tensor": "t5", "kind": "update", "order": 2, "mechanisms": ["m1", "m2"], "evidence_refs": ["e0"]},
                {"id": "df3", "src": "lt3", "dst": "lt4", "tensor": "t4", "kind": "normalize", "order": 3, "mechanisms": ["m2"], "evidence_refs": ["e0"]},
                {"id": "df4", "src": "lt4", "dst": "lt5", "tensor": "t5", "kind": "update", "order": 4, "mechanisms": ["m2"], "evidence_refs": ["e0"]},
            ],
            "mechanism_topology": [
                {"id": "mt0", "src": "m0", "dst": "m2", "relation": "feeds", "tensors": ["t0", "t3", "t4", "t5"], "lifetimes": ["lt0", "lt3", "lt4", "lt5"], "evidence_refs": ["e0"]},
                {"id": "mt1", "src": "m1", "dst": "m2", "relation": "feeds", "tensors": ["t1", "t2", "t3", "t4", "t5"], "lifetimes": ["lt1", "lt2", "lt3", "lt4", "lt5"], "evidence_refs": ["e0"]},
                {"id": "mt2", "src": "m1", "dst": "m3", "relation": "gates", "tensors": ["t1", "t2"], "lifetimes": ["lt1", "lt2"], "evidence_refs": ["e0"]},
                {"id": "mt3", "src": "m3", "dst": "m1", "relation": "feeds", "tensors": ["t1", "t2"], "lifetimes": ["lt1", "lt2"], "evidence_refs": ["e0"]},
                {"id": "mt4", "src": "m2", "dst": "m4", "relation": "feeds", "tensors": ["t3", "t4", "t5"], "lifetimes": ["lt3", "lt4", "lt5"], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "attn2d_causal_softmax_v6",
                "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "flash.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    if kernel == "_attn_fwd":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "_attn_fwd",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"Z": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
                "artifacts": {"ttgir_path": "attn_fwd.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep q/state resident", "scope": "q_state", "tensors": ["Q"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "online reduce", "scope": "softmax", "tensors": ["Out"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "avoid score matrix", "scope": "scores", "tensors": ["scores"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "pipeline loads", "scope": "kv_loop", "tensors": ["K", "V"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "qkv_stage", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_m", "block_kv"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "online_softmax_reduce", "category": "communication", "supports_goals": ["g1", "g2"], "attrs": {}, "dims": ["block_kv"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "mask_causal_apply", "category": "communication", "supports_goals": ["g2"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "prefetch_pipeline", "category": "pipeline", "supports_goals": ["g3"], "attrs": {}, "dims": ["pipeline_stages"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_m", "role": "query_tile", "candidates": [8, 4], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "block_kv", "role": "kv_tile", "candidates": [32, 16], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "pipeline_stages", "role": "pipeline_depth", "candidates": [2], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "attn_fwd_tiled_v3",
                "bindings": {"ATTN_FWD_BLOCK_M": 8, "ATTN_FWD_BLOCK_KV": 32},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "attn_fwd.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    if kernel == "masked_softmax2d":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "masked_softmax2d",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 64},
                "artifacts": {"ttgir_path": "masked_softmax2d.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "row reduction", "scope": "softmax", "tensors": ["output"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "mask inline", "scope": "mask", "tensors": ["mask"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "vector row path", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "row_reduction", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "mask_apply", "category": "communication", "supports_goals": ["g2"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "vector_row_path", "category": "mapping", "supports_goals": ["g3"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [{"name": "block_threads", "role": "thread_block", "candidates": [64, 128], "constraints": [], "evidence_refs": ["e0"]}],
            "source_oracle": {"kernel_kind": "row_masked_softmax_axis1_v1", "bindings": {}, "arch": "sm90", "compiler_stack": "python", "evidence_refs": ["e1"]},
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "masked_softmax2d.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    if kernel == "softmax_inner":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "softmax_inner",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 64},
                "artifacts": {"ttgir_path": "softmax_inner.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "row reduction", "scope": "softmax", "tensors": ["output"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "avoid extra buffer", "scope": "softmax", "tensors": ["scores"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "vector row path", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "row_reduction", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "vector_row_path", "category": "mapping", "supports_goals": ["g3"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [{"name": "block_threads", "role": "thread_block", "candidates": [64, 128], "constraints": [], "evidence_refs": ["e0"]}],
            "source_oracle": {"kernel_kind": "row_softmax_axis1_triton_v1", "bindings": {"SOFTMAX_BLOCK_THREADS": 64}, "arch": "sm90", "compiler_stack": "python", "evidence_refs": ["e1"]},
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "softmax_inner.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    if kernel == "row_sum":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "row_sum",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 256},
                "artifacts": {"ttgir_path": "row_sum.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "reduction_tree_balance", "summary": "balanced reduction tree", "scope": "reduce", "tensors": ["output"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "memory_coalescing", "summary": "vector row loads", "scope": "load", "tensors": ["input"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "vector_row_path", "category": "mapping", "supports_goals": ["g2"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "row_reduction", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "shared_staging", "category": "staging", "supports_goals": ["g0", "g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_threads", "role": "thread_block", "candidates": [128, 64], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "vector_width", "role": "vector_width", "candidates": [2, 1], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "row_sum_axis1_v2",
                "bindings": {"ROW_REDUCE_BLOCK_THREADS": 128, "ROW_REDUCE_VECTOR_WIDTH": 2, "ROW_REDUCE_SHARED_STAGE": 1},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "row_sum.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    if kernel == "row_max":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "row_max",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 256},
                "artifacts": {"ttgir_path": "row_max.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "reduction_tree_balance", "summary": "balanced reduction tree", "scope": "reduce", "tensors": ["output"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "memory_coalescing", "summary": "vector row loads", "scope": "load", "tensors": ["input"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "tile_load_stage", "category": "mapping", "supports_goals": ["g2"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "warp_reduction_tree", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "block_synchronization", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_threads", "role": "thread_block", "candidates": [128, 64], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "vector_width", "role": "vector_width", "candidates": [2, 1], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "row_max_axis1_v2",
                "bindings": {"ROW_REDUCE_BLOCK_THREADS": 128, "ROW_REDUCE_VECTOR_WIDTH": 2, "ROW_REDUCE_SHARED_STAGE": 1},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "row_max.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    if kernel == "layer_norm_persistent":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "layer_norm_persistent",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 64},
                "artifacts": {"ttgir_path": "layer_norm_persistent.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "persistent_row_state", "summary": "cache row statistics", "scope": "norm", "tensors": ["mean", "rstd"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "memory_coalescing", "summary": "vectorized row path", "scope": "load", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "affine_epilogue_fusion", "summary": "fuse affine writeback", "scope": "epilogue", "tensors": ["weight", "bias"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "warp_reduction", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "persistent_row_cache", "category": "staging", "supports_goals": ["g1"], "attrs": {}, "dims": ["persistent_row"], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "tile_load_stage", "category": "mapping", "supports_goals": ["g2"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m4", "tag": "affine_epilogue", "category": "fusion", "supports_goals": ["g3"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m5", "tag": "row_parallel_axis", "category": "mapping", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m6", "tag": "block_synchronization", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_threads", "role": "thread_block", "candidates": [32, 64], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "vector_width", "role": "vector_width", "candidates": [2, 1], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "persistent_row", "role": "persistent_row", "candidates": [1], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "resident_bytes", "role": "resident_bytes", "candidates": [256], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "tensors": [
                {"id": "t0", "name": "input", "role": "input_row", "aliases": ["x"], "shape_refs": ["N"], "evidence_refs": ["e0"]},
                {"id": "t1", "name": "row_stats", "role": "row_stats", "aliases": ["mean", "rstd"], "evidence_refs": ["e0"]},
                {"id": "t2", "name": "out", "role": "affine_out", "evidence_refs": ["e0"]},
            ],
            "tensor_lifetimes": [
                {
                    "id": "lt0",
                    "tensor": "t0",
                    "region": "row_reduce",
                    "storage": "shared",
                    "start": "row_load",
                    "end": "row_reduce",
                    "producer_mechanisms": ["m3"],
                    "consumer_mechanisms": ["m0", "m1"],
                    "supports_goals": ["g0", "g2"],
                    "dims": ["vector_width", "resident_bytes"],
                    "bytes_hint": 256,
                    "reuse_window": "row_reduce",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt1",
                    "tensor": "t1",
                    "region": "row_stats",
                    "storage": "shared",
                    "start": "row_reduce",
                    "end": "affine_epilogue",
                    "producer_mechanisms": ["m1", "m2"],
                    "consumer_mechanisms": ["m4"],
                    "supports_goals": ["g1"],
                    "dims": ["persistent_row"],
                    "bytes_hint": 8,
                    "reuse_window": "row_epilogue",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt2",
                    "tensor": "t2",
                    "region": "affine_epilogue",
                    "storage": "register",
                    "start": "affine_epilogue",
                    "end": "store",
                    "producer_mechanisms": ["m4"],
                    "consumer_mechanisms": [],
                    "supports_goals": ["g3"],
                    "dims": [],
                    "evidence_refs": ["e0"],
                },
            ],
            "dataflow_edges": [
                {"id": "df0", "src": "lt0", "dst": "lt1", "tensor": "t1", "kind": "reduce", "order": 0, "mechanisms": ["m0", "m1"], "evidence_refs": ["e0"]},
                {"id": "df1", "src": "lt1", "dst": "lt2", "tensor": "t2", "kind": "epilogue", "order": 1, "mechanisms": ["m2", "m4"], "evidence_refs": ["e0"]},
            ],
            "mechanism_topology": [
                {"id": "mt0", "src": "m3", "dst": "m0", "relation": "vectorizes", "tensors": ["t0"], "lifetimes": ["lt0"], "evidence_refs": ["e0"]},
                {"id": "mt1", "src": "m0", "dst": "m2", "relation": "feeds", "tensors": ["t0", "t1"], "lifetimes": ["lt0", "lt1"], "evidence_refs": ["e0"]},
                {"id": "mt2", "src": "m1", "dst": "m2", "relation": "feeds", "tensors": ["t1"], "lifetimes": ["lt1"], "evidence_refs": ["e0"]},
                {"id": "mt3", "src": "m2", "dst": "m4", "relation": "feeds", "tensors": ["t1", "t2"], "lifetimes": ["lt1", "lt2"], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "layer_norm_axis1_v1",
                "bindings": {"LAYER_NORM_BLOCK_THREADS": 32, "LAYER_NORM_VECTOR_WIDTH": 2, "LAYER_NORM_PERSISTENT_ROW": 1},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "layer_norm_persistent.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    if kernel == "add2d":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "add2d",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 256},
                "artifacts": {"ttgir_path": "add2d.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "tile resident", "scope": "tile", "tensors": ["A", "B"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "memory_coalescing", "summary": "vector io", "scope": "load_store", "tensors": ["A", "B", "C"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "register compute", "scope": "compute", "tensors": ["C"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "wide cta", "scope": "grid", "tensors": ["C"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "blocked_register_layout", "category": "tiling", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads", "vector_width"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "vector_global_io", "category": "mapping", "supports_goals": ["g1"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "elementwise_add_primitive", "category": "primitive", "supports_goals": ["g2"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "two_axis_grid_mapping", "category": "mapping", "supports_goals": ["g3"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_threads", "role": "thread_block", "candidates": [256, 512, 128], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "vector_width", "role": "vector_width", "candidates": [4, 2, 1], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "elementwise_v1",
                "bindings": {"ELEMENTWISE_BLOCK_THREADS": 256, "ELEMENTWISE_VECTOR_WIDTH": 4},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "add2d.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    if kernel == "exp2d":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "exp2d",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 256},
                "artifacts": {"ttgir_path": "exp2d.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "tile resident", "scope": "tile", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "memory_coalescing", "summary": "vector io", "scope": "load_store", "tensors": ["input", "output"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "register compute", "scope": "compute", "tensors": ["output"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "wide cta", "scope": "grid", "tensors": ["output"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "blocked_register_layout", "category": "tiling", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads", "vector_width"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "vector_global_io", "category": "mapping", "supports_goals": ["g1"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "elementwise_exp_primitive", "category": "primitive", "supports_goals": ["g2"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "two_axis_grid_mapping", "category": "mapping", "supports_goals": ["g3"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_threads", "role": "thread_block", "candidates": [256, 512, 128], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "vector_width", "role": "vector_width", "candidates": [4, 2, 1], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "elementwise_v1",
                "bindings": {"ELEMENTWISE_BLOCK_THREADS": 256, "ELEMENTWISE_VECTOR_WIDTH": 4},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "exp2d.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    if kernel == "group_norm_kernel":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "group_norm_kernel",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"N": 16, "C": 128, "HW": 256, "num_groups": 128, "group_size": 1},
                "artifacts": {"ttgir_path": "group_norm_kernel.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep group tile resident", "scope": "tile", "tensors": ["X"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "reduction_tree_balance", "summary": "warp reduction", "scope": "reduce", "tensors": ["Mean", "Rstd"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "memory_coalescing", "summary": "vector group io", "scope": "load_store", "tensors": ["X", "Y"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "fused_epilogue_avoid_writeback", "summary": "fuse affine epilogue", "scope": "epilogue", "tensors": ["W", "B"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "group_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "warp_reduction", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "online_normalization", "category": "fusion", "supports_goals": ["g1"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "affine_fused_epilogue", "category": "fusion", "supports_goals": ["g3"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m4", "tag": "blocked_layout", "category": "tiling", "supports_goals": ["g0", "g2"], "attrs": {}, "dims": ["block_threads", "vector_width"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_threads", "role": "thread_block", "candidates": [256, 128], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "vector_width", "role": "vector_width", "candidates": [4, 2, 1], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "group_norm_v1",
                "bindings": {"GROUP_NORM_BLOCK_THREADS": 256, "GROUP_NORM_VECTOR_WIDTH": 4},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "group_norm_kernel.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    return {
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
            {"id": "g0", "tag": "operand_reuse", "summary": "reuse ab tiles", "scope": "k_loop", "tensors": ["A", "B"], "evidence_refs": ["e0"]},
            {"id": "g1", "tag": "mma_acceleration", "summary": "use mma", "scope": "mainloop", "tensors": ["A", "B"], "evidence_refs": ["e0"]},
            {"id": "g2", "tag": "fused_epilogue_avoid_writeback", "summary": "keep epilogue fused", "scope": "epilogue", "tensors": ["bias"], "evidence_refs": ["e0"]},
            {"id": "g3", "tag": "latency_hiding", "summary": "prefetch tiles", "scope": "k_loop", "tensors": ["A", "B"], "evidence_refs": ["e0"]},
        ],
        "mechanisms": [
            {"id": "m0", "tag": "ab_tile_stage", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["tile_m", "tile_n", "tile_k"], "evidence_refs": ["e0"]},
            {"id": "m1", "tag": "mma_core", "category": "primitive", "supports_goals": ["g1"], "attrs": {}, "dims": ["tile_m", "tile_n", "tile_k"], "evidence_refs": ["e0"]},
            {"id": "m2", "tag": "epilogue_fused_writeback", "category": "fusion", "supports_goals": ["g2"], "attrs": {}, "dims": ["tile_m", "tile_n"], "evidence_refs": ["e0"]},
            {"id": "m3", "tag": "prefetch_pipeline", "category": "pipeline", "supports_goals": ["g3"], "attrs": {}, "dims": ["pipeline_stages"], "evidence_refs": ["e0"]},
        ],
        "dims": [
            {"name": "tile_m", "role": "m_tile", "candidates": [32, 64], "constraints": [], "evidence_refs": ["e0"]},
            {"name": "tile_n", "role": "n_tile", "candidates": [16, 32], "constraints": [], "evidence_refs": ["e0"]},
            {"name": "tile_k", "role": "k_tile", "candidates": [16, 32], "constraints": [], "evidence_refs": ["e0"]},
            {"name": "pipeline_stages", "role": "pipeline_depth", "candidates": [2], "constraints": [], "evidence_refs": ["e0"]},
        ],
        "source_oracle": {
            "kernel_kind": "matmul_mma_tf32_v1",
            "bindings": {"MMA_BM": 32, "MMA_BN": 32, "MMA_BK": 32},
            "arch": "sm90",
            "compiler_stack": "python",
            "evidence_refs": ["e1"],
        },
        "evidence": [
            {"id": "e0", "kind": "ttgir_line", "path": "matmul.ttgir:1", "summary": "ttgir evidence"},
            {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
        ],
    }


def _write_seed(*, out_dir: Path, kernel: str) -> None:
    save_org_seed = load_org_attr("org.io", "save_org_seed")
    validate_org_doc = load_org_attr("org.schema", "validate_org_doc")
    org = validate_org_doc(_seed_payload(kernel=kernel))
    save_org_seed(
        path=out_dir / f"{kernel}.org_seed.json",
        kernel=str(kernel),
        triton_provider="native",
        backend_target="cuda_5090d",
        org=org,
        raw_json=org.to_json_dict(),
        llm_trace={"provider": "test", "cached": True},
        quality={"diff_ok": True, "static_ok": True, "contract_level": "test"},
    )


def test_force_cache_apply_flash_attention2d_requires_ttgir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    _write_seed(out_dir=tmp_path, kernel="flash_attention2d")
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="flash_attention2d",
        out_dir=tmp_path,
        desc=None,
        intent=_dummy_intent("flash_attention2d"),
        report=report,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("error") == "ttgir_missing"


def test_force_cache_apply_flash_attention2d_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    monkeypatch.delenv("INTENTIR_ORG_SOURCE_ARCH", raising=False)
    _write_seed(out_dir=tmp_path, kernel="flash_attention2d")
    ttgir = tmp_path / "flash.ttgir"
    ptx = tmp_path / "flash.ptx"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @flash_attention2d_kernel(%Q_ptr: !tt.ptr<f32>) {\n    %pid_q = tt.get_program_id x : i32\n    %k_33 = tt.load %k_32, %k_25, %cst_2 : tensor<32x64x!tt.ptr<f32>, #blocked>\n    %m_ij = "tt.reduce"(%scores_46) <{axis = 0 : i32}> ({\n    ^bb0(%lhs: f32, %rhs: f32):\n      %max = arith.maxnumf %lhs, %rhs : f32\n      tt.reduce.return %max : f32\n    }) : (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> f32\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    ptx.write_text("cp.async.cg.shared.global;\nshfl.sync.bfly;\nbar.sync 0;\n", encoding="utf-8")
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="flash_attention2d",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="flash_attention2d", ttgir_path=ttgir, ptx_path=ptx),
        intent=_dummy_intent("flash_attention2d"),
        report=report,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    evidence_source = (report["org"] or {}).get("evidence_source", {})
    assert evidence_source.get("primary") == "ttgir"
    assert (report["org"] or {}).get("compiler_stack") == "python"
    assert (report["org"] or {}).get("compiler_cpp_wave") in {"", "wave2"}
    assert evidence_source.get("ptx_available") is True
    assert str(evidence_source.get("ptx_path") or "").endswith("flash.ptx")
    assert isinstance((report["org"] or {}).get("hardware_model"), dict)
    assert ((report["org"] or {}).get("hardware_model") or {}).get("arch_cluster") == "cuda_tc_mid_smem"
    source_oracle_facts = json.loads(Path(str((report["org"] or {}).get("source_oracle_facts_path"))).read_text(encoding="utf-8"))
    assert source_oracle_facts["available"] is True
    assert source_oracle_facts["oracle"]["arch"] == "sm90"
    assert source_oracle_facts["oracle"]["kernel_kind"] == "attn2d_causal_softmax_v6"
    assert source_oracle_facts["oracle"]["bindings"]["ATTN_SCORE_WARPS"] == 6
    assert Path(str((report["org"] or {}).get("ttgir_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("ptx_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("source_oracle_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("hardware_model_path"))).is_file()
    assert (tmp_path / "flash_attention2d.org_plan.json").is_file()


def test_force_cache_apply_attn_fwd_requires_ttgir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    _write_seed(out_dir=tmp_path, kernel="_attn_fwd")
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="_attn_fwd",
        out_dir=tmp_path,
        desc=None,
        intent=_dummy_intent("_attn_fwd"),
        report=report,
        shape_bindings={"Z": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("error") == "ttgir_missing"


def test_force_cache_apply_attn_fwd_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    monkeypatch.delenv("INTENTIR_ORG_SOURCE_ARCH", raising=False)
    _write_seed(out_dir=tmp_path, kernel="_attn_fwd")
    ttgir = tmp_path / "attn_fwd.ttgir"
    ptx = tmp_path / "attn_fwd.ptx"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @_attn_fwd(%Q: !tt.ptr<f32>, %K: !tt.ptr<f32>, %V: !tt.ptr<f32>, %attn_mask: !tt.ptr<f32>) {\n    %q_0 = tt.load %Qv, %mask_q, %cst : tensor<16x64x!tt.ptr<f32>, #blocked>\n    %acc = scf.for %tile = %c0_i32 to %KV_CTX step %c16_i32 iter_args(%m = %neg_inf) -> (f32) {\n      %k_0 = tt.load %Kv, %mask_k, %cst : tensor<64x16x!tt.ptr<f32>, #blocked>\n      %v_0 = tt.load %Vv, %mask_v, %cst : tensor<16x64x!tt.ptr<f32>, #blocked>\n      %pred_causal = arith.cmpi sle, %kv, %q : i1\n      %m_ij = "tt.reduce"(%scores) <{axis = 1 : i32}> ({\n      ^bb0(%lhs: f32, %rhs: f32):\n        %max = arith.maxnumf %lhs, %rhs : f32\n        tt.reduce.return %max : f32\n      }) : (tensor<16x16xf32, #blocked>) -> tensor<16xf32, #blocked>\n      scf.yield %m\n    }\n    %dot = tt.dot %a, %b : tensor<16x64xf32, #blocked> * tensor<64x16xf32, #blocked>\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    ptx.write_text("cp.async.cg.shared.global;\nshfl.sync.bfly;\nbar.sync 0;\n", encoding="utf-8")
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="_attn_fwd",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="_attn_fwd", ttgir_path=ttgir, ptx_path=ptx),
        intent=_dummy_intent("_attn_fwd"),
        report=report,
        shape_bindings={"Z": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    source_oracle_facts = json.loads(Path(str((report["org"] or {}).get("source_oracle_facts_path"))).read_text(encoding="utf-8"))
    assert source_oracle_facts["available"] is True
    assert source_oracle_facts["oracle"]["kernel_kind"] == "attn_fwd_tiled_v3"
    assert Path(str((report["org"] or {}).get("ttgir_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("ptx_facts_path"))).is_file()
    assert (tmp_path / "_attn_fwd.org_plan.json").is_file()
    assert (tmp_path / "_attn_fwd.org_candidates.txt").is_file()


def test_force_cache_apply_masked_softmax2d_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    _write_seed(out_dir=tmp_path, kernel="masked_softmax2d")
    ttgir = tmp_path / "masked_softmax2d.ttgir"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @masked_softmax2d_kernel(%inp_ptr: !tt.ptr<f32>, %mask_ptr: !tt.ptr<i1>, %out_ptr: !tt.ptr<f32>, %M: i32, %N: i32) {\n    %x_12 = tt.load %x_11, %in_bounds_8, %cst : tensor<256x!tt.ptr<f32>, #blocked>\n    %m_15 = tt.load %m_14, %in_bounds_6, %cst_1 : tensor<256x!tt.ptr<i8>, #blocked>\n    %x_17 = arith.select %m_16, %x_12, %cst_0 : tensor<256xi1, #blocked>, tensor<256xf32, #blocked>\n    %mx = "tt.reduce"(%x_17) <{axis = 0 : i32}> ({\n    ^bb0(%lhs: f32, %rhs: f32):\n      %max = arith.maxnumf %lhs, %rhs : f32\n      tt.reduce.return %max : f32\n    }) : (tensor<256xf32, #blocked>) -> f32\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="masked_softmax2d",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="masked_softmax2d", ttgir_path=ttgir),
        intent=_dummy_intent("masked_softmax2d"),
        report=report,
        shape_bindings={"M": 4, "N": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    assert (tmp_path / "masked_softmax2d.org_plan.json").is_file()
    assert (tmp_path / "masked_softmax2d.org_candidates.txt").is_file()


def test_force_cache_apply_softmax_inner_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    _write_seed(out_dir=tmp_path, kernel="softmax_inner")
    ttgir = tmp_path / "softmax_inner.ttgir"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @softmax_kernel_inner(%output_ptr: !tt.ptr<f32>, %input_ptr: !tt.ptr<f32>, %M: i32, %N: i32) {\n    %inp = tt.load %input_ptrs, %mask_3, %cst : tensor<64x!tt.ptr<f32>, #blocked>\n    %m = "tt.reduce"(%inp) <{axis = 0 : i32}> ({\n    ^bb0(%lhs: f32, %rhs: f32):\n      %max = arith.maxnumf %lhs, %rhs : f32\n      tt.reduce.return %max : f32\n    }) : (tensor<64xf32, #blocked>) -> f32\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="softmax_inner",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="softmax_inner", ttgir_path=ttgir),
        intent=_dummy_intent("softmax_inner"),
        report=report,
        shape_bindings={"M": 4, "N": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    assert (tmp_path / "softmax_inner.org_plan.json").is_file()
    assert (tmp_path / "softmax_inner.org_candidates.txt").is_file()


def test_force_cache_apply_matmul_fused_epilogue_requires_ttgir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    _write_seed(out_dir=tmp_path, kernel="matmul_fused_epilogue2d")
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="matmul_fused_epilogue2d",
        out_dir=tmp_path,
        desc=None,
        intent=_dummy_intent("matmul_fused_epilogue2d"),
        report=report,
        shape_bindings={"M": 32, "N": 32, "K": 32},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("error") == "ttgir_missing"


def test_force_cache_apply_matmul_fused_epilogue_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    monkeypatch.delenv("INTENTIR_ORG_SOURCE_ARCH", raising=False)
    _write_seed(out_dir=tmp_path, kernel="matmul_fused_epilogue2d")
    ttgir = tmp_path / "matmul.ttgir"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @matmul_fused_epilogue2d_kernel(%A: !tt.ptr<f32>) {\n    %pid_m = tt.get_program_id x : i32\n    %dot = tt.dot %a, %b : tensor<16x16xf32, #blocked> * tensor<16x16xf32, #blocked>\n    %layout = ttg.convert_layout %dot : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked>\n    tt.store %out, %layout : tensor<16x16x!tt.ptr<f32>, #blocked>\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="matmul_fused_epilogue2d",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="matmul_fused_epilogue2d", ttgir_path=ttgir),
        intent=_dummy_intent("matmul_fused_epilogue2d"),
        report=report,
        shape_bindings={"M": 32, "N": 32, "K": 32},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    assert (report["org"] or {}).get("compiler_stack") == "python"
    assert (report["org"] or {}).get("compiler_cpp_wave") in {"", "wave2"}
    assert ((report["org"] or {}).get("hardware_model") or {}).get("arch_cluster") == "cuda_tc_mid_smem"
    source_oracle_facts = json.loads(Path(str((report["org"] or {}).get("source_oracle_facts_path"))).read_text(encoding="utf-8"))
    assert source_oracle_facts["available"] is True
    assert source_oracle_facts["oracle"]["arch"] == "sm90"
    assert source_oracle_facts["oracle"]["bindings"]["MMA_ASYNC_COPY"] == 1
    assert Path(str((report["org"] or {}).get("ttgir_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("ptx_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("source_oracle_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("hardware_model_path"))).is_file()
    assert (tmp_path / "matmul_fused_epilogue2d.org_plan.json").is_file()
    assert (tmp_path / "matmul_fused_epilogue2d.org_candidates.txt").read_text(encoding="utf-8").splitlines()[3] == "matmul_tile_v2"


def test_force_cache_apply_flash_attention2d_records_compile_checks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "2")
    _write_seed(out_dir=tmp_path, kernel="flash_attention2d")
    ttgir = tmp_path / "flash.ttgir"
    ptx = tmp_path / "flash.ptx"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @flash_attention2d_kernel(%Q_ptr: !tt.ptr<f32>) {\n    %q_18 = tt.load %Q_ptr, %q_mask, %cst : tensor<64x!tt.ptr<f32>, #blocked>\n    %k_33 = tt.load %K_ptr, %k_25, %cst_2 : tensor<32x64x!tt.ptr<f32>, #blocked>\n    %v_43 = tt.load %V_ptr, %k_25, %cst_2 : tensor<32x64x!tt.ptr<f32>, #blocked>\n    %m_ij = "tt.reduce"(%scores_46) <{axis = 0 : i32}> ({\n    ^bb0(%lhs: f32, %rhs: f32):\n      %max = arith.maxnumf %lhs, %rhs : f32\n      tt.reduce.return %max : f32\n    }) : (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> f32\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    ptx.write_text("cp.async.cg.shared.global;\nshfl.sync.bfly;\nbar.sync 0;\n", encoding="utf-8")
    monkeypatch.setattr(
        "pipeline.triton.org_bridge._run_compile_check_candidates",
        lambda **_: [
            {
                "candidate": "attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6",
                "kernel_kind": "attn2d_causal_softmax_v6",
                "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6},
                "report_path": "/tmp/fake/report.json",
                "contract_path": "/tmp/fake/contract.json",
                "ptx_path": "/tmp/fake/kernel.ptx",
                "entry": "flash_attention2d",
                "requested_sm": "sm_120",
                "effective_sm": "sm_120",
                "downleveled": False,
                "ok": True,
                "error": "",
            }
        ],
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}, "mlir": {"toolchain": {"tools": {}}, "downstream_cuda_std_llvm_contract_exec_meta": {"cuda_requested_sm": "sm_120", "cuda_effective_sm": "sm_120", "cuda_target_downleveled": False}}}
    _run_org_plugin(
        spec_name="flash_attention2d",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="flash_attention2d", ttgir_path=ttgir, ptx_path=ptx),
        intent=_dummy_intent("flash_attention2d"),
        report=report,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    plan = json.loads((tmp_path / "flash_attention2d.org_plan.json").read_text(encoding="utf-8"))
    assert len(plan["compile_checks"]) == 1
    assert plan["realizations"][0]["effective_sm"] == "sm_120"
    assert ((report["org"] or {}).get("compile_checks_count")) == 1


def test_compile_check_candidates_unknown_kernel_use_inline_backend_compile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "1")
    seen: list[str] = []

    def _fake_inline_compile_check(**kwargs):
        seen.append(str(kwargs["spec_name"]))
        return (
            True,
            {
                "mlir": {
                    "downstream_cuda_contract_path": str(tmp_path / "fake.contract.json"),
                    "downstream_cuda_contract_exec_meta": {
                        "cuda_ptx_path": str(tmp_path / "fake.kernel.ptx"),
                        "cuda_ptx_entries": ["liger_swiglu"],
                        "cuda_requested_sm": "sm_120",
                        "cuda_effective_sm": "sm_120",
                        "cuda_target_downleveled": False,
                    },
                }
            },
            "",
        )

    monkeypatch.setattr("pipeline.triton.org_bridge._run_inline_compile_check", _fake_inline_compile_check)
    checks = org_bridge._run_compile_check_candidates(
        spec_name="liger_swiglu",
        out_dir=tmp_path,
        backend_target="cuda_5090d",
        target_arch="sm120",
        candidates=[SimpleNamespace(kernel_kind="elementwise_v1", bindings={"ELEMENTWISE_BLOCK_THREADS": 128, "ELEMENTWISE_VECTOR_WIDTH": 4})],
        intent=_dummy_intent("liger_swiglu"),
        shape_bindings={"M": 128, "N": 1024},
        toolchain_model={"requires_real_mlir": False},
    )
    assert seen == ["liger_swiglu"]
    assert len(checks) == 1
    assert checks[0]["ok"] is True
    assert checks[0]["ptx_path"].endswith("fake.kernel.ptx")


def test_compile_check_candidates_unknown_kernel_real_mlir_allows_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "1")
    seen_env: list[dict[str, str]] = []

    def _fake_inline_compile_check(**kwargs):
        seen_env.append(dict(kwargs.get("env_updates") or {}))
        return (
            True,
            {
                "mlir": {
                    "downstream_cuda_std_llvm_contract_path": str(tmp_path / "fake.contract.json"),
                    "downstream_cuda_std_llvm_contract_exec_meta": {
                        "cuda_ptx_path": str(tmp_path / "fake.kernel.ptx"),
                        "cuda_ptx_entries": ["liger_rms_norm"],
                        "cuda_requested_sm": "sm_120",
                        "cuda_effective_sm": "sm_120",
                        "cuda_target_downleveled": False,
                    },
                }
            },
            "",
        )

    monkeypatch.setattr("pipeline.triton.org_bridge._run_inline_compile_check", _fake_inline_compile_check)
    checks = org_bridge._run_compile_check_candidates(
        spec_name="liger_rms_norm",
        out_dir=tmp_path,
        backend_target="cuda_5090d",
        target_arch="sm120",
        candidates=[SimpleNamespace(kernel_kind="rms_norm_axis1_v2", bindings={})],
        intent=_dummy_intent("liger_rms_norm"),
        shape_bindings={"M": 128, "N": 1024},
        toolchain_model={"requires_real_mlir": True, "cuda_real_mlir_wave": "wave25"},
    )
    assert len(seen_env) == 1
    assert seen_env[0]["INTENTIR_REAL_MLIR"] == "1"
    assert seen_env[0]["INTENTIR_CUDA_REAL_MLIR_ALLOW_UNKNOWN"] == "1"
    assert seen_env[0]["INTENTIR_CUDA_REAL_MLIR_WAVE"] == "wave25"
    assert len(checks) == 1
    assert checks[0]["ok"] is True
    assert checks[0]["ptx_path"].endswith("fake.kernel.ptx")


def test_force_cache_apply_row_sum_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    _write_seed(out_dir=tmp_path, kernel="row_sum")
    ttgir = tmp_path / "row_sum.ttgir"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @row_sum_kernel(%inp_ptr: !tt.ptr<f32>) {\n    %x_12 = tt.load %inp_ptr, %in_bounds, %cst : tensor<256x!tt.ptr<f32>, #blocked>\n    %sum = "tt.reduce"(%x_12) <{axis = 0 : i32}> ({\n    ^bb0(%lhs: f32, %rhs: f32):\n      %acc = arith.addf %lhs, %rhs : f32\n      tt.reduce.return %acc : f32\n    }) : (tensor<256xf32, #blocked>) -> f32\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="row_sum",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="row_sum", ttgir_path=ttgir),
        intent=_dummy_intent("row_sum"),
        report=report,
        shape_bindings={"M": 4, "N": 256},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    assert (tmp_path / "row_sum.org_plan.json").is_file()
    assert (tmp_path / "row_sum.org_candidates.txt").read_text(encoding="utf-8").splitlines()[3] == "row_sum_axis1_v2:ROW_REDUCE_BLOCK_THREADS=128,ROW_REDUCE_SHARED_STAGE=1,ROW_REDUCE_VECTOR_WIDTH=2"


def test_force_cache_apply_row_max_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    _write_seed(out_dir=tmp_path, kernel="row_max")
    ttgir = tmp_path / "row_max.ttgir"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @row_max_kernel(%inp_ptr: !tt.ptr<f32>) {\n    %x_12 = tt.load %inp_ptr, %in_bounds, %cst : tensor<256x!tt.ptr<f32>, #blocked>\n    %mx = "tt.reduce"(%x_12) <{axis = 0 : i32}> ({\n    ^bb0(%lhs: f32, %rhs: f32):\n      %acc = arith.maxnumf %lhs, %rhs : f32\n      tt.reduce.return %acc : f32\n    }) : (tensor<256xf32, #blocked>) -> f32\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="row_max",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="row_max", ttgir_path=ttgir),
        intent=_dummy_intent("row_max"),
        report=report,
        shape_bindings={"M": 4, "N": 256},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    assert (tmp_path / "row_max.org_plan.json").is_file()
    assert (tmp_path / "row_max.org_candidates.txt").read_text(encoding="utf-8").splitlines()[3] == "row_max_axis1_v2:ROW_REDUCE_BLOCK_THREADS=128,ROW_REDUCE_SHARED_STAGE=1,ROW_REDUCE_VECTOR_WIDTH=2"


def test_force_cache_apply_layer_norm_persistent_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    _write_seed(out_dir=tmp_path, kernel="layer_norm_persistent")
    ttgir = tmp_path / "layer_norm_persistent.ttgir"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @layer_norm_persistent_kernel(%inp_ptr: !tt.ptr<f32>, %weight_ptr: !tt.ptr<f32>, %bias_ptr: !tt.ptr<f32>) {\n    %x_12 = tt.load %inp_ptr, %in_bounds, %cst : tensor<64x!tt.ptr<f32>, #blocked>\n    %mx = "tt.reduce"(%x_12) <{axis = 0 : i32}> ({\n    ^bb0(%lhs: f32, %rhs: f32):\n      %acc = arith.addf %lhs, %rhs : f32\n      tt.reduce.return %acc : f32\n    }) : (tensor<64xf32, #blocked>) -> f32\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="layer_norm_persistent",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="layer_norm_persistent", ttgir_path=ttgir),
        intent=_dummy_intent("layer_norm_persistent"),
        report=report,
        shape_bindings={"M": 4, "N": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    assert (tmp_path / "layer_norm_persistent.org_plan.json").is_file()
    assert (tmp_path / "layer_norm_persistent.org_candidates.txt").read_text(encoding="utf-8").splitlines()[3] == "layer_norm_axis1_v1:LAYER_NORM_BLOCK_THREADS=32,LAYER_NORM_PERSISTENT_ROW=1,LAYER_NORM_VECTOR_WIDTH=2"


def test_force_cache_apply_add2d_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    _write_seed(out_dir=tmp_path, kernel="add2d")
    ttgir = tmp_path / "add2d.ttgir"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>\nmodule attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @add2d_kernel(%A_ptr: !tt.ptr<f32>, %B_ptr: !tt.ptr<f32>, %C_ptr: !tt.ptr<f32>) {\n    %a = tt.load %A_ptrs, %mask, %cst : tensor<1024x!tt.ptr<f32>, #blocked>\n    %b = tt.load %B_ptrs, %mask, %cst : tensor<1024x!tt.ptr<f32>, #blocked>\n    %c = arith.addf %a, %b : tensor<1024xf32, #blocked>\n    tt.store %C_ptrs, %c, %mask : tensor<1024x!tt.ptr<f32>, #blocked>\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="add2d",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="add2d", ttgir_path=ttgir),
        intent=_dummy_intent("add2d"),
        report=report,
        shape_bindings={"M": 4, "N": 256},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    assert (tmp_path / "add2d.org_plan.json").is_file()
    assert (tmp_path / "add2d.org_candidates.txt").read_text(encoding="utf-8").splitlines()[3] == "elementwise_v1:ELEMENTWISE_BLOCK_THREADS=256,ELEMENTWISE_VECTOR_WIDTH=4"


def test_force_cache_apply_exp2d_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    _write_seed(out_dir=tmp_path, kernel="exp2d")
    ttgir = tmp_path / "exp2d.ttgir"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>\nmodule attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @exp2d_kernel(%A_ptr: !tt.ptr<f32>, %C_ptr: !tt.ptr<f32>) {\n    %a = tt.load %A_ptrs, %mask, %cst : tensor<1024x!tt.ptr<f32>, #blocked>\n    %c = math.exp %a : tensor<1024xf32, #blocked>\n    tt.store %C_ptrs, %c, %mask : tensor<1024x!tt.ptr<f32>, #blocked>\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="exp2d",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="exp2d", ttgir_path=ttgir),
        intent=_dummy_intent("exp2d"),
        report=report,
        shape_bindings={"M": 4, "N": 256},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    assert (tmp_path / "exp2d.org_plan.json").is_file()
    assert (tmp_path / "exp2d.org_candidates.txt").read_text(encoding="utf-8").splitlines()[3] == "elementwise_v1:ELEMENTWISE_BLOCK_THREADS=256,ELEMENTWISE_VECTOR_WIDTH=4"


def test_force_cache_apply_group_norm_kernel_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_ORG_COMPILE_TOPK", "0")
    _write_seed(out_dir=tmp_path, kernel="group_norm_kernel")
    ttgir = tmp_path / "group_norm_kernel.ttgir"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>\nmodule attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @group_norm_kernel(%X: !tt.ptr<f32>, %Y: !tt.ptr<f32>) {\n    %x = tt.load %Xptrs, %mask, %cst : tensor<1024x!tt.ptr<f32>, #blocked>\n    %sum = "tt.reduce"(%x) <{axis = 0 : i32}> ({\n    ^bb0(%lhs: f32, %rhs: f32):\n      %acc = arith.addf %lhs, %rhs : f32\n      tt.reduce.return %acc : f32\n    }) : (tensor<1024xf32, #blocked>) -> f32\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="group_norm_kernel",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="group_norm_kernel", ttgir_path=ttgir),
        intent=_dummy_intent("group_norm_kernel"),
        report=report,
        shape_bindings={"N": 16, "C": 128, "HW": 256, "num_groups": 128, "group_size": 1},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    assert (tmp_path / "group_norm_kernel.org_plan.json").is_file()
    assert (tmp_path / "group_norm_kernel.org_candidates.txt").read_text(encoding="utf-8").splitlines()[3] == "group_norm_v1:GROUP_NORM_BLOCK_THREADS=256,GROUP_NORM_VECTOR_WIDTH=4"
