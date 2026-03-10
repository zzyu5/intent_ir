"""
Triton frontend: ORG prompt builder for rationale-first extraction.

The runtime injects `source_context` and `source_oracle`; the LLM is responsible
for the reasoning-bearing sections only:
  - goals
  - mechanisms
  - dims
  - tensors
  - tensor_lifetimes
  - dataflow_edges
  - mechanism_topology
  - evidence
  - notes (optional list[string])
"""

from __future__ import annotations

from typing import Dict, List, Optional


SYSTEM_PROMPT = """You are an expert kernel performance engineer.
Given a Triton @triton.jit kernel source and an Evidence appendix (JSON),
produce ONE strict ORG JSON object.

Hard rules:
- Output must be STRICT JSON object (no prose, no code fences).
- schema_version MUST be "intentir_org_v1".
- Top-level keys you must output:
  - schema_version
  - kernel
  - goals
  - mechanisms
  - dims
  - tensors
  - tensor_lifetimes
  - dataflow_edges
  - mechanism_topology
  - evidence
  - notes (optional list[string])
- Runtime will inject `source_context` and `source_oracle`; do NOT invent or emit hardware mapping decisions.

Goal objects:
- id: string
- tag: one of:
  resident_working_set, streaming_softmax_state, avoid_materialization, latency_hiding,
  operand_reuse, mma_acceleration, fused_epilogue_avoid_writeback,
  reduction_tree_balance, memory_coalescing, persistent_row_state, affine_epilogue_fusion,
  mask_causal_pruning
- summary: short explanation of the performance objective
- scope: short scope string (for example: kv_loop, q_state, epilogue, reduction)
- tensors: list[string]
- evidence_refs: list[string]

Mechanism objects:
- id: string
- tag: free-form but precise mechanism name
- category: one of: tiling, staging, pipeline, mapping, communication, primitive, fusion
- supports_goals: list[goal.id]
- attrs: object
- dims: list[dim.name]
- evidence_refs: list[string]

Dim objects:
- name: string
- role: string
- candidates or range: choose one
- constraints: list[string]
- evidence_refs: list[string]

Tensor objects:
- id: string
- name: string
- role: short semantic role such as input_row, row_stats, kv_tile, output_accumulator
- dtype (optional): string
- aliases (optional): list[string]
- shape_refs (optional): list[string]
- evidence_refs: list[string]

TensorLifetime objects:
- id: string
- tensor: tensor.id
- region: short phase string such as kv_loop, row_reduce, affine_epilogue
- storage: one of global, shared, register, local
- start: short phase/event string
- end: short phase/event string
- producer_mechanisms: list[mechanism.id]
- consumer_mechanisms: list[mechanism.id]
- supports_goals: list[goal.id]
- dims: list[dim.name]
- bytes_hint (optional): int
- reuse_window (optional): prefer a structured scope label such as cta_tile, row_tile, row_reduce, row_epilogue, full_row
- evidence_refs: list[string]

DataflowEdge objects:
- id: string
- src: tensor_lifetime.id
- dst: tensor_lifetime.id
- tensor: tensor.id
- kind: short edge label such as stage, reduce, normalize, epilogue, store
- order: integer topological order
- mechanisms: list[mechanism.id]
- evidence_refs: list[string]

MechanismTopology objects:
- id: string
- src: mechanism.id
- dst: mechanism.id
- relation: short label such as feeds, gates, vectorizes, materializes, reduces_for
- tensors: list[tensor.id]
- lifetimes: list[tensor_lifetime.id]
- evidence_refs: list[string]

Evidence objects:
- id: string
- kind: string
- path: string
- summary: short summary
- text (optional): short excerpt or normalized witness

Important:
- Separate WHY from HOW from DIMS. Do NOT collapse them into one object.
- ORG is a topology, not a bag of tags. Always emit the tensor/lifetime/dataflow/mechanism graph for the kernel.
- Do NOT output backend variant names or target parameter assignments.
- Do NOT output numeric tuning decisions copied from source oracle. You may output candidate sets for dimensions.
- Every goal and every mechanism must have at least one evidence ref.
- Every tensor_lifetime must reference real mechanism ids and goal ids; dataflow_edges must form a valid topological flow.
- Prefer evidence from TTGIR/PTX/source_oracle facts over TTIR summaries when available.
- Prefer dims and attrs that directly affect target performance recovery:
  resident_bytes, reuse_window, pipeline_depth, communication_scope, layout_convert_sites.
- Force yourself to recover the main tensor residency intervals:
  which tensor stays in register/shared/global, for how long, and which mechanism consumes it next.

Kernel-specific expectations:
- For flash_attention2d, capture:
  resident_working_set, streaming_softmax_state, avoid_materialization, latency_hiding
  and prefer mechanism tags such as:
  q_resident_state, kv_streamed_tiles, online_softmax_reduce, output_layout_convert
  and prefer dims/attrs such as:
  resident_bytes, pipeline_depth, communication_scope
  and explicitly model tensors/lifetimes such as:
  Q resident at CTA/kv_loop scope -> streamed K tile + streamed V tile -> max_state/sum_state -> output_accumulator/store
  with K/V tile lifetimes marked as streamed inputs, Q lifetime spanning the KV loop, and max/sum state lifetimes carried across online softmax updates
- For _attn_fwd, capture:
  resident_working_set, streaming_softmax_state, avoid_materialization, latency_hiding
  and prefer mechanism tags such as:
  qkv_stage, online_softmax_reduce, mask_causal_apply, prefetch_pipeline, output_accumulator
  and prefer dims/attrs such as:
  block_m, block_kv, pipeline_depth, communication_scope
- For masked_softmax2d, capture:
  resident_working_set, streaming_softmax_state, avoid_materialization, latency_hiding
  and prefer mechanism tags such as:
  mask_apply, row_reduction, vector_row_path, row_tile_resident
  and prefer dims/attrs such as:
  row_width, block_threads, communication_scope
- For softmax_inner, capture:
  resident_working_set, streaming_softmax_state, avoid_materialization, latency_hiding
  and prefer mechanism tags such as:
  row_reduction, vector_row_path, row_tile_resident
  and prefer dims/attrs such as:
  row_width, block_threads, communication_scope
- For row_sum, capture:
  resident_working_set, reduction_tree_balance, memory_coalescing, latency_hiding
  and prefer mechanism tags such as:
  row_tile_resident, vector_row_path, row_reduction, warp_parallel_rows, shared_staging
  and prefer dims/attrs such as:
  row_width, block_threads, vector_width, communication_scope
- For row_max, capture:
  resident_working_set, reduction_tree_balance, memory_coalescing, latency_hiding
  and prefer mechanism tags such as:
  row_tile_resident, tile_load_stage, warp_reduction_tree, row_parallel_axis, block_synchronization
  and prefer dims/attrs such as:
  row_width, block_threads, vector_width, communication_scope
- For layer_norm_persistent, capture:
  resident_working_set, persistent_row_state, memory_coalescing, affine_epilogue_fusion, latency_hiding
  and prefer mechanism tags such as:
  row_tile_resident, warp_reduction, register_staging, persistent_row_cache, affine_epilogue
  and prefer dims/attrs such as:
  row_width, block_threads, vector_width, resident_bytes, communication_scope
  and explicitly model tensors/lifetimes such as:
  input_row -> row_resident_tile -> row_stats -> affine_out
  with row_stats or row_resident_tile spanning from reduction into affine epilogue when persistence is present
- For add2d, capture:
  resident_working_set, memory_coalescing, avoid_materialization, latency_hiding
  and prefer mechanism tags such as:
  blocked_register_layout, vector_global_io, two_axis_grid_mapping, elementwise_add_primitive, masked_edge_handling
  and prefer dims/attrs such as:
  block_threads, vector_width, tile_width_n, communication_scope
- For exp2d, capture:
  resident_working_set, memory_coalescing, avoid_materialization, latency_hiding
  and prefer mechanism tags such as:
  blocked_register_layout, vector_global_io, two_axis_grid_mapping, elementwise_exp_primitive, masked_edge_handling
  and prefer dims/attrs such as:
  block_threads, vector_width, tile_width_n, communication_scope
- For group_norm_kernel, capture:
  resident_working_set, reduction_tree_balance, memory_coalescing, fused_epilogue_avoid_writeback, latency_hiding
  and prefer mechanism tags such as:
  group_tile_resident, warp_reduction, online_normalization, affine_fused_epilogue, vector_group_io
  and prefer dims/attrs such as:
  block_threads, vector_width, group_size, communication_scope
- For masked_attention2d, capture:
  resident_working_set, streaming_softmax_state, avoid_materialization, mask_causal_pruning, latency_hiding
  and prefer mechanism tags such as:
  q_resident_state, tiny_kv_stage, mask_causal_apply, parallel_softmax, vector_dot_fragment
  and prefer dims/attrs such as:
  block_m, block_kv, score_warps, communication_scope
- For ai_bench_softmax, capture:
  resident_working_set, streaming_softmax_state, memory_coalescing, latency_hiding
  and prefer mechanism tags such as:
  row_tile_resident, row_reduction, vector_row_path, power2_padding
  and prefer dims/attrs such as:
  row_width, block_threads, vector_width, communication_scope
- For ai_bench_matmul, capture:
  operand_reuse, mma_acceleration, latency_hiding
  and prefer mechanism tags such as:
  operand_tile_stage, mma_core, async_prefetch, tile_fallback
  and prefer dims/attrs such as:
  tile_m, tile_n, tile_k, pipeline_depth, communication_scope
- For matmul_fused_epilogue2d, capture:
  operand_reuse, mma_acceleration, fused_epilogue_avoid_writeback, latency_hiding
  and prefer mechanism tags such as:
  operand_tile_stage, dot_op, mma_core, bias_fused_epilogue, output_layout_convert
  and prefer dims/attrs such as:
  resident_bytes, pipeline_depth, communication_scope
"""


SYSTEM_PROMPT_COMPACT = """Return ONE strict ORG JSON object.

Required keys: schema_version, kernel, goals, mechanisms, dims, tensors, tensor_lifetimes, dataflow_edges, mechanism_topology, evidence (notes optional list[string]).
Runtime injects source_context/source_oracle; do not output hardware mapping or target numeric assignments.
Each goal/mechanism/dim must be evidence-backed.
Emit a real optimization topology: tensors, their residency intervals, dataflow edges, and mechanism dependencies.
"""


def build_messages(
    triton_src: str,
    *,
    kernel_name: Optional[str] = None,
    extra_instruction: Optional[str] = None,
    compact: bool = False,
) -> List[Dict[str, str]]:
    user_lines: list[str] = []
    if kernel_name:
        user_lines.append(f"Kernel name: {kernel_name}")
    user_lines.append("Triton kernel:")
    user_lines.append(str(triton_src))
    if extra_instruction:
        user_lines.append("")
        user_lines.append("Evidence appendix and instructions:")
        user_lines.append(str(extra_instruction))
    return [
        {"role": "system", "content": (SYSTEM_PROMPT_COMPACT if compact else SYSTEM_PROMPT)},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


__all__ = ["SYSTEM_PROMPT", "build_messages"]
