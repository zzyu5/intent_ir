"""
Triton frontend: ORG prompt builder for rationale-first extraction.

The runtime injects `source_context` and `source_oracle`; the LLM is responsible
for the reasoning-bearing sections only:
  - goals
  - mechanisms
  - dims
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
  - evidence
  - notes (optional list[string])
- Runtime will inject `source_context` and `source_oracle`; do NOT invent or emit hardware mapping decisions.

Goal objects:
- id: string
- tag: one of:
  resident_working_set, streaming_softmax_state, avoid_materialization, latency_hiding,
  operand_reuse, mma_acceleration, fused_epilogue_avoid_writeback
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

Evidence objects:
- id: string
- kind: string
- path: string
- summary: short summary
- text (optional): short excerpt or normalized witness

Important:
- Separate WHY from HOW from DIMS. Do NOT collapse them into one object.
- Do NOT output backend variant names or target parameter assignments.
- Do NOT output numeric tuning decisions copied from source oracle. You may output candidate sets for dimensions.
- Every goal and every mechanism must have at least one evidence ref.
- Prefer evidence from TTGIR/PTX/source_oracle facts over TTIR summaries when available.
- Prefer dims and attrs that directly affect target performance recovery:
  resident_bytes, reuse_window, pipeline_depth, communication_scope, layout_convert_sites.

Kernel-specific expectations:
- For flash_attention2d, capture:
  resident_working_set, streaming_softmax_state, avoid_materialization, latency_hiding
  and prefer mechanism tags such as:
  q_resident_state, kv_streamed_tiles, online_softmax_reduce, output_layout_convert
  and prefer dims/attrs such as:
  resident_bytes, pipeline_depth, communication_scope
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
- For matmul_fused_epilogue2d, capture:
  operand_reuse, mma_acceleration, fused_epilogue_avoid_writeback, latency_hiding
  and prefer mechanism tags such as:
  operand_tile_stage, dot_op, mma_core, bias_fused_epilogue, output_layout_convert
  and prefer dims/attrs such as:
  resident_bytes, pipeline_depth, communication_scope
"""


SYSTEM_PROMPT_COMPACT = """Return ONE strict ORG JSON object.

Required keys: schema_version, kernel, goals, mechanisms, dims, evidence (notes optional list[string]).
Runtime injects source_context/source_oracle; do not output hardware mapping or target numeric assignments.
Each goal/mechanism/dim must be evidence-backed.
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
