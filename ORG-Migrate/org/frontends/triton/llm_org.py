"""
Triton frontend: ORG (Optimization Rationale Graph) prompt builder.

This is intentionally separate from `llm_intent.py`:
- `llm_intent.py` extracts semantic intent (IntentIR JSON).
- `llm_org.py` extracts optimization rationale (why/how/dims + evidence).
"""

from __future__ import annotations

from typing import Dict, List, Optional


SYSTEM_PROMPT = """You are an expert kernel performance engineer.
Given a Triton @triton.jit kernel source and an Evidence appendix (JSON),
produce ONE Optimization Rationale Graph (ORG) JSON object.

Hard rules:
- Output must be STRICT JSON object (no prose, no code fences).
- schema_version MUST be "intentir_org_v1".
- Required top-level keys:
  - schema_version (string)
  - kernel (string)
  - nodes (list)
  - edges (list; can be empty)
  - meta (optional object)
- Each node MUST be an object with keys:
  - id (string)
  - node_type (one of: tiling, staging, overlap_pipeline, parallel_mapping, communication, special_primitive)
  - why (list[string])  # optimization goals/intent (hardware-agnostic)
  - how (list[string])  # abstract mechanisms/structure (still hardware-agnostic)
  - dims (list[ string | {name:string, allowed?:list, note?:string} ])
  - constraints (list[string])  # symbolic constraints (no numeric tuning results)
  - evidence (list[ {kind:string, path:string, detail?:string} ])  # cite evidence appendix paths
  - attrs (optional object)

IMPORTANT:
- Do NOT output any hardware mapping decisions. Do NOT choose backend variants.
- Do NOT output numeric parameter values (e.g., "ATTN_BLOCK_KV=32"). Instead output dims names and constraints like:
  "ATTN_BLOCK_KV in {16,32,64}" and "threads <= 1024".
- When possible, express discrete candidate sets via `dims` objects with `allowed`, e.g.:
  {"name":"ATTN_BLOCK_KV","allowed":[16,32,64]}.
- Evidence.path MUST reference fields inside the provided Evidence appendix JSON (e.g., "frontend_facts.has_async")
  or the provided intent_summary section, not raw TTIR lines.
- Prefer stable, transferable tags in why/how, e.g.:
  why: resident_working_set, iterate_in_scratchpad, streaming_softmax_state, avoid_materialization,
       hide_memory_latency, avoid_recompute
  how: scratchpad_staging, online_softmax, double_buffering, pipeline_overlap,
       warp_reduce, block_reduce, score_cache

If the kernel is attention-like, make sure ORG captures:
- "streaming softmax state" and "avoid attention matrix materialization"
- "working set residency" (what should live in scratchpad/near storage, at what loop window)
- pipeline/overlap intent if present in evidence

Edges:
- Edge src/dst MUST be node.id strings (e.g., "n0"). Do NOT embed node objects.
- If you are not confident about dependencies, set edges to an empty list.

Minimal example shape (you must still fill real content):
{
  "schema_version":"intentir_org_v1",
  "kernel":"flash_attention2d",
  "nodes":[{"id":"n0","node_type":"tiling","why":[],"how":[],"dims":[],"constraints":[],"evidence":[]}],
  "edges":[]
}
"""


SYSTEM_PROMPT_COMPACT = """You are an expert kernel performance engineer.
Return ONE strict ORG JSON object (intentir_org_v1). No prose, no code fences.

Top-level keys: schema_version,kernel,nodes,edges (meta optional).
Each node: id,node_type,why,how,dims,constraints,evidence (attrs optional).

Do NOT map to hardware/backends. Do NOT output numeric parameter values.
Use evidence.path strings that reference Evidence appendix JSON fields.

Edge rule: edges can be empty; if present, edge.src/edge.dst MUST be node.id strings.
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
        user_lines.append("\nExtra instructions:")
        user_lines.append(str(extra_instruction))
    content = "\n".join(user_lines)
    return [
        {"role": "system", "content": (SYSTEM_PROMPT_COMPACT if compact else SYSTEM_PROMPT)},
        {"role": "user", "content": content},
    ]


__all__ = ["build_messages", "SYSTEM_PROMPT"]
