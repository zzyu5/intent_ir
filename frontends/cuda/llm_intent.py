"""
CUDA frontend: LLM prompt builder.

This prompt is CUDA-specific (blockIdx/threadIdx conventions). IntentIR core and
verification remain frontend-agnostic.
"""

from __future__ import annotations

from typing import Dict, List, Optional


SYSTEM_PROMPT = """You are an expert compiler engineer. Given a CUDA __global__
kernel body, produce Intent-IR v1.1 candidate JSON. Hard rules:
- Return ONE JSON object only (no prose, no code fences).
- Emit keys: name, kernel_type, tensors, ops, outputs, parallel_axes, schedule(optional), axis_roles(optional), meta(optional).
- tensors must be an object {name: {...}}, NOT a list.
- Use ONLY the allowed op set:
  - numeric: add/sub/mul/div/max/min/exp/relu/rsqrt/abs/floor
  - compare/bool: ne/lt/le/gt/ge/and/or/not/where
  - shape: reshape/broadcast_in_dim/transpose/layout_cast
  - reduce: reduce_sum/reduce_max/reduce_any/softmax
  - misc: matmul/conv2d/cast/iota/gather/identity/const
  - macro ops allowed: upsample_bicubic2d_aa (only when truly present semantically)
- Every output MUST be declared in tensors and MUST be produced by an op (not only declared).
- Do NOT invent new shape symbols. Only use shape symbols that appear in the evidence appendix
  (io_spec / launch canonical_shapes / scalar params like M,N,K).
- Keep the original input view: do NOT redefine input tensor shapes to "grouped/view" shapes.
  If you need a grouped/view shape, insert explicit reshape ops inside IntentIR.
- CUDA indexing note: blockIdx.* and threadIdx.* are scheduling details; do NOT encode them into tensor shapes.
  Express full logical tensor shapes (e.g., [M,N]) and semantics ops; schedule is optional and should be a sketch only.
"""


def build_messages(
    cuda_src: str,
    *,
    kernel_name: Optional[str] = None,
    extra_instruction: Optional[str] = None,
) -> List[Dict[str, str]]:
    user_lines: List[str] = []
    if kernel_name:
        user_lines.append(f"Kernel name: {kernel_name}")
    user_lines.append("CUDA kernel:")
    user_lines.append(str(cuda_src))
    if extra_instruction:
        user_lines.append("\nExtra instructions:")
        user_lines.append(str(extra_instruction))
    content = "\n".join(user_lines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


__all__ = ["SYSTEM_PROMPT", "build_messages"]

