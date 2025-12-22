"""
LLM-driven intent extraction helpers (lives under intent_ir namespace).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .llm_client import DEFAULT_MODEL, LLMClientError, LLMResponse, chat_completion


SYSTEM_PROMPT = """You are an expert compiler engineer. Given the Triton
@triton.jit kernel body, produce Intent-IR v1.1 candidate JSON. Hard rules:
- Emit JSON object with: name, kernel_type, tensors, ops, outputs,
  parallel_axes, schedule(optional), axis_roles(optional), meta(optional).
- tensors must be an object {name: {...}}, NOT a list. Each tensor has dtype in
  {f16,bf16,f32,i32,...}, shape list of symbols/ints, layout row_major unless explicit.
  IMPORTANT: keep argument tensor shapes as declared in the kernel signature; if you need
  a grouped/view shape (e.g., [num_groups, group_size, HW]) insert explicit reshape ops
  instead of redefining the original tensor shape.
- Use the standard op set: matmul, reduce_sum, reduce_max, reduce_any, softmax,
  add/sub/mul/div/max/min/exp/relu, broadcast_in_dim, transpose, reshape,
  layout_cast, conv2d, rsqrt, identity, const, custom_call.
  Do NOT invent new names or intent.elemwise.
- reshape MUST include attrs.shape (non-empty list). broadcast_in_dim MUST have out_shape + broadcast_dims. transpose MUST have perm.
- Every output tensor MUST be declared in tensors and produced by an op. If the kernel writes Mean/Rstd or similar, add the corresponding reduce/assign ops.
- reduce_any MUST include dims/axis. Softmax must be present for attention kernels.
- Reduce dims/axis must be integer axis indices (after any reshape/transpose), not symbolic axis names.
- Scalars/constants (eps, num_elements, group_size, sm_scale, BLOCK_*) must be explicit: const ops or attrs; NEVER leave them implicit.
- op.inputs reference tensors or prior op outputs only; scalars/constants go in attrs if needed.
- For groupnorm kernels: Mean/Rstd must be shaped as [N, num_groups] (optionally [N,num_groups,1] if keepdims=true).
- For layernorm kernels: Mean/Rstd must be shaped as [M] (optionally [M,1] if keepdims=true).
  Use explicit reshape/broadcast ops for any view changes; do NOT fake shapes by redefining inputs.
- axis_roles: {axis: role} with role in {spatial,reduction,batch,channel}; do NOT invert.
- parallel_axes: list of axis strings, and every axis must appear in some tensor shape; do not invent axes.
- schedule may include tile_m/tile_n/tile_k/vec_width/axis_bindings/vec_axis/parallel_axes; if unknown, omit rather than guess.
- For optional tensors, either mark optional:true or omit consistently.
- For complex kernels that cannot be expressed in the primitive op set, use `custom_call`
  with attrs.callee and explicit tensor/scalar inputs. Example:
    - op: "custom_call", attrs: {"callee": "upsample_bicubic2d_aa"}
    - inputs: [Input, reciprocal_scale_h, reciprocal_scale_w]
Return a single JSON object with no prose and no code fences."""


def _strip_code_fence(text: str) -> str:
    fence = re.compile(r"^```(?:json)?|```$", re.MULTILINE)
    return fence.sub("", text).strip()


def _parse_json_block(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # fallback: grab substring between first { and last } and try stripping trailing commas
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                # remove trailing commas before } or ]
                cleaned2 = re.sub(r",\\s*([}\\]])", r"\\1", snippet)
                return json.loads(cleaned2)
        raise


def build_messages(
    triton_src: str,
    *,
    kernel_name: Optional[str] = None,
    extra_instruction: Optional[str] = None,
) -> List[Dict[str, str]]:
    user_lines = []
    if kernel_name:
        user_lines.append(f"Kernel name: {kernel_name}")
    user_lines.append("Triton kernel:")
    user_lines.append(triton_src)
    if extra_instruction:
        user_lines.append("\nExtra instructions:")
        user_lines.append(extra_instruction)
    content = "\n".join(user_lines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def extract_intent_json(
    triton_src: str,
    *,
    kernel_name: Optional[str] = None,
    extra_instruction: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    **chat_kwargs: Any,
) -> Dict[str, Any]:
    messages = build_messages(
        triton_src,
        kernel_name=kernel_name,
        extra_instruction=extra_instruction,
    )
    last_err = None
    chat_kwargs = dict(chat_kwargs)
    chat_kwargs.setdefault("max_tokens", 800)
    # Model/provider fallback is handled in llm_client.chat_completion; keep retries here
    # strictly for "output format" (JSON) issues to avoid burning rate limits.
    for attempt in range(2):
        response: LLMResponse = chat_completion(messages, model=model, stream=False, **chat_kwargs)
        raw_text = response.first_message()
        try:
            return _parse_json_block(raw_text)
        except json.JSONDecodeError as e:
            last_err = e
            # dump raw for debugging
            try:
                import tempfile, time

                fname = tempfile.gettempdir() + f"/llm_fail_{kernel_name or 'kernel'}_{int(time.time())}_{attempt}.txt"
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(raw_text)
            except Exception:
                pass
            # augment user message with error hint and retry
            messages[-1]["content"] += (
                f"\nPrevious attempt failed to parse JSON: {e}. Please return STRICT JSON (no trailing commas/comments), keep concise."
            )
            continue
    snippet = "" if last_err is None else str(last_err)
    raise LLMClientError(f"Failed to parse LLM JSON after retries: {snippet}")


__all__ = ["extract_intent_json", "build_messages"]
