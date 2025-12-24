"""
LLM-driven intent extraction helpers (lives under intent_ir namespace).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .llm_client import DEFAULT_MODEL, LLMClientError, LLMResponse, candidate_models, chat_completion


SYSTEM_PROMPT = """You are an expert compiler engineer. Given the Triton
@triton.jit kernel body, produce Intent-IR v1.1 candidate JSON. Hard rules:
- Emit JSON object with: name, kernel_type, tensors, ops, outputs,
  parallel_axes, schedule(optional), axis_roles(optional), meta(optional).
- tensors must be an object {name: {...}}, NOT a list. Each tensor has dtype in
  {f16,bf16,f32,i32,...}, shape list of symbols/ints, layout row_major unless explicit.
  IMPORTANT: keep argument tensor shapes as declared in the kernel signature; if you need
  a grouped/view shape (e.g., [num_groups, group_size, HW]) insert explicit reshape ops
  instead of redefining the original tensor shape.
- Use the standard op set (primitives):
  - numeric: add/sub/mul/div/max/min/exp/relu/rsqrt/abs/floor
  - compare/bool: ne/lt/le/gt/ge/and/or/not/where
  - shape: reshape/broadcast_in_dim/transpose/layout_cast
  - reduce: reduce_sum/reduce_max/reduce_any/softmax
  - misc: matmul/conv2d/cast/iota/gather/identity/const
- Macro ops are allowed when they represent a well-known semantic operator and will
  be lowered/expanded later by the compiler. Currently allowed macro ops:
  - upsample_bicubic2d_aa: bicubic AA upsample (NCHW). When you use this macro op, include implementation-detail attrs:
    - Prefer a structured attrs.impl object with nested sections:
      impl.kernel, impl.index_plan, impl.composition, impl.hoist
    - impl.kernel should be machine-usable (not just a name):
      - name: "keys_cubic"
      - a: number (e.g., -0.5)
      - invscale: number (usually 1.0)
      - segments: list[ {t_max: number, coeffs: [c0,c1,c2,c3]} ] where poly(t)=c0+c1*t+c2*t^2+c3*t^3
        (for Keys cubic: seg1 t<1 uses [1,0,-(a+3),(a+2)]; seg2 t<2 uses [-4a,8a,-5a,a]; outside is 0)
    - impl.index_plan must specify rounding/clamp semantics:
      - center_offset (0.5), start_offset (0.5), span_end_offset (0.5), tap_offset (0.5), clamp_low (0.0), support (2.0), tap_enable ("k < span_size")
    - impl.composition must specify structure choices:
      - separable (true), compute_order ("x_then_y"), normalize_weights (true), other_value (0.0), mask_policy ("mask_y(dy) & mask_x(dx)")
    - Also include flat shortcuts: a, support, invscale, kernel, separable, compute_order, normalize_weights, and brief formula strings (center/start/span/mask policies).
  Do NOT invent arbitrary new op names beyond this allowed list.
- reshape MUST include attrs.shape (non-empty list). broadcast_in_dim MUST have out_shape + broadcast_dims. transpose MUST have perm.
- Do NOT invent new shape symbols. Any symbol used in tensor shapes or reshape/broadcast/iota shapes must come from the kernel signature (or existing symbols already present in tensor shapes).
- iota MUST use attrs.axis (int) and attrs.shape; do NOT use attrs.dimension. cast MUST use attrs.to; do NOT use attrs.dtype.
- Every output tensor MUST be declared in tensors and produced by an op. If the kernel writes Mean/Rstd or similar, add the corresponding reduce/assign ops.
- reduce_any MUST include dims/axis. Softmax must be present for attention kernels.
- Reduce dims/axis must be integer axis indices (after any reshape/transpose), not symbolic axis names.
- IMPORTANT groupnorm semantics: reduce_sum computes SUM; you must normalize by num_elements=group_size*HW
  (mean = sum/num_elements; var = sumsq/num_elements; rstd = rsqrt(var+eps)). Use reduce_sum(attrs.scale=...) or explicit div ops.
- IMPORTANT layernorm semantics: normalize by N (mean = sum/N; var = sumsq/N).
- Scalars/constants (eps, num_elements, group_size, sm_scale, BLOCK_*) must be explicit. Prefer modeling them as `const` ops (scalar tensors) and pass them as normal op inputs.
  Do NOT use ad-hoc scalar-in-attrs shorthands for arithmetic ops (e.g., add with attrs.scalar); keep arithmetic ops as 2-input ops.
- Scalar runtime parameters that appear in the kernel signature (e.g., reciprocal_scale_h/w) must be declared in tensors as shape=[] and referenced directly; do NOT replace them with const("reciprocal_scale_h").
- Do NOT model scheduling/tiling details (program_id, tl.constexpr BLOCK_*). Never put BLOCK_* into tensor shapes or iota.shape. Treat ow/oh as full logical ranges [OW] and [OH] rather than per-program tiles.
- Do NOT put BLOCK_*/TILE_* into tensor shapes or iota.shape. Treat ow/oh as full logical ranges [OW] and [OH] rather than per-program tiles.
  However, you SHOULD emit a schedule sketch when tl.constexpr block/tile parameters exist:
  - Put BLOCK_*/TILE_* into schedule.tile_m/tile_n/tile_k or schedule.memory_hint.
  - Bind them to real axes via schedule.axis_bindings (e.g., BLOCK_M->M, BLOCK_N->N; BLOCK_Y->OH, BLOCK_X->OW).
- op.inputs reference tensors or prior op outputs only; represent scalar values via `const` ops (or scalar tensors from the kernel signature).
- For groupnorm kernels: Mean/Rstd must be shaped as [N, num_groups] (optionally [N,num_groups,1] if keepdims=true).
- For layernorm kernels: Mean/Rstd must be shaped as [M] (optionally [M,1] if keepdims=true).
  Use explicit reshape/broadcast ops for any view changes; do NOT fake shapes by redefining inputs.
- axis_roles: {axis: role} with role in {spatial,reduction,batch,channel}; do NOT invert.
- parallel_axes: list of axis strings, and every axis must appear in some tensor shape; do not invent axes.
- schedule may include tile_m/tile_n/tile_k/vec_width/axis_bindings/vec_axis/parallel_axes; if unknown, omit rather than guess.
- For optional tensors, either mark optional:true or omit consistently.
- For complex kernels (e.g., bicubic upsample), prefer a semantic macro op instead of a huge low-level graph:
  - Use op `upsample_bicubic2d_aa` with 1 input (the input tensor) and output (the output tensor).
  - Optional attrs: {"a": -0.5}.
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
    last_err: Exception | None = None
    chat_kwargs = dict(chat_kwargs)
    chat_kwargs.setdefault("max_tokens", 800)
    # NOTE: llm_client.chat_completion only fails over on HTTP-level errors.
    # Here we add *content-level* fallback: if a provider returns non-JSON prose,
    # try the other provider/model without stalling the whole pipeline.
    model_candidates = candidate_models(model)

    # Keep retries here strictly for "output format" (JSON) issues.
    for attempt in range(2):
        for m in model_candidates:
            response: LLMResponse = chat_completion(messages, model=m, stream=False, **chat_kwargs)
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
                continue
        # augment user message with error hint and retry once (helps with trailing commas/fences)
        if last_err is not None:
            messages[-1]["content"] += (
                f"\nPrevious attempt failed to parse JSON: {last_err}. Please return STRICT JSON (no prose/code fences), keep concise."
            )
    snippet = "" if last_err is None else str(last_err)
    raise LLMClientError(f"Failed to parse LLM JSON after retries: {snippet}")


__all__ = ["extract_intent_json", "build_messages"]
