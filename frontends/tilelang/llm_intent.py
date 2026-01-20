"""
TileLang frontend: LLM prompt builder for IntentIR extraction.

The TileLang "source" is TVMScript emitted from a `tvm.tir.PrimFunc` produced by
TileLang (`tilelang.language as T`). The prompt focuses on semantic extraction
instead of mirroring low-level thread/block details.
"""

from __future__ import annotations

from typing import Dict, List, Optional


SYSTEM_PROMPT = """You are an expert compiler engineer. Given a TileLang kernel
(TVMScript / tvm.tir.PrimFunc), produce Intent-IR v1.1 candidate JSON. Hard rules:
- Emit a single JSON object with keys:
  name, kernel_type, tensors, ops, outputs, parallel_axes,
  schedule(optional), axis_roles(optional), meta(optional).
- `tensors` MUST be an object {name: {...}}, NOT a list.
- Keep argument tensor shapes as declared by the kernel signature / buffer shapes.
  If you need a grouped/view shape (e.g., [N,G,group_size,HW]) insert explicit
  reshape ops instead of redefining the original tensor shape.
- You MAY replace a constant extent in an interface tensor shape (e.g., 16) with a
  symbol from the Evidence appendix `launch.canonical_shapes` (e.g., Q_CTX), but
  ONLY when the numeric values match exactly. Treat these symbols as compile-time
  constants; do not invent any other new symbols.
- Use the standard op set (primitives):
  - numeric: add/sub/mul/div/max/min/exp/relu/rsqrt/abs/floor
  - compare/bool: ne/lt/le/gt/ge/and/or/not/where
  - shape: reshape/broadcast_in_dim/transpose/layout_cast
  - reduce: reduce_sum/reduce_max/reduce_any/softmax
  - misc: matmul/conv2d/cast/iota/gather/identity/const
- Macro ops are allowed when they represent a well-known semantic operator and
  will be lowered/expanded later by the compiler. Currently allowed macro ops:
  - upsample_bicubic2d_aa (NCHW). Include useful implementation-detail attrs if available.
  Do NOT invent arbitrary new op names beyond this allowed list.
- reshape MUST include attrs.shape (non-empty list).
  broadcast_in_dim MUST have attrs.out_shape + attrs.broadcast_dims.
  transpose MUST have attrs.perm.
- iota MUST use attrs.axis (int) + attrs.shape; omit attrs.dtype unless you need a non-default dtype.
- Do NOT invent arbitrary new shape symbols. Any symbol used in tensor shapes or
  reshape/broadcast/iota shapes must come from:
  - existing kernel signature symbols (if any), OR
  - Evidence appendix: `launch.canonical_shapes` keys (compile-time constants).
- When you use reshape/iota/broadcast_in_dim, prefer using canonical shape symbols
  (from `launch.canonical_shapes`) in attrs.shape / attrs.out_shape instead of raw
  integers, as long as the numeric values match exactly.
- Every output tensor MUST be declared in tensors and produced by an op.
  If the kernel writes Mean/Rstd or similar, add the corresponding reduce/compute ops.
- Each op.output name MUST be unique across the ops list (no duplicates).
- Reduce dims/axis must be integer axis indices (after any reshape/transpose), not symbolic names.
- IMPORTANT groupnorm semantics:
  reduce_sum computes SUM; normalize by num_elements=group_size*HW
  (mean=sum/num_elements; var=sumsq/num_elements - mean^2; rstd=rsqrt(var+eps)).
  Prefer explicit arithmetic ops; model eps/num_elements/group_size as `const` ops (scalar tensors)
  unless they are runtime scalar tensors in the kernel signature.
- IMPORTANT layernorm semantics: normalize by N (mean=sum/N; var=sumsq/N - mean^2).
- IMPORTANT attention kernels:
  - Represent score computation as `matmul` + `mul(sm_scale)` + `softmax`, then `matmul` with V.
  - You MUST use the `softmax` primitive op for attention/softmax; do NOT expand it into
    exp/reduce_max/reduce_sum/sub/div chains unless explicitly asked.
  - For causal masks, prefer this canonical IntentIR pattern (to keep cross-frontend graphs consistent):
    scores_scaled=mul(scores_raw, sm_scale)  # sm_scale is scalar; do NOT broadcast it
    query_pos=iota(shape=[Q_CTX],axis=0); key_pos=iota(shape=[KV_CTX],axis=0);
    query_bc=broadcast_in_dim(query_pos,out_shape=[Q_CTX,KV_CTX],broadcast_dims=[0]);
    key_bc=broadcast_in_dim(key_pos,out_shape=[Q_CTX,KV_CTX],broadcast_dims=[1]);
    causal_mask = gt(key_bc, query_bc); neg_inf=const(-1e9);
    neg_inf_bc=broadcast_in_dim(neg_inf,out_shape=[Q_CTX,KV_CTX],broadcast_dims=[]);
    scores_masked = where(causal_mask, neg_inf_bc, scores_scaled); then softmax(axis=1).
  - Do NOT implement matmul via ad-hoc reduce_sum with wrong axes; prefer `matmul` primitive.
- Do NOT model low-level thread/block/program mapping in tensor shapes or iota.shape.
  Treat logical axes (M/N/K/OH/OW/etc) as full ranges. You MAY emit a schedule sketch:
  - schedule.tile_m/tile_n/tile_k/vec_width/pipeline_depth when obvious tile constants exist.
  - schedule.axis_bindings binds tile keys to logical axes (e.g., tile_n->N).
- parallel_axes is a list of axis strings, and every axis must appear in some tensor shape;
  do not invent axes.
- axis_roles values must be in {spatial,reduction,batch,channel}.
- If `launch.canonical_shapes` provides meaningful symbols for this kernel, prefer emitting
  `axis_roles` for those symbols (e.g., M/N/K, Q_CTX/KV_CTX/HEAD_DIM, N/C/HW, OH/OW, etc).
Return JSON only: no prose, no code fences."""


SYSTEM_PROMPT_COMPACT = """You are an expert compiler engineer. Convert the given
TileLang TVMScript (tvm.tir.PrimFunc) into ONE Intent-IR v1.1 JSON object (no prose).

Required keys: name, kernel_type, tensors, ops, outputs, parallel_axes (schedule/axis_roles/meta optional).
Rules:
- tensors is an object {name:{dtype,shape,layout}}, NOT a list.
- outputs is a list; every output must be declared in tensors and produced by some op.
- Allowed ops only: add/sub/mul/div/max/min/exp/relu/rsqrt/abs/floor,
  ne/lt/le/gt/ge/and/or/not/where, reshape/broadcast_in_dim/transpose/layout_cast,
  reduce_sum/reduce_max/reduce_any/softmax, matmul/conv2d/cast/iota/gather/identity/const,
  macro: upsample_bicubic2d_aa (only when semantically appropriate).
- Do NOT invent arbitrary new shape symbols; use only symbols from the kernel signature,
  and/or Evidence appendix `launch.canonical_shapes` keys (compile-time constants).
  You MAY replace constant extents in interface tensor shapes with canonical_shapes symbols
  only when values match exactly.
- Prefer using canonical_shapes symbols in attrs.shape/attrs.out_shape too (when values match).
- Prefer emitting axis_roles when canonical shape symbols exist (see Evidence appendix).
- Prefer `softmax` op over manual exp/reduce chains (especially for attention kernels).
- For iota, omit attrs.dtype unless non-default (default i32).
- Keep original input view shapes; use reshape ops for any grouped/view computation.
"""


def build_messages(
    tilelang_src: str,
    *,
    kernel_name: Optional[str] = None,
    extra_instruction: Optional[str] = None,
    compact: bool = False,
) -> List[Dict[str, str]]:
    user_lines: List[str] = []
    if kernel_name:
        user_lines.append(f"Kernel name: {kernel_name}")
    user_lines.append("TileLang TVMScript (PrimFunc):")
    user_lines.append(tilelang_src)
    if extra_instruction:
        user_lines.append("\nExtra instructions:")
        user_lines.append(extra_instruction)
    return [
        {"role": "system", "content": (SYSTEM_PROMPT_COMPACT if compact else SYSTEM_PROMPT)},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


__all__ = ["SYSTEM_PROMPT", "build_messages"]
