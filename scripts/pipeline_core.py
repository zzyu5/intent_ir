"""
Reusable helpers for the full pipeline (LLM -> IntentIR -> TTIR -> validation).

This keeps `scripts/full_pipeline_verify.py` thin; new kernels can reuse the same
helpers by constructing a KernelSpec and calling `run_pipeline_for_spec`.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import triton

from intent_ir.ir_types import IntentIRValidationError
from intent_ir.llm_intent import LLMClientError, extract_intent_json
from intent_ir.macro_expand import expand_macros
from intent_ir.macro_spec import enrich_intent_macros
from intent_ir.parser_llm import CandidateIntent, LLMJsonParseError, parse_candidate_json
from intent_ir.printer_mlir_like import print_mlir_like
from intent_ir.ir_types import ScheduleSketch
from triton_frontend.certificate import build_certificate
from triton_frontend.contract import evaluate_contract
from triton_frontend.facts import extract_constraints, extract_facts
from triton_frontend.static_validate import static_validate
from verify.diff_runner import run_diff
from verify.gen_cases import TestCase, generate_cases
from triton_frontend.certificate import SemanticCertificate
from verify.metamorphic import run_bounded_exhaustive, run_metamorphic_suite
from verify.mutation import run_mutation_kill


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class KernelSpec:
    name: str
    module: str
    attr: str  # attribute on module for source, e.g., "fn.src" or "src"
    runner: Callable[[TestCase], Dict[str, np.ndarray]]  # reference runner (launch Triton kernel)
    canonical_shapes: Dict[str, int]
    vary_axes: List[str]
    exclude_axes: Optional[List[str]] = None
    normalize_shapes: Optional[Callable[[Dict[str, int]], Dict[str, int]]] = None


def _get_src(mod_name: str, attr: str) -> str:
    mod = __import__(mod_name, fromlist=["dummy"])
    obj = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    return str(obj)


def prepare_dump_and_cache_dirs(base_dir: Path, kernel_name: str, *, clean: bool = True) -> tuple[Path, Path]:
    """
    Use per-kernel dump/cache directories so each run reliably produces TTIR.
    """
    dump_dir = base_dir / "_triton_dump" / kernel_name
    cache_dir = base_dir / "_triton_cache" / kernel_name
    if clean:
        shutil.rmtree(dump_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = str(dump_dir)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
    return dump_dir, cache_dir


def find_latest_ttir(dump_dir: Path, name_hint: str) -> Optional[Path]:
    ttirs = sorted(dump_dir.rglob("*.ttir"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in ttirs:
        if name_hint in p.name:
            return p
    return ttirs[0] if ttirs else None


def llm_to_intent(src: str, kernel_name: str, feedback: Optional[List[str]] = None) -> CandidateIntent:
    base_extra = (
        "Return STRICT JSON (no prose/code fences). "
        "Every output must be produced by ops (no placeholder outputs). "
        "Keep argument tensor shapes exactly as the kernel signature; if you need a view (e.g., group view), insert explicit reshape ops (do not redefine inputs). "
        "Arithmetic ops (add/sub/mul/div/max/min) must have 2 inputs; model scalars as const ops (shape=[]), not ad-hoc scalar attrs. "
        "Reduce dims/axis must be integer axis indices (after any reshape/transpose). "
        "For groupnorm stats, Mean/Rstd must be [N, num_groups] (optionally [N,num_groups,1]); do NOT flatten into 1D. "
        "IMPORTANT groupnorm semantics: reduce_sum computes SUM; you must normalize by num_elements=group_size*HW "
        "(mean = reduce_sum(X)/num_elements; var = reduce_sum((X-mean)^2)/num_elements; rstd = rsqrt(var+eps)). "
        "You may encode normalization as reduce_sum(attrs.scale='1.0/(group_size*HW)') or explicit div ops. "
        "For layernorm stats, Mean/Rstd must be [M] (optionally [M,1]). "
        "IMPORTANT layernorm semantics: normalize by N (mean = sum/N; var = sumsq/N). "
        "Do NOT use BLOCK_*/TILE_* in tensor shapes or iota.shape. "
        "You SHOULD emit schedule sketch if the kernel has tl.constexpr block/tile parameters: put them in schedule.tile_*/schedule.memory_hint "
        "and bind them to axes via schedule.axis_bindings (e.g., BLOCK_M->M, BLOCK_N->N; BLOCK_Y->OH, BLOCK_X->OW). "
        "iota uses attrs.axis (not dimension); cast uses attrs.to (not dtype). "
        "Prefer primitive ops unless a semantic macro op is explicitly allowed."
    )
    lower_src = src.lower()
    if "bicubic" in lower_src or "cubic" in lower_src:
        base_extra += (
            " Prefer using the semantic macro op `upsample_bicubic2d_aa` (single op) instead of expanding into dozens of primitive ops. "
            "Macro form: one op with op='upsample_bicubic2d_aa', inputs=[input_tensor], output=output_tensor. "
            "For macro attrs, include BOTH a structured impl object (preferred) and a few human-readable formula strings: "
            "attrs.impl = {"
            "kernel:{name:'keys_cubic', a:-0.5, invscale:1.0, segments:[{t_max:1.0, coeffs:[1,0,-(a+3),(a+2)]},{t_max:2.0, coeffs:[-4*a,8*a,-5*a,a]}]}, "
            "index_plan:{center_offset:0.5, support:2.0, start_offset:0.5, clamp_low:0.0, tap_enable:'k<span_size'}, "
            "composition:{separable:true, compute_order:'x_then_y', normalize_weights:true, other_value:0.0}, "
            "hoist:['span_start','span_size','weights_x','weights_y','masks_x','masks_y']"
            "}. "
            "Also include flat shortcuts: a/support/invscale/kernel/separable/compute_order/normalize_weights/mask_policy/other_value/hoist and brief formulas. "
            "Do NOT invent composite shape symbols like IH_IW/OH_OW; use existing symbols (N,C,IH,IW,OH,OW) only."
        )
    # Keep prompt kernel-name agnostic: the LLM should infer patterns from the code itself.
    if feedback:
        base_extra += "\nFeedback from last attempt:\n- " + "\n- ".join(feedback)
    last_err = None
    for attempt in range(2):
        try:
            js = extract_intent_json(
                src,
                kernel_name=kernel_name,
                temperature=0,
                max_tokens=6000,
                extra_instruction=base_extra if attempt == 0 else base_extra + f" Previous attempt failed: {last_err}",
            )
            return parse_candidate_json(js)
        except (LLMJsonParseError, IntentIRValidationError) as e:
            last_err = e
            continue
    raise last_err or Exception("LLM/parse failed")


def _ensure_schedule(intent, *, kernel_name: str, triton_src: str) -> None:
    """
    Keep schedule visible in IntentIR.
    - Prefer schedule produced by LLM.
    - If missing, attach a lightweight ScheduleSketch derived from common tl.constexpr names.
    This is a *sketch* (can be symbolic) and is used by backend mapping + reporting.
    """
    existing = intent.schedule
    if existing is not None:
        # If tiles/axis_bindings are present, keep as-is. Otherwise, we can still
        # infer tile bindings and *merge* them while preserving memory_hint.
        if any(getattr(existing, k) for k in ("tile_m", "tile_n", "tile_k", "vec_width", "pipeline_depth")):
            return
        if existing.axis_bindings:
            return

    # Heuristic: infer from tl.constexpr argument names.
    # We avoid parsing full Triton; just look for common names.
    text = str(triton_src).upper()
    hint_keys = set()
    if existing is not None and isinstance(existing.memory_hint, dict):
        hint_keys = {str(k).upper() for k in existing.memory_hint.keys()}
    axis_bindings = {}
    tile_m = None
    tile_n = None
    tile_k = None

    def has(name: str) -> bool:
        n = str(name).upper()
        return (n in text) or (n in hint_keys)

    # 2D tiles in OW/OH space (upsample-like)
    if has("BLOCK_X") and has("BLOCK_Y"):
        tile_m = "BLOCK_Y"
        tile_n = "BLOCK_X"
        axis_bindings = {"tile_m": "OH", "tile_n": "OW"}
    # Row/col blocking (reduce_any)
    elif has("BLOCK_M") and has("BLOCK_N"):
        tile_m = "BLOCK_M"
        tile_n = "BLOCK_N"
        axis_bindings = {"tile_m": "M", "tile_n": "N"}
    # Softmax-style tiling
    elif has("TILE_N") and has("TILE_K"):
        tile_n = "TILE_N"
        tile_k = "TILE_K"
        axis_bindings = {"tile_n": "N", "tile_k": "K"}
    elif has("TILE_N"):
        tile_n = "TILE_N"
        axis_bindings = {"tile_n": "N"}
    # GroupNorm-style blocking
    elif has("BLOCK_GROUP_SIZE") and has("BLOCK_HW_SIZE"):
        tile_m = "BLOCK_GROUP_SIZE"
        tile_n = "BLOCK_HW_SIZE"
        axis_bindings = {"tile_m": "group_size", "tile_n": "HW"}

    if tile_m is None and tile_n is None and tile_k is None:
        # Still attach an empty schedule to keep the field present (paper narrative).
        if existing is None:
            intent.schedule = ScheduleSketch()
        return

    # Merge into existing schedule (preserve memory_hint/parallel_axes/etc).
    if existing is None:
        intent.schedule = ScheduleSketch(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, axis_bindings=axis_bindings)
    else:
        intent.schedule = ScheduleSketch(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            vec_width=existing.vec_width,
            pipeline_depth=existing.pipeline_depth,
            axis_bindings=dict(axis_bindings),
            vec_axis=existing.vec_axis,
            parallel_axes=list(existing.parallel_axes),
            memory_hint=dict(existing.memory_hint),
        )


def make_cases(
    spec: KernelSpec,
    intent: CandidateIntent,
    constraints,
    limit: int = 8,
    *,
    extra_tile_hints: Optional[List[int]] = None,
    cert: Optional[SemanticCertificate] = None,
) -> List[TestCase]:
    base = dict(spec.canonical_shapes)
    if spec.normalize_shapes:
        base = spec.normalize_shapes(base)
    cases = [TestCase(shapes=base, dtypes={}, seed=0)]
    tile_hints = []
    if extra_tile_hints:
        tile_hints.extend(int(x) for x in extra_tile_hints if isinstance(x, int))
    if intent.intent.schedule:
        for k in ("tile_m", "tile_n", "tile_k", "vec_width"):
            v = getattr(intent.intent.schedule, k)
            if isinstance(v, int):
                tile_hints.append(v)
    # If certificate indicates masks, force needs_mask-like behavior in case gen.
    if cert and (cert.mask_constraints or cert.needs_mask):
        from triton_frontend.facts import TTIRConstraints

        constraints = constraints or TTIRConstraints(needs_mask=True, suggested_edge_cases=[])
        constraints.needs_mask = True

    extra_sizes: List[int] = []
    if cert:
        # Heuristic: add numeric literals from mask constraints and range ends as candidate sizes.
        import re

        num_re = re.compile(r"(-?\d+)")
        for cs in cert.mask_constraints.values():
            for c in cs:
                for m in num_re.findall(c):
                    try:
                        v = int(m)
                        if 0 < v <= 2048:
                            extra_sizes.append(v)
                            extra_sizes.append(max(1, v - 1))
                            extra_sizes.append(v + 1)
                    except Exception:
                        pass
        for spec_range in (cert.index_symbols.get("ranges") or {}).values():
            try:
                end = int(spec_range.get("end"))
                if 0 < end <= 2048:
                    extra_sizes.append(end)
                    extra_sizes.append(max(1, end - 1))
                    extra_sizes.append(end + 1)
            except Exception:
                pass
        extra_sizes = sorted(set(extra_sizes))

    generated = generate_cases(
        intent.intent,
        constraints=constraints,
        limit=limit,
        seed=1,
        tile_hints=tile_hints,
        axes=spec.vary_axes,
        exclude_axes=spec.exclude_axes,
        extra_sizes=extra_sizes,
    )
    for c in generated:
        merged = dict(spec.canonical_shapes)
        merged.update(c.shapes)
        if spec.normalize_shapes:
            merged = spec.normalize_shapes(merged)
        c.shapes = merged
        cases.append(c)
    return cases


# ---------------------- default kernel runners ----------------------

def _run_any_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from tritonops.any import any_kernel_dim

    M = case.shapes.get("M", 4)
    N = case.shapes.get("N", 8)
    device = "cuda"
    if case.inputs and "inp" in case.inputs:
        inp_np = case.inputs["inp"]
        inp = torch.as_tensor(inp_np, device=device)
        if inp.dtype != torch.float32:
            inp = inp.to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"inp shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        # Deterministic, non-degenerate pattern: ensure both True/False rows appear.
        inp = torch.zeros((M, N), device=device, dtype=torch.float32)
        for m in range(int(M)):
            if m % 2 == 1:
                inp[m, (m * 3) % int(N)] = 1.0
    out = torch.empty((M,), device=device, dtype=torch.bool)
    grid = lambda meta: ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],)
    any_kernel_dim[grid](inp, out, M, N)
    torch.cuda.synchronize()
    return {"inp": inp.cpu().numpy(), "out": out.cpu().numpy()}


def _run_groupnorm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from tritonops.groupnorm import group_norm_kernel

    N = case.shapes.get("N", 2)
    C = case.shapes.get("C", 4)
    HW = case.shapes.get("HW", 4)
    num_groups = case.shapes.get("num_groups", case.shapes.get("group", 2))
    if C % num_groups != 0:
        raise ValueError(f"groupnorm requires C divisible by num_groups for reshape-based IntentIR: C={C} num_groups={num_groups}")
    group_size = C // num_groups

    def next_power_of_2(x: int) -> int:
        return 1 if x <= 1 else 1 << (x - 1).bit_length()

    block_group_size = next_power_of_2(group_size)
    block_hw_size = next_power_of_2(HW)

    device = "cuda"
    if case.inputs and "X" in case.inputs:
        x_np = case.inputs["X"]
        x = torch.as_tensor(x_np, device=device)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if tuple(x.shape) != (N, C, HW):
            raise ValueError(f"X shape mismatch: got {tuple(x.shape)} expected {(N, C, HW)}")
    else:
        x = torch.randn((N, C, HW), device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    if case.inputs and "W" in case.inputs:
        w_np = case.inputs["W"]
        w = torch.as_tensor(w_np, device=device)
        if w.dtype != torch.float32:
            w = w.to(torch.float32)
        if tuple(w.shape) != (C,):
            raise ValueError(f"W shape mismatch: got {tuple(w.shape)} expected {(C,)}")
    else:
        w = torch.ones((C,), device=device, dtype=torch.float32)
    if case.inputs and "B" in case.inputs:
        b_np = case.inputs["B"]
        b = torch.as_tensor(b_np, device=device)
        if b.dtype != torch.float32:
            b = b.to(torch.float32)
        if tuple(b.shape) != (C,):
            raise ValueError(f"B shape mismatch: got {tuple(b.shape)} expected {(C,)}")
    else:
        b = torch.zeros((C,), device=device, dtype=torch.float32)
    mean = torch.empty((N * num_groups,), device=device, dtype=torch.float32)
    rstd = torch.empty((N * num_groups,), device=device, dtype=torch.float32)
    grid = lambda meta: (N * num_groups,)
    group_norm_kernel[grid](
        x,
        y,
        w,
        b,
        mean,
        rstd,
        group_size,
        C,
        HW,
        num_groups,
        1e-5,
        BLOCK_GROUP_SIZE=block_group_size,
        BLOCK_HW_SIZE=block_hw_size,
    )
    torch.cuda.synchronize()
    return {
        "X": x.cpu().numpy(),
        "W": w.cpu().numpy(),
        "B": b.cpu().numpy(),
        "Y": y.cpu().numpy(),
        "Mean": mean.view(N, num_groups).cpu().numpy(),
        "Rstd": rstd.view(N, num_groups).cpu().numpy(),
    }


def _run_attention_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from tritonops.attention import _attn_fwd

    batch = case.shapes.get("Z", case.shapes.get("batch", 1))
    q_numhead = case.shapes.get("q_numhead", 1)
    kv_numhead = case.shapes.get("kv_numhead", 1)
    Q_CTX = case.shapes.get("Q_CTX", 128)
    KV_CTX = case.shapes.get("KV_CTX", 128)
    HEAD_DIM = case.shapes.get("HEAD_DIM", 64)
    device = "cuda"
    if case.inputs and "Q" in case.inputs:
        Q = torch.as_tensor(case.inputs["Q"], device=device).to(torch.float32)
    else:
        Q = torch.randn((batch, q_numhead, Q_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    if case.inputs and "K" in case.inputs:
        K = torch.as_tensor(case.inputs["K"], device=device).to(torch.float32)
    else:
        K = torch.randn((batch, kv_numhead, KV_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    if case.inputs and "V" in case.inputs:
        V = torch.as_tensor(case.inputs["V"], device=device).to(torch.float32)
    else:
        V = torch.randn((batch, kv_numhead, KV_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    if case.inputs and "attn_mask" in case.inputs:
        attn_mask = torch.as_tensor(case.inputs["attn_mask"], device=device).to(torch.float32)
    else:
        attn_mask = torch.zeros((batch, q_numhead, Q_CTX, KV_CTX), device=device, dtype=torch.float32)
    Out = torch.empty((batch, q_numhead, Q_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    if case.inputs and "sm_scale" in case.inputs:
        sm_scale = float(np.asarray(case.inputs["sm_scale"]).reshape(()))
    else:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    block_m = 16
    block_n = 16
    meta_cfg = {"BLOCK_M": block_m, "BLOCK_N": block_n, "STAGE": 1, "HAS_ATTN_MASK": 0, "PRE_LOAD_V": 0}
    grid = lambda meta: (triton.cdiv(Q_CTX, meta_cfg["BLOCK_M"]), batch * q_numhead)
    kernel = _attn_fwd.fn
    stride_q_batch, stride_q_head, stride_q_seqlen, stride_q_headsize = Q.stride()
    stride_k_batch, stride_k_head, stride_k_seqlen, stride_k_headsize = K.stride()
    stride_v_batch, stride_v_head, stride_v_seqlen, stride_v_headsize = V.stride()
    stride_o_batch, stride_o_head, stride_o_seqlen, stride_o_headsize = Out.stride()
    stride_attn_mask_batch, stride_attn_mask_head, stride_attn_mask_q_seqlen, stride_attn_mask_kv_seqlen = attn_mask.stride()
    kernel[grid](
        Q,
        K,
        V,
        attn_mask,
        sm_scale,
        Out,
        stride_q_batch,
        stride_q_head,
        stride_q_seqlen,
        stride_q_headsize,
        stride_k_batch,
        stride_k_head,
        stride_k_seqlen,
        stride_k_headsize,
        stride_v_batch,
        stride_v_head,
        stride_v_seqlen,
        stride_v_headsize,
        stride_attn_mask_batch,
        stride_attn_mask_head,
        stride_attn_mask_q_seqlen,
        stride_attn_mask_kv_seqlen,
        stride_o_batch,
        stride_o_head,
        stride_o_seqlen,
        stride_o_headsize,
        KV_CTX,
        q_numhead,
        kv_numhead,
        Q_CTX,
        KV_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=meta_cfg["BLOCK_M"],
        BLOCK_N=meta_cfg["BLOCK_N"],
        STAGE=meta_cfg["STAGE"],
        HAS_ATTN_MASK=meta_cfg["HAS_ATTN_MASK"],
        PRE_LOAD_V=meta_cfg["PRE_LOAD_V"],
    )
    torch.cuda.synchronize()
    return {
        "Z": np.array(batch, dtype=np.int32),
        "Q": Q.cpu().numpy(),
        "K": K.cpu().numpy(),
        "V": V.cpu().numpy(),
        "attn_mask": attn_mask.cpu().numpy(),
        "sm_scale": np.array(sm_scale, dtype=np.float32),
        "Out": Out.cpu().numpy(),
    }


def _run_softmax_reference(case: TestCase) -> Dict[str, np.ndarray]:
    """
    Simple 2D softmax over the last dim.
    Uses the inner kernel directly with explicit meta-params to avoid heuristic dependency.
    """
    from tritonops.softmax import softmax_kernel_inner

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    device = "cuda"
    # Use stable tensor names ("input"/"output") so LLM/IntentIR aligns with baseline I/O.
    if case.inputs and "input" in case.inputs:
        x_np = case.inputs["input"]
        x = torch.as_tensor(x_np, device=device).to(torch.float32)
        if tuple(x.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(x.shape)} expected {(M, N)}")
    else:
        x = torch.randn((M, N), device=device, dtype=torch.float32)
    out = torch.empty((M, N), device=device, dtype=torch.float32)
    # single program per row
    grid = (M, 1, 1)
    TILE_N = 1 if N <= 1 else min(1024, 1 << (N - 1).bit_length())
    ONE_TILE_PER_CTA = 1 if N <= TILE_N else 0
    softmax_kernel_inner[grid](out, x, M, N, TILE_N=TILE_N, ONE_TILE_PER_CTA=ONE_TILE_PER_CTA)
    torch.cuda.synchronize()
    return {"input": x.cpu().numpy(), "output": out.cpu().numpy()}


def _run_layernorm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    """
    LayerNorm forward over last dim (shape [M, N]).
    Returns out + per-row mean/rstd (shape [M]).
    """
    from tritonops.layernorm import layer_norm_persistent_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    eps = float(case.shapes.get("eps", 1e-5))
    device = "cuda"

    if case.inputs and "in_ptr" in case.inputs:
        x = torch.as_tensor(case.inputs["in_ptr"], device=device).to(torch.float32)
        if tuple(x.shape) != (M, N):
            raise ValueError(f"in_ptr shape mismatch: got {tuple(x.shape)} expected {(M, N)}")
    else:
        x = torch.randn((M, N), device=device, dtype=torch.float32)

    if case.inputs and "weight_ptr" in case.inputs:
        w = torch.as_tensor(case.inputs["weight_ptr"], device=device).to(torch.float32)
        if tuple(w.shape) != (N,):
            raise ValueError(f"weight_ptr shape mismatch: got {tuple(w.shape)} expected {(N,)}")
    else:
        w = torch.ones((N,), device=device, dtype=torch.float32)

    if case.inputs and "bias_ptr" in case.inputs:
        b = torch.as_tensor(case.inputs["bias_ptr"], device=device).to(torch.float32)
        if tuple(b.shape) != (N,):
            raise ValueError(f"bias_ptr shape mismatch: got {tuple(b.shape)} expected {(N,)}")
    else:
        b = torch.zeros((N,), device=device, dtype=torch.float32)

    y = torch.empty((M, N), device=device, dtype=torch.float32)
    mean = torch.empty((M,), device=device, dtype=torch.float32)
    rstd = torch.empty((M,), device=device, dtype=torch.float32)
    grid = (M, 1, 1)
    layer_norm_persistent_kernel[grid](x, y, w, b, mean, rstd, M, N, eps)
    torch.cuda.synchronize()
    return {
        "in_ptr": x.cpu().numpy(),
        "weight_ptr": w.cpu().numpy(),
        "bias_ptr": b.cpu().numpy(),
        "out_ptr": y.cpu().numpy(),
        "out_mean_ptr": mean.cpu().numpy(),
        "out_rstd_ptr": rstd.cpu().numpy(),
        "eps": np.array(eps, dtype=np.float32),
    }


def _run_upsample_bicubic2d_aa_reference(case: TestCase) -> Dict[str, np.ndarray]:
    """
    Upsample bicubic2d AA: input/output are NCHW float32.
    This kernel is expected to be OUT_OF_SCOPE for current contract/IR, but we still
    run it to get real TTIR + baseline artifacts for future extensions.
    """
    from tritonops.upsample_bicubic2d_aa import upsample_bicubic2d_aa_kernel

    N = int(case.shapes.get("N", 1))
    C = int(case.shapes.get("C", 1))
    IH = int(case.shapes.get("IH", 4))
    IW = int(case.shapes.get("IW", 4))
    OH = int(case.shapes.get("OH", 4))
    OW = int(case.shapes.get("OW", 4))
    device = "cuda"
    # Use stable names ("Input"/"Output") to match typical LLM naming from this kernel.
    if case.inputs and "Input" in case.inputs:
        inp = torch.as_tensor(case.inputs["Input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (N, C, IH, IW):
            raise ValueError(f"Input shape mismatch: got {tuple(inp.shape)} expected {(N, C, IH, IW)}")
    else:
        inp = torch.randn((N, C, IH, IW), device=device, dtype=torch.float32)
    out = torch.empty((N, C, OH, OW), device=device, dtype=torch.float32)
    reciprocal_scale_h = float(IH) / float(OH) if OH > 0 else 1.0
    reciprocal_scale_w = float(IW) / float(OW) if OW > 0 else 1.0
    grid = lambda meta: (triton.cdiv(OW, meta["BLOCK_X"]), triton.cdiv(OH, meta["BLOCK_Y"]))
    upsample_bicubic2d_aa_kernel[grid](
        out,
        inp,
        N,
        C,
        OH,
        OW,
        IH,
        IW,
        reciprocal_scale_h,
        reciprocal_scale_w,
    )
    torch.cuda.synchronize()
    return {
        "Input": inp.cpu().numpy(),
        "Output": out.cpu().numpy(),
        # Expose scalars so LLM may reference them (as const values) without crashing the interpreter.
        "reciprocal_scale_h": np.array(reciprocal_scale_h, dtype=np.float32),
        "reciprocal_scale_w": np.array(reciprocal_scale_w, dtype=np.float32),
    }


def default_kernel_specs() -> List[KernelSpec]:
    def _norm_groupnorm(shapes: Dict[str, int]) -> Dict[str, int]:
        out = dict(shapes)
        if "C" not in out:
            return out
        c = int(out["C"])
        requested_g = int(out.get("num_groups", out.get("group", 1)))
        requested_g = max(1, min(requested_g, c))
        divisors: List[int] = []
        d = 1
        while d * d <= c:
            if c % d == 0:
                divisors.append(d)
                if d != c // d:
                    divisors.append(c // d)
            d += 1
        divisors = sorted(set(divisors))
        best_g = min(divisors, key=lambda x: (abs(x - requested_g), -x))
        out["num_groups"] = int(best_g)
        out.pop("group", None)
        out["group_size"] = c // int(best_g)
        return out

    return [
        KernelSpec(
            name="any_kernel_dim",
            module="tritonops.any",
            attr="any_kernel_dim.fn.src",
            runner=_run_any_reference,
            canonical_shapes={"M": 4, "N": 8},
            vary_axes=["M", "N"],
        ),
        KernelSpec(
            name="group_norm_kernel",
            module="tritonops.groupnorm",
            attr="group_norm_kernel.src",
            runner=_run_groupnorm_reference,
            canonical_shapes={"N": 2, "C": 4, "HW": 4, "num_groups": 2},
            vary_axes=["N", "C", "HW", "num_groups"],
            exclude_axes=["group_size"],
            normalize_shapes=_norm_groupnorm,
        ),
        KernelSpec(
            name="_attn_fwd",
            module="tritonops.attention",
            attr="_attn_fwd.fn.src",
            runner=_run_attention_reference,
            canonical_shapes={"Z": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
            vary_axes=["Q_CTX", "KV_CTX"],
            exclude_axes=["Z", "q_numhead", "kv_numhead", "HEAD_DIM"],
        ),
        KernelSpec(
            name="softmax_inner",
            module="tritonops.softmax",
            attr="softmax_kernel_inner.fn.src",
            runner=_run_softmax_reference,
            canonical_shapes={"M": 4, "N": 64},
            vary_axes=["M", "N"],
        ),
        KernelSpec(
            name="layer_norm_persistent",
            module="tritonops.layernorm",
            attr="layer_norm_persistent_kernel.fn.src",
            runner=_run_layernorm_reference,
            canonical_shapes={"M": 4, "N": 64},
            vary_axes=["M", "N"],
        ),
        KernelSpec(
            name="upsample_bicubic2d_aa",
            module="tritonops.upsample_bicubic2d_aa",
            attr="upsample_bicubic2d_aa_kernel.fn.src",
            runner=_run_upsample_bicubic2d_aa_reference,
            canonical_shapes={"N": 1, "C": 1, "IH": 4, "IW": 4, "OH": 4, "OW": 4},
            vary_axes=["IH", "IW", "OH", "OW"],
        ),
    ]


def run_pipeline_for_spec(spec: KernelSpec, *, out_dir: Path, cases_limit: int = 8) -> Dict[str, object]:
    """
    Run the full pipeline for a single kernel spec and write artifacts under out_dir.
    Returns the report dict (also written to JSON).
    """
    report: Dict[str, object] = {"kernel": spec.name}
    print(f"[{spec.name}] stage1: prepare triton dump/cache", flush=True)
    dump_dir, cache_dir = prepare_dump_and_cache_dirs(out_dir, spec.name, clean=True)
    report["triton_dump_dir"] = str(dump_dir)
    report["triton_cache_dir"] = str(cache_dir)

    # 1) Load source
    print(f"[{spec.name}] stage2: load triton source", flush=True)
    src = _get_src(spec.module, spec.attr)
    (out_dir / f"{spec.name}.triton_src.txt").write_text(src, encoding="utf-8")
    report["triton_src_path"] = str(out_dir / f"{spec.name}.triton_src.txt")

    # 2) LLM -> IntentIR
    print(f"[{spec.name}] stage3: LLM -> IntentIR (may take a while)", flush=True)
    feedback: List[str] = []
    cand: CandidateIntent | None = None
    cand_expanded: CandidateIntent | None = None
    for attempt in range(2):
        try:
            cand = llm_to_intent(src, spec.name, feedback=feedback)
            _ensure_schedule(cand.intent, kernel_name=spec.name, triton_src=src)
            enrich_intent_macros(cand.intent)
            intent_mlir = print_mlir_like(cand.intent)
            (out_dir / f"{spec.name}.intentir.mlir").write_text(intent_mlir, encoding="utf-8")
            expanded_intent = expand_macros(cand.intent)
            cand_expanded = CandidateIntent(
                intent=expanded_intent,
                problem_params=dict(cand.problem_params),
                schedule_params=dict(cand.schedule_params),
                raw_json=dict(cand.raw_json),
            )
            (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(
                print_mlir_like(expanded_intent), encoding="utf-8"
            )
            break
        except Exception as e:
            feedback = [f"Previous failure: {e}"]
            cand = None
            cand_expanded = None
            continue
    if cand is None:
        raise RuntimeError(f"LLM/Intent parse failed after retries for {spec.name}: {'; '.join(feedback)}")
    # Ensure schedule is attached even if the LLM emits only partial schedule fields.
    _ensure_schedule(cand.intent, kernel_name=spec.name, triton_src=src)
    report["intent"] = cand.intent.to_json_dict()
    if cand_expanded is not None:
        _ensure_schedule(cand_expanded.intent, kernel_name=spec.name, triton_src=src)
    report["intent_expanded"] = (cand_expanded.intent.to_json_dict() if cand_expanded is not None else None)

    # 3) Launch kernel once to dump TTIR
    print(f"[{spec.name}] stage4: launch triton once (dump TTIR + baseline)", flush=True)
    baseline_shapes = dict(spec.canonical_shapes)
    if spec.normalize_shapes:
        baseline_shapes = spec.normalize_shapes(baseline_shapes)
    baseline_case = TestCase(shapes=baseline_shapes, dtypes={}, seed=0)
    baseline_io = spec.runner(baseline_case)
    # Add non-destructive IO name aliases so downstream stages can refer to either
    # Input/Output or input/output (and *_ptr variants) without breaking.
    try:
        from verify.diff_runner import _with_io_aliases as _with_io_aliases_for_diff

        baseline_io = _with_io_aliases_for_diff(cand.intent, baseline_io)
    except Exception:
        pass
    # Cache baseline IO (inputs + outputs) as an .npz artifact so downstream stages
    # (e.g., Task6 remote run) can reuse it without re-launching Triton.
    try:
        total_bytes = 0
        for v in baseline_io.values():
            arr = np.asarray(v)
            total_bytes += int(arr.size) * int(arr.dtype.itemsize)
        # Avoid writing huge blobs (e.g., attention) into artifacts by default.
        if total_bytes <= 16 * 1024 * 1024:
            npz_path = out_dir / f"{spec.name}.baseline.npz"
            np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in baseline_io.items()})
            report["baseline"] = {
                "shapes": dict(baseline_case.shapes),
                "seed": int(baseline_case.seed),
                "npz_path": str(npz_path),
                "keys": sorted(list(baseline_io.keys())),
                "bytes": int(total_bytes),
            }
        else:
            report["baseline"] = {
                "shapes": dict(baseline_case.shapes),
                "seed": int(baseline_case.seed),
                "npz_path": None,
                "keys": sorted(list(baseline_io.keys())),
                "bytes": int(total_bytes),
                "skipped": "baseline too large to cache (over 16MB)",
            }
    except Exception as e:
        report["baseline"] = {"shapes": dict(baseline_case.shapes), "seed": int(baseline_case.seed), "error": str(e)}
    ttir_path = find_latest_ttir(dump_dir, spec.name)
    constraints = None
    cert = None
    contract = None
    sv = None
    if ttir_path and ttir_path.exists():
        print(f"[{spec.name}] stage5: Task4 facts/contract/certificate + static validation", flush=True)
        ttir_text = ttir_path.read_text()
        ttir_copy = out_dir / f"{spec.name}.ttir"
        ttir_copy.write_text(ttir_text, encoding="utf-8")
        facts = extract_facts(ttir_text)
        constraints = extract_constraints(ttir_text, facts=facts)
        contract = evaluate_contract(facts)
        cert = build_certificate(ttir_text, facts=facts)
        sv = static_validate((cand_expanded.intent if cand_expanded is not None else cand.intent), cert)
        report["ttir_path"] = str(ttir_path)
        report["contract"] = {"level": contract.level, "reasons": list(contract.reasons), "signals": dict(contract.signals)}
        report["certificate"] = cert.to_json_dict()
        report["static_validation"] = {
            "ok": bool(sv.ok),
            "reasons": list(sv.reasons),
            "obligations": [{"id": o.id, "status": o.status, "detail": o.detail} for o in sv.obligations],
        }
        if not sv.ok:
            # Try one more round with feedback
            try:
                cand = llm_to_intent(src, spec.name, feedback=sv.reasons)
                enrich_intent_macros(cand.intent)
                intent_mlir = print_mlir_like(cand.intent)
                (out_dir / f"{spec.name}.intentir.mlir").write_text(intent_mlir, encoding="utf-8")
                report["intent"] = cand.intent.to_json_dict()
                expanded_intent = expand_macros(cand.intent)
                cand_expanded = CandidateIntent(
                    intent=expanded_intent,
                    problem_params=dict(cand.problem_params),
                    schedule_params=dict(cand.schedule_params),
                    raw_json=dict(cand.raw_json),
                )
                (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(
                    print_mlir_like(expanded_intent), encoding="utf-8"
                )
                report["intent_expanded"] = expanded_intent.to_json_dict()
            except Exception:
                pass
    else:
        report["ttir_path"] = None

    # 4) Stage B diff
    print(f"[{spec.name}] stage6: Task5 cases + diff", flush=True)
    cand_for_run = cand_expanded or cand
    cases = make_cases(
        spec,
        cand_for_run,
        constraints,
        limit=cases_limit,
        extra_tile_hints=cert.tile_hints if cert else None,
        cert=cert,
    )
    report["cases"] = [dict(c.shapes) for c in cases]
    diffs, cex = run_diff(cand_for_run.intent, spec.runner, cases)
    if diffs:
        worst = max(diffs, key=lambda d: (not d.ok, d.max_abs_err))
        report["diff"] = {
            "ok": bool(all(d.ok for d in diffs)),
            "worst": {"summary": worst.summary, "max_abs": float(worst.max_abs_err), "max_rel": float(worst.max_rel_err)},
            "results": [
                {
                    "case_shapes": dict(cases[i].shapes),
                    "ok": bool(diffs[i].ok),
                    "summary": diffs[i].summary,
                    "max_abs": float(diffs[i].max_abs_err),
                    "max_rel": float(diffs[i].max_rel_err),
                }
                for i in range(min(len(cases), len(diffs)))
            ],
        }
    else:
        report["diff"] = {"ok": False, "error": "no diff results"}
    if cex:
        report["counterexamples"] = [
            {"shapes": dict(cx.case.shapes), "summary": cx.diff.summary, "hints": list(cx.hints)} for cx in cex[:3]
        ]

    # If dynamic diff fails, do one bounded LLM repair round using concrete feedback.
    # This is deliberately conservative (1 retry) to respect LLM rate limits.
    if diffs and not all(d.ok for d in diffs):
        worst_summary = (report.get("diff") or {}).get("worst", {}).get("summary")
        ce0 = (report.get("counterexamples") or [{}])[0]
        feedback3: List[str] = []
        if spec.name == "group_norm_kernel":
            feedback3 += [
                "Your groupnorm math is wrong: reduce_sum returns SUM. You must divide by num_elements=group_size*HW for mean and var.",
                "Implement mean = reduce_sum(X, dims=[2,3], keepdims=true, scale='1.0/(group_size*HW)').",
                "Implement var = reduce_sum((X-mean)^2, dims=[2,3], keepdims=true, scale='1.0/(group_size*HW)').",
                "Then rstd = rsqrt(var + eps).",
            ]
        if spec.name == "layer_norm_persistent":
            feedback3 += [
                "Your layernorm math is wrong: reduce_sum returns SUM. You must divide by N for mean/var.",
            ]
        if worst_summary:
            feedback3.append(f"Observed diff failure: {worst_summary}")
        if ce0.get("shapes"):
            feedback3.append(f"Counterexample shapes: {ce0.get('shapes')}")

        if feedback3:
            try:
                cand_fix = llm_to_intent(src, spec.name, feedback=feedback3)
                _ensure_schedule(cand_fix.intent, kernel_name=spec.name, triton_src=src)
                (out_dir / f"{spec.name}.intentir.mlir").write_text(print_mlir_like(cand_fix.intent), encoding="utf-8")
                expanded_fix = expand_macros(cand_fix.intent)
                (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(print_mlir_like(expanded_fix), encoding="utf-8")
                report["intent"] = cand_fix.intent.to_json_dict()
                report["intent_expanded"] = expanded_fix.to_json_dict()
                cand_for_run = CandidateIntent(
                    intent=expanded_fix,
                    problem_params=dict(cand_fix.problem_params),
                    schedule_params=dict(cand_fix.schedule_params),
                    raw_json=dict(cand_fix.raw_json),
                )
                # Re-run diff with a small case set to confirm the repair.
                cases_fix = make_cases(
                    spec,
                    cand_for_run,
                    constraints,
                    limit=min(4, cases_limit),
                    extra_tile_hints=cert.tile_hints if cert else None,
                    cert=cert,
                )
                report["cases"] = [dict(c.shapes) for c in cases_fix]
                diffs_fix, cex_fix = run_diff(cand_for_run.intent, spec.runner, cases_fix)
                diffs, cex = diffs_fix, cex_fix
                if diffs:
                    worst = max(diffs, key=lambda d: (not d.ok, d.max_abs_err))
                    report["diff"] = {
                        "ok": bool(all(d.ok for d in diffs)),
                        "worst": {"summary": worst.summary, "max_abs": float(worst.max_abs_err), "max_rel": float(worst.max_rel_err)},
                        "results": [
                            {
                                "case_shapes": dict(cases_fix[i].shapes),
                                "ok": bool(diffs[i].ok),
                                "summary": diffs[i].summary,
                                "max_abs": float(diffs[i].max_abs_err),
                                "max_rel": float(diffs[i].max_rel_err),
                            }
                            for i in range(min(len(cases_fix), len(diffs)))
                        ],
                    }
                if cex:
                    report["counterexamples"] = [
                        {"shapes": dict(cx.case.shapes), "summary": cx.diff.summary, "hints": list(cx.hints)} for cx in cex[:3]
                    ]
            except Exception:
                pass

    # 5) Stage C + mutation-kill if Stage B passed
    if diffs and all(d.ok for d in diffs):
        meta = run_metamorphic_suite(spec.name, cand_for_run.intent, spec.runner, base_case=cases[0])
        bounded = run_bounded_exhaustive(spec.name, cand_for_run.intent, spec.runner)
        report["stage_c"] = {
            "metamorphic": {
                "ok": bool(meta.ok),
                "results": [{"relation": r.relation, "ok": bool(r.ok), "detail": r.detail} for r in meta.results],
            },
            "bounded_exhaustive": {
                "ok": bool(bounded.ok),
                "checked": int(bounded.checked),
                "total": int(bounded.total),
                "detail": bounded.detail,
                "first_failure": (dict(bounded.first_failure_case.shapes) if bounded.first_failure_case else None),
                "first_failure_summary": bounded.first_failure_summary,
            },
        }
        if cert is not None:
            mut = run_mutation_kill(
                spec.name,
                intent=cand_for_run.intent,
                cert=cert,
                run_ref_fn=spec.runner,
                diff_cases=cases[:2],
                metamorphic_base_case=cases[0],
                n_mutants=16,
                seed=0,
            )
            report["mutation_kill"] = {
                "kill_rate": float(mut.kill_rate),
                "total": int(mut.total),
                "killed": int(mut.killed),
                "survived": int(mut.survived),
                "killed_by_stage": dict(mut.killed_by_stage),
                "outcomes": [{"mutant_id": o.mutant_id, "killed_by": o.killed_by, "detail": o.detail, "diff_summary": o.diff_summary} for o in mut.outcomes],
            }
    return report


__all__ = [
    "KernelSpec",
    "default_kernel_specs",
    "run_pipeline_for_spec",
    "prepare_dump_and_cache_dirs",
    "find_latest_ttir",
    "llm_to_intent",
    "make_cases",
]
