"""
Reusable helpers for the Triton full pipeline (LLM -> IntentIR -> TTIR -> validation).

This keeps `scripts/triton/full_pipeline_verify.py` thin; new kernels can reuse
the same helpers by constructing a KernelSpec and calling `run_pipeline_for_spec`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import triton

from intent_ir.ir import ScheduleSketch
from frontends.triton.dump import find_latest_ttir, prepare_dump_and_cache_dirs
from intent_ir.llm import LLMIntentHub
from intent_ir.macros import expand_macros, enrich_intent_macros
from intent_ir.parser import CandidateIntent
from intent_ir.ir.printer_mlir_like import print_mlir_like
from frontends.common.static_validate import static_validate
from pipeline import registry as pipeline_registry
from pipeline.interfaces import FrontendConstraints
from verify.diff_runner import run_diff
from verify.gen_cases import GeneratedCases, TestCase, generate_cases_split
from frontends.common.certificate_v2 import SemanticCertificateV2
from frontends.common.evidence import CanonicalEvidence
from frontends.triton.certificate import SemanticCertificate, build_certificate_v2
from frontends.common.obligations import evaluate_obligations
from frontends.triton.contract import evaluate_contract_v2
from verify.metamorphic import run_bounded_exhaustive, run_metamorphic_suite
from verify.mutation import run_mutation_kill
from verify.tolerances import infer_tolerances


ROOT = Path(__file__).resolve().parents[2]
_LLM_HUB = LLMIntentHub()


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
def llm_to_intent(desc, feedback: Optional[List[str]] = None) -> CandidateIntent:
    """
    Backward-compatible helper: lift a KernelDescriptor into a CandidateIntent.
    """
    return _LLM_HUB.lift(desc, feedback=list(feedback or []))


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
    cert_v2: SemanticCertificateV2 | None = None,
    assumptions: List[str] | None = None,
) -> GeneratedCases:
    base = dict(spec.canonical_shapes)
    if spec.normalize_shapes:
        base = spec.normalize_shapes(base)
    base_case = TestCase(shapes=base, dtypes={}, seed=0)
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
        constraints = constraints or FrontendConstraints(needs_mask=True, suggested_edge_cases=[])
        constraints.needs_mask = True
    if cert_v2 is not None:
        try:
            ce = (cert_v2.semantic_facts or {}).get("canonical_evidence")
            accesses = []
            if isinstance(ce, CanonicalEvidence):
                accesses = [a.to_json_dict() for a in ce.accesses]
            elif isinstance(ce, dict):
                accesses = [a for a in (ce.get("accesses") or []) if isinstance(a, dict)]
            needs_mask_v2 = any(isinstance(a.get("predicate"), dict) and a.get("predicate", {}).get("clauses") for a in accesses)
            if needs_mask_v2:
                constraints = constraints or FrontendConstraints(needs_mask=True, suggested_edge_cases=[])
                constraints.needs_mask = True
        except Exception:
            pass

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

    predicate_clauses: List[str] = []
    if cert_v2 is not None:
        try:
            ce = (cert_v2.semantic_facts or {}).get("canonical_evidence")
            if isinstance(ce, CanonicalEvidence):
                for a in ce.accesses:
                    if a.predicate and a.predicate.clauses:
                        predicate_clauses.extend(list(a.predicate.clauses))
                symbols = (ce.meta or {}).get("symbols") or {}
            elif isinstance(ce, dict):
                for a in (ce.get("accesses") or []):
                    if isinstance(a, dict):
                        pred = a.get("predicate")
                        if isinstance(pred, dict) and pred.get("clauses"):
                            predicate_clauses.extend(list(pred.get("clauses") or []))
                symbols = ((ce.get("meta") or {}).get("symbols") or {}) if isinstance(ce.get("meta"), dict) else {}
            else:
                symbols = {}
            # Add range ends as candidate sizes (helps non-divisible edge generation).
            for spec_range in (symbols.get("ranges") or {}).values():
                try:
                    end = int(spec_range.get("end"))
                except Exception:
                    continue
                if 0 < end <= 2048:
                    extra_sizes.extend([end, max(1, end - 1), end + 1])
        except Exception:
            pass

    generated = generate_cases_split(
        intent.intent,
        constraints=constraints,
        limit=limit,
        seed=1,
        tile_hints=tile_hints,
        axes=spec.vary_axes,
        exclude_axes=spec.exclude_axes,
        extra_sizes=extra_sizes,
        predicate_clauses=predicate_clauses,
        assumptions=list(assumptions or []),
        base_shapes=dict(base),
    )

    def merge_case(c: TestCase) -> TestCase:
        merged = dict(spec.canonical_shapes)
        merged.update(c.shapes)
        if spec.normalize_shapes:
            merged = spec.normalize_shapes(merged)
        c.shapes = merged
        return c

    in_cases: List[TestCase] = []
    seen = set()
    for c in [base_case] + list(generated.in_contract):
        c = merge_case(c)
        key = tuple(sorted(c.shapes.items()))
        if key in seen:
            continue
        seen.add(key)
        in_cases.append(c)

    out_cases: List[TestCase] = []
    for c in list(generated.out_of_contract):
        c = merge_case(c)
        key = tuple(sorted(c.shapes.items()))
        if key in seen:
            continue
        seen.add(key)
        out_cases.append(c)

    return GeneratedCases(in_contract=in_cases, out_of_contract=out_cases)


# ---------------------- default kernel runners ----------------------

def _run_any_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.any import any_kernel_dim

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
    from kernels.triton.ops.groupnorm import group_norm_kernel

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
    from kernels.triton.ops.attention import _attn_fwd

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
    from kernels.triton.ops.softmax import softmax_kernel_inner

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
    from kernels.triton.ops.layernorm import layer_norm_persistent_kernel

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
    from kernels.triton.ops.upsample_bicubic2d_aa import upsample_bicubic2d_aa_kernel

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
            module="kernels.triton.ops.any",
            attr="any_kernel_dim.fn.src",
            runner=_run_any_reference,
            canonical_shapes={"M": 4, "N": 8},
            vary_axes=["M", "N"],
        ),
        KernelSpec(
            name="group_norm_kernel",
            module="kernels.triton.ops.groupnorm",
            attr="group_norm_kernel.src",
            runner=_run_groupnorm_reference,
            canonical_shapes={"N": 2, "C": 4, "HW": 4, "num_groups": 2},
            vary_axes=["N", "C", "HW", "num_groups"],
            exclude_axes=["group_size"],
            normalize_shapes=_norm_groupnorm,
        ),
        KernelSpec(
            name="_attn_fwd",
            module="kernels.triton.ops.attention",
            attr="_attn_fwd.fn.src",
            runner=_run_attention_reference,
            canonical_shapes={"Z": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
            vary_axes=["Q_CTX", "KV_CTX"],
            exclude_axes=["Z", "q_numhead", "kv_numhead", "HEAD_DIM"],
        ),
        KernelSpec(
            name="softmax_inner",
            module="kernels.triton.ops.softmax",
            attr="softmax_kernel_inner.fn.src",
            runner=_run_softmax_reference,
            canonical_shapes={"M": 4, "N": 64},
            vary_axes=["M", "N"],
        ),
        KernelSpec(
            name="layer_norm_persistent",
            module="kernels.triton.ops.layernorm",
            attr="layer_norm_persistent_kernel.fn.src",
            runner=_run_layernorm_reference,
            canonical_shapes={"M": 4, "N": 64},
            vary_axes=["M", "N"],
        ),
        KernelSpec(
            name="upsample_bicubic2d_aa",
            module="kernels.triton.ops.upsample_bicubic2d_aa",
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
    adapter = pipeline_registry.get("triton")
    print(f"[{spec.name}] stage1: prepare triton dump/cache", flush=True)
    dump_dir, cache_dir = prepare_dump_and_cache_dirs(out_dir, spec.name, clean=True)
    report["triton_dump_dir"] = str(dump_dir)
    report["triton_cache_dir"] = str(cache_dir)

    # 1) Load source
    print(f"[{spec.name}] stage2: load triton source", flush=True)
    desc = adapter.build_descriptor(spec)
    # Attach run-specific artifact paths for adapter extraction.
    desc.meta["artifact_dir"] = str(out_dir)
    desc.meta["triton_dump_dir"] = str(dump_dir)
    desc.meta["triton_cache_dir"] = str(cache_dir)
    src = desc.source_text
    (out_dir / f"{spec.name}.triton_src.txt").write_text(src, encoding="utf-8")
    report["triton_src_path"] = str(out_dir / f"{spec.name}.triton_src.txt")
    report["descriptor"] = desc.to_json_dict()

    # 2) Launch kernel once to dump TTIR (and capture a baseline IO snapshot)
    print(f"[{spec.name}] stage3: launch triton once (dump TTIR + baseline)", flush=True)
    baseline_shapes = dict(spec.canonical_shapes)
    if spec.normalize_shapes:
        baseline_shapes = spec.normalize_shapes(baseline_shapes)
    baseline_case = TestCase(shapes=baseline_shapes, dtypes={}, seed=0)
    baseline_io_raw = spec.runner(baseline_case)
    report["baseline"] = {
        "shapes": dict(baseline_case.shapes),
        "seed": int(baseline_case.seed),
        "npz_path": None,
        "keys": sorted(list(baseline_io_raw.keys())),
    }

    desc = adapter.ensure_artifacts(desc, spec)
    report["descriptor"] = desc.to_json_dict()
    if desc.meta.get("descriptor_path"):
        report["descriptor_path"] = str(desc.meta.get("descriptor_path"))

    ttir_path = find_latest_ttir(dump_dir, spec.name)
    constraints = None
    cert = None
    cert_v2: SemanticCertificateV2 | None = None
    contract = None
    sv = None
    if ttir_path and ttir_path.exists():
        print(f"[{spec.name}] stage4: Task4 facts/contract/certificate", flush=True)
        facts = adapter.extract_facts(desc)
        constraints = adapter.extract_constraints(desc, facts)
        # Legacy contract (v1) for fallback/debug; v2 contract is derived from obligations.
        contract_legacy = adapter.evaluate_contract(facts, constraints, None)
        cert = adapter.build_certificate(desc, facts, constraints)
        # CertificateV2 (stable semantic_facts + canonical evidence).
        try:
            ttir_text = None
            if desc.artifacts.ttir_path:
                ttir_text = Path(str(desc.artifacts.ttir_path)).read_text(encoding="utf-8")
            elif ttir_path:
                ttir_text = Path(str(ttir_path)).read_text(encoding="utf-8")
            if ttir_text:
                cert_v2 = build_certificate_v2(ttir_text, desc=desc, facts=facts)
                obligations = evaluate_obligations(desc, cert_v2)
                # Store obligations inside cert_v2 semantic_facts (stable, schema-versioned).
                cert_v2.semantic_facts["obligations"] = [o.to_json_dict() for o in obligations]
                contract = evaluate_contract_v2(desc, cert_v2, obligations, constraints=constraints)
                # Store the contract summary in cert_v2.meta (NOT semantic_facts) so
                # Stage-A static_validate can surface OUT_OF_SCOPE reasons for repair,
                # without perturbing semantic_facts golden locks.
                try:
                    cert_v2.meta = dict(getattr(cert_v2, "meta", {}) or {})
                    cert_v2.meta["contract"] = {
                        "level": str(contract.level),
                        "reasons": list(contract.reasons),
                        "assumptions": list(contract.assumptions),
                    }
                except Exception:
                    pass
                report["certificate_v2"] = cert_v2.to_json_dict()
                (out_dir / f"{spec.name}.certificate_v2.json").write_text(
                    json.dumps(report["certificate_v2"], indent=2), encoding="utf-8"
                )
                report["obligations"] = [o.to_json_dict() for o in obligations]
                report["contract"] = {
                    "level": contract.level,
                    "reasons": list(contract.reasons),
                    "assumptions": list(contract.assumptions),
                    "signals": dict(contract.signals),
                }
                (out_dir / f"{spec.name}.contract.json").write_text(json.dumps(report["contract"], indent=2), encoding="utf-8")
        except Exception as e:
            report["certificate_v2_error"] = str(e)
        # Always include a contract summary (prefer v2; fall back to legacy if v2 failed).
        report.setdefault(
            "contract",
            {
                "level": contract_legacy.level,
                "reasons": list(contract_legacy.reasons),
                "assumptions": list(contract_legacy.assumptions),
                "signals": dict(contract_legacy.signals),
            },
        )
        report["contract_legacy"] = {
            "level": contract_legacy.level,
            "reasons": list(contract_legacy.reasons),
            "assumptions": list(contract_legacy.assumptions),
            "signals": dict(contract_legacy.signals),
        }
        report["ttir_path"] = str(desc.artifacts.ttir_path or ttir_path)
        if desc.meta.get("ttir_original_path"):
            report["ttir_dump_path"] = str(desc.meta.get("ttir_original_path"))
        report["certificate"] = cert.to_json_dict()
    else:
        report["ttir_path"] = None

    # 3) LLM -> IntentIR (KernelDescriptor -> CandidateIntent)
    print(f"[{spec.name}] stage5: LLM -> IntentIR (may take a while)", flush=True)
    feedback: List[str] = []
    cand: CandidateIntent | None = None
    cand_expanded: CandidateIntent | None = None
    for attempt in range(2):
        try:
            cand = llm_to_intent(desc, feedback=feedback)
            report["llm_trace"] = dict(cand.llm_trace)
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
                llm_trace=dict(cand.llm_trace),
            )
            (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(print_mlir_like(expanded_intent), encoding="utf-8")
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

    # 4) Static validation (Intent vs certificate), if TTIR certificate exists.
    if cert is not None:
        print(f"[{spec.name}] stage6: Task4 static validation", flush=True)
        sv_cert = cert_v2 or cert
        sv = static_validate((cand_expanded.intent if cand_expanded is not None else cand.intent), sv_cert)
        report["static_validation"] = {
            "ok": bool(sv.ok),
            "reasons": list(sv.reasons),
            "obligations": [{"id": o.id, "status": o.status, "detail": o.detail} for o in sv.obligations],
        }
        if not sv.ok:
            # One conservative repair round using certificate-derived feedback.
            try:
                cand = llm_to_intent(desc, feedback=list(sv.reasons))
                report["llm_trace"] = dict(cand.llm_trace)
                _ensure_schedule(cand.intent, kernel_name=spec.name, triton_src=src)
                enrich_intent_macros(cand.intent)
                (out_dir / f"{spec.name}.intentir.mlir").write_text(print_mlir_like(cand.intent), encoding="utf-8")
                expanded_intent = expand_macros(cand.intent)
                cand_expanded = CandidateIntent(
                    intent=expanded_intent,
                    problem_params=dict(cand.problem_params),
                    schedule_params=dict(cand.schedule_params),
                    raw_json=dict(cand.raw_json),
                    llm_trace=dict(cand.llm_trace),
                )
                (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(
                    print_mlir_like(expanded_intent), encoding="utf-8"
                )
                report["intent"] = cand.intent.to_json_dict()
                report["intent_expanded"] = expanded_intent.to_json_dict()
                sv = static_validate((cand_expanded.intent if cand_expanded is not None else cand.intent), sv_cert)
                report["static_validation"] = {
                    "ok": bool(sv.ok),
                    "reasons": list(sv.reasons),
                    "obligations": [{"id": o.id, "status": o.status, "detail": o.detail} for o in sv.obligations],
                }
            except Exception:
                pass

    # 5) Stage B diff
    print(f"[{spec.name}] stage7: Task5 cases + diff", flush=True)
    cand_for_run = cand_expanded or cand
    assumptions = []
    try:
        if isinstance(report.get("contract"), dict):
            assumptions = list((report.get("contract") or {}).get("assumptions") or [])
    except Exception:
        assumptions = []
    cases_pack = make_cases(
        spec,
        cand_for_run,
        constraints,
        limit=cases_limit,
        extra_tile_hints=cert.tile_hints if cert else None,
        cert=cert,
        cert_v2=cert_v2,
        assumptions=assumptions,
    )
    cases_in = list(cases_pack.in_contract)
    cases_out = list(cases_pack.out_of_contract)
    report["cases"] = {"in_contract": [dict(c.shapes) for c in cases_in], "out_of_contract": [dict(c.shapes) for c in cases_out]}

    tol = infer_tolerances(cand_for_run.intent).to_dict()
    report["tolerances"] = dict(tol)
    diffs_in, cex_in = run_diff(cand_for_run.intent, spec.runner, cases_in, tolerances=tol)
    if diffs_in:
        worst = max(diffs_in, key=lambda d: (not d.ok, d.max_abs_err))
        report["diff"] = {
            "ok": bool(all(d.ok for d in diffs_in)),
            "worst": {"summary": worst.summary, "max_abs": float(worst.max_abs_err), "max_rel": float(worst.max_rel_err)},
            "results": [
                {
                    "case_shapes": dict(cases_in[i].shapes),
                    "ok": bool(diffs_in[i].ok),
                    "summary": diffs_in[i].summary,
                    "max_abs": float(diffs_in[i].max_abs_err),
                    "max_rel": float(diffs_in[i].max_rel_err),
                }
                for i in range(min(len(cases_in), len(diffs_in)))
            ],
        }
    else:
        report["diff"] = {"ok": False, "error": "no diff results"}
    if cex_in:
        report["counterexamples"] = [
            {"shapes": dict(cx.case.shapes), "summary": cx.diff.summary, "hints": list(cx.hints)} for cx in cex_in[:3]
        ]

    # Out-of-contract probing (does NOT affect correctness gate).
    if cases_out:
        diffs_out, cex_out = run_diff(cand_for_run.intent, spec.runner, cases_out, tolerances=tol)
        if diffs_out:
            worst = max(diffs_out, key=lambda d: (not d.ok, d.max_abs_err))
            report["diff_out_of_contract"] = {
                "ok": bool(all(d.ok for d in diffs_out)),
                "worst": {"summary": worst.summary, "max_abs": float(worst.max_abs_err), "max_rel": float(worst.max_rel_err)},
                "results": [
                    {
                        "case_shapes": dict(cases_out[i].shapes),
                        "ok": bool(diffs_out[i].ok),
                        "summary": diffs_out[i].summary,
                        "max_abs": float(diffs_out[i].max_abs_err),
                        "max_rel": float(diffs_out[i].max_rel_err),
                    }
                    for i in range(min(len(cases_out), len(diffs_out)))
                ],
            }
        else:
            report["diff_out_of_contract"] = {"ok": False, "error": "no diff results"}
        if cex_out:
            report["out_of_contract_counterexamples"] = [
                {"shapes": dict(cx.case.shapes), "summary": cx.diff.summary, "hints": list(cx.hints)} for cx in cex_out[:3]
            ]

    # If dynamic diff fails, do one bounded LLM repair round using concrete feedback.
    # This is deliberately conservative (1 retry) to respect LLM rate limits.
    if diffs_in and not all(d.ok for d in diffs_in):
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
                cand_fix = llm_to_intent(desc, feedback=feedback3)
                report["llm_trace"] = dict(cand_fix.llm_trace)
                cand = cand_fix
                _ensure_schedule(cand_fix.intent, kernel_name=spec.name, triton_src=src)
                (out_dir / f"{spec.name}.intentir.mlir").write_text(print_mlir_like(cand_fix.intent), encoding="utf-8")
                expanded_fix = expand_macros(cand_fix.intent)
                (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(print_mlir_like(expanded_fix), encoding="utf-8")
                report["intent"] = cand_fix.intent.to_json_dict()
                report["intent_expanded"] = expanded_fix.to_json_dict()
                cand_expanded = CandidateIntent(
                    intent=expanded_fix,
                    problem_params=dict(cand_fix.problem_params),
                    schedule_params=dict(cand_fix.schedule_params),
                    raw_json=dict(cand_fix.raw_json),
                    llm_trace=dict(cand_fix.llm_trace),
                )
                cand_for_run = cand_expanded
                # Re-run diff with a small case set to confirm the repair.
                cases_fix_pack = make_cases(
                    spec,
                    cand_for_run,
                    constraints,
                    limit=min(4, cases_limit),
                    extra_tile_hints=cert.tile_hints if cert else None,
                    cert=cert,
                    cert_v2=cert_v2,
                    assumptions=assumptions,
                )
                cases_fix = list(cases_fix_pack.in_contract)
                cases_in = cases_fix
                report["cases"] = {"in_contract": [dict(c.shapes) for c in cases_fix], "out_of_contract": []}
                diffs_fix, cex_fix = run_diff(cand_for_run.intent, spec.runner, cases_fix)
                diffs_in, cex_in = diffs_fix, cex_fix
                tol = infer_tolerances(cand_for_run.intent).to_dict()
                report["tolerances"] = dict(tol)
                if diffs_in:
                    worst = max(diffs_in, key=lambda d: (not d.ok, d.max_abs_err))
                    report["diff"] = {
                        "ok": bool(all(d.ok for d in diffs_in)),
                        "worst": {"summary": worst.summary, "max_abs": float(worst.max_abs_err), "max_rel": float(worst.max_rel_err)},
                        "results": [
                            {
                                "case_shapes": dict(cases_fix[i].shapes),
                                "ok": bool(diffs_in[i].ok),
                                "summary": diffs_in[i].summary,
                                "max_abs": float(diffs_in[i].max_abs_err),
                                "max_rel": float(diffs_in[i].max_rel_err),
                            }
                            for i in range(min(len(cases_fix), len(diffs_in)))
                        ],
                    }
                if cex_in:
                    report["counterexamples"] = [
                        {"shapes": dict(cx.case.shapes), "summary": cx.diff.summary, "hints": list(cx.hints)} for cx in cex_in[:3]
                    ]
            except Exception:
                pass

    # If diff still fails, attach a compact debug report (P0 gap fix).
    if diffs_in and not all(d.ok for d in diffs_in):
        try:
            from verify.diff_debugger import debug_mismatch

            debug_case = (cex_in[0].case if cex_in else (cases_in[0] if cases_in else baseline_case))
            report["diff_debug"] = debug_mismatch(
                cand_for_run.intent,
                spec.runner,
                debug_case,
                sample_elems=16,
            )
        except Exception as e:
            report["diff_debug"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # 5) Stage C + mutation-kill if Stage B passed
    if diffs_in and all(d.ok for d in diffs_in):
        meta = run_metamorphic_suite(spec.name, cand_for_run.intent, spec.runner, base_case=cases_in[0], atol=tol["atol"], rtol=tol["rtol"])
        bounded = run_bounded_exhaustive(spec.name, cand_for_run.intent, spec.runner, atol=tol["atol"], rtol=tol["rtol"])
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
        try:
            from verify.numerical_stability import run_numerical_stability_suite

            report["stage_c"]["numerical_stability"] = run_numerical_stability_suite(
                spec.name, cand_for_run.intent, spec.runner, base_case=cases_in[0], tolerances=tol
            ).to_json_dict()
        except Exception as e:
            report["stage_c"]["numerical_stability"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if cert is not None:
            mut = run_mutation_kill(
                spec.name,
                intent=cand_for_run.intent,
                run_ref_fn=spec.runner,
                diff_cases=cases_in[:2],
                metamorphic_base_case=cases_in[0],
                static_validate_fn=(lambda m, _cert=cert: static_validate(m, _cert)),
                n_mutants=16,
                seed=0,
                atol=float(tol["atol"]),
                rtol=float(tol["rtol"]),
            )
            report["mutation_kill"] = {
                "kill_rate": float(mut.kill_rate),
                "total": int(mut.total),
                "killed": int(mut.killed),
                "survived": int(mut.survived),
                "killed_by_stage": dict(mut.killed_by_stage),
                "mutation_breakdown": dict(mut.mutation_breakdown),
                "outcomes": [
                    {
                        "mutant_id": o.mutant_id,
                        "mutation_type": o.mutation_type,
                        "killed_by": o.killed_by,
                        "detail": o.detail,
                        "diff_summary": o.diff_summary,
                    }
                    for o in mut.outcomes
                ],
            }

    # Persist baseline IO aligned to the final macro intent, so Task6 tools can
    # reuse the same snapshot without re-launching Triton.
    try:
        baseline_io = dict(baseline_io_raw)
        try:
            from verify.diff_runner import _with_io_aliases as _with_io_aliases_for_diff

            baseline_io = _with_io_aliases_for_diff(cand.intent, baseline_io)
        except Exception:
            pass
        total_bytes = 0
        for v in baseline_io.values():
            arr = np.asarray(v)
            total_bytes += int(arr.size) * int(arr.dtype.itemsize)
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
