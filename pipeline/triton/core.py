"""
Reusable helpers for the Triton full pipeline (LLM -> IntentIR -> TTIR -> validation).

This keeps `scripts/triton/full_pipeline_verify.py` thin; new kernels can reuse
the same helpers by constructing a KernelSpec and calling `run_pipeline_for_spec`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
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
from frontends.common.obligations import O3_MASK_IMPLIES_INBOUNDS, evaluate_obligations
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
    # Compile-time tile/constexpr values used by the reference launcher.
    # Used by paper experiments (freeze-tile baseline) to reuse frontend tiling.
    constexpr: Dict[str, int] = field(default_factory=dict)
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


def _attach_access_witness_meta(intent, *, cert_v2: SemanticCertificateV2 | None, canonical_shapes: Dict[str, int] | None) -> None:
    """
    Attach a compact access witness summary to `intent.meta`.

    This is derived from CertificateV2 CanonicalEvidence (not raw source code),
    so it is safe to consume for evidence-guided tuning/codegen.
    """
    if cert_v2 is None:
        return
    try:
        from frontends.common.access_witness import build_stride_summary  # noqa: PLC0415

        shape_bindings: Dict[str, int] = {}
        for k, v in (canonical_shapes or {}).items():
            if isinstance(k, str) and isinstance(v, (int, float)):
                shape_bindings[str(k)] = int(v)

        ss = build_stride_summary(cert_v2.to_json_dict(), shape_bindings=shape_bindings)
        if ss is None:
            return
        j = ss.to_json_dict()
        tp = j.get("tensor_penalty") if isinstance(j.get("tensor_penalty"), dict) else {}
        top = sorted(((str(k), float(v)) for k, v in tp.items()), key=lambda kv: kv[1], reverse=True)[:8]
        meta = dict(getattr(intent, "meta", {}) or {})
        # Contract V2 summary (obligation-driven) can drive safe fastpath decisions.
        try:
            contract = (getattr(cert_v2, "meta", {}) or {}).get("contract")
            if isinstance(contract, dict):
                meta["contract_v2"] = {
                    "level": str(contract.get("level") or ""),
                    "reasons": list(contract.get("reasons") or []) if isinstance(contract.get("reasons"), list) else [],
                    "assumptions": list(contract.get("assumptions") or []) if isinstance(contract.get("assumptions"), list) else [],
                }
        except Exception:
            pass
        # Schedule-hint V2: drift-allowed tuning priors (tile_hints, symbol domains).
        try:
            sh = getattr(cert_v2, "schedule_hints", {}) or {}
            if isinstance(sh, dict):
                meta["schedule_hints_v2"] = {
                    "tile_hints": list(sh.get("tile_hints") or []) if isinstance(sh.get("tile_hints"), list) else [],
                    "symbol_ranges": dict(sh.get("symbol_ranges") or {}) if isinstance(sh.get("symbol_ranges"), dict) else {},
                }
        except Exception:
            pass
        meta["access_witness"] = {
            "dominant_axis": j.get("dominant_axis"),
            "dominant_range": j.get("dominant_range"),
            "dominant_range_len": j.get("dominant_range_len"),
            "has_contiguous_range": bool(j.get("has_contiguous_range")),
            "tensor_penalty_top": top,
            "notes": list(j.get("notes") or []) if isinstance(j.get("notes"), list) else [],
        }
        intent.meta = meta
    except Exception:
        return


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

    counterexample_models: List[Dict[str, int]] = []
    if cert_v2 is not None:
        try:
            obs = (cert_v2.semantic_facts or {}).get("obligations")
            if isinstance(obs, list):
                for o in obs:
                    if not isinstance(o, dict):
                        continue
                    if o.get("id") != O3_MASK_IMPLIES_INBOUNDS:
                        continue
                    wit = o.get("witness") if isinstance(o.get("witness"), dict) else {}
                    for ac in (wit.get("access_checks") or []):
                        if not isinstance(ac, dict):
                            continue
                        for d in (ac.get("dims") or []):
                            if not isinstance(d, dict):
                                continue
                            cx = d.get("counterexample")
                            if not isinstance(cx, dict):
                                continue
                            assigns = cx.get("assignments")
                            if not isinstance(assigns, dict) or not assigns:
                                continue
                            model: Dict[str, int] = {}
                            for k, v in assigns.items():
                                if isinstance(k, str) and isinstance(v, (int, float)):
                                    model[str(k)] = int(v)
                            if model:
                                counterexample_models.append(model)
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
        counterexample_models=counterexample_models,
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


def _run_add2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.add2d import add2d_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    device = "cuda"
    if case.inputs and "A" in case.inputs:
        a = torch.as_tensor(case.inputs["A"], device=device).to(torch.float32)
        if tuple(a.shape) != (M, N):
            raise ValueError(f"A shape mismatch: got {tuple(a.shape)} expected {(M, N)}")
    else:
        a = torch.randn((M, N), device=device, dtype=torch.float32)
    if case.inputs and "B" in case.inputs:
        b = torch.as_tensor(case.inputs["B"], device=device).to(torch.float32)
        if tuple(b.shape) != (M, N):
            raise ValueError(f"B shape mismatch: got {tuple(b.shape)} expected {(M, N)}")
    else:
        b = torch.randn((M, N), device=device, dtype=torch.float32)
    c = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))
    add2d_kernel[grid](a, b, c, M, N, BLOCK_N=256)
    torch.cuda.synchronize()
    return {"A": a.cpu().numpy(), "B": b.cpu().numpy(), "C": c.cpu().numpy()}


def _run_transpose2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.transpose2d import transpose2d_kernel

    M = int(case.shapes.get("M", 16))
    N = int(case.shapes.get("N", 16))
    device = "cuda"
    if case.inputs and "inp" in case.inputs:
        inp = torch.as_tensor(case.inputs["inp"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"inp shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    out = torch.empty((N, M), device=device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    transpose2d_kernel[grid](inp, out, M, N, BLOCK_M=32, BLOCK_N=32)
    torch.cuda.synchronize()
    return {"inp": inp.cpu().numpy(), "out": out.cpu().numpy()}


def _run_relu2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.relu2d import relu2d_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    device = "cuda"
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    out = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))
    relu2d_kernel[grid](inp, out, M, N, BLOCK_N=256)
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "output": out.cpu().numpy()}


def _run_add_bias2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.add_bias2d import add_bias2d_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    device = "cuda"
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    if case.inputs and "bias" in case.inputs:
        bias = torch.as_tensor(case.inputs["bias"], device=device).to(torch.float32)
        if tuple(bias.shape) != (N,):
            raise ValueError(f"bias shape mismatch: got {tuple(bias.shape)} expected {(N,)}")
    else:
        bias = torch.randn((N,), device=device, dtype=torch.float32)
    out = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))
    add_bias2d_kernel[grid](inp, bias, out, M, N, BLOCK_N=256)
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "bias": bias.cpu().numpy(), "output": out.cpu().numpy()}


def _run_where2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.where2d import where2d_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    device = "cuda"
    if case.inputs and "A" in case.inputs:
        a = torch.as_tensor(case.inputs["A"], device=device).to(torch.float32)
        if tuple(a.shape) != (M, N):
            raise ValueError(f"A shape mismatch: got {tuple(a.shape)} expected {(M, N)}")
    else:
        a = torch.randn((M, N), device=device, dtype=torch.float32)
    if case.inputs and "B" in case.inputs:
        b = torch.as_tensor(case.inputs["B"], device=device).to(torch.float32)
        if tuple(b.shape) != (M, N):
            raise ValueError(f"B shape mismatch: got {tuple(b.shape)} expected {(M, N)}")
    else:
        b = torch.randn((M, N), device=device, dtype=torch.float32)
    c = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))
    where2d_kernel[grid](a, b, c, M, N, BLOCK_N=256)
    torch.cuda.synchronize()
    return {"A": a.cpu().numpy(), "B": b.cpu().numpy(), "C": c.cpu().numpy()}


def _run_row_sum_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.row_sum import row_sum_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 256))
    device = "cuda"
    if N > 1024:
        raise ValueError(f"row_sum requires N<=1024, got N={N}")
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    out = torch.empty((M,), device=device, dtype=torch.float32)
    block_n = 1 if N <= 1 else min(1024, 1 << (int(N) - 1).bit_length())
    grid = (M,)
    row_sum_kernel[grid](inp, out, M, N, BLOCK_N=block_n)
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "output": out.cpu().numpy()}


def _run_exp2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.exp2d import exp2d_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    device = "cuda"
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    out = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))
    exp2d_kernel[grid](inp, out, M, N, BLOCK_N=256)
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "output": out.cpu().numpy()}


def _run_floor2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.floor2d import floor2d_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    device = "cuda"
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    out = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))
    floor2d_kernel[grid](inp, out, M, N, BLOCK_N=256)
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "output": out.cpu().numpy()}


def _run_clamp2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.clamp2d import clamp2d_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    device = "cuda"
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)

    lo = -0.5
    hi = 0.5
    if case.inputs and "lo" in case.inputs:
        lo = float(np.asarray(case.inputs["lo"]).reshape(()))
    if case.inputs and "hi" in case.inputs:
        hi = float(np.asarray(case.inputs["hi"]).reshape(()))

    out = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))
    clamp2d_kernel[grid](inp, out, M, N, float(lo), float(hi), BLOCK_N=256)
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "output": out.cpu().numpy(), "lo": np.array(lo, dtype=np.float32), "hi": np.array(hi, dtype=np.float32)}


def _run_row_max_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.row_max import row_max_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 256))
    device = "cuda"
    if N > 1024:
        raise ValueError(f"row_max requires N<=1024, got N={N}")
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    out = torch.empty((M,), device=device, dtype=torch.float32)
    block_n = 1 if N <= 1 else min(1024, 1 << (int(N) - 1).bit_length())
    grid = (M,)
    row_max_kernel[grid](inp, out, M, N, BLOCK_N=block_n)
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "output": out.cpu().numpy()}


def _run_copy2d_divmod_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.copy2d_divmod import copy2d_divmod_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    device = "cuda"
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    out = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = lambda meta: (M * triton.cdiv(N, meta["BLOCK_N"]),)
    copy2d_divmod_kernel[grid](inp, out, M, N, BLOCK_N=256)
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "output": out.cpu().numpy()}


def _run_matmul_relu2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.matmul_relu2d import matmul_relu2d_kernel

    M = int(case.shapes.get("M", 32))
    N = int(case.shapes.get("N", 32))
    K = int(case.shapes.get("K", 32))
    device = "cuda"

    if case.inputs and "A" in case.inputs:
        a = torch.as_tensor(case.inputs["A"], device=device).to(torch.float32)
        if tuple(a.shape) != (M, K):
            raise ValueError(f"A shape mismatch: got {tuple(a.shape)} expected {(M, K)}")
    else:
        a = torch.randn((M, K), device=device, dtype=torch.float32)
    if case.inputs and "B" in case.inputs:
        b = torch.as_tensor(case.inputs["B"], device=device).to(torch.float32)
        if tuple(b.shape) != (K, N):
            raise ValueError(f"B shape mismatch: got {tuple(b.shape)} expected {(K, N)}")
    else:
        b = torch.randn((K, N), device=device, dtype=torch.float32)
    c = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_relu2d_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=16,
    )
    torch.cuda.synchronize()
    return {"A": a.cpu().numpy(), "B": b.cpu().numpy(), "C": c.cpu().numpy()}


def _run_rms_norm2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.rms_norm2d import rms_norm2d_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    eps = float(case.shapes.get("eps", 1e-5))
    device = "cuda"
    if N > 1024:
        raise ValueError(f"rms_norm2d requires N<=1024, got N={N}")
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    if case.inputs and "weight" in case.inputs:
        w = torch.as_tensor(case.inputs["weight"], device=device).to(torch.float32)
        if tuple(w.shape) != (N,):
            raise ValueError(f"weight shape mismatch: got {tuple(w.shape)} expected {(N,)}")
    else:
        w = torch.ones((N,), device=device, dtype=torch.float32)
    out = torch.empty((M, N), device=device, dtype=torch.float32)
    rstd = torch.empty((M,), device=device, dtype=torch.float32)
    block_n = 1 if N <= 1 else min(1024, 1 << (int(N) - 1).bit_length())
    grid = (M,)
    rms_norm2d_kernel[grid](inp, w, out, rstd, M, N, eps, BLOCK_N=block_n)
    torch.cuda.synchronize()
    return {
        "input": inp.cpu().numpy(),
        "weight": w.cpu().numpy(),
        "output": out.cpu().numpy(),
        "rstd": rstd.cpu().numpy(),
        "eps": np.array(eps, dtype=np.float32),
    }


def _run_rms_norm_residual2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    """
    Fused residual + RMSNorm over last dim (shape [M, N]).
    Returns out + per-row rstd (shape [M]).
    """
    from kernels.triton.ops.rms_norm_residual2d import rms_norm_residual2d_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    eps = float(case.shapes.get("eps", 1e-5))
    device = "cuda"
    if N > 1024:
        raise ValueError(f"rms_norm_residual2d requires N<=1024, got N={N}")
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    if case.inputs and "residual" in case.inputs:
        residual = torch.as_tensor(case.inputs["residual"], device=device).to(torch.float32)
        if tuple(residual.shape) != (M, N):
            raise ValueError(f"residual shape mismatch: got {tuple(residual.shape)} expected {(M, N)}")
    else:
        residual = torch.randn((M, N), device=device, dtype=torch.float32)
    if case.inputs and "weight" in case.inputs:
        w = torch.as_tensor(case.inputs["weight"], device=device).to(torch.float32)
        if tuple(w.shape) != (N,):
            raise ValueError(f"weight shape mismatch: got {tuple(w.shape)} expected {(N,)}")
    else:
        w = torch.ones((N,), device=device, dtype=torch.float32)
    if case.inputs and "bias" in case.inputs:
        b = torch.as_tensor(case.inputs["bias"], device=device).to(torch.float32)
        if tuple(b.shape) != (N,):
            raise ValueError(f"bias shape mismatch: got {tuple(b.shape)} expected {(N,)}")
    else:
        b = torch.zeros((N,), device=device, dtype=torch.float32)

    out = torch.empty((M, N), device=device, dtype=torch.float32)
    rstd = torch.empty((M,), device=device, dtype=torch.float32)
    block_n = 1 if N <= 1 else min(1024, 1 << (int(N) - 1).bit_length())
    grid = (M,)
    rms_norm_residual2d_kernel[grid](inp, residual, w, b, out, rstd, M, N, eps, BLOCK_N=block_n)
    torch.cuda.synchronize()
    return {
        "input": inp.cpu().numpy(),
        "residual": residual.cpu().numpy(),
        "weight": w.cpu().numpy(),
        "bias": b.cpu().numpy(),
        "output": out.cpu().numpy(),
        "rstd": rstd.cpu().numpy(),
        "eps": np.array(eps, dtype=np.float32),
    }


def _run_layer_norm_residual2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    """
    Fused residual + LayerNorm forward over last dim (shape [M, N]).
    Returns out + per-row mean/rstd (shape [M]).
    """
    from kernels.triton.ops.layer_norm_residual2d import layer_norm_residual2d_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    eps = float(case.shapes.get("eps", 1e-5))
    device = "cuda"
    if N > 1024:
        raise ValueError(f"layer_norm_residual2d requires N<=1024, got N={N}")

    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    if case.inputs and "residual" in case.inputs:
        residual = torch.as_tensor(case.inputs["residual"], device=device).to(torch.float32)
        if tuple(residual.shape) != (M, N):
            raise ValueError(f"residual shape mismatch: got {tuple(residual.shape)} expected {(M, N)}")
    else:
        residual = torch.randn((M, N), device=device, dtype=torch.float32)
    if case.inputs and "weight" in case.inputs:
        w = torch.as_tensor(case.inputs["weight"], device=device).to(torch.float32)
        if tuple(w.shape) != (N,):
            raise ValueError(f"weight shape mismatch: got {tuple(w.shape)} expected {(N,)}")
    else:
        w = torch.ones((N,), device=device, dtype=torch.float32)
    if case.inputs and "bias" in case.inputs:
        b = torch.as_tensor(case.inputs["bias"], device=device).to(torch.float32)
        if tuple(b.shape) != (N,):
            raise ValueError(f"bias shape mismatch: got {tuple(b.shape)} expected {(N,)}")
    else:
        b = torch.zeros((N,), device=device, dtype=torch.float32)

    out = torch.empty((M, N), device=device, dtype=torch.float32)
    mean = torch.empty((M,), device=device, dtype=torch.float32)
    rstd = torch.empty((M,), device=device, dtype=torch.float32)
    block_n = 1 if N <= 1 else min(1024, 1 << (int(N) - 1).bit_length())
    grid = (M,)
    layer_norm_residual2d_kernel[grid](inp, residual, w, b, out, mean, rstd, M, N, eps, BLOCK_N=block_n)
    torch.cuda.synchronize()
    return {
        "input": inp.cpu().numpy(),
        "residual": residual.cpu().numpy(),
        "weight": w.cpu().numpy(),
        "bias": b.cpu().numpy(),
        "output": out.cpu().numpy(),
        "mean": mean.cpu().numpy(),
        "rstd": rstd.cpu().numpy(),
        "eps": np.array(eps, dtype=np.float32),
    }


def _run_matmul_bias_relu2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.matmul_bias_relu2d import matmul_bias_relu2d_kernel

    M = int(case.shapes.get("M", 32))
    N = int(case.shapes.get("N", 32))
    K = int(case.shapes.get("K", 32))
    device = "cuda"

    if case.inputs and "A" in case.inputs:
        a = torch.as_tensor(case.inputs["A"], device=device).to(torch.float32)
        if tuple(a.shape) != (M, K):
            raise ValueError(f"A shape mismatch: got {tuple(a.shape)} expected {(M, K)}")
    else:
        a = torch.randn((M, K), device=device, dtype=torch.float32)
    if case.inputs and "B" in case.inputs:
        b = torch.as_tensor(case.inputs["B"], device=device).to(torch.float32)
        if tuple(b.shape) != (K, N):
            raise ValueError(f"B shape mismatch: got {tuple(b.shape)} expected {(K, N)}")
    else:
        b = torch.randn((K, N), device=device, dtype=torch.float32)
    if case.inputs and "bias" in case.inputs:
        bias = torch.as_tensor(case.inputs["bias"], device=device).to(torch.float32)
        if tuple(bias.shape) != (N,):
            raise ValueError(f"bias shape mismatch: got {tuple(bias.shape)} expected {(N,)}")
    else:
        bias = torch.randn((N,), device=device, dtype=torch.float32)
    c = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_bias_relu2d_kernel[grid](
        a,
        b,
        bias,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=16,
    )
    torch.cuda.synchronize()
    return {"A": a.cpu().numpy(), "B": b.cpu().numpy(), "bias": bias.cpu().numpy(), "C": c.cpu().numpy()}


def _run_rowmask_where2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.rowmask_where2d import rowmask_where2d_kernel

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    device = "cuda"
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    if case.inputs and "row_mask" in case.inputs:
        rm = torch.as_tensor(case.inputs["row_mask"], device=device).to(torch.bool)
        if tuple(rm.shape) != (M,):
            raise ValueError(f"row_mask shape mismatch: got {tuple(rm.shape)} expected {(M,)}")
    else:
        # deterministic-ish: mask odd rows
        rm = (torch.arange(M, device=device) % 2 == 0)
    out = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))
    rowmask_where2d_kernel[grid](inp, rm, out, M, N, BLOCK_N=256)
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "row_mask": rm.cpu().numpy(), "output": out.cpu().numpy()}


def _run_masked_softmax2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.masked_softmax2d import masked_softmax2d

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    if N > 256:
        raise ValueError(f"masked_softmax2d expects N<=256 (single-tile softmax), got N={N}")
    device = "cuda"
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    if case.inputs and "mask" in case.inputs:
        mask = torch.as_tensor(case.inputs["mask"], device=device).to(torch.bool)
        if tuple(mask.shape) != (N,):
            raise ValueError(f"mask shape mismatch: got {tuple(mask.shape)} expected {(N,)}")
    else:
        # deterministic-ish: keep every 3rd element masked out, ensure at least one True.
        if N <= 0:
            raise ValueError("N must be positive")
        mask = (torch.arange(N, device=device) % 3 != 0)
        mask[0] = True
    out = masked_softmax2d(inp, mask)
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "mask": mask.cpu().numpy(), "output": out.cpu().numpy()}


def _run_grouped_row_sum2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.grouped_row_sum2d import grouped_row_sum2d

    M = int(case.shapes.get("M", 4))
    N = int(case.shapes.get("N", 64))
    group_size = int(case.shapes.get("group_size", 4))
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if N % group_size != 0:
        raise ValueError(f"grouped_row_sum2d expects N divisible by group_size, got N={N} group_size={group_size}")
    device = "cuda"
    if case.inputs and "input" in case.inputs:
        inp = torch.as_tensor(case.inputs["input"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"input shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    out = grouped_row_sum2d(inp, group_size=group_size)
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "output": out.cpu().numpy()}


def _run_mlp2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.mlp2d import mlp2d

    M = int(case.shapes.get("M", 32))
    N = int(case.shapes.get("N", 32))
    K = int(case.shapes.get("K", 32))
    H = int(case.shapes.get("H", 32))
    device = "cuda"

    if case.inputs and "A" in case.inputs:
        A = torch.as_tensor(case.inputs["A"], device=device).to(torch.float32)
        if tuple(A.shape) != (M, K):
            raise ValueError(f"A shape mismatch: got {tuple(A.shape)} expected {(M, K)}")
    else:
        A = torch.randn((M, K), device=device, dtype=torch.float32)
    if case.inputs and "W1" in case.inputs:
        W1 = torch.as_tensor(case.inputs["W1"], device=device).to(torch.float32)
        if tuple(W1.shape) != (K, H):
            raise ValueError(f"W1 shape mismatch: got {tuple(W1.shape)} expected {(K, H)}")
    else:
        W1 = torch.randn((K, H), device=device, dtype=torch.float32)
    if case.inputs and "b1" in case.inputs:
        b1 = torch.as_tensor(case.inputs["b1"], device=device).to(torch.float32)
        if tuple(b1.shape) != (H,):
            raise ValueError(f"b1 shape mismatch: got {tuple(b1.shape)} expected {(H,)}")
    else:
        b1 = torch.randn((H,), device=device, dtype=torch.float32)
    if case.inputs and "W2" in case.inputs:
        W2 = torch.as_tensor(case.inputs["W2"], device=device).to(torch.float32)
        if tuple(W2.shape) != (H, N):
            raise ValueError(f"W2 shape mismatch: got {tuple(W2.shape)} expected {(H, N)}")
    else:
        W2 = torch.randn((H, N), device=device, dtype=torch.float32)
    if case.inputs and "b2" in case.inputs:
        b2 = torch.as_tensor(case.inputs["b2"], device=device).to(torch.float32)
        if tuple(b2.shape) != (N,):
            raise ValueError(f"b2 shape mismatch: got {tuple(b2.shape)} expected {(N,)}")
    else:
        b2 = torch.randn((N,), device=device, dtype=torch.float32)

    C = mlp2d(A, W1, b1, W2, b2)
    torch.cuda.synchronize()
    return {
        "A": A.cpu().numpy(),
        "W1": W1.cpu().numpy(),
        "b1": b1.cpu().numpy(),
        "W2": W2.cpu().numpy(),
        "b2": b2.cpu().numpy(),
        "C": C.cpu().numpy(),
    }


def _run_gather2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.gather2d import gather2d

    M = int(case.shapes.get("M", 16))
    N = int(case.shapes.get("N", 64))
    L = int(case.shapes.get("L", 256))
    device = "cuda"
    if case.inputs and "inp" in case.inputs:
        inp = torch.as_tensor(case.inputs["inp"], device=device).to(torch.float32)
        if tuple(inp.shape) != (M, N):
            raise ValueError(f"inp shape mismatch: got {tuple(inp.shape)} expected {(M, N)}")
    else:
        inp = torch.randn((M, N), device=device, dtype=torch.float32)
    # Indices must be in-bounds for both numpy reference and RVV backend.
    if case.inputs and "row_idx" in case.inputs and "col_idx" in case.inputs:
        row_idx = torch.as_tensor(case.inputs["row_idx"], device=device).to(torch.int32)
        col_idx = torch.as_tensor(case.inputs["col_idx"], device=device).to(torch.int32)
        if tuple(row_idx.shape) != (L,) or tuple(col_idx.shape) != (L,):
            raise ValueError(f"index shape mismatch: row_idx={tuple(row_idx.shape)} col_idx={tuple(col_idx.shape)} expected {(L,)}")
    else:
        # deterministic-ish indices
        row_idx = (torch.arange(L, device=device, dtype=torch.int32) % max(1, M)).to(torch.int32)
        col_idx = (torch.arange(L, device=device, dtype=torch.int32) % max(1, N)).to(torch.int32)
    out = gather2d(inp, row_idx, col_idx)
    torch.cuda.synchronize()
    return {"inp": inp.cpu().numpy(), "row_idx": row_idx.cpu().numpy(), "col_idx": col_idx.cpu().numpy(), "out": out.cpu().numpy()}


def _run_masked_attention2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.masked_attention2d import masked_attention2d

    Q_CTX = int(case.shapes.get("Q_CTX", 16))
    KV_CTX = int(case.shapes.get("KV_CTX", 16))
    HEAD_DIM = int(case.shapes.get("HEAD_DIM", 16))
    device = "cuda"
    if case.inputs and "Q" in case.inputs:
        Q = torch.as_tensor(case.inputs["Q"], device=device).to(torch.float32)
        if tuple(Q.shape) != (Q_CTX, HEAD_DIM):
            raise ValueError(f"Q shape mismatch: got {tuple(Q.shape)} expected {(Q_CTX, HEAD_DIM)}")
    else:
        Q = torch.randn((Q_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    if case.inputs and "K" in case.inputs:
        K = torch.as_tensor(case.inputs["K"], device=device).to(torch.float32)
        if tuple(K.shape) != (KV_CTX, HEAD_DIM):
            raise ValueError(f"K shape mismatch: got {tuple(K.shape)} expected {(KV_CTX, HEAD_DIM)}")
    else:
        K = torch.randn((KV_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    if case.inputs and "V" in case.inputs:
        V = torch.as_tensor(case.inputs["V"], device=device).to(torch.float32)
        if tuple(V.shape) != (KV_CTX, HEAD_DIM):
            raise ValueError(f"V shape mismatch: got {tuple(V.shape)} expected {(KV_CTX, HEAD_DIM)}")
    else:
        V = torch.randn((KV_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    Out = masked_attention2d(Q, K, V)
    torch.cuda.synchronize()
    sm_scale = float(1.0 / (float(HEAD_DIM) ** 0.5))
    return {
        "Q": Q.cpu().numpy(),
        "K": K.cpu().numpy(),
        "V": V.cpu().numpy(),
        "sm_scale": np.array(sm_scale, dtype=np.float32),
        "Out": Out.cpu().numpy(),
    }


def _run_flash_attention2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.flash_attention2d import flash_attention2d

    Q_CTX = int(case.shapes.get("Q_CTX", 64))
    KV_CTX = int(case.shapes.get("KV_CTX", Q_CTX))
    HEAD_DIM = int(case.shapes.get("HEAD_DIM", 64))
    device = "cuda"

    if case.inputs and "Q" in case.inputs:
        Q = torch.as_tensor(case.inputs["Q"], device=device).to(torch.float32)
        if tuple(Q.shape) != (Q_CTX, HEAD_DIM):
            raise ValueError(f"Q shape mismatch: got {tuple(Q.shape)} expected {(Q_CTX, HEAD_DIM)}")
    else:
        Q = torch.randn((Q_CTX, HEAD_DIM), device=device, dtype=torch.float32)

    if case.inputs and "K" in case.inputs:
        K = torch.as_tensor(case.inputs["K"], device=device).to(torch.float32)
        if tuple(K.shape) != (KV_CTX, HEAD_DIM):
            raise ValueError(f"K shape mismatch: got {tuple(K.shape)} expected {(KV_CTX, HEAD_DIM)}")
    else:
        K = torch.randn((KV_CTX, HEAD_DIM), device=device, dtype=torch.float32)

    if case.inputs and "V" in case.inputs:
        V = torch.as_tensor(case.inputs["V"], device=device).to(torch.float32)
        if tuple(V.shape) != (KV_CTX, HEAD_DIM):
            raise ValueError(f"V shape mismatch: got {tuple(V.shape)} expected {(KV_CTX, HEAD_DIM)}")
    else:
        V = torch.randn((KV_CTX, HEAD_DIM), device=device, dtype=torch.float32)

    Out = flash_attention2d(Q, K, V)
    torch.cuda.synchronize()
    sm_scale = float(1.0 / (float(HEAD_DIM) ** 0.5))
    return {
        "Q": Q.cpu().numpy(),
        "K": K.cpu().numpy(),
        "V": V.cpu().numpy(),
        "sm_scale": np.array(sm_scale, dtype=np.float32),
        "Out": Out.cpu().numpy(),
    }


def _run_matmul_fused_epilogue2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.matmul_fused_epilogue2d import matmul_fused_epilogue2d

    M = int(case.shapes.get("M", 32))
    N = int(case.shapes.get("N", 32))
    K = int(case.shapes.get("K", 32))
    device = "cuda"
    if case.inputs and "A" in case.inputs:
        A = torch.as_tensor(case.inputs["A"], device=device).to(torch.float32)
        if tuple(A.shape) != (M, K):
            raise ValueError(f"A shape mismatch: got {tuple(A.shape)} expected {(M, K)}")
    else:
        A = torch.randn((M, K), device=device, dtype=torch.float32)
    if case.inputs and "B" in case.inputs:
        B = torch.as_tensor(case.inputs["B"], device=device).to(torch.float32)
        if tuple(B.shape) != (K, N):
            raise ValueError(f"B shape mismatch: got {tuple(B.shape)} expected {(K, N)}")
    else:
        B = torch.randn((K, N), device=device, dtype=torch.float32)
    if case.inputs and "bias" in case.inputs:
        bias = torch.as_tensor(case.inputs["bias"], device=device).to(torch.float32)
        if tuple(bias.shape) != (N,):
            raise ValueError(f"bias shape mismatch: got {tuple(bias.shape)} expected {(N,)}")
    else:
        bias = torch.randn((N,), device=device, dtype=torch.float32)
    if case.inputs and "row_mask" in case.inputs:
        row_mask = torch.as_tensor(case.inputs["row_mask"], device=device).to(torch.bool)
        if tuple(row_mask.shape) != (M,):
            raise ValueError(f"row_mask shape mismatch: got {tuple(row_mask.shape)} expected {(M,)}")
    else:
        row_mask = (torch.arange(M, device=device) % 2 == 0)
    if case.inputs and "col_mask" in case.inputs:
        col_mask = torch.as_tensor(case.inputs["col_mask"], device=device).to(torch.bool)
        if tuple(col_mask.shape) != (N,):
            raise ValueError(f"col_mask shape mismatch: got {tuple(col_mask.shape)} expected {(N,)}")
    else:
        col_mask = (torch.arange(N, device=device) % 3 != 0)
        if N >= 1:
            col_mask[0] = True
    C = matmul_fused_epilogue2d(A, B, bias, row_mask, col_mask)
    torch.cuda.synchronize()
    return {
        "A": A.cpu().numpy(),
        "B": B.cpu().numpy(),
        "bias": bias.cpu().numpy(),
        "row_mask": row_mask.cpu().numpy(),
        "col_mask": col_mask.cpu().numpy(),
        "C": C.cpu().numpy(),
    }


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


def _run_ai_bench_matmul_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.ai_bench_matmul import ai_bench_matmul_kernel

    M = int(case.shapes.get("M", 256))
    N = int(case.shapes.get("N", 512))
    K = int(case.shapes.get("K", 256))
    device = "cuda"

    A = torch.randn((M, K), device=device, dtype=torch.float32)
    B = torch.randn((K, N), device=device, dtype=torch.float32)
    C = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    ai_bench_matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=64,
        BLOCK_N=16,
        BLOCK_K=16,
    )
    torch.cuda.synchronize()
    return {"A": A.cpu().numpy(), "B": B.cpu().numpy(), "C": C.cpu().numpy()}


def _run_ai_bench_dropout_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.ai_bench_dropout import ai_bench_dropout_kernel

    n_elements = int(case.shapes.get("n_elements", 1048576))
    device = "cuda"
    # Match the external baseline defaults (AI-Benchmark): p=0.5, seed=123.
    p = 0.5
    seed = 123

    x = torch.randn((n_elements,), device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    ai_bench_dropout_kernel[grid](x, y, n_elements, p, seed, BLOCK_SIZE=32)
    torch.cuda.synchronize()
    return {
        "x": x.cpu().numpy(),
        "out": y.cpu().numpy(),
        "p": np.array(p, dtype=np.float32),
        "seed": np.array(seed, dtype=np.int64),
    }


def _run_ai_bench_softmax_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.ai_bench_softmax import ai_bench_softmax_kernel

    R = int(case.shapes.get("R", 1823))
    C = int(case.shapes.get("C", 781))
    device = "cuda"

    inp = torch.randn((R, C), device=device, dtype=torch.float32)
    out = torch.empty_like(inp)
    block = 1 << (int(C) - 1).bit_length()
    if block > 1024:
        block = 1024
    ai_bench_softmax_kernel[(R,)](out, inp, R, C, BLOCK_SIZE=block)
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "output": out.cpu().numpy()}


def _run_ai_bench_layernorm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.ai_bench_layernorm import ai_bench_layernorm_fwd_kernel

    M = int(case.shapes.get("M", 1151))
    N = int(case.shapes.get("N", 8192))
    eps = float(case.shapes.get("eps", 1e-5))
    device = "cuda"

    X = torch.randn((M, N), device=device, dtype=torch.float32)
    W = torch.randn((N,), device=device, dtype=torch.float32)
    B = torch.randn((N,), device=device, dtype=torch.float32)
    Y = torch.empty_like(X)
    Mean = torch.empty((M,), device=device, dtype=torch.float32)
    Rstd = torch.empty((M,), device=device, dtype=torch.float32)
    ai_bench_layernorm_fwd_kernel[(M,)](X, Y, W, B, Mean, Rstd, M, N, eps, BLOCK_SIZE=16)
    torch.cuda.synchronize()
    return {"X": X.cpu().numpy(), "W": W.cpu().numpy(), "B": B.cpu().numpy(), "Y": Y.cpu().numpy(), "Mean": Mean.cpu().numpy(), "Rstd": Rstd.cpu().numpy(), "eps": np.array(eps, dtype=np.float32)}


def _run_ai_bench_rope_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.ai_bench_rope import ai_bench_rope_fwd_kernel

    SEQ_LEN = int(case.shapes.get("SEQ_LEN", 128))
    BATCH_NUM = int(case.shapes.get("BATCH_NUM", 4))
    HEAD_NUM = int(case.shapes.get("HEAD_NUM", 2))
    HEAD_DIM = int(case.shapes.get("HEAD_DIM", 128))
    if HEAD_DIM % 2 != 0:
        raise ValueError(f"rope requires even HEAD_DIM, got {HEAD_DIM}")
    device = "cuda"

    inp = torch.randn((SEQ_LEN, BATCH_NUM, HEAD_NUM, HEAD_DIM), device=device, dtype=torch.float32)
    out = torch.empty_like(inp)
    cos = torch.randn((SEQ_LEN, HEAD_DIM // 2), device=device, dtype=torch.float32)
    sin = torch.randn((SEQ_LEN, HEAD_DIM // 2), device=device, dtype=torch.float32)
    grid = (HEAD_NUM, BATCH_NUM, SEQ_LEN)
    ai_bench_rope_fwd_kernel[grid](
        inp,
        out,
        cos,
        sin,
        SEQ_LEN,
        BATCH_NUM,
        HEAD_NUM,
        HEAD_DIM,
        BLOCK_SIZE=32,
    )
    torch.cuda.synchronize()
    return {"input": inp.cpu().numpy(), "cos": cos.cpu().numpy(), "sin": sin.cpu().numpy(), "output": out.cpu().numpy()}


def _run_ai_bench_correlation_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.ai_bench_correlation import ai_bench_correlation_kernel

    out_channel = int(case.shapes.get("out_channel", 5))
    in_channel = int(case.shapes.get("in_channel", 58))
    height = int(case.shapes.get("height", 112))
    width = int(case.shapes.get("width", 88))
    out_shift = int(case.shapes.get("out_shift", 0))
    device = "cuda"

    in_size = in_channel * height * width
    vals0 = (torch.arange(in_size, device=device) % 16).to(torch.int8).reshape((in_channel, height, width))
    vals1 = (torch.arange(in_size, device=device) % 35).to(torch.int8).reshape((in_channel, height, width))
    out = torch.empty((out_channel, height, width), device=device, dtype=torch.int8)
    grid = lambda meta: (
        triton.cdiv(width, meta["BLOCK_W"]),
        triton.cdiv(height, meta["BLOCK_H"]),
        out_channel,
    )
    ai_bench_correlation_kernel[grid](
        vals0,
        vals1,
        out,
        out_channel,
        in_channel,
        height,
        width,
        out_shift,
        BLOCK_H=1,
        BLOCK_W=8,
        BLOCK_IC=64,
    )
    torch.cuda.synchronize()
    return {
        "src0": vals0.cpu().numpy(),
        "src1": vals1.cpu().numpy(),
        "out": out.cpu().numpy(),
        "out_shift": np.array(out_shift, dtype=np.int32),
    }


def _run_ai_bench_resize_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.ai_bench_resize import ai_bench_resize_kernel

    channel = int(case.shapes.get("C", case.shapes.get("channel", 3)))
    height = int(case.shapes.get("H", case.shapes.get("height", 512)))
    width = int(case.shapes.get("W", case.shapes.get("width", 512)))
    device = "cuda"

    in_size = channel * height * width
    src = (torch.arange(in_size, device=device) % 17).to(torch.int8).reshape((channel, height, width))
    out = torch.empty((channel, 2 * height, 2 * width), device=device, dtype=torch.int8)
    grid = lambda meta: (2 * height, channel, triton.cdiv(2 * width, meta["BLOCK_W"]))
    ai_bench_resize_kernel[grid](src, out, channel, height, width, BLOCK_W=128)
    torch.cuda.synchronize()
    return {"src": src.cpu().numpy(), "out": out.cpu().numpy()}


def _run_ai_bench_warp_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from kernels.triton.ops.ai_bench_warp import ai_bench_warp_kernel

    channel = int(case.shapes.get("C", case.shapes.get("channel", 3)))
    height = int(case.shapes.get("H", case.shapes.get("height", 1024)))
    width = int(case.shapes.get("W", case.shapes.get("width", 1024)))
    device = "cuda"

    in_size = channel * height * width
    src = (torch.arange(in_size, device=device) % 17).to(torch.int8).reshape((channel, height, width))
    offset = torch.zeros((height, width), device=device, dtype=torch.int16)
    out = torch.empty((channel, height, width), device=device, dtype=torch.int8)
    grid = lambda meta: (height, channel, triton.cdiv(width, meta["BLOCK_W"]))
    ai_bench_warp_kernel[grid](src, offset, out, channel, height, width, BLOCK_W=128)
    torch.cuda.synchronize()
    return {"src": src.cpu().numpy(), "offset": offset.cpu().numpy(), "out": out.cpu().numpy()}


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


def _upsample_bicubic2d_aa_fallback_intent():
    """
    Deterministic macro IntentIR for bicubic AA upsample.

    This is a resilience fallback when LLM providers are unavailable (5xx/429/403),
    because the raw Triton kernel body is extremely long and can trigger proxy
    instability. The macro op will still be expanded by the compiler later.
    """
    from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType  # noqa: PLC0415

    rm = TensorLayout(kind="row_major", params={})
    tensors = {
        "I": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "IH"), Dim("sym", "IW")], layout=rm),
        "reciprocal_scale_h": TensorType(dtype="f32", shape=[], layout=rm),
        "reciprocal_scale_w": TensorType(dtype="f32", shape=[], layout=rm),
        "O": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "OH"), Dim("sym", "OW")], layout=rm),
    }
    ops = [
        Op(
            op="upsample_bicubic2d_aa",
            inputs=["I"],
            output="O",
            attrs={"a": -0.5, "support": 2.0, "invscale": 1.0, "separable": True, "normalize_weights": True},
        )
    ]
    schedule = ScheduleSketch(tile_m="BLOCK_Y", tile_n="BLOCK_X", tile_k=None, vec_width=1, pipeline_depth=1)
    return IntentFunction(
        name="upsample_bicubic2d_aa",
        tensors=tensors,
        ops=ops,
        outputs=["O"],
        schedule=schedule,
        axis_roles={"N": "batch", "C": "channel", "IH": "spatial", "IW": "spatial", "OH": "spatial", "OW": "spatial"},
    )


def _attn_fwd_fallback_intent():
    """
    Deterministic, compiler-style IntentIR for the Triton _attn_fwd kernel.

    Keeps original view shapes:
      Q, Out: [Z, q_numhead, Q_CTX, HEAD_DIM]
      K, V:   [Z, kv_numhead, KV_CTX, HEAD_DIM]
      attn_mask: [Z, q_numhead, Q_CTX, KV_CTX] (may be all-zeros if HAS_ATTN_MASK=0)
    """
    from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType  # noqa: PLC0415

    rm = TensorLayout(kind="row_major", params={})
    tensors = {
        "Q": TensorType(
            dtype="f32",
            shape=[Dim("sym", "Z"), Dim("sym", "q_numhead"), Dim("sym", "Q_CTX"), Dim("sym", "HEAD_DIM")],
            layout=rm,
        ),
        "K": TensorType(
            dtype="f32",
            shape=[Dim("sym", "Z"), Dim("sym", "kv_numhead"), Dim("sym", "KV_CTX"), Dim("sym", "HEAD_DIM")],
            layout=rm,
        ),
        "V": TensorType(
            dtype="f32",
            shape=[Dim("sym", "Z"), Dim("sym", "kv_numhead"), Dim("sym", "KV_CTX"), Dim("sym", "HEAD_DIM")],
            layout=rm,
        ),
        "attn_mask": TensorType(
            dtype="f32",
            shape=[Dim("sym", "Z"), Dim("sym", "q_numhead"), Dim("sym", "Q_CTX"), Dim("sym", "KV_CTX")],
            layout=rm,
        ),
        "sm_scale": TensorType(dtype="f32", shape=[], layout=rm),
        "Out": TensorType(
            dtype="f32",
            shape=[Dim("sym", "Z"), Dim("sym", "q_numhead"), Dim("sym", "Q_CTX"), Dim("sym", "HEAD_DIM")],
            layout=rm,
        ),
    }
    ops: list[Op] = []
    ops.append(Op(op="transpose", inputs=["K"], output="K_t", attrs={"perm": [0, 1, 3, 2]}))
    tensors["K_t"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "Z"), Dim("sym", "kv_numhead"), Dim("sym", "HEAD_DIM"), Dim("sym", "KV_CTX")],
        layout=rm,
    )

    ops.append(Op(op="matmul", inputs=["Q", "K_t"], output="scores", attrs={}))
    tensors["scores"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "Z"), Dim("sym", "q_numhead"), Dim("sym", "Q_CTX"), Dim("sym", "KV_CTX")],
        layout=rm,
    )

    # Add optional attention mask (when mask is all zeros, this is a no-op).
    ops.append(Op(op="add", inputs=["scores", "attn_mask"], output="scores_m", attrs={}))
    tensors["scores_m"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "Z"), Dim("sym", "q_numhead"), Dim("sym", "Q_CTX"), Dim("sym", "KV_CTX")],
        layout=rm,
    )

    ops.append(Op(op="mul", inputs=["scores_m", "sm_scale"], output="scores_s", attrs={}))
    tensors["scores_s"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "Z"), Dim("sym", "q_numhead"), Dim("sym", "Q_CTX"), Dim("sym", "KV_CTX")],
        layout=rm,
    )

    ops.append(Op(op="softmax", inputs=["scores_s"], output="probs", attrs={"axis": -1}))
    tensors["probs"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "Z"), Dim("sym", "q_numhead"), Dim("sym", "Q_CTX"), Dim("sym", "KV_CTX")],
        layout=rm,
    )

    ops.append(Op(op="matmul", inputs=["probs", "V"], output="Out", attrs={}))
    schedule = ScheduleSketch(tile_m="BLOCK_M", tile_n="BLOCK_N", tile_k="BLOCK_N", vec_width=1, pipeline_depth=1)
    return IntentFunction(
        name="_attn_fwd",
        tensors=tensors,
        ops=ops,
        outputs=["Out"],
        schedule=schedule,
        # Keep axis roles within the v1.1 allowed set (batch/channel/spatial/reduction).
        axis_roles={
            "Z": "batch",
            "q_numhead": "channel",
            "kv_numhead": "channel",
            "Q_CTX": "spatial",
            "KV_CTX": "spatial",
            "HEAD_DIM": "channel",
        },
    )


def _tilelang_deterministic_intent_for(kernel_name: str):
    """
    Resilience fallback: reuse TileLang's deterministic intent builders.

    This is used only when LLM providers are unavailable/quota-limited, to keep
    the end-to-end pipeline (static_validate/diff/remote) usable.
    """
    try:
        from pipeline.tilelang.core import coverage_kernel_specs  # noqa: PLC0415

        for s in coverage_kernel_specs():
            if getattr(s, "name", None) == str(kernel_name):
                return s.intent_builder()
    except Exception:
        return None
    return None


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


def coverage_kernel_specs() -> List[KernelSpec]:
    """
    P3: expanded kernel coverage suite.

    Keep `default_kernel_specs()` as the fast 6-kernel smoke set; grow this list
    gradually as we add more representative kernels.
    """
    specs = list(default_kernel_specs())
    def _norm_row_sum(shapes: Dict[str, int]) -> Dict[str, int]:
        out = dict(shapes)
        if "N" in out:
            out["N"] = max(1, min(int(out["N"]), 1024))
        if "M" in out:
            out["M"] = max(1, int(out["M"]))
        return out

    def _norm_masked_softmax(shapes: Dict[str, int]) -> Dict[str, int]:
        out = dict(shapes)
        if "N" in out:
            # masked_softmax2d kernel is a single-tile softmax (N <= 256)
            out["N"] = max(1, min(int(out["N"]), 256))
        if "M" in out:
            out["M"] = max(1, int(out["M"]))
        return out

    def _norm_grouped_row_sum(shapes: Dict[str, int]) -> Dict[str, int]:
        out = dict(shapes)
        gs = int(out.get("group_size", 4))
        if gs <= 0:
            gs = 1
        out["group_size"] = gs
        if "N" in out:
            n = max(1, min(int(out["N"]), 256))
            if n < gs:
                n = gs
            # round down to nearest multiple (keep >= gs)
            n = max(gs, (n // gs) * gs)
            out["N"] = int(n)
        if "M" in out:
            out["M"] = max(1, int(out["M"]))
        return out

    def _norm_masked_attention2d(shapes: Dict[str, int]) -> Dict[str, int]:
        out = dict(shapes)
        out["Q_CTX"] = max(1, int(out.get("Q_CTX", 16)))
        out["KV_CTX"] = max(1, min(int(out.get("KV_CTX", 16)), 64))
        hd = int(out.get("HEAD_DIM", 16))
        allowed = [16, 32, 64]
        out["HEAD_DIM"] = int(min(allowed, key=lambda x: abs(int(x) - hd)))
        return out

    def _norm_flash_attention2d(shapes: Dict[str, int]) -> Dict[str, int]:
        out = dict(shapes)
        q = max(1, min(int(out.get("Q_CTX", 64)), 256))
        out["Q_CTX"] = int(q)
        # Keep KV_CTX aligned with Q_CTX for causal self-attention by default.
        out["KV_CTX"] = int(q)
        hd = int(out.get("HEAD_DIM", 64))
        allowed = [16, 32, 64]
        out["HEAD_DIM"] = int(min(allowed, key=lambda x: abs(int(x) - hd)))
        return out

    specs.extend(
        [
            KernelSpec(
                name="add2d",
                module="kernels.triton.ops.add2d",
                attr="add2d_kernel.src",
                runner=_run_add2d_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
            ),
            KernelSpec(
                name="transpose2d",
                module="kernels.triton.ops.transpose2d",
                attr="transpose2d_kernel.src",
                runner=_run_transpose2d_reference,
                canonical_shapes={"M": 16, "N": 16},
                vary_axes=["M", "N"],
            ),
            KernelSpec(
                name="relu2d",
                module="kernels.triton.ops.relu2d",
                attr="relu2d_kernel.src",
                runner=_run_relu2d_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
            ),
            KernelSpec(
                name="add_bias2d",
                module="kernels.triton.ops.add_bias2d",
                attr="add_bias2d_kernel.src",
                runner=_run_add_bias2d_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
            ),
            KernelSpec(
                name="where2d",
                module="kernels.triton.ops.where2d",
                attr="where2d_kernel.src",
                runner=_run_where2d_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
            ),
            KernelSpec(
                name="row_sum",
                module="kernels.triton.ops.row_sum",
                attr="row_sum_kernel.src",
                runner=_run_row_sum_reference,
                canonical_shapes={"M": 4, "N": 256},
                vary_axes=["M", "N"],
                normalize_shapes=_norm_row_sum,
            ),
            KernelSpec(
                name="exp2d",
                module="kernels.triton.ops.exp2d",
                attr="exp2d_kernel.src",
                runner=_run_exp2d_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
            ),
            KernelSpec(
                name="floor2d",
                module="kernels.triton.ops.floor2d",
                attr="floor2d_kernel.src",
                runner=_run_floor2d_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
            ),
            KernelSpec(
                name="clamp2d",
                module="kernels.triton.ops.clamp2d",
                attr="clamp2d_kernel.src",
                runner=_run_clamp2d_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
            ),
            KernelSpec(
                name="row_max",
                module="kernels.triton.ops.row_max",
                attr="row_max_kernel.src",
                runner=_run_row_max_reference,
                canonical_shapes={"M": 4, "N": 256},
                vary_axes=["M", "N"],
                normalize_shapes=_norm_row_sum,
            ),
            KernelSpec(
                name="copy2d_divmod",
                module="kernels.triton.ops.copy2d_divmod",
                attr="copy2d_divmod_kernel.src",
                runner=_run_copy2d_divmod_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
            ),
            KernelSpec(
                name="matmul_relu2d",
                module="kernels.triton.ops.matmul_relu2d",
                attr="matmul_relu2d_kernel.src",
                runner=_run_matmul_relu2d_reference,
                canonical_shapes={"M": 32, "N": 32, "K": 32},
                vary_axes=["M", "N", "K"],
            ),
            KernelSpec(
                name="rms_norm2d",
                module="kernels.triton.ops.rms_norm2d",
                attr="rms_norm2d_kernel.src",
                runner=_run_rms_norm2d_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
                normalize_shapes=_norm_row_sum,
            ),
            KernelSpec(
                name="rms_norm_residual2d",
                module="kernels.triton.ops.rms_norm_residual2d",
                attr="rms_norm_residual2d_kernel.src",
                runner=_run_rms_norm_residual2d_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
                normalize_shapes=_norm_row_sum,
            ),
            KernelSpec(
                name="layer_norm_residual2d",
                module="kernels.triton.ops.layer_norm_residual2d",
                attr="layer_norm_residual2d_kernel.src",
                runner=_run_layer_norm_residual2d_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
                normalize_shapes=_norm_row_sum,
            ),
            KernelSpec(
                name="matmul_bias_relu2d",
                module="kernels.triton.ops.matmul_bias_relu2d",
                attr="matmul_bias_relu2d_kernel.src",
                runner=_run_matmul_bias_relu2d_reference,
                canonical_shapes={"M": 32, "N": 32, "K": 32},
                vary_axes=["M", "N", "K"],
            ),
            KernelSpec(
                name="rowmask_where2d",
                module="kernels.triton.ops.rowmask_where2d",
                attr="rowmask_where2d_kernel.src",
                runner=_run_rowmask_where2d_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
            ),
            KernelSpec(
                name="masked_softmax2d",
                module="kernels.triton.ops.masked_softmax2d",
                attr="masked_softmax2d_kernel.src",
                runner=_run_masked_softmax2d_reference,
                canonical_shapes={"M": 4, "N": 64},
                vary_axes=["M", "N"],
                normalize_shapes=_norm_masked_softmax,
            ),
            KernelSpec(
                name="grouped_row_sum2d",
                module="kernels.triton.ops.grouped_row_sum2d",
                attr="grouped_row_sum2d_kernel.src",
                runner=_run_grouped_row_sum2d_reference,
                canonical_shapes={"M": 4, "N": 64, "group_size": 4},
                vary_axes=["M", "N"],
                exclude_axes=[],
                normalize_shapes=_norm_grouped_row_sum,
            ),
            KernelSpec(
                name="mlp2d",
                module="kernels.triton.ops.mlp2d",
                attr="mlp2d_kernel.src",
                runner=_run_mlp2d_reference,
                canonical_shapes={"M": 32, "N": 32, "K": 32, "H": 32},
                vary_axes=["M"],
            ),
            KernelSpec(
                name="gather2d",
                module="kernels.triton.ops.gather2d",
                attr="gather2d_kernel.src",
                runner=_run_gather2d_reference,
                canonical_shapes={"M": 16, "N": 64, "L": 256},
                vary_axes=["M"],
            ),
            KernelSpec(
                name="masked_attention2d",
                module="kernels.triton.ops.masked_attention2d",
                attr="masked_attention2d_kernel.src",
                runner=_run_masked_attention2d_reference,
                canonical_shapes={"Q_CTX": 16, "KV_CTX": 16, "HEAD_DIM": 16},
                vary_axes=["Q_CTX", "KV_CTX"],
                exclude_axes=["HEAD_DIM"],
                normalize_shapes=_norm_masked_attention2d,
            ),
            KernelSpec(
                name="flash_attention2d",
                module="kernels.triton.ops.flash_attention2d",
                attr="flash_attention2d_kernel.src",
                runner=_run_flash_attention2d_reference,
                canonical_shapes={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
                vary_axes=["Q_CTX"],
                exclude_axes=["KV_CTX", "HEAD_DIM"],
                normalize_shapes=_norm_flash_attention2d,
            ),
            KernelSpec(
                name="matmul_fused_epilogue2d",
                module="kernels.triton.ops.matmul_fused_epilogue2d",
                attr="matmul_fused_epilogue2d_kernel.src",
                runner=_run_matmul_fused_epilogue2d_reference,
                canonical_shapes={"M": 32, "N": 32, "K": 32},
                vary_axes=["M"],
            ),
            # AI-Benchmark (Triton-CPU) kernel equivalents used for Experiment A baselines.
            KernelSpec(
                name="ai_bench_matmul",
                module="kernels.triton.ops.ai_bench_matmul",
                attr="ai_bench_matmul_kernel.src",
                runner=_run_ai_bench_matmul_reference,
                canonical_shapes={"M": 256, "N": 512, "K": 256},
                vary_axes=["M", "N", "K"],
                constexpr={"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 16},
            ),
            KernelSpec(
                name="ai_bench_dropout",
                module="kernels.triton.ops.ai_bench_dropout",
                attr="ai_bench_dropout_kernel.src",
                runner=_run_ai_bench_dropout_reference,
                canonical_shapes={"n_elements": 1048576},
                vary_axes=["n_elements"],
                constexpr={"BLOCK_SIZE": 32},
            ),
            KernelSpec(
                name="ai_bench_softmax",
                module="kernels.triton.ops.ai_bench_softmax",
                attr="ai_bench_softmax_kernel.src",
                runner=_run_ai_bench_softmax_reference,
                canonical_shapes={"R": 1823, "C": 781},
                vary_axes=["R", "C"],
            ),
            KernelSpec(
                name="ai_bench_layernorm",
                module="kernels.triton.ops.ai_bench_layernorm",
                attr="ai_bench_layernorm_fwd_kernel.src",
                runner=_run_ai_bench_layernorm_reference,
                canonical_shapes={"M": 1151, "N": 8192},
                vary_axes=["M", "N"],
                constexpr={"BLOCK_SIZE": 16},
            ),
            KernelSpec(
                name="ai_bench_correlation",
                module="kernels.triton.ops.ai_bench_correlation",
                attr="ai_bench_correlation_kernel.src",
                runner=_run_ai_bench_correlation_reference,
                canonical_shapes={"out_channel": 5, "in_channel": 58, "height": 112, "width": 88, "out_shift": 0},
                vary_axes=[],
                constexpr={"BLOCK_H": 1, "BLOCK_W": 8, "BLOCK_IC": 64},
            ),
            KernelSpec(
                name="ai_bench_resize",
                module="kernels.triton.ops.ai_bench_resize",
                attr="ai_bench_resize_kernel.src",
                runner=_run_ai_bench_resize_reference,
                canonical_shapes={"C": 3, "H": 512, "W": 512, "OH": 1024, "OW": 1024},
                vary_axes=[],
                constexpr={"BLOCK_W": 128},
            ),
            KernelSpec(
                name="ai_bench_rope",
                module="kernels.triton.ops.ai_bench_rope",
                attr="ai_bench_rope_fwd_kernel.src",
                runner=_run_ai_bench_rope_reference,
                canonical_shapes={"SEQ_LEN": 128, "BATCH_NUM": 4, "HEAD_NUM": 2, "HEAD_DIM": 128},
                vary_axes=["SEQ_LEN", "BATCH_NUM", "HEAD_NUM", "HEAD_DIM"],
                exclude_axes=["HEAD_DIM"],
                constexpr={"BLOCK_SIZE": 32},
            ),
            KernelSpec(
                name="ai_bench_warp",
                module="kernels.triton.ops.ai_bench_warp",
                attr="ai_bench_warp_kernel.src",
                runner=_run_ai_bench_warp_reference,
                canonical_shapes={"C": 3, "H": 1024, "W": 1024},
                vary_axes=[],
                constexpr={"BLOCK_W": 128},
            ),
        ]
    )
    return specs


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
    # Some macro/complex kernels can easily trigger provider instability or long
    # waits. Keep the default full pipeline usable by allowing deterministic
    # fallbacks, while still permitting LLM forcing via env vars.
    use_llm_for_macro = str(os.getenv("INTENTIR_TRITON_UPSAMPLE_USE_LLM", "0")).strip() in {"1", "true", "yes", "on"}
    use_llm_for_attn = str(os.getenv("INTENTIR_TRITON_ATTN_USE_LLM", "0")).strip() in {"1", "true", "yes", "on"}
    if spec.name == "upsample_bicubic2d_aa" and not use_llm_for_macro:
        fb_intent = _upsample_bicubic2d_aa_fallback_intent()
        cand = CandidateIntent(intent=fb_intent, problem_params={}, schedule_params={}, raw_json={"fallback": True}, llm_trace={})
        report["llm_fallback"] = {"used": True, "kind": "macro_deterministic", "reason": "default (set INTENTIR_TRITON_UPSAMPLE_USE_LLM=1 to force LLM)"}
        enrich_intent_macros(cand.intent)
        mlir_txt = print_mlir_like(cand.intent)
        (out_dir / f"{spec.name}.intentir.mlir").write_text(mlir_txt, encoding="utf-8")
        (out_dir / f"{spec.name}.intentir.fallback.mlir").write_text(mlir_txt, encoding="utf-8")
        expanded_intent = expand_macros(cand.intent)
        cand_expanded = CandidateIntent(
            intent=expanded_intent,
            problem_params={},
            schedule_params={},
            raw_json={"fallback": True},
            llm_trace={},
        )
        exp_txt = print_mlir_like(expanded_intent)
        (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(exp_txt, encoding="utf-8")
        (out_dir / f"{spec.name}.intentir.fallback.expanded.mlir").write_text(exp_txt, encoding="utf-8")
    elif spec.name == "_attn_fwd" and not use_llm_for_attn:
        fb_intent = _attn_fwd_fallback_intent()
        cand = CandidateIntent(intent=fb_intent, problem_params={}, schedule_params={}, raw_json={"fallback": True}, llm_trace={})
        report["llm_fallback"] = {"used": True, "kind": "deterministic_attn", "reason": "default (set INTENTIR_TRITON_ATTN_USE_LLM=1 to force LLM)"}
        enrich_intent_macros(cand.intent)
        mlir_txt = print_mlir_like(cand.intent)
        (out_dir / f"{spec.name}.intentir.mlir").write_text(mlir_txt, encoding="utf-8")
        (out_dir / f"{spec.name}.intentir.fallback.mlir").write_text(mlir_txt, encoding="utf-8")
        expanded_intent = expand_macros(cand.intent)
        cand_expanded = CandidateIntent(
            intent=expanded_intent,
            problem_params={},
            schedule_params={},
            raw_json={"fallback": True},
            llm_trace={},
        )
        exp_txt = print_mlir_like(expanded_intent)
        (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(exp_txt, encoding="utf-8")
        (out_dir / f"{spec.name}.intentir.fallback.expanded.mlir").write_text(exp_txt, encoding="utf-8")
    else:
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
        # Resilience fallback for macro-heavy kernels when providers are unstable.
        if spec.name == "upsample_bicubic2d_aa":
            fb_intent = _upsample_bicubic2d_aa_fallback_intent()
            cand = CandidateIntent(intent=fb_intent, problem_params={}, schedule_params={}, raw_json={"fallback": True}, llm_trace={})
            report["llm_fallback"] = {"used": True, "kind": "macro_deterministic", "reason": "; ".join(feedback) if feedback else "LLM failed"}
            enrich_intent_macros(cand.intent)
            mlir_txt = print_mlir_like(cand.intent)
            (out_dir / f"{spec.name}.intentir.mlir").write_text(mlir_txt, encoding="utf-8")
            (out_dir / f"{spec.name}.intentir.fallback.mlir").write_text(mlir_txt, encoding="utf-8")
            expanded_intent = expand_macros(cand.intent)
            cand_expanded = CandidateIntent(
                intent=expanded_intent,
                problem_params={},
                schedule_params={},
                raw_json={"fallback": True},
                llm_trace={},
            )
            exp_txt = print_mlir_like(expanded_intent)
            (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(exp_txt, encoding="utf-8")
            (out_dir / f"{spec.name}.intentir.fallback.expanded.mlir").write_text(exp_txt, encoding="utf-8")
        else:
            fb_intent = (_attn_fwd_fallback_intent() if spec.name == "_attn_fwd" else _tilelang_deterministic_intent_for(spec.name))
            if fb_intent is None:
                raise RuntimeError(f"LLM/Intent parse failed after retries for {spec.name}: {'; '.join(feedback)}")
            cand = CandidateIntent(intent=fb_intent, problem_params={}, schedule_params={}, raw_json={"fallback": True}, llm_trace={})
            report["llm_fallback"] = {
                "used": True,
                "kind": ("deterministic_attn" if spec.name == "_attn_fwd" else "tilelang_deterministic"),
                "reason": "; ".join(feedback) if feedback else "LLM failed",
            }
            enrich_intent_macros(cand.intent)
            mlir_txt = print_mlir_like(cand.intent)
            (out_dir / f"{spec.name}.intentir.mlir").write_text(mlir_txt, encoding="utf-8")
            (out_dir / f"{spec.name}.intentir.fallback.mlir").write_text(mlir_txt, encoding="utf-8")
            expanded_intent = expand_macros(cand.intent)
            cand_expanded = CandidateIntent(
                intent=expanded_intent,
                problem_params={},
                schedule_params={},
                raw_json={"fallback": True},
                llm_trace={},
            )
            exp_txt = print_mlir_like(expanded_intent)
            (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(exp_txt, encoding="utf-8")
            (out_dir / f"{spec.name}.intentir.fallback.expanded.mlir").write_text(exp_txt, encoding="utf-8")
    # Ensure schedule is attached even if the LLM emits only partial schedule fields.
    _ensure_schedule(cand.intent, kernel_name=spec.name, triton_src=src)
    _attach_access_witness_meta(cand.intent, cert_v2=cert_v2, canonical_shapes=dict(spec.canonical_shapes))
    report["intent"] = cand.intent.to_json_dict()
    if cand_expanded is not None:
        _ensure_schedule(cand_expanded.intent, kernel_name=spec.name, triton_src=src)
        _attach_access_witness_meta(cand_expanded.intent, cert_v2=cert_v2, canonical_shapes=dict(spec.canonical_shapes))
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
