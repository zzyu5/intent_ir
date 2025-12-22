"""
Task5-style verification for the three real Triton ops (any, groupnorm, attention).

Steps per op:
1) Call LLM to extract Intent-IR JSON from the Triton kernel source.
2) Parse into IntentFunction.
3) Run the real Triton kernel to get reference outputs.
4) Run the Intent interpreter and diff against reference.

Note: This is a thin driver; it reuses verify.diff_runner and interpreter.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Dict, Callable

import numpy as np
import torch
import triton

ROOT = __file__.rsplit("/", 2)[0]
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from intent_ir.llm_intent import extract_intent_json
from intent_ir.parser_llm import parse_candidate_json, CandidateIntent
from verify.gen_cases import TestCase, generate_cases
from verify.diff_runner import run_diff
from triton_frontend.facts import extract_facts, extract_constraints
from triton_frontend.contract import evaluate_contract
from triton_frontend.certificate import build_certificate
from triton_frontend.static_validate import static_validate
from pathlib import Path
from verify.interpreter import execute_intent


@dataclass
class OpResult:
    name: str
    intent: CandidateIntent
    diffs: list
    counterexamples: list


def _intent_from_kernel(src: str, kernel_name: str) -> CandidateIntent:
    js = extract_intent_json(
        src,
        kernel_name=kernel_name,
        temperature=0,
        max_tokens=2000,
        extra_instruction=(
            "Outputs must be produced by ops; include all writes (mean/rstd, etc). "
            "Model scalars as const ops or attrs; do not leave them implicit."
        ),
    )
    return parse_candidate_json(js)


def _run_any_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from tritonops.any import any_kernel_dim

    M = case.shapes.get("M", 4)
    N = case.shapes.get("N", 8)
    device = "cuda"
    inp = torch.randint(0, 2, (M, N), device=device, dtype=torch.float32)
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
    # Triton kernel expects BLOCK_* to be power-of-two; round up for launch params.
    def next_power_of_2(x: int) -> int:
        return 1 if x <= 1 else 1 << (x - 1).bit_length()

    block_group_size = next_power_of_2(group_size)
    block_hw_size = next_power_of_2(HW)
    device = "cuda"
    x = torch.randn((N, C, HW), device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    w = torch.ones((C,), device=device, dtype=torch.float32)
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
        # Canonical structured stats: [N, num_groups]
        "Mean": mean.view(N, num_groups).cpu().numpy(),
        "Rstd": rstd.view(N, num_groups).cpu().numpy(),
    }


def _run_attention_reference(case: TestCase) -> Dict[str, np.ndarray]:
    from tritonops.attention import _attn_fwd

    batch = case.shapes.get("batch", 1)
    q_numhead = case.shapes.get("q_numhead", 1)
    kv_numhead = case.shapes.get("kv_numhead", 1)
    Q_CTX = case.shapes.get("Q_CTX", 16)
    KV_CTX = case.shapes.get("KV_CTX", 16)
    HEAD_DIM = case.shapes.get("HEAD_DIM", 16)
    device = "cuda"
    Q = torch.randn((batch, q_numhead, Q_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    K = torch.randn((batch, kv_numhead, KV_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    V = torch.randn((batch, kv_numhead, KV_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    attn_mask = torch.zeros((batch, q_numhead, Q_CTX, KV_CTX), device=device, dtype=torch.float32)
    Out = torch.empty((batch, q_numhead, Q_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    block_m = 16
    block_n = min(16, HEAD_DIM)  # satisfy tl.static_assert(BLOCK_N <= HEAD_DIM)
    meta_cfg = {"BLOCK_M": block_m, "BLOCK_N": block_n, "STAGE": 1, "HAS_ATTN_MASK": 0, "PRE_LOAD_V": 0}
    grid = lambda meta: (
        triton.cdiv(Q_CTX, meta_cfg["BLOCK_M"]),
        batch * q_numhead,
    )
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
        "Q": Q.cpu().numpy(),
        "K": K.cpu().numpy(),
        "V": V.cpu().numpy(),
        "attn_mask": attn_mask.cpu().numpy(),
        "sm_scale": np.array(sm_scale, dtype=np.float32),
        "Out": Out.cpu().numpy(),
    }


def _find_latest_ttir(dump_dir: Path, name_hint: str) -> Path | None:
    ttirs = sorted(dump_dir.rglob("*.ttir"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in ttirs:
        if name_hint in p.name:
            return p
    return ttirs[0] if ttirs else None


def _verify_one(name: str, src: str, kernel_name: str, ref_fn: Callable[[TestCase], Dict[str, np.ndarray]], shapes: Dict[str, int]) -> OpResult:
    intent = _intent_from_kernel(src, kernel_name)
    dump_dir = Path(os.environ.get("TRITON_DUMP_DIR", "/tmp/triton_dump"))
    dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_KERNEL_DUMP", "1")
    # prime TTIR for constraints
    ref_fn(TestCase(shapes=shapes, seed=0, dtypes={}))
    constraints = None
    ttir_path = _find_latest_ttir(dump_dir, kernel_name)
    if ttir_path and ttir_path.exists():
        ttir_text = ttir_path.read_text()
        facts = extract_facts(ttir_text)
        constraints = extract_constraints(ttir_text, facts=facts)
        contract = evaluate_contract(facts)
        cert = build_certificate(ttir_text, facts=facts)
        sv = static_validate(intent.intent, cert)
        print(f"[{name}] TTIR contract={contract.level} reasons={contract.reasons}")
        print(f"[{name}] certificate kernel_kind={cert.kernel_kind} tile_hints={cert.tile_hints} obligations={[f'{o.id}:{o.status}' for o in sv.obligations]}")
    # Generate edge/boundary cases based on Intent axes; include a seeded canonical shape.
    cases = [TestCase(shapes=shapes, seed=0, dtypes={})]
    tile_hints = []
    if intent.intent.schedule:
        for k in ("tile_m", "tile_n", "tile_k", "vec_width"):
            v = getattr(intent.intent.schedule, k)
            if isinstance(v, int):
                tile_hints.append(v)
    # Keep cases in-scope per kernel kind (avoid varying constexpr-only axes like HEAD_DIM).
    vary_axes = None
    exclude_axes = None
    if name == "any_kernel_dim":
        vary_axes = ["M", "N"]
    elif name == "group_norm_kernel":
        vary_axes = ["N", "C", "HW", "num_groups"]
        exclude_axes = ["group_size"]
    elif name == "_attn_fwd":
        vary_axes = ["Q_CTX", "KV_CTX"]
        exclude_axes = ["HEAD_DIM", "batch", "q_numhead", "kv_numhead"]

    gen = generate_cases(
        intent.intent,
        constraints=constraints,
        limit=3,
        seed=1,
        tile_hints=tile_hints,
        axes=vary_axes,
        exclude_axes=exclude_axes,
    )
    for c in gen:
        merged = dict(shapes)
        merged.update(c.shapes)
        # Keep groupnorm cases in-scope for reshape-based IntentIR: enforce C % num_groups == 0.
        if name == "group_norm_kernel" and "C" in merged:
            c_val = int(merged["C"])
            g_req = int(merged.get("num_groups", merged.get("group", 1)))
            g_req = max(1, min(g_req, c_val))
            divisors = []
            d = 1
            while d * d <= c_val:
                if c_val % d == 0:
                    divisors.append(d)
                    if d != c_val // d:
                        divisors.append(c_val // d)
                d += 1
            divisors = sorted(set(divisors))
            best_g = min(divisors, key=lambda x: (abs(x - g_req), -x))
            merged["num_groups"] = int(best_g)
            merged.pop("group", None)
        c.shapes = merged
    cases.extend(gen)
    diffs, cex = run_diff(intent.intent, ref_fn, cases)
    return OpResult(name=name, intent=intent, diffs=diffs, counterexamples=cex)


def main():
    results = []
    from tritonops.any import any_kernel_dim
    results.append(
        _verify_one(
            "any_kernel_dim",
            any_kernel_dim.fn.src,
            "any_kernel_dim",
            _run_any_reference,
            {"M": 4, "N": 8},
        )
    )
    from tritonops.groupnorm import group_norm_kernel
    results.append(
        _verify_one(
            "group_norm_kernel",
            group_norm_kernel.src,
            "group_norm_kernel",
            _run_groupnorm_reference,
            {"N": 2, "C": 4, "HW": 4, "num_groups": 2},
        )
    )
    from tritonops.attention import _attn_fwd
    results.append(
        _verify_one(
            "_attn_fwd",
            _attn_fwd.fn.src,
            "_attn_fwd",
            _run_attention_reference,
            {"batch": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
        )
    )

    for r in results:
        worst = max(r.diffs, key=lambda d: (not d.ok, d.max_abs_err))
        status = "OK" if all(d.ok for d in r.diffs) else f"FAIL ({worst.summary})"
        print(f"{r.name}: {status}, max_abs={worst.max_abs_err:.4e}, max_rel={worst.max_rel_err:.4e}")
        if r.counterexamples:
            c0 = r.counterexamples[0]
            print("  counterexample shapes:", c0.case.shapes)
            print("  hint:", "; ".join(c0.hints))


if __name__ == "__main__":
    main()
