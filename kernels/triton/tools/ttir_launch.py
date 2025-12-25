"""
Launch helpers for existing Triton ops to trigger TTIR dump without modifying kernels.
These functions run a minimal instance of the original kernels.
"""

from __future__ import annotations

import torch
import triton
import sys
from pathlib import Path

# Ensure package imports work when run as script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.triton.ops.any import any_kernel_dim
from kernels.triton.ops.groupnorm import group_norm_kernel
from kernels.triton.ops.attention import _attn_fwd
from kernels.triton.support import runtime


def launch_any_kernel_dim():
    # Minimal 2D tensor for any_dim
    device = "cuda"
    M, N = 4, 8
    inp = torch.randint(0, 2, (M, N), device=device, dtype=torch.float32)
    out = torch.empty((M, 1), device=device, dtype=torch.bool)
    grid = lambda meta: ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],)
    any_kernel_dim[grid](inp, out, M, N)
    torch.cuda.synchronize()
    return out


def launch_group_norm_kernel():
    device = "cuda"
    N = 2
    C = 4
    HW = 4
    group = 2
    group_size = (C + group - 1) // group
    x = torch.randn((N, C, HW), device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    mean = torch.empty((N * group,), device=device, dtype=torch.float32)
    rstd = torch.empty((N * group,), device=device, dtype=torch.float32)
    # W/B optional: use None
    grid = lambda meta: (N * group,)
    group_norm_kernel[grid](
        x,
        y,
        None,
        None,
        mean,
        rstd,
        group_size,
        C,
        HW,
        group,
        1e-5,
        BLOCK_GROUP_SIZE=group_size,
        BLOCK_HW_SIZE=HW,
    )
    torch.cuda.synchronize()
    return y, mean, rstd


def launch_attention_kernel():
    device = "cuda"
    batch = 1
    q_numhead = 1
    kv_numhead = 1
    Q_CTX = 8
    KV_CTX = 8
    HEAD_DIM = 16
    # shapes: [B, H, S, D]
    Q = torch.randn((batch, q_numhead, Q_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    K = torch.randn((batch, kv_numhead, KV_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    V = torch.randn((batch, kv_numhead, KV_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    attn_mask = torch.zeros((batch, q_numhead, Q_CTX, KV_CTX), device=device, dtype=torch.float32)
    Out = torch.empty((batch, q_numhead, Q_CTX, HEAD_DIM), device=device, dtype=torch.float32)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    # strides in elements
    stride_q_batch, stride_q_head, stride_q_seqlen, stride_q_headsize = Q.stride()
    stride_k_batch, stride_k_head, stride_k_seqlen, stride_k_headsize = K.stride()
    stride_v_batch, stride_v_head, stride_v_seqlen, stride_v_headsize = V.stride()
    stride_o_batch, stride_o_head, stride_o_seqlen, stride_o_headsize = Out.stride()
    stride_attn_mask_batch, stride_attn_mask_head, stride_attn_mask_q_seqlen, stride_attn_mask_kv_seqlen = attn_mask.stride()

    meta_cfg = {"BLOCK_M": 16, "BLOCK_N": 16, "STAGE": 2, "HAS_ATTN_MASK": 0, "PRE_LOAD_V": 0}
    grid = lambda meta: (
        triton.cdiv(Q_CTX, meta_cfg["BLOCK_M"]),
        batch * q_numhead,
    )
    # Bypass autotune by calling underlying jit function
    kernel = _attn_fwd.fn
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
        KV_CTX,  # Z
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
    return Out


__all__ = ["launch_any_kernel_dim", "launch_group_norm_kernel", "launch_attention_kernel"]
