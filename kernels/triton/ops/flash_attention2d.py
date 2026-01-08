import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention2d_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    sm_scale,
    Q_CTX,
    KV_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    FlashAttention-style causal attention (2D, single-head).

    - Q:   [Q_CTX, HEAD_DIM]
    - K/V: [KV_CTX, HEAD_DIM]
    - Out: [Q_CTX, HEAD_DIM]

    Uses an online softmax accumulator (m_i, l_i, acc) and streams K/V in blocks.
    """
    pid_q = tl.program_id(0)

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q_ptr + pid_q * HEAD_DIM + offs_d, mask=(pid_q < Q_CTX) & (offs_d < HEAD_DIM), other=0.0).to(tl.float32)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.zeros((), tl.float32)
    acc = tl.zeros((HEAD_DIM,), tl.float32)

    for start_kv in range(0, KV_CTX, BLOCK_KV):
        offs_kv = start_kv + tl.arange(0, BLOCK_KV)
        kv_mask = offs_kv < KV_CTX

        k = tl.load(
            K_ptr + offs_kv[:, None] * HEAD_DIM + offs_d[None, :],
            mask=kv_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_ptr + offs_kv[:, None] * HEAD_DIM + offs_d[None, :],
            mask=kv_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(kv_mask, scores, -float("inf"))
        # Causal mask: disallow keys beyond the query index.
        scores = tl.where(offs_kv > pid_q, -float("inf"), scores)

        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)

        p = tl.exp(scores - m_new)
        l_ij = tl.sum(p, axis=0)

        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(Out_ptr + pid_q * HEAD_DIM + offs_d, out, mask=(pid_q < Q_CTX) & (offs_d < HEAD_DIM))


def flash_attention2d(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Convenience wrapper around `flash_attention2d_kernel` for coverage testing.
    """
    if Q.dtype != torch.float32 or K.dtype != torch.float32 or V.dtype != torch.float32:
        raise TypeError("flash_attention2d expects float32 tensors")
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("flash_attention2d expects rank-2 tensors")
    q_ctx, head_dim = Q.shape
    kv_ctx, head_dim2 = K.shape
    kv2, head_dim3 = V.shape
    if int(head_dim2) != int(head_dim) or int(head_dim3) != int(head_dim):
        raise ValueError("flash_attention2d expects matching HEAD_DIM")
    if int(kv2) != int(kv_ctx):
        raise ValueError("flash_attention2d expects K/V with same KV_CTX")
    if int(head_dim) not in (16, 32, 64):
        raise ValueError("flash_attention2d expects HEAD_DIM in {16,32,64} for this coverage kernel")

    out = torch.empty((q_ctx, head_dim), device=Q.device, dtype=torch.float32)
    sm_scale = 1.0 / math.sqrt(float(head_dim))
    grid = (int(q_ctx),)
    flash_attention2d_kernel[grid](
        Q,
        K,
        V,
        out,
        float(sm_scale),
        int(q_ctx),
        int(kv_ctx),
        HEAD_DIM=int(head_dim),
        BLOCK_KV=32,
    )
    return out


__all__ = ["flash_attention2d_kernel", "flash_attention2d"]

