import math

import torch
import triton
import triton.language as tl


@triton.jit
def masked_attention2d_kernel(
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
    # One program per query row; require KV_CTX <= BLOCK_KV for simplicity.
    pid_q = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q_ptr + pid_q * HEAD_DIM + offs_d, mask=(pid_q < Q_CTX) & (offs_d < HEAD_DIM), other=0.0).to(tl.float32)

    offs_kv = tl.arange(0, BLOCK_KV)
    k = tl.load(
        K_ptr + offs_kv[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_kv[:, None] < KV_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(k * q[None, :], axis=1) * sm_scale

    # Mask out-of-range keys (KV_CTX may be < BLOCK_KV).
    in_range = offs_kv < KV_CTX
    scores = tl.where(in_range, scores, -1.0e9)
    # Causal mask: disallow keys beyond the query index.
    causal = offs_kv > pid_q
    scores = tl.where(causal & in_range & (pid_q < Q_CTX), -1.0e9, scores)

    mx = tl.max(scores, axis=0)
    ex = tl.exp(scores - mx)
    sm = tl.sum(ex, axis=0)
    probs = ex / sm

    v = tl.load(
        V_ptr + offs_kv[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_kv[:, None] < KV_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    out = tl.sum(v * probs[:, None], axis=0)
    tl.store(Out_ptr + pid_q * HEAD_DIM + offs_d, out, mask=(pid_q < Q_CTX) & (offs_d < HEAD_DIM))


def masked_attention2d(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    if Q.dtype != torch.float32 or K.dtype != torch.float32 or V.dtype != torch.float32:
        raise TypeError("masked_attention2d expects float32 tensors")
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("masked_attention2d expects rank-2 tensors")
    q_ctx, d = Q.shape
    kv_ctx, d2 = K.shape
    kv2, d3 = V.shape
    if int(d2) != int(d) or int(d3) != int(d):
        raise ValueError("masked_attention2d expects matching HEAD_DIM")
    if int(kv2) != int(kv_ctx):
        raise ValueError("masked_attention2d expects K/V with same KV_CTX")
    if int(kv_ctx) > 64:
        raise ValueError("masked_attention2d expects KV_CTX<=64")
    if int(d) not in (16, 32, 64):
        raise ValueError("masked_attention2d expects HEAD_DIM in {16,32,64} for this coverage kernel")
    out = torch.empty((q_ctx, d), device=Q.device, dtype=torch.float32)
    sm_scale = 1.0 / math.sqrt(float(d))
    grid = (int(q_ctx),)
    masked_attention2d_kernel[grid](Q, K, V, out, float(sm_scale), int(q_ctx), int(kv_ctx), HEAD_DIM=int(d), BLOCK_KV=64)
    return out


__all__ = ["masked_attention2d_kernel", "masked_attention2d"]
