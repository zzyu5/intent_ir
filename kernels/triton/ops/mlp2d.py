import torch
import triton
import triton.language as tl


@triton.jit
def mlp2d_kernel(
    A_ptr,
    W1_ptr,
    b1_ptr,
    W2_ptr,
    b2_ptr,
    C_ptr,
    M,
    N,
    K,
    H,
    stride_am,
    stride_ak,
    stride_w1k,
    stride_w1h,
    stride_w2h,
    stride_w2n,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)

    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    h0 = 0
    while h0 < H:
        h = h0 + offs_h
        # hidden segment: [BLOCK_M, BLOCK_H]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        w1_ptrs = W1_ptr + offs_k[:, None] * stride_w1k + h[None, :] * stride_w1h
        acc1 = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

        k0 = 0
        while k0 < K:
            k = k0 + offs_k
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0).to(tl.float32)
            w1 = tl.load(w1_ptrs, mask=(k[:, None] < K) & (h[None, :] < H), other=0.0).to(tl.float32)
            acc1 += tl.dot(a, w1, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * stride_ak
            w1_ptrs += BLOCK_K * stride_w1k
            k0 += BLOCK_K

        b1 = tl.load(b1_ptr + h, mask=(h < H), other=0.0).to(tl.float32)
        acc1 = tl.maximum(acc1 + b1[None, :], 0.0)

        w2_ptrs = W2_ptr + h[:, None] * stride_w2h + offs_n[None, :] * stride_w2n
        w2 = tl.load(w2_ptrs, mask=(h[:, None] < H) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        acc2 += tl.dot(acc1, w2, out_dtype=tl.float32)

        h0 += BLOCK_H

    b2 = tl.load(b2_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc2 = acc2 + b2[None, :]
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc2, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def mlp2d(A: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    for name, t, rank in [("A", A, 2), ("W1", W1, 2), ("W2", W2, 2)]:
        if t.dtype != torch.float32:
            raise TypeError(f"{name} must be float32")
        if t.ndim != rank:
            raise ValueError(f"{name} must be rank-{rank}")
    for name, t in [("b1", b1), ("b2", b2)]:
        if t.dtype != torch.float32:
            raise TypeError(f"{name} must be float32")
        if t.ndim != 1:
            raise ValueError(f"{name} must be rank-1")
    m, k = A.shape
    k1, h = W1.shape
    h2, n = W2.shape
    if int(k1) != int(k):
        raise ValueError(f"shape mismatch: A={tuple(A.shape)} W1={tuple(W1.shape)}")
    if int(h2) != int(h):
        raise ValueError(f"shape mismatch: W1={tuple(W1.shape)} W2={tuple(W2.shape)}")
    if int(b1.shape[0]) != int(h):
        raise ValueError(f"shape mismatch: b1={tuple(b1.shape)} expected ({h},)")
    if int(b2.shape[0]) != int(n):
        raise ValueError(f"shape mismatch: b2={tuple(b2.shape)} expected ({n},)")
    C = torch.empty((m, n), device=A.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    mlp2d_kernel[grid](
        A,
        W1,
        b1,
        W2,
        b2,
        C,
        m,
        n,
        k,
        h,
        A.stride(0),
        A.stride(1),
        W1.stride(0),
        W1.stride(1),
        W2.stride(0),
        W2.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=16,
        BLOCK_H=16,
    )
    return C


__all__ = ["mlp2d_kernel", "mlp2d"]

