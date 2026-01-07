import torch
import triton
import triton.language as tl


@triton.jit
def matmul_relu2d_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        k = k0 + offs_k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k0 += BLOCK_K

    acc = tl.maximum(acc, 0.0)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_relu2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise TypeError("matmul_relu2d expects float32 tensors")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("matmul_relu2d expects rank-2 tensors")
    m, k = A.shape
    k2, n = B.shape
    if k2 != k:
        raise ValueError(f"shape mismatch: A={tuple(A.shape)} B={tuple(B.shape)}")
    C = torch.empty((m, n), device=A.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    matmul_relu2d_kernel[grid](
        A,
        B,
        C,
        m,
        n,
        k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=16,
    )
    return C


__all__ = ["matmul_relu2d_kernel", "matmul_relu2d"]

