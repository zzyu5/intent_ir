import torch
import triton
import triton.language as tl


@triton.jit
def add2d_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_nb = tl.program_id(1)
    offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
    m_ok = pid_m < M
    mask = m_ok & (offs_n < N)
    a = tl.load(A_ptr + pid_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + pid_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    tl.store(C_ptr + pid_m * N + offs_n, a + b, mask=mask)


def add2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise TypeError("add2d expects float32 tensors")
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: A={tuple(A.shape)} B={tuple(B.shape)}")
    if A.ndim != 2:
        raise ValueError(f"add2d expects rank-2 tensors, got {A.ndim}")
    M, N = A.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))
    add2d_kernel[grid](A, B, C, M, N, BLOCK_N=256)
    return C


__all__ = ["add2d_kernel", "add2d"]

