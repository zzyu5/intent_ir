import torch
import triton
import triton.language as tl


@triton.jit
def where2d_kernel(
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
    mask = (pid_m < M) & (offs_n < N)
    a = tl.load(A_ptr + pid_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + pid_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    out = tl.where(a > b, a, b)
    tl.store(C_ptr + pid_m * N + offs_n, out, mask=mask)


def where2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise TypeError("where2d expects float32 tensors")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"where2d expects rank-2 tensors, got A.ndim={A.ndim} B.ndim={B.ndim}")
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: A={tuple(A.shape)} B={tuple(B.shape)}")
    m, n = A.shape
    C = torch.empty((m, n), device=A.device, dtype=torch.float32)
    grid = lambda meta: (m, triton.cdiv(n, meta["BLOCK_N"]))
    where2d_kernel[grid](A, B, C, m, n, BLOCK_N=256)
    return C


__all__ = ["where2d_kernel", "where2d"]

