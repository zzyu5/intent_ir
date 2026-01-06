import torch
import triton
import triton.language as tl


@triton.jit
def transpose2d_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x = tl.load(
        inp_ptr + offs_m[:, None] * N + offs_n[None, :],
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    ).to(tl.float32)
    x_t = tl.trans(x)
    tl.store(
        out_ptr + offs_n[:, None] * M + offs_m[None, :],
        x_t,
        mask=(offs_n[:, None] < N) & (offs_m[None, :] < M),
    )


def transpose2d(inp: torch.Tensor) -> torch.Tensor:
    if inp.dtype != torch.float32:
        raise TypeError("transpose2d expects float32 tensor")
    if inp.ndim != 2:
        raise ValueError(f"transpose2d expects rank-2 tensor, got {inp.ndim}")
    M, N = inp.shape
    out = torch.empty((N, M), device=inp.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    transpose2d_kernel[grid](inp, out, M, N, BLOCK_M=32, BLOCK_N=32)
    return out


__all__ = ["transpose2d_kernel", "transpose2d"]

