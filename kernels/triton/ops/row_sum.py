import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = (pid_m < M) & (offs_n < N)
    x = tl.load(inp_ptr + pid_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + pid_m, s, mask=(pid_m < M))


def row_sum(inp: torch.Tensor) -> torch.Tensor:
    if inp.dtype != torch.float32:
        raise TypeError("row_sum expects float32 tensor")
    if inp.ndim != 2:
        raise ValueError(f"row_sum expects rank-2 tensor, got {inp.ndim}")
    m, n = inp.shape
    if n > 1024:
        raise ValueError(f"row_sum supports N<=1024 for now, got N={n}")
    out = torch.empty((m,), device=inp.device, dtype=torch.float32)
    block_n = 1 if n <= 1 else min(1024, 1 << (int(n) - 1).bit_length())
    grid = (m,)
    row_sum_kernel[grid](inp, out, m, n, BLOCK_N=block_n)
    return out


__all__ = ["row_sum_kernel", "row_sum"]

