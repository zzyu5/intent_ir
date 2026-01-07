import torch
import triton
import triton.language as tl


@triton.jit
def grouped_row_sum2d_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    G,
    GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs = pid_g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = (pid_m < M) & (pid_g < G) & (offs < N)
    x = tl.load(inp_ptr + pid_m * N + offs, mask=mask, other=0.0).to(tl.float32)
    sm = tl.sum(x, axis=0)
    tl.store(out_ptr + pid_m * G + pid_g, sm, mask=(pid_m < M) & (pid_g < G))


def grouped_row_sum2d(inp: torch.Tensor, group_size: int) -> torch.Tensor:
    if inp.dtype != torch.float32:
        raise TypeError("grouped_row_sum2d expects float32 tensor")
    if inp.ndim != 2:
        raise ValueError(f"grouped_row_sum2d expects rank-2 tensor, got {inp.ndim}")
    if int(group_size) <= 0:
        raise ValueError("group_size must be positive")
    m, n = inp.shape
    if int(n) % int(group_size) != 0:
        raise ValueError(f"N must be divisible by group_size, got N={n} group_size={group_size}")
    g = int(n) // int(group_size)
    out = torch.empty((m, g), device=inp.device, dtype=torch.float32)
    grid = (m, g)
    grouped_row_sum2d_kernel[grid](inp, out, m, n, g, GROUP_SIZE=int(group_size))
    return out


__all__ = ["grouped_row_sum2d_kernel", "grouped_row_sum2d"]

