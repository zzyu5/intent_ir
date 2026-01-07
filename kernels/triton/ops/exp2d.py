import torch
import triton
import triton.language as tl


@triton.jit
def exp2d_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_nb = tl.program_id(1)
    offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (pid_m < M) & (offs_n < N)
    x = tl.load(inp_ptr + pid_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    y = tl.exp(x)
    tl.store(out_ptr + pid_m * N + offs_n, y, mask=mask)


def exp2d(inp: torch.Tensor) -> torch.Tensor:
    if inp.dtype != torch.float32:
        raise TypeError("exp2d expects float32 tensor")
    if inp.ndim != 2:
        raise ValueError(f"exp2d expects rank-2 tensor, got {inp.ndim}")
    m, n = inp.shape
    out = torch.empty((m, n), device=inp.device, dtype=torch.float32)
    grid = lambda meta: (m, triton.cdiv(n, meta["BLOCK_N"]))
    exp2d_kernel[grid](inp, out, m, n, BLOCK_N=256)
    return out


__all__ = ["exp2d_kernel", "exp2d"]

