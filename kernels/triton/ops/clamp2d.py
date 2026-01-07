import torch
import triton
import triton.language as tl


@triton.jit
def clamp2d_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    lo,
    hi,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_nb = tl.program_id(1)
    offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (pid_m < M) & (offs_n < N)
    x = tl.load(inp_ptr + pid_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x, lo)
    y = tl.minimum(y, hi)
    tl.store(out_ptr + pid_m * N + offs_n, y, mask=mask)


def clamp2d(inp: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    if inp.dtype != torch.float32:
        raise TypeError("clamp2d expects float32 tensor")
    if inp.ndim != 2:
        raise ValueError(f"clamp2d expects rank-2 tensor, got {inp.ndim}")
    m, n = inp.shape
    out = torch.empty((m, n), device=inp.device, dtype=torch.float32)
    grid = lambda meta: (m, triton.cdiv(n, meta["BLOCK_N"]))
    clamp2d_kernel[grid](inp, out, m, n, float(lo), float(hi), BLOCK_N=256)
    return out


__all__ = ["clamp2d_kernel", "clamp2d"]

