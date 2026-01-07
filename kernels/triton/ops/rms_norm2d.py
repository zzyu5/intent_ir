import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm2d_kernel(
    inp_ptr,
    weight_ptr,
    out_ptr,
    rstd_ptr,
    M,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = (pid_m < M) & (offs_n < N)
    x = tl.load(inp_ptr + pid_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    sumsq = tl.sum(x * x, axis=0)
    n_f = tl.full((), N, tl.float32)
    mean = sumsq / n_f
    rstd = tl.rsqrt(mean + eps)
    y = x * rstd * w
    tl.store(out_ptr + pid_m * N + offs_n, y, mask=mask)
    tl.store(rstd_ptr + pid_m, rstd, mask=(pid_m < M))


def rms_norm2d(inp: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    if inp.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("rms_norm2d expects float32 tensors")
    if inp.ndim != 2 or weight.ndim != 1:
        raise ValueError("rms_norm2d expects inp rank-2 and weight rank-1")
    m, n = inp.shape
    if int(weight.shape[0]) != int(n):
        raise ValueError(f"shape mismatch: weight={tuple(weight.shape)} expected ({n},)")
    if n > 1024:
        raise ValueError(f"rms_norm2d supports N<=1024 for now, got N={n}")
    out = torch.empty((m, n), device=inp.device, dtype=torch.float32)
    rstd = torch.empty((m,), device=inp.device, dtype=torch.float32)
    block_n = 1 if n <= 1 else min(1024, 1 << (int(n) - 1).bit_length())
    grid = (m,)
    rms_norm2d_kernel[grid](inp, weight, out, rstd, m, n, float(eps), BLOCK_N=block_n)
    return out, rstd


__all__ = ["rms_norm2d_kernel", "rms_norm2d"]

