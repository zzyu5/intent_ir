import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_residual2d_kernel(
    inp_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
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
    r = tl.load(residual_ptr + pid_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)

    z = x + r + b
    sumsq = tl.sum(z * z, axis=0)
    n_f = tl.full((), N, tl.float32)
    mean_sq = sumsq / n_f
    rstd = tl.rsqrt(mean_sq + eps)

    y = z * rstd * w
    tl.store(out_ptr + pid_m * N + offs_n, y, mask=mask)
    tl.store(rstd_ptr + pid_m, rstd, mask=(pid_m < M))


def rms_norm_residual2d(
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused residual + RMSNorm:

      z[m,n] = inp[m,n] + residual[m,n] + bias[n]
      rstd[m] = rsqrt(mean(z[m,:]^2) + eps)
      out[m,n] = z[m,n] * rstd[m] * weight[n]
    """
    for name, t, rank in [("inp", inp, 2), ("residual", residual, 2)]:
        if t.dtype != torch.float32:
            raise TypeError(f"{name} must be float32")
        if t.ndim != rank:
            raise ValueError(f"{name} must be rank-{rank}")
    for name, t in [("weight", weight), ("bias", bias)]:
        if t.dtype != torch.float32:
            raise TypeError(f"{name} must be float32")
        if t.ndim != 1:
            raise ValueError(f"{name} must be rank-1")

    if tuple(inp.shape) != tuple(residual.shape):
        raise ValueError(f"shape mismatch: inp={tuple(inp.shape)} residual={tuple(residual.shape)}")
    m, n = inp.shape
    if int(weight.shape[0]) != int(n):
        raise ValueError(f"shape mismatch: weight={tuple(weight.shape)} expected ({n},)")
    if int(bias.shape[0]) != int(n):
        raise ValueError(f"shape mismatch: bias={tuple(bias.shape)} expected ({n},)")
    if int(n) > 1024:
        raise ValueError(f"rms_norm_residual2d supports N<=1024 for now, got N={n}")

    out = torch.empty((m, n), device=inp.device, dtype=torch.float32)
    rstd = torch.empty((m,), device=inp.device, dtype=torch.float32)
    block_n = 1 if n <= 1 else min(1024, 1 << (int(n) - 1).bit_length())
    grid = (int(m),)
    rms_norm_residual2d_kernel[grid](inp, residual, weight, bias, out, rstd, int(m), int(n), float(eps), BLOCK_N=block_n)
    return out, rstd


__all__ = ["rms_norm_residual2d_kernel", "rms_norm_residual2d"]

