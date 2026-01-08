import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_residual2d_kernel(
    inp_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
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

    z = x + r
    sum_z = tl.sum(z, axis=0)
    n_f = tl.full((), N, tl.float32)
    mean = sum_z / n_f

    dz = z - mean
    var = tl.sum(dz * dz, axis=0) / n_f
    rstd = tl.rsqrt(var + eps)

    y = dz * rstd * w + b
    tl.store(out_ptr + pid_m * N + offs_n, y, mask=mask)
    tl.store(mean_ptr + pid_m, mean, mask=(pid_m < M))
    tl.store(rstd_ptr + pid_m, rstd, mask=(pid_m < M))


def layer_norm_residual2d(
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused residual + LayerNorm:

      z[m,n] = inp[m,n] + residual[m,n]
      mean[m] = mean(z[m,:])
      rstd[m] = rsqrt(mean((z-mean)^2) + eps)
      out[m,n] = (z[m,n]-mean[m]) * rstd[m] * weight[n] + bias[n]
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
        raise ValueError(f"layer_norm_residual2d supports N<=1024 for now, got N={n}")

    out = torch.empty((m, n), device=inp.device, dtype=torch.float32)
    mean = torch.empty((m,), device=inp.device, dtype=torch.float32)
    rstd = torch.empty((m,), device=inp.device, dtype=torch.float32)
    block_n = 1 if n <= 1 else min(1024, 1 << (int(n) - 1).bit_length())
    grid = (int(m),)
    layer_norm_residual2d_kernel[grid](
        inp,
        residual,
        weight,
        bias,
        out,
        mean,
        rstd,
        int(m),
        int(n),
        float(eps),
        BLOCK_N=block_n,
    )
    return out, mean, rstd


__all__ = ["layer_norm_residual2d_kernel", "layer_norm_residual2d"]

