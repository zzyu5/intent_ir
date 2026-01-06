import torch
import triton
import triton.language as tl


@triton.jit
def add_bias2d_kernel(
    inp_ptr,
    bias_ptr,
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
    b = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    tl.store(out_ptr + pid_m * N + offs_n, x + b, mask=mask)


def add_bias2d(inp: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if inp.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("add_bias2d expects float32 tensors")
    if inp.ndim != 2:
        raise ValueError(f"add_bias2d expects rank-2 input, got {inp.ndim}")
    if bias.ndim != 1:
        raise ValueError(f"add_bias2d expects rank-1 bias, got {bias.ndim}")
    m, n = inp.shape
    if bias.shape[0] != n:
        raise ValueError(f"bias shape mismatch: bias={tuple(bias.shape)} expected ({n},)")
    out = torch.empty((m, n), device=inp.device, dtype=torch.float32)
    grid = lambda meta: (m, triton.cdiv(n, meta["BLOCK_N"]))
    add_bias2d_kernel[grid](inp, bias, out, m, n, BLOCK_N=256)
    return out


__all__ = ["add_bias2d_kernel", "add_bias2d"]

