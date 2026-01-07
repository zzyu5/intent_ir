import torch
import triton
import triton.language as tl


@triton.jit
def copy2d_divmod_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_nb = pid % grid_n
    offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (pid_m < M) & (offs_n < N)
    x = tl.load(inp_ptr + pid_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid_m * N + offs_n, x, mask=mask)


def copy2d_divmod(inp: torch.Tensor) -> torch.Tensor:
    if inp.dtype != torch.float32:
        raise TypeError("copy2d_divmod expects float32 tensor")
    if inp.ndim != 2:
        raise ValueError(f"copy2d_divmod expects rank-2 tensor, got {inp.ndim}")
    m, n = inp.shape
    out = torch.empty((m, n), device=inp.device, dtype=torch.float32)
    grid = lambda meta: (m * triton.cdiv(n, meta["BLOCK_N"]),)
    copy2d_divmod_kernel[grid](inp, out, m, n, BLOCK_N=256)
    return out


__all__ = ["copy2d_divmod_kernel", "copy2d_divmod"]

