import torch
import triton
import triton.language as tl


@triton.jit
def masked_softmax2d_kernel(
    inp_ptr,
    mask_ptr,
    out_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_nb = tl.program_id(1)
    offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
    in_bounds = (pid_m < M) & (offs_n < N)
    x = tl.load(inp_ptr + pid_m * N + offs_n, mask=in_bounds, other=-float("inf")).to(tl.float32)
    m = tl.load(mask_ptr + offs_n, mask=(offs_n < N), other=False)
    x = tl.where(m, x, -1.0e9)
    mx = tl.max(x, axis=0)
    ex = tl.exp(x - mx)
    sm = tl.sum(ex, axis=0)
    y = ex / sm
    tl.store(out_ptr + pid_m * N + offs_n, y, mask=in_bounds)


def masked_softmax2d(inp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if inp.dtype != torch.float32:
        raise TypeError("masked_softmax2d expects float32 inp")
    if mask.dtype != torch.bool:
        raise TypeError("masked_softmax2d expects bool mask")
    if inp.ndim != 2 or mask.ndim != 1:
        raise ValueError("masked_softmax2d expects inp rank-2 and mask rank-1")
    m, n = inp.shape
    if int(mask.shape[0]) != int(n):
        raise ValueError(f"mask shape mismatch: got {tuple(mask.shape)} expected ({n},)")
    # Ensure at least one unmasked element (avoid div-by-zero).
    if not bool(torch.any(mask)):
        raise ValueError("mask must have at least one True element")
    out = torch.empty((m, n), device=inp.device, dtype=torch.float32)
    grid = lambda meta: (m, triton.cdiv(n, meta["BLOCK_N"]))
    masked_softmax2d_kernel[grid](inp, mask, out, m, n, BLOCK_N=256)
    return out


__all__ = ["masked_softmax2d_kernel", "masked_softmax2d"]

