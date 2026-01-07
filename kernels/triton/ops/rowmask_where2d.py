import torch
import triton
import triton.language as tl


@triton.jit
def rowmask_where2d_kernel(
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
    x = tl.load(inp_ptr + pid_m * N + offs_n, mask=in_bounds, other=0.0).to(tl.float32)
    m = tl.load(mask_ptr + pid_m, mask=(pid_m < M), other=False)
    y = tl.where(m, x, 0.0)
    tl.store(out_ptr + pid_m * N + offs_n, y, mask=in_bounds)


def rowmask_where2d(inp: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    if inp.dtype != torch.float32 or row_mask.dtype != torch.bool:
        raise TypeError("rowmask_where2d expects inp=f32 and row_mask=bool")
    if inp.ndim != 2 or row_mask.ndim != 1:
        raise ValueError("rowmask_where2d expects inp rank-2 and row_mask rank-1")
    m, n = inp.shape
    if int(row_mask.shape[0]) != int(m):
        raise ValueError(f"shape mismatch: row_mask={tuple(row_mask.shape)} expected ({m},)")
    out = torch.empty((m, n), device=inp.device, dtype=torch.float32)
    grid = lambda meta: (m, triton.cdiv(n, meta["BLOCK_N"]))
    rowmask_where2d_kernel[grid](inp, row_mask, out, m, n, BLOCK_N=256)
    return out


__all__ = ["rowmask_where2d_kernel", "rowmask_where2d"]

