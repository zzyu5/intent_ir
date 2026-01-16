import torch
import triton
import triton.language as tl


@triton.jit
def ai_bench_softmax_kernel(
    out_ptr,
    in_ptr,
    R,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = in_ptr + row_idx * C
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < C
    x = tl.load(row_start + col, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    out_row = out_ptr + row_idx * C
    tl.store(out_row + col, y, mask=mask)


def launch(R: int, C: int, *, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn((R, C), device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    # One row per program; use a power-of-two block size for reduction.
    block = 1 << (int(C) - 1).bit_length()
    if block > 1024:
        block = 1024
    ai_bench_softmax_kernel[(R,)](y, x, R, C, BLOCK_SIZE=block)
    return x, y
