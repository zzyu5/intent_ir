import torch
import triton
import triton.language as tl


@triton.jit
def ai_bench_dropout_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    rnd = tl.rand(seed, offsets)
    keep = rnd > p
    out = tl.where(keep, x / (1.0 - p), 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def launch(
    n_elements: int,
    *,
    p: float = 0.5,
    seed: int = 123,
    block_size: int = 32,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn((n_elements,), device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    ai_bench_dropout_kernel[grid](x, y, n_elements, float(p), int(seed), BLOCK_SIZE=int(block_size))
    return x, y

