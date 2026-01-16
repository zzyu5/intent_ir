import torch
import triton
import triton.language as tl


@triton.jit
def ai_bench_warp_kernel(
    src_ptr,  # *int8, [C, H, W]
    offset_ptr,  # *int16, [H, W]
    out_ptr,  # *int8, [C, H, W]
    channel,
    height,
    width,
    BLOCK_W: tl.constexpr,
):
    pid_h = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_w = tl.program_id(axis=2)

    h_idx = pid_h
    w_idx = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w_idx < width

    offset_idx = h_idx * width + w_idx
    offset_val = tl.load(offset_ptr + offset_idx, mask=mask, other=0).to(tl.int16)

    offset_int = (offset_val >> 8).to(tl.int8)
    offset_fraction = ((offset_val << 8) >> 8).to(tl.int8)

    indvar = w_idx.to(tl.int8)
    right_idx = (indvar - offset_int).to(tl.int8)
    left_idx = (right_idx - 1).to(tl.int8)

    src_base = pid_c * height * width + h_idx * width
    right_mask = mask & (right_idx >= 0)
    left_mask = mask & (left_idx >= 0)
    right_val = tl.load(src_ptr + src_base + right_idx, mask=right_mask, other=0).to(tl.int8)
    left_val = tl.load(src_ptr + src_base + left_idx, mask=left_mask, other=0).to(tl.int8)

    out = (right_val.to(tl.int16) << 8) + (left_val - right_val).to(tl.int16) * offset_fraction.to(tl.int16)
    out = (out >> 8).to(tl.int8)

    out_idx = src_base + w_idx
    tl.store(out_ptr + out_idx, out, mask=mask)


def launch(
    channel: int,
    height: int,
    width: int,
    *,
    block_w: int = 128,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    in_size = int(channel) * int(height) * int(width)
    vals = (torch.arange(in_size, device=device) % 17).to(torch.int8)
    src = vals.reshape((channel, height, width))
    offset = torch.zeros((height, width), device=device, dtype=torch.int16)
    out = torch.empty((channel, height, width), device=device, dtype=torch.int8)
    grid = lambda meta: (
        height,
        channel,
        triton.cdiv(width, meta["BLOCK_W"]),
    )
    ai_bench_warp_kernel[grid](src, offset, out, channel, height, width, BLOCK_W=int(block_w))
    return src, offset, out

