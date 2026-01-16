import torch
import triton
import triton.language as tl


@triton.jit
def ai_bench_resize_kernel(
    src_ptr,  # *int8, [C, H, W]
    out_ptr,  # *int8, [C, 2H, 2W]
    channel,
    height,
    width,
    BLOCK_W: tl.constexpr,
):
    pid_h = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_w = tl.program_id(axis=2)

    dst_h = 2 * height
    dst_w = 2 * width
    hw_fl = 7
    factor = 1 << hw_fl

    h_idx = pid_h
    input_y = h_idx << (hw_fl - 1)
    y0 = input_y >> hw_fl
    h1 = input_y - (y0 << hw_fl)
    h0 = factor - h1
    y1 = tl.minimum(y0 + 1, height - 1)

    w_idx = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w_idx < dst_w
    input_x = w_idx << (hw_fl - 1)
    x0 = input_x >> hw_fl
    x1 = tl.minimum(x0 + 1, width - 1)
    w1 = input_x - (x0 << hw_fl)
    w0 = factor - w1

    src_off = pid_c * height * width
    src0_row = src_ptr + src_off + y0 * width
    src1_row = src_ptr + src_off + y1 * width
    y0x0 = tl.load(src0_row + x0, mask=mask, other=0).to(tl.int16)
    y0x1 = tl.load(src0_row + x1, mask=mask, other=0).to(tl.int16)
    y1x0 = tl.load(src1_row + x0, mask=mask, other=0).to(tl.int16)
    y1x1 = tl.load(src1_row + x1, mask=mask, other=0).to(tl.int16)

    sum1 = (y0x0.to(tl.int32) * w0.to(tl.int32) + y0x1.to(tl.int32) * w1.to(tl.int32)) >> hw_fl
    sum2 = (y1x0.to(tl.int32) * w0.to(tl.int32) + y1x1.to(tl.int32) * w1.to(tl.int32)) >> hw_fl
    out = (sum1 * h0.to(tl.int32) + sum2 * h1.to(tl.int32)) >> hw_fl

    out_idx = pid_c * dst_h * dst_w + h_idx * dst_w + w_idx
    tl.store(out_ptr + out_idx, out.to(tl.int8), mask=mask)


def launch(
    channel: int,
    height: int,
    width: int,
    *,
    block_w: int = 128,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    in_size = int(channel) * int(height) * int(width)
    vals = (torch.arange(in_size, device=device) % 17).to(torch.int8)
    src = vals.reshape((channel, height, width))
    out = torch.empty((channel, 2 * height, 2 * width), device=device, dtype=torch.int8)
    grid = lambda meta: (
        2 * height,
        channel,
        triton.cdiv(2 * width, meta["BLOCK_W"]),
    )
    ai_bench_resize_kernel[grid](src, out, channel, height, width, BLOCK_W=int(block_w))
    return src, out

