import torch
import triton
import triton.language as tl


@triton.jit
def ai_bench_correlation_kernel(
    src0_ptr,  # *int8, [in_channel, H, W]
    src1_ptr,  # *int8, [in_channel, H, W]
    out_ptr,  # *int8, [out_channel, H, W]
    out_channel,
    in_channel,
    height,
    width,
    out_shift,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)

    w_idx = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
    h_idx = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)

    bound = (h_idx[:, None] < height) & (w_idx[None, :] < width) & (w_idx[None, :] >= pid_z)
    offsets = h_idx[:, None] * width + w_idx[None, :]
    hw = height * width

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
    for k in tl.static_range(0, BLOCK_IC):
        k_in = k < in_channel
        m = bound & k_in
        src0 = tl.load(src0_ptr + k * hw + offsets, mask=m, other=0).to(tl.int16)
        src1 = tl.load(src1_ptr + k * hw + offsets - pid_z, mask=m, other=0).to(tl.int16)
        acc += src0.to(tl.int32) * src1.to(tl.int32)

    out_idx = pid_z * hw + offsets
    out = (acc >> out_shift).to(tl.int8)
    tl.store(out_ptr + out_idx, out, mask=bound)


def launch(
    out_channel: int,
    in_channel: int,
    height: int,
    width: int,
    *,
    out_shift: int = 0,
    block_h: int = 1,
    block_w: int = 8,
    block_ic: int = 64,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Match AI-Benchmark input pattern (small int8 ranges; avoids overflow in int16 accum).
    in_size = int(in_channel) * int(height) * int(width)
    vals0 = (torch.arange(in_size, device=device) % 16).to(torch.int8)
    vals1 = (torch.arange(in_size, device=device) % 35).to(torch.int8)
    src0 = vals0.reshape((in_channel, height, width))
    src1 = vals1.reshape((in_channel, height, width))
    out = torch.empty((out_channel, height, width), device=device, dtype=torch.int8)
    grid = lambda meta: (
        triton.cdiv(width, meta["BLOCK_W"]),
        triton.cdiv(height, meta["BLOCK_H"]),
        out_channel,
    )
    ai_bench_correlation_kernel[grid](
        src0,
        src1,
        out,
        out_channel,
        in_channel,
        height,
        width,
        int(out_shift),
        BLOCK_H=int(block_h),
        BLOCK_W=int(block_w),
        BLOCK_IC=int(block_ic),
    )
    return src0, src1, out

