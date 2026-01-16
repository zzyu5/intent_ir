import torch
import triton
import triton.language as tl


@triton.jit
def ai_bench_rope_fwd_kernel(
    input_ptr,  # [SEQ, BATCH, HEAD, HEAD_DIM]
    output_ptr,
    cos_ptr,  # [SEQ, HEAD_DIM/2]
    sin_ptr,  # [SEQ, HEAD_DIM/2]
    SEQ_LEN,
    BATCH_NUM,
    HEAD_NUM,
    HEAD_DIM,
    BLOCK_SIZE: tl.constexpr,
):
    pid_head = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_seq = tl.program_id(axis=2)

    head_dim_mid = HEAD_DIM // 2
    cos_stride = head_dim_mid
    sin_stride = head_dim_mid
    base = ((pid_seq * BATCH_NUM + pid_batch) * HEAD_NUM + pid_head) * HEAD_DIM
    for off in range(0, head_dim_mid, BLOCK_SIZE):
        head_dim_offset = off + tl.arange(0, BLOCK_SIZE)
        mask = head_dim_offset < head_dim_mid

        cos = tl.load(cos_ptr + pid_seq * cos_stride + head_dim_offset, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + pid_seq * sin_stride + head_dim_offset, mask=mask, other=0.0)

        x1 = tl.load(input_ptr + base + head_dim_offset, mask=mask, other=0.0)
        x2 = tl.load(input_ptr + base + head_dim_mid + head_dim_offset, mask=mask, other=0.0)

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        tl.store(output_ptr + base + head_dim_offset, y1, mask=mask)
        tl.store(output_ptr + base + head_dim_mid + head_dim_offset, y2, mask=mask)


def launch(seq_len: int, batch_num: int, head_num: int, head_dim: int, *, device: str = "cuda") -> dict[str, torch.Tensor]:
    assert head_dim % 2 == 0
    x = torch.randn((seq_len, batch_num, head_num, head_dim), device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    cos = torch.randn((seq_len, head_dim // 2), device=device, dtype=torch.float32)
    sin = torch.randn((seq_len, head_dim // 2), device=device, dtype=torch.float32)
    grid = (head_num, batch_num, seq_len)
    ai_bench_rope_fwd_kernel[grid](
        x,
        out,
        cos,
        sin,
        seq_len,
        batch_num,
        head_num,
        head_dim,
        BLOCK_SIZE=32,
    )
    return {"input": x, "cos": cos, "sin": sin, "output": out}
