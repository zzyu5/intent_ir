import torch
import triton
import triton.language as tl


@triton.jit
def ai_bench_layernorm_fwd_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_row = X + row * N
    Y_row = Y + row * N

    # mean
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X_row + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += x
    mean = tl.sum(acc, axis=0) / N

    # variance
    acc2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X_row + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        acc2 += x * x
    var = tl.sum(acc2, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y_row + cols, y, mask=mask)


def launch(M: int, N: int, *, device: str = "cuda") -> dict[str, torch.Tensor]:
    x = torch.randn((M, N), device=device, dtype=torch.float32)
    w = torch.randn((N,), device=device, dtype=torch.float32)
    b = torch.randn((N,), device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    mean = torch.empty((M,), device=device, dtype=torch.float32)
    rstd = torch.empty((M,), device=device, dtype=torch.float32)
    ai_bench_layernorm_fwd_kernel[(M,)](x, y, w, b, mean, rstd, M, N, 1e-5, BLOCK_SIZE=16)
    return {"X": x, "W": w, "B": b, "Y": y, "Mean": mean, "Rstd": rstd}
