"""
Self-contained Triton kernels and launch helpers for TTIR dumping/testing.

These kernels are simplified versions of attention/groupnorm/any to make sure
we can trigger JIT and collect TTIR even when full project dependencies are
missing. Each launch_* function runs the kernel once with small tensors.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ------------------------- Attention-like kernel ------------------------- #
@triton.jit
def attention_smoke(Q, K, V, Out, N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    if pid_m >= N_CTX:
        return
    offs = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + pid_m * HEAD_DIM + offs)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for n in range(0, N_CTX):
        k = tl.load(K + n * HEAD_DIM + offs)
        score = tl.sum(q * k, axis=0)
        v = tl.load(V + n * HEAD_DIM + offs)
        acc += score * v
    tl.store(Out + pid_m * HEAD_DIM + offs, acc)


def launch_attention_smoke():
    device = "cuda"
    N_CTX = 8
    HEAD_DIM = 16
    Q = torch.randn(N_CTX, HEAD_DIM, device=device, dtype=torch.float32)
    K = torch.randn(N_CTX, HEAD_DIM, device=device, dtype=torch.float32)
    V = torch.randn(N_CTX, HEAD_DIM, device=device, dtype=torch.float32)
    Out = torch.empty_like(Q)
    grid = (N_CTX,)
    attention_smoke[grid](Q, K, V, Out, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM)
    torch.cuda.synchronize()


# ------------------------- Groupnorm-like kernel ------------------------- #
@triton.jit
def groupnorm_smoke(X, Y, C, HW, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C * HW
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / (C * HW)
    var = tl.sum((x - mean) * (x - mean), axis=0) / (C * HW)
    rstd = tl.rsqrt(var + eps)
    y = (x - mean) * rstd
    tl.store(Y + offs, y, mask=mask)


def launch_groupnorm_smoke():
    device = "cuda"
    C = 4
    HW = 8
    BLOCK = C * HW
    X = torch.randn(C * HW, device=device, dtype=torch.float32)
    Y = torch.empty_like(X)
    grid = ((C * HW + BLOCK - 1) // BLOCK,)
    groupnorm_smoke[grid](X, Y, C, HW, eps=1e-5, BLOCK=BLOCK)
    torch.cuda.synchronize()


# ------------------------- Any-reduction-like kernel ------------------------- #
@triton.jit
def any_smoke(inp, out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(inp + offs, mask=mask, other=0.0)
    nonzero = vals != 0
    any_val = tl.max(nonzero, axis=0)
    tl.store(out + pid, any_val)


def launch_any_smoke():
    device = "cuda"
    N = 128
    BLOCK = 32
    inp = torch.randint(0, 2, (N,), device=device, dtype=torch.int32)
    out = torch.empty(( (N + BLOCK - 1) // BLOCK,), device=device, dtype=torch.int32)
    grid = ((N + BLOCK - 1) // BLOCK,)
    any_smoke[grid](inp, out, N, BLOCK=BLOCK)
    torch.cuda.synchronize()


SMOKE_KERNELS = {
    "attention_smoke": (attention_smoke, {"Q": "*fp32", "K": "*fp32", "V": "*fp32", "Out": "*fp32", "N_CTX": "i32", "HEAD_DIM": "i32"}, launch_attention_smoke),
    "groupnorm_smoke": (groupnorm_smoke, {"X": "*fp32", "Y": "*fp32", "C": "i32", "HW": "i32"}, launch_groupnorm_smoke),
    "any_smoke": (any_smoke, {"inp": "*i32", "out": "*i32", "N": "i32"}, launch_any_smoke),
}


__all__ = [
    "attention_smoke",
    "groupnorm_smoke",
    "any_smoke",
    "launch_attention_smoke",
    "launch_groupnorm_smoke",
    "launch_any_smoke",
    "SMOKE_KERNELS",
]
