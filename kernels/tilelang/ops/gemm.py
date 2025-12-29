from __future__ import annotations

from typing import Dict

import numpy as np

from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType
from verify.gen_cases import TestCase

from .spec import TileLangKernelSpec


def make_gemm_relu_prim_func(
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 32,
    num_stages: int = 3,
    threads: int = 128,
    in_dtype: str = "float16",
    out_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Real TileLang GEMM(+ReLU) kernel.

    Notes:
    - We keep a first-class `PrimFunc` because Task4 (facts/certificate) needs to parse TIR.
    - The runtime pipeline will compile/execute this PrimFunc on CUDA to produce baseline IO.
    """
    import tilelang.language as T  # noqa: PLC0415

    M = T.dynamic("M")
    N = T.dynamic("N")
    K = T.dynamic("K")

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), in_dtype)
            B_shared = T.alloc_shared((block_k, block_n), in_dtype)
            C_local = T.alloc_fragment((block_m, block_n), accum_dtype)
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(A[by * block_m, ko * block_k], A_shared)
                T.copy(B[ko * block_k, bx * block_n], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_m, block_n):
                C_local[i, j] = T.max(C_local[i, j], 0)
            T.copy(C_local, C[by * block_m, bx * block_n])

    return main


def gemm_relu_reference(case: TestCase) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(case.seed))
    M = int(case.shapes["M"])
    N = int(case.shapes["N"])
    K = int(case.shapes["K"])
    A = rng.standard_normal((M, K), dtype=np.float32).astype(np.float16)
    B = rng.standard_normal((K, N), dtype=np.float32).astype(np.float16)
    C = np.maximum((A.astype(np.float32) @ B.astype(np.float32)), 0.0).astype(np.float16)
    return {"A": A, "B": B, "C": C}


def gemm_relu_intent() -> IntentFunction:
    rm = TensorLayout(kind="row_major", params={})
    tensors = {
        "A": TensorType(dtype="f16", shape=[Dim("sym", "M"), Dim("sym", "K")], layout=rm),
        "B": TensorType(dtype="f16", shape=[Dim("sym", "K"), Dim("sym", "N")], layout=rm),
        "C": TensorType(dtype="f16", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
    }
    ops = [
        Op(op="matmul", inputs=["A", "B"], output="mm", attrs={}),
        Op(op="relu", inputs=["mm"], output="C", attrs={}),
    ]
    schedule = ScheduleSketch(tile_m=128, tile_n=128, tile_k=32, vec_width=1, pipeline_depth=3)
    return IntentFunction(
        name="tilelang_gemm_relu",
        tensors=tensors,
        ops=ops,
        outputs=["C"],
        schedule=schedule,
        axis_roles={"M": "spatial", "N": "spatial", "K": "reduction"},
    )


def gemm_spec() -> TileLangKernelSpec:
    # Keep canonical sizes modest so pipeline runs quickly; users can override shapes/cases.
    return TileLangKernelSpec(
        name="tilelang_gemm",
        prim_func=make_gemm_relu_prim_func(block_m=128, block_n=128, block_k=32, num_stages=3, threads=128),
        arg_names=["A", "B", "C", "M", "N", "K"],
        canonical_shapes={"M": 256, "N": 256, "K": 256},
        vary_axes=["M", "N", "K"],
        runner=gemm_relu_reference,
        intent_builder=gemm_relu_intent,
        exclude_axes=[],
        constexpr_names=[],
    )


__all__ = [
    "make_gemm_relu_prim_func",
    "gemm_relu_reference",
    "gemm_relu_intent",
    "gemm_spec",
]


if __name__ == "__main__":
    # Standalone sanity check (mirrors your local demo, but avoids running on import).
    import torch

    from frontends.tilelang.runtime import run_tilelang_kernel_io

    M = 1024
    N = 1024
    K = 1024
    prim = make_gemm_relu_prim_func(block_m=128, block_n=128, block_k=32, num_stages=3, threads=128)
    case = TestCase(shapes={"M": M, "N": N, "K": K}, dtypes={}, seed=0)
    ref = gemm_relu_reference(case)
    io = run_tilelang_kernel_io(prim, bindings=case.shapes, inputs_np={"A": ref["A"], "B": ref["B"]})

    a = torch.from_numpy(ref["A"]).cuda()
    b = torch.from_numpy(ref["B"]).cuda()
    ref_c = torch.relu((a.float() @ b.float())).half()
    torch.testing.assert_close(torch.from_numpy(io["C"]).cuda(), ref_c, rtol=1e-2, atol=1e-2)
    print("tilelang_gemm: OK")
