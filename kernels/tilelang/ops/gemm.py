from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType
from verify.gen_cases import TestCase


def make_gemm_prim_func(
    *,
    block_m: int = 8,
    block_n: int = 8,
    block_k: int = 8,
    num_stages: int = 1,
    threads: int = 128,
    in_dtype: str = "float16",
    out_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Real TileLang kernel (PrimFunc) for GEMM.

    This is intentionally small and "anchor-strong" for the TileLang MVP:
      - T.copy regions are explicit (structured indexing).
      - T.gemm provides a robust dot anchor.
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = T.dynamic("N")
    K = T.dynamic("K")

    A_shape = (M, K)
    B_shape = (K, N)
    A_shared_shape = (block_m, block_k)
    B_shared_shape = (block_k, block_n)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_m, block_n), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(A[by * block_m, k * block_k], A_shared)
                T.copy(B[k * block_k, bx * block_n], B_shared)
                T.gemm(A_shared, B_shared, C_local, False, False)
            T.copy(C_local, C[by * block_m, bx * block_n])

    return main


def gemm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(case.seed))
    M = int(case.shapes["M"])
    N = int(case.shapes["N"])
    K = int(case.shapes["K"])
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    C = A @ B
    return {"A": A, "B": B, "C": C}


def gemm_intent() -> IntentFunction:
    rm = TensorLayout(kind="row_major", params={})
    tensors = {
        "A": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "K")], layout=rm),
        "B": TensorType(dtype="f32", shape=[Dim("sym", "K"), Dim("sym", "N")], layout=rm),
        "C": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
    }
    ops = [Op(op="matmul", inputs=["A", "B"], output="C", attrs={})]
    schedule = ScheduleSketch(tile_m=8, tile_n=8, tile_k=16, vec_width=1, pipeline_depth=1)
    return IntentFunction(
        name="tilelang_gemm",
        tensors=tensors,
        ops=ops,
        outputs=["C"],
        schedule=schedule,
        axis_roles={"M": "spatial", "N": "spatial", "K": "reduction"},
    )


@dataclass
class TileLangKernelSpec:
    name: str
    prim_func: Any
    arg_names: List[str]
    canonical_shapes: Dict[str, int]
    vary_axes: List[str]
    runner: Callable[[TestCase], Dict[str, np.ndarray]]
    intent_builder: Callable[[], IntentFunction]
    exclude_axes: List[str] | None = None
    constexpr_names: List[str] | None = None


def gemm_spec() -> TileLangKernelSpec:
    return TileLangKernelSpec(
        name="tilelang_gemm",
        prim_func=make_gemm_prim_func(block_m=8, block_n=8, block_k=8, num_stages=1, threads=128),
        arg_names=["A", "B", "C", "M", "N", "K"],
        canonical_shapes={"M": 16, "N": 16, "K": 16},
        vary_axes=["M", "N", "K"],
        runner=gemm_reference,
        intent_builder=gemm_intent,
        exclude_axes=[],
        constexpr_names=[],
    )


__all__ = ["TileLangKernelSpec", "make_gemm_prim_func", "gemm_reference", "gemm_intent", "gemm_spec"]
