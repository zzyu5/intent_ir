from typing import Dict

import numpy as np

from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType
from verify.gen_cases import TestCase

from .spec import TileLangKernelSpec


def make_any_kernel_dim_prim_func(*, n: int = 16, threads: int = 128):
    """
    Row-wise `any` reduction: out[m] = any(inp[m, :]).

    This mirrors the spirit of Triton's `any_kernel_dim` (reduce over the last axis).
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        out: T.Tensor((M,), "bool"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(inp[pid_m, 0], row)
            mx = T.alloc_fragment((1,), "float32")
            T.reduce_max(row, mx, dim=0)
            out[pid_m] = mx[0] != T.float32(0.0)

    return main


def any_kernel_dim_reference(case: TestCase) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(case.seed))
    M = int(case.shapes["M"])
    N = int(case.shapes["N"])
    inp = rng.integers(0, 2, size=(M, N)).astype(np.float32)
    out = np.any(inp != 0, axis=1)
    return {"inp": inp, "out": out}


def any_kernel_dim_intent() -> IntentFunction:
    rm = TensorLayout(kind="row_major", params={})
    tensors = {
        "inp": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
        "out": TensorType(dtype="bool", shape=[Dim("sym", "M")], layout=rm),
    }
    ops = [
        Op(op="reduce_any", inputs=["inp"], output="out", attrs={"axes": [1], "keepdims": False}),
    ]
    schedule = ScheduleSketch(tile_m=16, tile_n=16, tile_k=None, vec_width=1, pipeline_depth=1)
    return IntentFunction(
        name="any_kernel_dim",
        tensors=tensors,
        ops=ops,
        outputs=["out"],
        schedule=schedule,
        axis_roles={"M": "spatial", "N": "reduction"},
    )


def any_kernel_dim_spec() -> TileLangKernelSpec:
    return TileLangKernelSpec(
        name="any_kernel_dim",
        prim_func=make_any_kernel_dim_prim_func(n=16, threads=128),
        arg_names=["inp", "out", "M", "N"],
        canonical_shapes={"M": 16, "N": 16},
        vary_axes=["M"],
        runner=any_kernel_dim_reference,
        intent_builder=any_kernel_dim_intent,
        exclude_axes=[],
        constexpr_names=[],
    )


__all__ = [
    "make_any_kernel_dim_prim_func",
    "any_kernel_dim_reference",
    "any_kernel_dim_intent",
    "any_kernel_dim_spec",
]
