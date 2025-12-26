from typing import Dict

import numpy as np

from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType
from verify.gen_cases import TestCase

from .spec import TileLangKernelSpec


def make_softmax_inner_prim_func(*, n: int = 16, threads: int = 128):
    """
    Softmax over the last axis, with auxiliary outputs (row_max/row_sum) to make
    reductions explicit in the extracted evidence/contract.
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        input_ptr: T.Tensor((M, N), "float32"),
        output_ptr: T.Tensor((M, N), "float32"),
        row_max_ptr: T.Tensor((M,), "float32"),
        row_sum_ptr: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(input_ptr[pid_m, 0], row)

            mx = T.alloc_fragment((1,), "float32")
            T.reduce_max(row, mx, dim=0)
            row_max_ptr[pid_m] = mx[0]

            exp_row = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                exp_row[r] = T.exp(row[r] - mx[0])

            sm = T.alloc_fragment((1,), "float32")
            T.reduce_sum(exp_row, sm, dim=0)
            row_sum_ptr[pid_m] = sm[0]

            out_row = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                out_row[r] = exp_row[r] / sm[0]
            T.copy(out_row, output_ptr[pid_m, 0])

    return main


def softmax_inner_reference(case: TestCase) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(case.seed))
    M = int(case.shapes["M"])
    N = int(case.shapes["N"])
    x = rng.standard_normal((M, N), dtype=np.float32)
    x_max = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - x_max)
    y = e / np.sum(e, axis=1, keepdims=True)
    return {"input_ptr": x, "output_ptr": y}


def softmax_inner_intent() -> IntentFunction:
    rm = TensorLayout(kind="row_major", params={})
    tensors = {
        "input_ptr": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
        "output_ptr": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
    }
    ops = [Op(op="softmax", inputs=["input_ptr"], output="output_ptr", attrs={"axis": -1})]
    schedule = ScheduleSketch(tile_m=16, tile_n=16, tile_k=None, vec_width=1, pipeline_depth=1)
    return IntentFunction(
        name="softmax_inner",
        tensors=tensors,
        ops=ops,
        outputs=["output_ptr"],
        schedule=schedule,
        axis_roles={"M": "spatial", "N": "channel"},
    )


def softmax_inner_spec() -> TileLangKernelSpec:
    return TileLangKernelSpec(
        name="softmax_inner",
        prim_func=make_softmax_inner_prim_func(n=16, threads=128),
        arg_names=["output_ptr", "input_ptr", "row_max_ptr", "row_sum_ptr", "M", "N"],
        canonical_shapes={"M": 16, "N": 16},
        vary_axes=["M"],
        runner=softmax_inner_reference,
        intent_builder=softmax_inner_intent,
        exclude_axes=[],
        constexpr_names=[],
    )


__all__ = [
    "make_softmax_inner_prim_func",
    "softmax_inner_reference",
    "softmax_inner_intent",
    "softmax_inner_spec",
]
