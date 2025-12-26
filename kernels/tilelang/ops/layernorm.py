from typing import Dict

import numpy as np

from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType
from verify.gen_cases import TestCase

from .spec import TileLangKernelSpec


def make_layer_norm_persistent_prim_func(*, n: int = 16, threads: int = 128):
    """
    LayerNorm over the last axis (N), with explicit Mean/Rstd outputs.
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        in_ptr: T.Tensor((M, N), "float32"),
        out_ptr: T.Tensor((M, N), "float32"),
        weight_ptr: T.Tensor((N,), "float32"),
        bias_ptr: T.Tensor((N,), "float32"),
        out_mean_ptr: T.Tensor((M,), "float32"),
        out_rstd_ptr: T.Tensor((M,), "float32"),
    ):
        eps = T.float32(1e-5)
        with T.Kernel(M, threads=threads) as (pid,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(in_ptr[pid, 0], row)
            w = T.alloc_fragment((N,), "float32")
            T.copy(weight_ptr[0], w)
            b = T.alloc_fragment((N,), "float32")
            T.copy(bias_ptr[0], b)

            s = T.alloc_fragment((1,), "float32")
            T.reduce_sum(row, s, dim=0)
            mean = s[0] / T.float32(N)
            out_mean_ptr[pid] = mean

            diff = T.alloc_fragment((N,), "float32")
            sq = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                d = row[r] - mean
                diff[r] = d
                sq[r] = d * d

            ss = T.alloc_fragment((1,), "float32")
            T.reduce_sum(sq, ss, dim=0)
            var = ss[0] / T.float32(N)
            rstd = T.rsqrt(var + eps)
            out_rstd_ptr[pid] = rstd

            out_row = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                out_row[r] = diff[r] * rstd * w[r] + b[r]
            T.copy(out_row, out_ptr[pid, 0])

    return main


def layer_norm_persistent_reference(case: TestCase) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(case.seed))
    M = int(case.shapes["M"])
    N = int(case.shapes["N"])
    x = rng.standard_normal((M, N), dtype=np.float32)
    w = rng.standard_normal((N,), dtype=np.float32)
    b = rng.standard_normal((N,), dtype=np.float32)
    eps = np.float32(1e-5)
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    y = (x - mean) * rstd * w[None, :] + b[None, :]
    return {
        "in_ptr": x,
        "weight_ptr": w,
        "bias_ptr": b,
        "out_ptr": y.astype(np.float32),
        "out_mean_ptr": mean.reshape(-1).astype(np.float32),
        "out_rstd_ptr": rstd.reshape(-1).astype(np.float32),
    }


def layer_norm_persistent_intent() -> IntentFunction:
    rm = TensorLayout(kind="row_major", params={})
    tensors = {
        "in_ptr": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
        "weight_ptr": TensorType(dtype="f32", shape=[Dim("sym", "N")], layout=rm),
        "bias_ptr": TensorType(dtype="f32", shape=[Dim("sym", "N")], layout=rm),
        "out_ptr": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
        "out_mean_ptr": TensorType(dtype="f32", shape=[Dim("sym", "M")], layout=rm),
        "out_rstd_ptr": TensorType(dtype="f32", shape=[Dim("sym", "M")], layout=rm),
    }
    ops: list[Op] = []

    # sum_row: [M,1]
    ops.append(Op(op="reduce_sum", inputs=["in_ptr"], output="sum_row", attrs={"axes": [1], "keepdims": True}))
    tensors["sum_row"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("const", 1)], layout=rm)

    ops.append(Op(op="div", inputs=["sum_row"], output="mean_row", attrs={"divisor": "N"}))
    tensors["mean_row"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("const", 1)], layout=rm)

    ops.append(Op(op="sub", inputs=["in_ptr", "mean_row"], output="diff", attrs={}))
    tensors["diff"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm)

    ops.append(Op(op="mul", inputs=["diff", "diff"], output="sq", attrs={}))
    tensors["sq"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm)

    ops.append(Op(op="reduce_sum", inputs=["sq"], output="var_sum", attrs={"axes": [1], "keepdims": True}))
    tensors["var_sum"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("const", 1)], layout=rm)

    ops.append(Op(op="div", inputs=["var_sum"], output="var", attrs={"divisor": "N"}))
    tensors["var"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("const", 1)], layout=rm)

    ops.append(Op(op="const", inputs=[], output="eps", attrs={"value": 1e-5, "dtype": "f32"}))
    tensors["eps"] = TensorType(dtype="f32", shape=[], layout=rm)

    ops.append(Op(op="add", inputs=["var", "eps"], output="var_eps", attrs={}))
    tensors["var_eps"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("const", 1)], layout=rm)

    ops.append(Op(op="rsqrt", inputs=["var_eps"], output="rstd_row", attrs={}))
    tensors["rstd_row"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("const", 1)], layout=rm)

    ops.append(Op(op="mul", inputs=["diff", "rstd_row"], output="xhat", attrs={}))
    tensors["xhat"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm)

    ops.append(
        Op(
            op="broadcast_in_dim",
            inputs=["weight_ptr"],
            output="w_b",
            attrs={"out_shape": ["M", "N"], "broadcast_dims": [1]},
        )
    )
    tensors["w_b"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm)
    ops.append(
        Op(
            op="broadcast_in_dim",
            inputs=["bias_ptr"],
            output="b_b",
            attrs={"out_shape": ["M", "N"], "broadcast_dims": [1]},
        )
    )
    tensors["b_b"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm)

    ops.append(Op(op="mul", inputs=["xhat", "w_b"], output="scaled", attrs={}))
    tensors["scaled"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm)
    ops.append(Op(op="add", inputs=["scaled", "b_b"], output="out_ptr", attrs={}))

    # mean/rstd outputs: reshape [M,1] -> [M]
    ops.append(Op(op="reshape", inputs=["mean_row"], output="out_mean_ptr", attrs={"shape": ["M"]}))
    ops.append(Op(op="reshape", inputs=["rstd_row"], output="out_rstd_ptr", attrs={"shape": ["M"]}))

    schedule = ScheduleSketch(tile_m=16, tile_n=16, tile_k=None, vec_width=1, pipeline_depth=1)
    return IntentFunction(
        name="layer_norm_persistent",
        tensors=tensors,
        ops=ops,
        outputs=["out_ptr", "out_mean_ptr", "out_rstd_ptr"],
        schedule=schedule,
        axis_roles={"M": "spatial", "N": "channel"},
    )


def layer_norm_persistent_spec() -> TileLangKernelSpec:
    return TileLangKernelSpec(
        name="layer_norm_persistent",
        prim_func=make_layer_norm_persistent_prim_func(n=16, threads=128),
        arg_names=["in_ptr", "out_ptr", "weight_ptr", "bias_ptr", "out_mean_ptr", "out_rstd_ptr", "M", "N"],
        canonical_shapes={"M": 16, "N": 16},
        vary_axes=["M"],
        runner=layer_norm_persistent_reference,
        intent_builder=layer_norm_persistent_intent,
        exclude_axes=[],
        constexpr_names=[],
    )


__all__ = [
    "make_layer_norm_persistent_prim_func",
    "layer_norm_persistent_reference",
    "layer_norm_persistent_intent",
    "layer_norm_persistent_spec",
]
