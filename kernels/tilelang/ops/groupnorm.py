from typing import Dict

import numpy as np

from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType
from verify.gen_cases import TestCase

from .spec import TileLangKernelSpec


def make_group_norm_kernel_prim_func(*, c: int = 64, hw: int = 16, num_groups: int = 4, threads: int = 128):
    """
    GroupNorm forward (N, C, HW) with Mean/Rstd outputs (N, G).
    """
    import tilelang.language as T

    N = T.dynamic("N")
    C = int(c)
    HW = int(hw)
    G = int(num_groups)
    if G <= 0 or C % G != 0:
        raise ValueError(f"invalid groupnorm config: C={C} num_groups={G}")
    group_size = C // G

    @T.prim_func
    def main(
        X: T.Tensor((N, C, HW), "float32"),
        Y: T.Tensor((N, C, HW), "float32"),
        W: T.Tensor((C,), "float32"),
        B: T.Tensor((C,), "float32"),
        Mean: T.Tensor((N, G), "float32"),
        Rstd: T.Tensor((N, G), "float32"),
    ):
        eps = T.float32(1e-5)
        denom = T.float32(group_size * HW)
        with T.Kernel(N, G, threads=threads) as (pid_n, pid_g):
            n = pid_n
            g = pid_g
            c0 = g * group_size

            x_tile = T.alloc_fragment((1, group_size, HW), "float32")
            T.copy(X[n, c0, 0], x_tile)
            w_tile = T.alloc_fragment((group_size,), "float32")
            T.copy(W[c0], w_tile)
            b_tile = T.alloc_fragment((group_size,), "float32")
            T.copy(B[c0], b_tile)

            # sum(x)
            sum_hw = T.alloc_fragment((1, group_size, 1), "float32")
            T.reduce_sum(x_tile, sum_hw, dim=2)
            sum_all = T.alloc_fragment((1, 1, 1), "float32")
            T.reduce_sum(sum_hw, sum_all, dim=1)
            mean = sum_all[0, 0, 0] / denom

            # sum(x^2)
            x2_tile = T.alloc_fragment((1, group_size, HW), "float32")
            for c1 in T.serial(group_size):
                for h1 in T.serial(HW):
                    v = x_tile[0, c1, h1]
                    x2_tile[0, c1, h1] = v * v
            sum2_hw = T.alloc_fragment((1, group_size, 1), "float32")
            T.reduce_sum(x2_tile, sum2_hw, dim=2)
            sum2_all = T.alloc_fragment((1, 1, 1), "float32")
            T.reduce_sum(sum2_hw, sum2_all, dim=1)
            mean2 = sum2_all[0, 0, 0] / denom
            var = mean2 - mean * mean
            rstd = T.rsqrt(var + eps)

            Mean[n, g] = mean
            Rstd[n, g] = rstd

            y_tile = T.alloc_fragment((1, group_size, HW), "float32")
            for c1 in T.serial(group_size):
                for h1 in T.serial(HW):
                    y_tile[0, c1, h1] = (x_tile[0, c1, h1] - mean) * rstd * w_tile[c1] + b_tile[c1]
            T.copy(y_tile, Y[n, c0, 0])

    return main


def group_norm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(case.seed))
    N = int(case.shapes["N"])
    C = int(case.shapes["C"])
    HW = int(case.shapes["HW"])
    G = int(case.shapes["num_groups"])
    if G <= 0 or C % G != 0:
        raise ValueError(f"invalid group config: C={C} num_groups={G}")
    group_size = C // G
    x = rng.standard_normal((N, C, HW), dtype=np.float32)
    w = rng.standard_normal((C,), dtype=np.float32)
    b = rng.standard_normal((C,), dtype=np.float32)
    eps = np.float32(1e-5)

    x4 = x.reshape(N, G, group_size, HW)
    mean = np.mean(x4, axis=(2, 3), keepdims=True)
    var = np.mean((x4 - mean) ** 2, axis=(2, 3), keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    x_hat = (x4 - mean) * rstd
    x_hat3 = x_hat.reshape(N, C, HW)
    y = x_hat3 * w[None, :, None] + b[None, :, None]
    return {
        "X": x,
        "W": w,
        "B": b,
        "Y": y.astype(np.float32),
        "Mean": mean.reshape(N, G).astype(np.float32),
        "Rstd": rstd.reshape(N, G).astype(np.float32),
    }


def group_norm_intent() -> IntentFunction:
    rm = TensorLayout(kind="row_major", params={})
    tensors = {
        "X": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm),
        "W": TensorType(dtype="f32", shape=[Dim("sym", "C")], layout=rm),
        "B": TensorType(dtype="f32", shape=[Dim("sym", "C")], layout=rm),
        "Y": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm),
        "Mean": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups")], layout=rm),
        "Rstd": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups")], layout=rm),
    }
    ops: list[Op] = []

    ops.append(
        Op(
            op="reshape",
            inputs=["X"],
            output="X4",
            attrs={"shape": ["N", "num_groups", "group_size", "HW"]},
        )
    )
    tensors["X4"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("sym", "group_size"), Dim("sym", "HW")],
        layout=rm,
    )

    ops.append(Op(op="reduce_sum", inputs=["X4"], output="sum", attrs={"axes": [2, 3], "keepdims": True}))
    tensors["sum"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)],
        layout=rm,
    )
    ops.append(Op(op="div", inputs=["sum"], output="mean4", attrs={"divisor": "num_elements"}))
    tensors["mean4"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)],
        layout=rm,
    )

    ops.append(Op(op="sub", inputs=["X4", "mean4"], output="diff", attrs={}))
    tensors["diff"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("sym", "group_size"), Dim("sym", "HW")],
        layout=rm,
    )
    ops.append(Op(op="mul", inputs=["diff", "diff"], output="sq", attrs={}))
    tensors["sq"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("sym", "group_size"), Dim("sym", "HW")],
        layout=rm,
    )
    ops.append(Op(op="reduce_sum", inputs=["sq"], output="var_sum", attrs={"axes": [2, 3], "keepdims": True}))
    tensors["var_sum"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)],
        layout=rm,
    )
    ops.append(Op(op="div", inputs=["var_sum"], output="var4", attrs={"divisor": "num_elements"}))
    tensors["var4"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)],
        layout=rm,
    )

    ops.append(Op(op="const", inputs=[], output="eps", attrs={"value": 1e-5, "dtype": "f32"}))
    tensors["eps"] = TensorType(dtype="f32", shape=[], layout=rm)
    ops.append(Op(op="add", inputs=["var4", "eps"], output="var_eps", attrs={}))
    tensors["var_eps"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)],
        layout=rm,
    )
    ops.append(Op(op="rsqrt", inputs=["var_eps"], output="rstd4", attrs={}))
    tensors["rstd4"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)],
        layout=rm,
    )

    ops.append(Op(op="mul", inputs=["diff", "rstd4"], output="xhat4", attrs={}))
    tensors["xhat4"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("sym", "group_size"), Dim("sym", "HW")],
        layout=rm,
    )

    ops.append(Op(op="reshape", inputs=["xhat4"], output="xhat3", attrs={"shape": ["N", "C", "HW"]}))
    tensors["xhat3"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm)

    ops.append(
        Op(
            op="broadcast_in_dim",
            inputs=["W"],
            output="W3",
            attrs={"out_shape": ["N", "C", "HW"], "broadcast_dims": [1]},
        )
    )
    tensors["W3"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm)
    ops.append(
        Op(
            op="broadcast_in_dim",
            inputs=["B"],
            output="B3",
            attrs={"out_shape": ["N", "C", "HW"], "broadcast_dims": [1]},
        )
    )
    tensors["B3"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm)

    ops.append(Op(op="mul", inputs=["xhat3", "W3"], output="scaled", attrs={}))
    tensors["scaled"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm)
    ops.append(Op(op="add", inputs=["scaled", "B3"], output="Y", attrs={}))

    # Mean/Rstd outputs: reshape [N,G,1,1] -> [N,G]
    ops.append(Op(op="reshape", inputs=["mean4"], output="Mean", attrs={"shape": ["N", "num_groups"]}))
    ops.append(Op(op="reshape", inputs=["rstd4"], output="Rstd", attrs={"shape": ["N", "num_groups"]}))

    schedule = ScheduleSketch(tile_m=16, tile_n=16, tile_k=None, vec_width=1, pipeline_depth=1)
    return IntentFunction(
        name="group_norm_kernel",
        tensors=tensors,
        ops=ops,
        outputs=["Y", "Mean", "Rstd"],
        schedule=schedule,
        axis_roles={"N": "batch", "C": "channel", "HW": "spatial", "num_groups": "channel"},
    )


def group_norm_kernel_spec() -> TileLangKernelSpec:
    return TileLangKernelSpec(
        name="group_norm_kernel",
        prim_func=make_group_norm_kernel_prim_func(c=64, hw=16, num_groups=4, threads=128),
        arg_names=[
            "X",
            "Y",
            "W",
            "B",
            "Mean",
            "Rstd",
            "N",
            "C",
            "HW",
            "num_groups",
        ],
        canonical_shapes={"N": 16, "C": 64, "HW": 16, "num_groups": 4},
        vary_axes=["N"],
        runner=group_norm_reference,
        intent_builder=group_norm_intent,
        exclude_axes=["group_size", "num_elements"],
        constexpr_names=[],
    )


__all__ = [
    "make_group_norm_kernel_prim_func",
    "group_norm_reference",
    "group_norm_intent",
    "group_norm_kernel_spec",
]
