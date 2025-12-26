from typing import Dict, Tuple

import numpy as np

from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType
from verify.gen_cases import TestCase

from .spec import TileLangKernelSpec


def make_upsample_bicubic2d_aa_prim_func(*, threads: int = 128):
    """
    A small TileLang PrimFunc for an upsample-like kernel.

    The TileLang frontend uses this PrimFunc only as "evidence" (anchors/accesses).
    The semantic diff uses the numpy/PyTorch reference + IntentIR interpreter.
    """
    import tilelang.language as T

    # Keep the PrimFunc tiny and deterministic.
    N = 1
    C = 1
    IH = 4
    IW = 4
    OH = 8
    OW = 8

    @T.prim_func
    def main(I: T.Tensor((N, C, IH, IW), "float32"), O: T.Tensor((N, C, OH, OW), "float32")):
        with T.Kernel(OH, OW, threads=threads) as (pid_oh, pid_ow):
            oh = pid_oh
            ow = pid_ow
            ih = oh // 2
            iw = ow // 2
            O[0, 0, oh, ow] = I[0, 0, ih, iw]

    return main


def _bicubic_reciprocal_scale(src_size: int, dst_size: int, align_corners: bool, scale: float | None) -> float:
    if align_corners:
        if dst_size > 1:
            return float(src_size - 1) / float(dst_size - 1)
        return 0.0
    if scale is not None and scale > 0:
        return 1.0 / float(scale)
    return float(src_size) / float(dst_size)


def upsample_bicubic2d_aa_reference(case: TestCase) -> Dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(int(case.seed))
    N = int(case.shapes.get("N", 1))
    C = int(case.shapes.get("C", 1))
    IH = int(case.shapes.get("IH", 4))
    IW = int(case.shapes.get("IW", 4))
    OH = int(case.shapes.get("OH", 8))
    OW = int(case.shapes.get("OW", 8))

    x = rng.standard_normal((N, C, IH, IW), dtype=np.float32)

    # Match kernels/triton/ops/upsample_bicubic2d_aa.py: align_corners=False by default.
    align_corners = False
    reciprocal_scale_h = _bicubic_reciprocal_scale(IH, OH, align_corners, scale=None)
    reciprocal_scale_w = _bicubic_reciprocal_scale(IW, OW, align_corners, scale=None)

    xt = torch.from_numpy(x)
    yt = F.interpolate(xt, size=(OH, OW), mode="bicubic", align_corners=align_corners, antialias=True)
    y = yt.numpy().astype(np.float32)

    return {
        "I": x,
        "reciprocal_scale_h": np.array(reciprocal_scale_h, dtype=np.float32),
        "reciprocal_scale_w": np.array(reciprocal_scale_w, dtype=np.float32),
        "O": y,
    }


def upsample_bicubic2d_aa_intent() -> IntentFunction:
    rm = TensorLayout(kind="row_major", params={})
    tensors = {
        "I": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "IH"), Dim("sym", "IW")], layout=rm),
        "reciprocal_scale_h": TensorType(dtype="f32", shape=[], layout=rm),
        "reciprocal_scale_w": TensorType(dtype="f32", shape=[], layout=rm),
        "O": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "OH"), Dim("sym", "OW")], layout=rm),
    }
    ops = [
        Op(
            op="upsample_bicubic2d_aa",
            inputs=["I"],
            output="O",
            attrs={"a": -0.5, "support": 2.0, "invscale": 1.0, "separable": True, "normalize_weights": True},
        )
    ]
    schedule = ScheduleSketch(tile_m="BLOCK_Y", tile_n="BLOCK_X", tile_k=None, vec_width=1, pipeline_depth=1)
    return IntentFunction(
        name="upsample_bicubic2d_aa",
        tensors=tensors,
        ops=ops,
        outputs=["O"],
        schedule=schedule,
        axis_roles={"N": "batch", "C": "channel", "IH": "spatial", "IW": "spatial", "OH": "spatial", "OW": "spatial"},
    )


def upsample_bicubic2d_aa_spec() -> TileLangKernelSpec:
    # Keep cases small so macro expansion + interpreter stay fast in CI.
    base: Dict[str, int] = {"N": 1, "C": 1, "IH": 4, "IW": 4, "OH": 8, "OW": 8}
    return TileLangKernelSpec(
        name="upsample_bicubic2d_aa",
        prim_func=make_upsample_bicubic2d_aa_prim_func(threads=128),
        arg_names=["I", "reciprocal_scale_h", "reciprocal_scale_w", "O", "N", "C", "IH", "IW", "OH", "OW"],
        canonical_shapes=base,
        vary_axes=[],
        runner=upsample_bicubic2d_aa_reference,
        intent_builder=upsample_bicubic2d_aa_intent,
        exclude_axes=[],
        constexpr_names=[],
    )


__all__ = [
    "make_upsample_bicubic2d_aa_prim_func",
    "upsample_bicubic2d_aa_reference",
    "upsample_bicubic2d_aa_intent",
    "upsample_bicubic2d_aa_spec",
]

