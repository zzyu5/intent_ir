"""
TileLang MVP full pipeline runner (PR#9).

This mirrors the Triton pipeline shape, but uses the TileLang adapter:
  TileLang DSL -> CertificateV2 -> obligations -> contract -> LLM->IntentIR -> diff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from frontends.common.contract_v2 import evaluate_contract_v2
from frontends.common.obligations import evaluate_obligations
from pipeline import registry as pipeline_registry
from pipeline.interfaces import FrontendConstraints
from verify.diff_runner import run_diff
from verify.gen_cases import GeneratedCases, TestCase, generate_cases_split
from verify.metamorphic import run_bounded_exhaustive, run_metamorphic_suite
from verify.mutation import run_mutation_kill
from verify.tolerances import infer_tolerances

from intent_ir.llm import LLMIntentHub
from intent_ir.macros import expand_macros, enrich_intent_macros
from intent_ir.parser import CandidateIntent
from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType
from intent_ir.ir.printer_mlir_like import print_mlir_like

from frontends.common.static_validate import static_validate
from frontends.tilelang.runtime import infer_written_global_buffers, run_tilelang_kernel_io

from kernels.tilelang.ops.any_kernel_dim import make_any_kernel_dim_prim_func
from kernels.tilelang.ops.groupnorm import make_group_norm_kernel_prim_func
from kernels.tilelang.ops.softmax_inner import make_softmax_inner_prim_func
from kernels.tilelang.ops.layernorm import make_layer_norm_persistent_prim_func
from kernels.tilelang.ops.upsample_bicubic2d_aa import make_upsample_bicubic2d_aa_prim_func
from kernels.tilelang.ops._attn_fwd import make_attn_fwd_prim_func
from kernels.tilelang.ops.gemm import make_gemm_relu_prim_func


ROOT = Path(__file__).resolve().parents[2]
_LLM_HUB = LLMIntentHub()


@dataclass
class KernelSpec:
    """
    TileLang pipeline kernel spec (internal).

    Keeping this spec in the pipeline (not under kernels/) mirrors the Triton layout:
    - `kernels/tilelang/ops/*`: clean TileLang kernels only
    - `pipeline/tilelang/core.py`: runners, deterministic intent builders, and suite composition
    """

    name: str
    prim_func: Any
    canonical_shapes: Dict[str, int]
    vary_axes: List[str]
    runner: Callable[[TestCase], Dict[str, np.ndarray]]
    intent_builder: Callable[[], IntentFunction]
    exclude_axes: List[str] | None = None
    constexpr_names: List[str] | None = None
    # For LLM evidence only (not used by TileLang runtime executor).
    arg_names: List[str] | None = None


def _rm_layout() -> TensorLayout:
    return TensorLayout(kind="row_major", params={})


def _any_kernel_dim_reference(case: TestCase) -> Dict[str, np.ndarray]:
    if case.inputs and "inp" in case.inputs:
        inp = np.asarray(case.inputs["inp"])
    else:
        rng = np.random.default_rng(int(case.seed))
        m = int(case.shapes["M"])
        n = int(case.shapes["N"])
        inp = rng.integers(0, 2, size=(m, n)).astype(np.float32)
    out = np.any(inp != 0, axis=1)
    return {"inp": inp, "out": out}


def _any_kernel_dim_intent() -> IntentFunction:
    rm = _rm_layout()
    tensors = {
        "inp": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
        "out": TensorType(dtype="bool", shape=[Dim("sym", "M")], layout=rm),
    }
    ops = [Op(op="reduce_any", inputs=["inp"], output="out", attrs={"axes": [1], "keepdims": False})]
    schedule = ScheduleSketch(tile_m=16, tile_n=16, tile_k=None, vec_width=1, pipeline_depth=1)
    return IntentFunction(
        name="any_kernel_dim",
        tensors=tensors,
        ops=ops,
        outputs=["out"],
        schedule=schedule,
        axis_roles={"M": "spatial", "N": "reduction"},
    )


def _group_norm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    n = int(case.shapes["N"])
    c = int(case.shapes["C"])
    hw = int(case.shapes["HW"])
    g = int(case.shapes["num_groups"])
    if g <= 0 or c % g != 0:
        raise ValueError(f"invalid group config: C={c} num_groups={g}")
    group_size = c // g
    if case.inputs and "X" in case.inputs:
        x = np.asarray(case.inputs["X"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        x = rng.standard_normal((n, c, hw), dtype=np.float32)
    if case.inputs and "W" in case.inputs:
        w = np.asarray(case.inputs["W"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        w = rng.standard_normal((c,), dtype=np.float32)
    if case.inputs and "B" in case.inputs:
        b = np.asarray(case.inputs["B"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        b = rng.standard_normal((c,), dtype=np.float32)
    eps = np.float32(1e-5)

    x4 = x.reshape(n, g, group_size, hw)
    mean = np.mean(x4, axis=(2, 3), keepdims=True)
    var = np.mean((x4 - mean) ** 2, axis=(2, 3), keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    x_hat = (x4 - mean) * rstd
    x_hat3 = x_hat.reshape(n, c, hw)
    y = x_hat3 * w[None, :, None] + b[None, :, None]
    return {
        "X": x,
        "W": w,
        "B": b,
        "Y": y.astype(np.float32),
        "Mean": mean.reshape(n, g).astype(np.float32),
        "Rstd": rstd.reshape(n, g).astype(np.float32),
    }


def _group_norm_intent() -> IntentFunction:
    rm = _rm_layout()
    tensors = {
        "X": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm),
        "W": TensorType(dtype="f32", shape=[Dim("sym", "C")], layout=rm),
        "B": TensorType(dtype="f32", shape=[Dim("sym", "C")], layout=rm),
        "Y": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm),
        "Mean": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups")], layout=rm),
        "Rstd": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups")], layout=rm),
    }
    ops: list[Op] = []

    ops.append(Op(op="reshape", inputs=["X"], output="X4", attrs={"shape": ["N", "num_groups", "group_size", "HW"]}))
    tensors["X4"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("sym", "group_size"), Dim("sym", "HW")],
        layout=rm,
    )

    ops.append(Op(op="reduce_sum", inputs=["X4"], output="sum", attrs={"axes": [2, 3], "keepdims": True}))
    tensors["sum"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)], layout=rm)
    ops.append(Op(op="div", inputs=["sum"], output="mean4", attrs={"divisor": "num_elements"}))
    tensors["mean4"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)], layout=rm)

    ops.append(Op(op="sub", inputs=["X4", "mean4"], output="diff", attrs={}))
    tensors["diff"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("sym", "group_size"), Dim("sym", "HW")], layout=rm)
    ops.append(Op(op="mul", inputs=["diff", "diff"], output="sq", attrs={}))
    tensors["sq"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("sym", "group_size"), Dim("sym", "HW")], layout=rm)
    ops.append(Op(op="reduce_sum", inputs=["sq"], output="var_sum", attrs={"axes": [2, 3], "keepdims": True}))
    tensors["var_sum"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)], layout=rm)
    ops.append(Op(op="div", inputs=["var_sum"], output="var4", attrs={"divisor": "num_elements"}))
    tensors["var4"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)], layout=rm)

    ops.append(Op(op="const", inputs=[], output="eps", attrs={"value": 1e-5, "dtype": "f32"}))
    tensors["eps"] = TensorType(dtype="f32", shape=[], layout=rm)
    ops.append(Op(op="add", inputs=["var4", "eps"], output="var_eps", attrs={}))
    tensors["var_eps"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)], layout=rm)
    ops.append(Op(op="rsqrt", inputs=["var_eps"], output="rstd4", attrs={}))
    tensors["rstd4"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)], layout=rm)

    ops.append(Op(op="mul", inputs=["diff", "rstd4"], output="xhat4", attrs={}))
    tensors["xhat4"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("sym", "group_size"), Dim("sym", "HW")], layout=rm)

    ops.append(Op(op="reshape", inputs=["xhat4"], output="xhat3", attrs={"shape": ["N", "C", "HW"]}))
    tensors["xhat3"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm)

    ops.append(Op(op="broadcast_in_dim", inputs=["W"], output="W3", attrs={"out_shape": ["N", "C", "HW"], "broadcast_dims": [1]}))
    tensors["W3"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm)
    ops.append(Op(op="broadcast_in_dim", inputs=["B"], output="B3", attrs={"out_shape": ["N", "C", "HW"], "broadcast_dims": [1]}))
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


def _softmax_inner_reference(case: TestCase) -> Dict[str, np.ndarray]:
    if case.inputs and "input_ptr" in case.inputs:
        x = np.asarray(case.inputs["input_ptr"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        m = int(case.shapes["M"])
        n = int(case.shapes["N"])
        x = rng.standard_normal((m, n), dtype=np.float32)
    x_max = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - x_max)
    y = e / np.sum(e, axis=1, keepdims=True)
    return {"input_ptr": x, "output_ptr": y}


def _softmax_inner_intent() -> IntentFunction:
    rm = _rm_layout()
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


def _layer_norm_persistent_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes["M"])
    n = int(case.shapes["N"])
    if case.inputs and "in_ptr" in case.inputs:
        x = np.asarray(case.inputs["in_ptr"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        x = rng.standard_normal((m, n), dtype=np.float32)
    if case.inputs and "weight_ptr" in case.inputs:
        w = np.asarray(case.inputs["weight_ptr"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        w = rng.standard_normal((n,), dtype=np.float32)
    if case.inputs and "bias_ptr" in case.inputs:
        b = np.asarray(case.inputs["bias_ptr"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        b = rng.standard_normal((n,), dtype=np.float32)
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


def _layer_norm_persistent_intent() -> IntentFunction:
    rm = _rm_layout()
    tensors = {
        "in_ptr": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
        "weight_ptr": TensorType(dtype="f32", shape=[Dim("sym", "N")], layout=rm),
        "bias_ptr": TensorType(dtype="f32", shape=[Dim("sym", "N")], layout=rm),
        "out_ptr": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
        "out_mean_ptr": TensorType(dtype="f32", shape=[Dim("sym", "M")], layout=rm),
        "out_rstd_ptr": TensorType(dtype="f32", shape=[Dim("sym", "M")], layout=rm),
    }
    ops: list[Op] = []

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

    ops.append(Op(op="broadcast_in_dim", inputs=["weight_ptr"], output="w_b", attrs={"out_shape": ["M", "N"], "broadcast_dims": [1]}))
    tensors["w_b"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm)
    ops.append(Op(op="broadcast_in_dim", inputs=["bias_ptr"], output="b_b", attrs={"out_shape": ["M", "N"], "broadcast_dims": [1]}))
    tensors["b_b"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm)

    ops.append(Op(op="mul", inputs=["xhat", "w_b"], output="scaled", attrs={}))
    tensors["scaled"] = TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm)
    ops.append(Op(op="add", inputs=["scaled", "b_b"], output="out_ptr", attrs={}))

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


def _attn_fwd_reference(case: TestCase) -> Dict[str, np.ndarray]:
    q_ctx = int(case.shapes.get("Q_CTX", 16))
    kv_ctx = int(case.shapes.get("KV_CTX", 16))
    head_dim = int(case.shapes.get("HEAD_DIM", 16))
    if case.inputs and "Q" in case.inputs:
        q = np.asarray(case.inputs["Q"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        q = rng.standard_normal((q_ctx, head_dim), dtype=np.float32)
    if case.inputs and "K" in case.inputs:
        k = np.asarray(case.inputs["K"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        k = rng.standard_normal((kv_ctx, head_dim), dtype=np.float32)
    if case.inputs and "V" in case.inputs:
        v = np.asarray(case.inputs["V"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        v = rng.standard_normal((kv_ctx, head_dim), dtype=np.float32)
    if case.inputs and "sm_scale" in case.inputs:
        sm_scale = np.asarray(case.inputs["sm_scale"], dtype=np.float32).reshape(())
    else:
        sm_scale = np.array(1.0 / np.sqrt(float(head_dim)), dtype=np.float32)

    scores = (q @ k.T) * sm_scale
    scores_max = np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores - scores_max)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    out = probs @ v
    return {"Q": q, "K": k, "V": v, "sm_scale": sm_scale, "Out": out.astype(np.float32)}


def _attn_fwd_intent() -> IntentFunction:
    rm = _rm_layout()
    tensors = {
        "Q": TensorType(dtype="f32", shape=[Dim("sym", "Q_CTX"), Dim("sym", "HEAD_DIM")], layout=rm),
        "K": TensorType(dtype="f32", shape=[Dim("sym", "KV_CTX"), Dim("sym", "HEAD_DIM")], layout=rm),
        "V": TensorType(dtype="f32", shape=[Dim("sym", "KV_CTX"), Dim("sym", "HEAD_DIM")], layout=rm),
        "sm_scale": TensorType(dtype="f32", shape=[], layout=rm),
        "Out": TensorType(dtype="f32", shape=[Dim("sym", "Q_CTX"), Dim("sym", "HEAD_DIM")], layout=rm),
    }

    ops: list[Op] = []
    ops.append(Op(op="transpose", inputs=["K"], output="K_t", attrs={"perm": [1, 0]}))
    tensors["K_t"] = TensorType(dtype="f32", shape=[Dim("sym", "HEAD_DIM"), Dim("sym", "KV_CTX")], layout=rm)

    ops.append(Op(op="matmul", inputs=["Q", "K_t"], output="scores", attrs={}))
    tensors["scores"] = TensorType(dtype="f32", shape=[Dim("sym", "Q_CTX"), Dim("sym", "KV_CTX")], layout=rm)

    ops.append(Op(op="mul", inputs=["scores", "sm_scale"], output="scores_s", attrs={}))
    tensors["scores_s"] = TensorType(dtype="f32", shape=[Dim("sym", "Q_CTX"), Dim("sym", "KV_CTX")], layout=rm)

    ops.append(Op(op="softmax", inputs=["scores_s"], output="probs", attrs={"axis": -1}))
    tensors["probs"] = TensorType(dtype="f32", shape=[Dim("sym", "Q_CTX"), Dim("sym", "KV_CTX")], layout=rm)

    ops.append(Op(op="matmul", inputs=["probs", "V"], output="Out", attrs={}))

    schedule = ScheduleSketch(tile_m=16, tile_n=16, tile_k=16, vec_width=1, pipeline_depth=1)
    return IntentFunction(
        name="_attn_fwd",
        tensors=tensors,
        ops=ops,
        outputs=["Out"],
        schedule=schedule,
        axis_roles={"Q_CTX": "spatial", "KV_CTX": "reduction", "HEAD_DIM": "channel"},
    )


def _bicubic_reciprocal_scale(src_size: int, dst_size: int, align_corners: bool, scale: float | None) -> float:
    if align_corners:
        if dst_size > 1:
            return float(src_size - 1) / float(dst_size - 1)
        return 0.0
    if scale is not None and scale > 0:
        return 1.0 / float(scale)
    return float(src_size) / float(dst_size)


def _upsample_bicubic2d_aa_reference(case: TestCase) -> Dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    n = int(case.shapes.get("N", 1))
    c = int(case.shapes.get("C", 1))
    ih = int(case.shapes.get("IH", 4))
    iw = int(case.shapes.get("IW", 4))
    oh = int(case.shapes.get("OH", 8))
    ow = int(case.shapes.get("OW", 8))

    if case.inputs and "I" in case.inputs:
        x = np.asarray(case.inputs["I"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        x = rng.standard_normal((n, c, ih, iw), dtype=np.float32)

    align_corners = False
    reciprocal_scale_h = _bicubic_reciprocal_scale(ih, oh, align_corners, scale=None)
    reciprocal_scale_w = _bicubic_reciprocal_scale(iw, ow, align_corners, scale=None)

    xt = torch.from_numpy(x)
    yt = F.interpolate(xt, size=(oh, ow), mode="bicubic", align_corners=align_corners, antialias=True)
    y = yt.numpy().astype(np.float32)

    return {
        "I": x,
        "reciprocal_scale_h": np.array(reciprocal_scale_h, dtype=np.float32),
        "reciprocal_scale_w": np.array(reciprocal_scale_w, dtype=np.float32),
        "O": y,
    }


def _upsample_bicubic2d_aa_intent() -> IntentFunction:
    rm = _rm_layout()
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


def _gemm_relu_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes["M"])
    n = int(case.shapes["N"])
    k = int(case.shapes["K"])
    if case.inputs and "A" in case.inputs:
        a = np.asarray(case.inputs["A"], dtype=np.float16)
    else:
        rng = np.random.default_rng(int(case.seed))
        a = rng.standard_normal((m, k), dtype=np.float32).astype(np.float16)
    if case.inputs and "B" in case.inputs:
        b = np.asarray(case.inputs["B"], dtype=np.float16)
    else:
        rng = np.random.default_rng(int(case.seed))
        b = rng.standard_normal((k, n), dtype=np.float32).astype(np.float16)
    c = np.maximum((a.astype(np.float32) @ b.astype(np.float32)), 0.0).astype(np.float16)
    return {"A": a, "B": b, "C": c}


def _gemm_relu_intent() -> IntentFunction:
    rm = _rm_layout()
    tensors = {
        "A": TensorType(dtype="f16", shape=[Dim("sym", "M"), Dim("sym", "K")], layout=rm),
        "B": TensorType(dtype="f16", shape=[Dim("sym", "K"), Dim("sym", "N")], layout=rm),
        "C": TensorType(dtype="f16", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
    }
    ops = [Op(op="matmul", inputs=["A", "B"], output="mm", attrs={}), Op(op="relu", inputs=["mm"], output="C", attrs={})]
    schedule = ScheduleSketch(tile_m=128, tile_n=128, tile_k=32, vec_width=1, pipeline_depth=3)
    return IntentFunction(
        name="tilelang_gemm_relu",
        tensors=tensors,
        ops=ops,
        outputs=["C"],
        schedule=schedule,
        axis_roles={"M": "spatial", "N": "spatial", "K": "reduction"},
    )


def _ensure_schedule_tilelang(intent, spec) -> None:
    """
    Keep schedule visible in IntentIR.
    - Prefer schedule produced by LLM.
    - If missing, fall back to the spec's deterministic intent schedule (sketch).
    """
    if getattr(intent, "schedule", None) is not None:
        return
    try:
        det = spec.intent_builder()
        intent.schedule = det.schedule
    except Exception:
        return


def _buffer_param_names(prim_func) -> List[str]:
    try:
        from tvm import tir  # noqa: PLC0415

        if not isinstance(prim_func, tir.PrimFunc):
            return []
        out: List[str] = []
        for p in list(prim_func.params):
            if p in prim_func.buffer_map:
                out.append(str(prim_func.buffer_map[p].name))
        return out
    except Exception:
        return []


def _run_tilelang_ref(spec: KernelSpec, case: TestCase) -> Dict[str, np.ndarray]:
    """
    Execute the real TileLang kernel to produce a reference IO snapshot.

    Inputs are generated by the spec's numpy reference runner (for deterministic RNG),
    but outputs come from executing `spec.prim_func` via tilelang.compile.
    """
    ref_io = spec.runner(case)
    prim_func = spec.prim_func
    buf_names = _buffer_param_names(prim_func)
    written = set(infer_written_global_buffers(prim_func))
    inputs_np = {k: np.asarray(v) for k, v in ref_io.items() if k in buf_names and k not in written}
    io = run_tilelang_kernel_io(prim_func, bindings=dict(case.shapes), inputs_np=inputs_np)
    # Keep scalar inputs from reference runner (e.g., eps) if present.
    for k, v in ref_io.items():
        if k not in io and k not in buf_names:
            io[k] = np.asarray(v)
    return io


def mvp_kernel_specs() -> List[KernelSpec]:
    """
    PR#9 original MVP: one anchor-strong kernel to prove the pipeline can host TileLang.
    """
    prim = make_gemm_relu_prim_func(block_m=128, block_n=128, block_k=32, num_stages=3, threads=128)
    return [
        KernelSpec(
            name="tilelang_gemm",
            prim_func=prim,
            arg_names=["A", "B", "C", "M", "N", "K"],
            canonical_shapes={"M": 256, "N": 256, "K": 256},
            vary_axes=["M", "N", "K"],
            runner=_gemm_relu_reference,
            intent_builder=_gemm_relu_intent,
            exclude_axes=[],
            constexpr_names=[],
        )
    ]


def default_kernel_specs() -> List[KernelSpec]:
    """
    Regression suite: mirror Triton's 6 representative kernels (by name).
    """
    return [
        KernelSpec(
            name="any_kernel_dim",
            prim_func=make_any_kernel_dim_prim_func(n=16, threads=128),
            arg_names=["inp", "out", "M", "N"],
            canonical_shapes={"M": 16, "N": 16},
            vary_axes=["M"],
            runner=_any_kernel_dim_reference,
            intent_builder=_any_kernel_dim_intent,
            exclude_axes=[],
            constexpr_names=[],
        ),
        KernelSpec(
            name="group_norm_kernel",
            prim_func=make_group_norm_kernel_prim_func(c=64, hw=16, num_groups=4, threads=128),
            arg_names=["X", "Y", "W", "B", "Mean", "Rstd", "N", "C", "HW", "num_groups"],
            canonical_shapes={"N": 16, "C": 64, "HW": 16, "num_groups": 4},
            vary_axes=["N"],
            runner=_group_norm_reference,
            intent_builder=_group_norm_intent,
            exclude_axes=["group_size", "num_elements"],
            constexpr_names=[],
        ),
        KernelSpec(
            name="_attn_fwd",
            prim_func=make_attn_fwd_prim_func(q_ctx=16, kv_ctx=16, head_dim=16, threads=128),
            arg_names=["Q", "K", "V", "sm_scale", "Out", "Q_CTX", "KV_CTX", "HEAD_DIM"],
            canonical_shapes={"Q_CTX": 16, "KV_CTX": 16, "HEAD_DIM": 16},
            vary_axes=[],
            runner=_attn_fwd_reference,
            intent_builder=_attn_fwd_intent,
            exclude_axes=[],
            constexpr_names=[],
        ),
        KernelSpec(
            name="softmax_inner",
            prim_func=make_softmax_inner_prim_func(n=16, threads=128),
            arg_names=["output_ptr", "input_ptr", "row_max_ptr", "row_sum_ptr", "M", "N"],
            canonical_shapes={"M": 16, "N": 16},
            vary_axes=["M"],
            runner=_softmax_inner_reference,
            intent_builder=_softmax_inner_intent,
            exclude_axes=[],
            constexpr_names=[],
        ),
        KernelSpec(
            name="layer_norm_persistent",
            prim_func=make_layer_norm_persistent_prim_func(n=16, threads=128),
            arg_names=["in_ptr", "out_ptr", "weight_ptr", "bias_ptr", "out_mean_ptr", "out_rstd_ptr", "M", "N"],
            canonical_shapes={"M": 16, "N": 16},
            vary_axes=["M"],
            runner=_layer_norm_persistent_reference,
            intent_builder=_layer_norm_persistent_intent,
            exclude_axes=[],
            constexpr_names=[],
        ),
        KernelSpec(
            name="upsample_bicubic2d_aa",
            prim_func=make_upsample_bicubic2d_aa_prim_func(threads=128),
            arg_names=["I", "reciprocal_scale_h", "reciprocal_scale_w", "O", "N", "C", "IH", "IW", "OH", "OW"],
            canonical_shapes={"N": 1, "C": 1, "IH": 4, "IW": 4, "OH": 8, "OW": 8},
            vary_axes=[],
            runner=_upsample_bicubic2d_aa_reference,
            intent_builder=_upsample_bicubic2d_aa_intent,
            exclude_axes=[],
            constexpr_names=[],
        ),
    ]


def run_pipeline_for_spec(
    spec: KernelSpec,
    *,
    out_dir: Path,
    cases_limit: int = 8,
    stage_c: bool = True,
    mutation_kill: bool = True,
    use_llm: bool = True,
    use_tilelang_runtime: bool = True,
    llm_model: Optional[str] = None,
) -> Dict[str, object]:
    report: Dict[str, object] = {"kernel": spec.name, "frontend": "tilelang"}
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = pipeline_registry.get("tilelang")

    print(f"[{spec.name}] stage1: build tilelang descriptor", flush=True)
    desc = adapter.build_descriptor(spec)
    desc.meta["artifact_dir"] = str(out_dir)
    (out_dir / f"{spec.name}.tilelang_tir.py").write_text(desc.source_text, encoding="utf-8")
    report["descriptor"] = desc.to_json_dict()

    print(f"[{spec.name}] stage2: ensure tilelang artifacts", flush=True)
    desc = adapter.ensure_artifacts(desc, spec)
    print(f"[{spec.name}] stage3: launch tilelang once (baseline)", flush=True)
    baseline_case = TestCase(shapes=dict(spec.canonical_shapes), dtypes={}, seed=0)
    baseline_io_raw = (_run_tilelang_ref(spec, baseline_case) if use_tilelang_runtime else spec.runner(baseline_case))
    report["baseline"] = {
        "shapes": dict(baseline_case.shapes),
        "seed": int(baseline_case.seed),
        "npz_path": None,
        "keys": sorted(list(baseline_io_raw.keys())),
        "source": ("tilelang_runtime" if use_tilelang_runtime else "numpy_reference"),
    }

    print(f"[{spec.name}] stage4: Task4 facts/constraints/certificate", flush=True)
    facts = adapter.extract_facts(desc)
    constraints: FrontendConstraints = adapter.extract_constraints(desc, facts)
    cert_v2 = adapter.build_certificate(desc, facts, constraints)
    obligations = evaluate_obligations(desc, cert_v2)
    cert_v2.semantic_facts["obligations"] = [o.to_json_dict() for o in obligations]
    contract = evaluate_contract_v2(desc, cert_v2, obligations, constraints=constraints)

    report["certificate_v2"] = cert_v2.to_json_dict()
    (out_dir / f"{spec.name}.certificate_v2.json").write_text(json.dumps(report["certificate_v2"], indent=2), encoding="utf-8")
    report["obligations"] = [o.to_json_dict() for o in obligations]
    report["contract"] = {
        "level": contract.level,
        "reasons": list(contract.reasons),
        "assumptions": list(contract.assumptions),
        "signals": dict(contract.signals),
    }
    (out_dir / f"{spec.name}.contract.json").write_text(json.dumps(report["contract"], indent=2), encoding="utf-8")

    # 2) IntentIR: LLM (default) or deterministic fallback (tests/CI).
    print(f"[{spec.name}] stage5: LLM -> IntentIR (may take a while)", flush=True)
    if use_llm:
        cand = _LLM_HUB.lift(desc, model=llm_model)
        enrich_intent_macros(cand.intent)
        _ensure_schedule_tilelang(cand.intent, spec)
        report["llm_trace"] = dict(cand.llm_trace or {})
    else:
        intent = spec.intent_builder()
        cand = CandidateIntent(intent=intent, llm_trace={"provider": "tilelang_deterministic"})

    (out_dir / f"{spec.name}.intentir.mlir").write_text(print_mlir_like(cand.intent), encoding="utf-8")
    report["intent"] = cand.intent.to_json_dict()

    cand_expanded: CandidateIntent | None = None
    try:
        expanded_intent = expand_macros(cand.intent)
        cand_expanded = CandidateIntent(
            intent=expanded_intent,
            problem_params=dict(cand.problem_params),
            schedule_params=dict(cand.schedule_params),
            raw_json=dict(cand.raw_json),
            llm_trace=dict(cand.llm_trace),
        )
        _ensure_schedule_tilelang(cand_expanded.intent, spec)
        (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(print_mlir_like(expanded_intent), encoding="utf-8")
        report["intent_expanded"] = expanded_intent.to_json_dict()
    except Exception as e:
        report["intent_expanded"] = None
        report["intent_expand_error"] = f"{type(e).__name__}: {e}"

    print(f"[{spec.name}] stage6: Task4 static validation", flush=True)
    sv = static_validate((cand_expanded.intent if cand_expanded is not None else cand.intent), cert_v2)
    report["static_validation"] = {
        "ok": bool(sv.ok),
        "reasons": list(sv.reasons),
        "obligations": [{"id": o.id, "status": o.status, "detail": o.detail} for o in sv.obligations],
    }
    if use_llm and not sv.ok:
        # One conservative repair round using certificate-derived feedback.
        try:
            cand_fix = _LLM_HUB.lift(desc, feedback=list(sv.reasons), model=llm_model)
            enrich_intent_macros(cand_fix.intent)
            _ensure_schedule_tilelang(cand_fix.intent, spec)
            report["llm_trace"] = dict(cand_fix.llm_trace or {})
            cand = cand_fix
            (out_dir / f"{spec.name}.intentir.mlir").write_text(print_mlir_like(cand.intent), encoding="utf-8")
            report["intent"] = cand.intent.to_json_dict()
            expanded_fix = expand_macros(cand.intent)
            cand_expanded = CandidateIntent(
                intent=expanded_fix,
                problem_params=dict(cand.problem_params),
                schedule_params=dict(cand.schedule_params),
                raw_json=dict(cand.raw_json),
                llm_trace=dict(cand.llm_trace),
            )
            _ensure_schedule_tilelang(cand_expanded.intent, spec)
            (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(print_mlir_like(expanded_fix), encoding="utf-8")
            report["intent_expanded"] = expanded_fix.to_json_dict()
            sv = static_validate((cand_expanded.intent if cand_expanded is not None else cand.intent), cert_v2)
            report["static_validation"] = {
                "ok": bool(sv.ok),
                "reasons": list(sv.reasons),
                "obligations": [{"id": o.id, "status": o.status, "detail": o.detail} for o in sv.obligations],
            }
        except Exception:
            pass

    # 3) Stage B: cases + diff
    print(f"[{spec.name}] stage7: Task5 cases + diff", flush=True)
    cand_for_run = cand_expanded or cand
    use_rt_ref = bool(use_tilelang_runtime) and (contract.level != "OUT_OF_SCOPE")
    run_ref_fn = (lambda c: _run_tilelang_ref(spec, c)) if use_rt_ref else spec.runner
    cases_pack: GeneratedCases = generate_cases_split(
        cand_for_run.intent,
        constraints=constraints,
        limit=int(cases_limit),
        seed=0,
        axes=list(spec.vary_axes),
        exclude_axes=list(spec.exclude_axes or []),
        assumptions=list(contract.assumptions),
        base_shapes=dict(spec.canonical_shapes),
    )
    cases_in = list(cases_pack.in_contract)
    cases_out = list(cases_pack.out_of_contract)
    report["cases"] = {"in_contract": [dict(c.shapes) for c in cases_in], "out_of_contract": [dict(c.shapes) for c in cases_out]}

    tol = infer_tolerances(cand_for_run.intent).to_dict()
    report["tolerances"] = dict(tol)
    diffs_in, cex_in = run_diff(cand_for_run.intent, run_ref_fn, cases_in, tolerances=tol)
    diff_ok = bool(diffs_in and all(d.ok for d in diffs_in))
    if diffs_in:
        worst = max(diffs_in, key=lambda d: (not d.ok, d.max_abs_err))
        report["diff"] = {
            "ok": bool(diff_ok),
            "worst": {"summary": worst.summary, "max_abs": float(worst.max_abs_err), "max_rel": float(worst.max_rel_err)},
            "results": [
                {
                    "case_shapes": dict(cases_in[i].shapes),
                    "ok": bool(diffs_in[i].ok),
                    "summary": str(diffs_in[i].summary),
                    "max_abs": float(diffs_in[i].max_abs_err),
                    "max_rel": float(diffs_in[i].max_rel_err),
                }
                for i in range(min(len(cases_in), len(diffs_in)))
            ],
        }
    else:
        report["diff"] = {"ok": False, "error": "no diff results"}
    if cex_in:
        report["counterexamples"] = [
            {"shapes": dict(cx.case.shapes), "summary": cx.diff.summary, "hints": list(cx.hints)} for cx in cex_in[:3]
        ]

    # If dynamic diff fails, do one bounded LLM repair round using concrete feedback.
    # This is deliberately conservative (1 retry) to respect LLM rate limits.
    if use_llm and diffs_in and not diff_ok:
        worst_summary = (report.get("diff") or {}).get("worst", {}).get("summary")
        ce0 = (report.get("counterexamples") or [{}])[0]
        feedback3: List[str] = []
        if spec.name == "group_norm_kernel":
            feedback3 += [
                "Your groupnorm math is wrong: reduce_sum returns SUM. You must divide by num_elements=group_size*HW for mean and var.",
                "Implement mean = reduce_sum(X, dims=[2,3], keepdims=true, scale='1.0/(group_size*HW)').",
                "Implement var = reduce_sum((X-mean)^2, dims=[2,3], keepdims=true, scale='1.0/(group_size*HW)').",
                "Then rstd = rsqrt(var + eps).",
            ]
        if spec.name == "layer_norm_persistent":
            feedback3 += ["Your layernorm math is wrong: reduce_sum returns SUM. You must divide by N for mean/var."]
        if worst_summary:
            feedback3.append(f"Observed diff failure: {worst_summary}")
        if ce0.get("shapes"):
            feedback3.append(f"Counterexample shapes: {ce0.get('shapes')}")

        if feedback3:
            try:
                cand_fix = _LLM_HUB.lift(desc, feedback=feedback3, model=llm_model)
                enrich_intent_macros(cand_fix.intent)
                _ensure_schedule_tilelang(cand_fix.intent, spec)
                report["llm_trace"] = dict(cand_fix.llm_trace or {})
                cand = cand_fix
                (out_dir / f"{spec.name}.intentir.mlir").write_text(print_mlir_like(cand.intent), encoding="utf-8")
                report["intent"] = cand.intent.to_json_dict()

                expanded_fix = expand_macros(cand.intent)
                cand_expanded = CandidateIntent(
                    intent=expanded_fix,
                    problem_params=dict(cand.problem_params),
                    schedule_params=dict(cand.schedule_params),
                    raw_json=dict(cand.raw_json),
                    llm_trace=dict(cand.llm_trace),
                )
                _ensure_schedule_tilelang(cand_expanded.intent, spec)
                (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(print_mlir_like(expanded_fix), encoding="utf-8")
                report["intent_expanded"] = expanded_fix.to_json_dict()
                cand_for_run = cand_expanded

                # Re-run diff with a small case set to confirm the repair.
                tol = infer_tolerances(cand_for_run.intent).to_dict()
                report["tolerances"] = dict(tol)
                cases_fix_pack: GeneratedCases = generate_cases_split(
                    cand_for_run.intent,
                    constraints=constraints,
                    limit=min(4, int(cases_limit)),
                    seed=0,
                    axes=list(spec.vary_axes),
                    exclude_axes=list(spec.exclude_axes or []),
                    assumptions=list(contract.assumptions),
                    base_shapes=dict(spec.canonical_shapes),
                )
                cases_in = list(cases_fix_pack.in_contract)
                report["cases"] = {"in_contract": [dict(c.shapes) for c in cases_in], "out_of_contract": []}
                diffs_in, cex_in = run_diff(cand_for_run.intent, run_ref_fn, cases_in, tolerances=tol)
                diff_ok = bool(diffs_in and all(d.ok for d in diffs_in))
                if diffs_in:
                    worst = max(diffs_in, key=lambda d: (not d.ok, d.max_abs_err))
                    report["diff"] = {
                        "ok": bool(diff_ok),
                        "worst": {"summary": worst.summary, "max_abs": float(worst.max_abs_err), "max_rel": float(worst.max_rel_err)},
                        "results": [
                            {
                                "case_shapes": dict(cases_in[i].shapes),
                                "ok": bool(diffs_in[i].ok),
                                "summary": str(diffs_in[i].summary),
                                "max_abs": float(diffs_in[i].max_abs_err),
                                "max_rel": float(diffs_in[i].max_rel_err),
                            }
                            for i in range(min(len(cases_in), len(diffs_in)))
                        ],
                    }
                else:
                    report["diff"] = {"ok": False, "error": "no diff results"}
                if cex_in:
                    report["counterexamples"] = [
                        {"shapes": dict(cx.case.shapes), "summary": cx.diff.summary, "hints": list(cx.hints)}
                        for cx in cex_in[:3]
                    ]
            except Exception:
                pass

    if diffs_in and not diff_ok:
        try:
            from verify.diff_debugger import debug_mismatch

            debug_case = (cex_in[0].case if cex_in else (cases_in[0] if cases_in else TestCase(shapes={}, dtypes={}, seed=0)))
            report["diff_debug"] = debug_mismatch(
                cand_for_run.intent,
                run_ref_fn,
                debug_case,
                sample_elems=16,
            )
        except Exception as e:
            report["diff_debug"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # Stage C (metamorphic / bounded exhaustive) intentionally uses the pure-numpy
    # reference runner. TileLang PrimFuncs are often specialized (e.g., fixed inner
    # dims like N=16), while Stage C enumerates many tiny shapes by design.
    # Using the numpy runner keeps Stage C frontend-agnostic and robust.
    stage_c_ref_fn = spec.runner
    if stage_c and diff_ok and cases_in:
        base_case = cases_in[0] if cases_in else TestCase(shapes=dict(spec.canonical_shapes), dtypes={}, seed=0)
        meta = run_metamorphic_suite(
            spec.name, cand_for_run.intent, stage_c_ref_fn, base_case=base_case, atol=tol["atol"], rtol=tol["rtol"]
        )
        bounded = run_bounded_exhaustive(
            spec.name, cand_for_run.intent, stage_c_ref_fn, atol=tol["atol"], rtol=tol["rtol"], max_cases=64
        )
        report["stage_c"] = {
            "metamorphic": {
                "ok": bool(meta.ok),
                "results": [{"relation": r.relation, "ok": bool(r.ok), "detail": r.detail} for r in meta.results],
            },
            "bounded_exhaustive": {
                "ok": bool(bounded.ok),
                "checked": int(bounded.checked),
                "total": int(bounded.total),
                "detail": bounded.detail,
                "first_failure": (dict(bounded.first_failure_case.shapes) if bounded.first_failure_case else None),
                "first_failure_summary": bounded.first_failure_summary,
            },
        }
        try:
            from verify.numerical_stability import run_numerical_stability_suite

            report["stage_c"]["numerical_stability"] = run_numerical_stability_suite(
                spec.name, cand_for_run.intent, stage_c_ref_fn, base_case=base_case, tolerances=tol
            ).to_json_dict()
        except Exception as e:
            report["stage_c"]["numerical_stability"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        report["stage_c"] = {"skipped": True, "reason": ("diff_failed" if stage_c and not diff_ok else "disabled_or_no_cases")}

    if mutation_kill and diff_ok and cases_in:
        base_case = TestCase(shapes=dict(spec.canonical_shapes), dtypes={}, seed=0)
        diff_cases = cases_in[:2] if cases_in else [base_case]
        metamorphic_base = cases_in[0] if cases_in else base_case
        mut = run_mutation_kill(
            spec.name,
            intent=cand_for_run.intent,
            run_ref_fn=stage_c_ref_fn,
            diff_cases=diff_cases,
            metamorphic_base_case=metamorphic_base,
            static_validate_fn=(lambda m, _cert=cert_v2: static_validate(m, _cert)),
            n_mutants=8,
            seed=0,
            atol=float(tol["atol"]),
            rtol=float(tol["rtol"]),
        )
        report["mutation_kill"] = {
            "kill_rate": float(mut.kill_rate),
            "total": int(mut.total),
            "killed": int(mut.killed),
            "survived": int(mut.survived),
            "killed_by_stage": dict(mut.killed_by_stage),
            "mutation_breakdown": dict(mut.mutation_breakdown),
            "outcomes": [
                {
                    "mutant_id": o.mutant_id,
                    "mutation_type": o.mutation_type,
                    "killed_by": o.killed_by,
                    "detail": o.detail,
                    "diff_summary": o.diff_summary,
                }
                for o in mut.outcomes
            ],
        }
    else:
        report["mutation_kill"] = {"skipped": True, "reason": ("diff_failed" if mutation_kill and not diff_ok else "disabled_or_no_cases")}

    # Persist baseline IO for Task6 tools (remote RVV / backend codegen smoke).
    try:
        baseline_source = "tilelang_runtime" if use_rt_ref else "numpy_reference"
        baseline_io = dict(baseline_io_raw) if use_rt_ref else dict(spec.runner(baseline_case))
        try:
            from verify.diff_runner import _with_io_aliases as _with_io_aliases_for_diff

            baseline_io = _with_io_aliases_for_diff(cand.intent, baseline_io)
        except Exception:
            pass
        total_bytes = 0
        for v in baseline_io.values():
            arr = np.asarray(v)
            total_bytes += int(arr.size) * int(arr.dtype.itemsize)
        if total_bytes <= 16 * 1024 * 1024:
            npz_path = out_dir / f"{spec.name}.baseline.npz"
            np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in baseline_io.items()})
            report["baseline"] = {
                "shapes": dict(baseline_case.shapes),
                "seed": int(baseline_case.seed),
                "npz_path": str(npz_path),
                "keys": sorted(list(baseline_io.keys())),
                "bytes": int(total_bytes),
                "source": baseline_source,
            }
        else:
            report["baseline"] = {
                "shapes": dict(baseline_case.shapes),
                "seed": int(baseline_case.seed),
                "npz_path": None,
                "keys": sorted(list(baseline_io.keys())),
                "bytes": int(total_bytes),
                "skipped": "baseline too large to cache (over 16MB)",
                "source": baseline_source,
            }
    except Exception as e:
        report["baseline"] = {"error": f"{type(e).__name__}: {e}"}

    return report


__all__ = ["KernelSpec", "mvp_kernel_specs", "default_kernel_specs", "run_pipeline_for_spec"]
