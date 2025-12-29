from typing import Dict

import numpy as np

from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType
from verify.gen_cases import TestCase

from .spec import TileLangKernelSpec


def make_attn_fwd_prim_func(
    *,
    q_ctx: int = 16,
    kv_ctx: int = 16,
    head_dim: int = 16,
    threads: int = 128,
    in_dtype: str = "float16",
    out_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    TileLang PrimFunc for attention forward (small fixed shapes).

    This implementation is intentionally simple and correctness-oriented:
      scores = (Q @ K^T) * sm_scale
      probs = softmax(scores, axis=-1)
      Out = probs @ V
    """
    import tilelang.language as T

    Q_CTX = int(q_ctx)
    KV_CTX = int(kv_ctx)
    HEAD_DIM = int(head_dim)

    @T.prim_func
    def main(
        Q: T.Tensor((Q_CTX, HEAD_DIM), "float32"),
        K: T.Tensor((KV_CTX, HEAD_DIM), "float32"),
        V: T.Tensor((KV_CTX, HEAD_DIM), "float32"),
        Out: T.Tensor((Q_CTX, HEAD_DIM), "float32"),
    ):
        with T.Kernel(Q_CTX, threads=threads) as (pid_q,):
            sm_scale = T.float32(1.0 / float(HEAD_DIM) ** 0.5)
            q_row = T.alloc_fragment((HEAD_DIM,), "float32")
            T.copy(Q[pid_q, 0], q_row)

            scores = T.alloc_fragment((KV_CTX,), "float32")
            tmp = T.alloc_fragment((HEAD_DIM,), "float32")
            acc = T.alloc_fragment((1,), "float32")
            for j in T.serial(KV_CTX):
                for k in T.serial(HEAD_DIM):
                    tmp[k] = q_row[k] * K[j, k]
                T.reduce_sum(tmp, acc, dim=0)
                scores[j] = acc[0] * sm_scale

            mx = T.alloc_fragment((1,), "float32")
            T.reduce_max(scores, mx, dim=0)

            exp_scores = T.alloc_fragment((KV_CTX,), "float32")
            for j in T.serial(KV_CTX):
                exp_scores[j] = T.exp(scores[j] - mx[0])

            sm = T.alloc_fragment((1,), "float32")
            T.reduce_sum(exp_scores, sm, dim=0)

            probs = T.alloc_fragment((KV_CTX,), "float32")
            for j in T.serial(KV_CTX):
                probs[j] = exp_scores[j] / sm[0]

            out_row = T.alloc_fragment((HEAD_DIM,), "float32")
            tmp2 = T.alloc_fragment((KV_CTX,), "float32")
            acc2 = T.alloc_fragment((1,), "float32")
            for d in T.serial(HEAD_DIM):
                for j in T.serial(KV_CTX):
                    tmp2[j] = probs[j] * V[j, d]
                T.reduce_sum(tmp2, acc2, dim=0)
                out_row[d] = acc2[0]
            T.copy(out_row, Out[pid_q, 0])

    return main


def attn_fwd_reference(case: TestCase) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(case.seed))
    q_ctx = int(case.shapes.get("Q_CTX", 16))
    kv_ctx = int(case.shapes.get("KV_CTX", 16))
    head_dim = int(case.shapes.get("HEAD_DIM", 16))
    q = rng.standard_normal((q_ctx, head_dim), dtype=np.float32)
    k = rng.standard_normal((kv_ctx, head_dim), dtype=np.float32)
    v = rng.standard_normal((kv_ctx, head_dim), dtype=np.float32)
    sm_scale = np.array(1.0 / np.sqrt(float(head_dim)), dtype=np.float32)

    scores = (q @ k.T) * sm_scale
    scores_max = np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores - scores_max)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    out = probs @ v
    return {"Q": q, "K": k, "V": v, "sm_scale": sm_scale, "Out": out.astype(np.float32)}


def attn_fwd_intent() -> IntentFunction:
    rm = TensorLayout(kind="row_major", params={})
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


def attn_fwd_spec() -> TileLangKernelSpec:
    return TileLangKernelSpec(
        name="_attn_fwd",
        prim_func=make_attn_fwd_prim_func(q_ctx=16, kv_ctx=16, head_dim=16, threads=128),
        arg_names=["Q", "K", "V", "sm_scale", "Out", "Q_CTX", "KV_CTX", "HEAD_DIM"],
        canonical_shapes={"Q_CTX": 16, "KV_CTX": 16, "HEAD_DIM": 16},
        vary_axes=[],
        runner=attn_fwd_reference,
        intent_builder=attn_fwd_intent,
        exclude_axes=[],
        constexpr_names=[],
    )


__all__ = [
    "make_attn_fwd_prim_func",
    "attn_fwd_reference",
    "attn_fwd_intent",
    "attn_fwd_spec",
]
