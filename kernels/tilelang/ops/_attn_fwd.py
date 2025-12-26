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
    TileLang PrimFunc for an attention-like anchor.

    Note: the TileLang frontend currently uses PrimFunc only for evidence extraction
    (anchors + structured indexing). The semantic diff for TileLang kernels is done
    via the numpy reference + IntentIR interpreter.

    This PrimFunc intentionally keeps the core anchor as a `T.gemm` (dot anchor).
    """
    import tilelang.language as T

    Q_CTX = int(q_ctx)
    KV_CTX = int(kv_ctx)
    HEAD_DIM = int(head_dim)

    if KV_CTX != HEAD_DIM:
        # Keep the dot anchor simple: Q@[KV_CTX,HEAD_DIM]^T should land in [Q_CTX,HEAD_DIM].
        raise ValueError("make_attn_fwd_prim_func currently expects kv_ctx == head_dim")

    @T.prim_func
    def main(
        Q: T.Tensor((Q_CTX, HEAD_DIM), in_dtype),
        K: T.Tensor((KV_CTX, HEAD_DIM), in_dtype),
        V: T.Tensor((KV_CTX, HEAD_DIM), in_dtype),
        sm_scale: T.Tensor((), "float32"),
        Out: T.Tensor((Q_CTX, HEAD_DIM), out_dtype),
    ):
        with T.Kernel(1, threads=threads) as (_pid,):
            Qs = T.alloc_shared((Q_CTX, HEAD_DIM), in_dtype)
            Ks = T.alloc_shared((KV_CTX, HEAD_DIM), in_dtype)
            O = T.alloc_fragment((Q_CTX, HEAD_DIM), accum_dtype)
            T.clear(O)

            T.copy(Q[0, 0], Qs)
            T.copy(K[0, 0], Ks)

            # Anchor: dot (GEMM). (Softmax/V application is not modeled here.)
            T.gemm(Qs, Ks, O, False, True)

            # Consume sm_scale so it shows in the signature (not used by this anchor).
            _ = sm_scale[()]  # noqa: F841
            T.copy(O, Out[0, 0])

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

