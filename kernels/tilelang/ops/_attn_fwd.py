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


__all__ = ["make_attn_fwd_prim_func"]
