def make_masked_attention2d_prim_func(
    *,
    q_ctx: int = 16,
    kv_ctx: int = 16,
    head_dim: int = 16,
    threads: int = 128,
):
    """
    Masked attention forward (small fixed shapes) with a causal mask:

      scores[j] = (Q[q] · K[j]) * sm_scale
      if j > q: scores[j] = -1e9
      probs = softmax(scores)
      Out[q, d] = sum_j probs[j] * V[j, d]
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
        sm_scale: T.Tensor((1,), "float32"),
        Out: T.Tensor((Q_CTX, HEAD_DIM), "float32"),
    ):
        with T.Kernel(Q_CTX, threads=threads) as (pid_q,):
            q_row = T.alloc_fragment((HEAD_DIM,), "float32")
            T.copy(Q[pid_q, 0], q_row)

            scores = T.alloc_fragment((KV_CTX,), "float32")
            tmp = T.alloc_fragment((HEAD_DIM,), "float32")
            acc = T.alloc_fragment((1,), "float32")
            neg = T.float32(-1.0e9)
            scale = sm_scale[0]
            for j in T.serial(KV_CTX):
                for k in T.serial(HEAD_DIM):
                    tmp[k] = q_row[k] * K[j, k]
                T.reduce_sum(tmp, acc, dim=0)
                s = acc[0] * scale
                scores[j] = neg if (j > pid_q) else s

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


__all__ = ["make_masked_attention2d_prim_func"]
