def make_flash_attention2d_prim_func(
    *,
    q_ctx: int = 64,
    kv_ctx: int = 64,
    head_dim: int = 64,
    block_kv: int = 16,
    threads: int = 128,
):
    """
    FlashAttention-style causal attention forward (small fixed shapes).

    This PrimFunc implements an online-softmax ("FlashAttention") accumulator:
      - stream K/V in blocks
      - maintain per-query-row (m_i, l_i) and output accumulator
      - apply causal mask (j > q -> masked)
    """
    import math

    import tilelang.language as T

    Q_CTX = int(q_ctx)
    KV_CTX = int(kv_ctx)
    HEAD_DIM = int(head_dim)
    BLOCK_KV = int(block_kv)
    if KV_CTX % BLOCK_KV != 0:
        raise ValueError("make_flash_attention2d_prim_func requires kv_ctx divisible by block_kv")

    sm_scale = float(1.0 / math.sqrt(float(HEAD_DIM)))

    @T.prim_func
    def main(
        Q: T.Tensor((Q_CTX, HEAD_DIM), "float32"),
        K: T.Tensor((KV_CTX, HEAD_DIM), "float32"),
        V: T.Tensor((KV_CTX, HEAD_DIM), "float32"),
        Out: T.Tensor((Q_CTX, HEAD_DIM), "float32"),
    ):
        with T.Kernel(Q_CTX, threads=threads) as (pid_q,):
            q_row = T.alloc_fragment((HEAD_DIM,), "float32")
            T.copy(Q[pid_q, 0], q_row)

            neg = T.float32(-1.0e9)
            one = T.float32(1.0)
            scale = T.float32(sm_scale)

            m_i = T.alloc_fragment((1,), "float32")
            l_i = T.alloc_fragment((1,), "float32")
            m_i[0] = neg
            l_i[0] = T.float32(0.0)

            acc = T.alloc_fragment((HEAD_DIM,), "float32")
            for d in T.serial(HEAD_DIM):
                acc[d] = T.float32(0.0)

            # scratch
            scores = T.alloc_fragment((BLOCK_KV,), "float32")
            exp_scores = T.alloc_fragment((BLOCK_KV,), "float32")
            mx = T.alloc_fragment((1,), "float32")
            sm = T.alloc_fragment((1,), "float32")
            tmp = T.alloc_fragment((HEAD_DIM,), "float32")
            dot_acc = T.alloc_fragment((1,), "float32")
            tmp2 = T.alloc_fragment((BLOCK_KV,), "float32")
            red = T.alloc_fragment((1,), "float32")

            for blk in T.serial(KV_CTX // BLOCK_KV):
                base = blk * BLOCK_KV
                # Scores for this KV block.
                for j in T.serial(BLOCK_KV):
                    kv = base + j
                    for d in T.serial(HEAD_DIM):
                        tmp[d] = q_row[d] * K[kv, d]
                    T.reduce_sum(tmp, dot_acc, dim=0)
                    s = dot_acc[0] * scale
                    # Causal mask: disallow keys beyond query index.
                    mf = T.cast(pid_q >= kv, "float32")
                    scores[j] = s * mf + neg * (one - mf)

                T.reduce_max(scores, mx, dim=0)
                m_new = T.max(m_i[0], mx[0])
                alpha = T.exp(m_i[0] - m_new)

                for j in T.serial(BLOCK_KV):
                    exp_scores[j] = T.exp(scores[j] - m_new)
                T.reduce_sum(exp_scores, sm, dim=0)

                # Update l_i.
                l_i[0] = l_i[0] * alpha + sm[0]

                # Update acc row (HEAD_DIM).
                for d in T.serial(HEAD_DIM):
                    for j in T.serial(BLOCK_KV):
                        kv = base + j
                        tmp2[j] = exp_scores[j] * V[kv, d]
                    T.reduce_sum(tmp2, red, dim=0)
                    acc[d] = acc[d] * alpha + red[0]

                m_i[0] = m_new

            out_row = T.alloc_fragment((HEAD_DIM,), "float32")
            for d in T.serial(HEAD_DIM):
                out_row[d] = acc[d] / l_i[0]
            T.copy(out_row, Out[pid_q, 0])

    return main


__all__ = ["make_flash_attention2d_prim_func"]

