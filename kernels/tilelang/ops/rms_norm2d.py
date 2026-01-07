def make_rms_norm2d_prim_func(*, n: int = 64, eps: float = 1e-5, threads: int = 128):
    """
    RMSNorm over last dim:
      rstd[m] = rsqrt(mean(inp[m,:]^2) + eps)
      out[m,n] = inp[m,n] * rstd[m] * weight[n]

    Keep N static and M dynamic.
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)
    eps_c = T.float32(float(eps))
    invN = T.float32(1.0 / float(N))

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        weight: T.Tensor((N,), "float32"),
        out: T.Tensor((M, N), "float32"),
        rstd: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(inp[pid_m, 0], row)
            w = T.alloc_fragment((N,), "float32")
            T.copy(weight[0], w)

            sq = T.alloc_fragment((N,), "float32")
            for r0 in T.serial(N):
                sq[r0] = row[r0] * row[r0]

            sm = T.alloc_fragment((1,), "float32")
            T.reduce_sum(sq, sm, dim=0)
            mean = sm[0] * invN
            r0 = T.rsqrt(mean + eps_c)
            rstd[pid_m] = r0

            out_row = T.alloc_fragment((N,), "float32")
            for r1 in T.serial(N):
                out_row[r1] = row[r1] * r0 * w[r1]
            T.copy(out_row, out[pid_m, 0])

    return main


__all__ = ["make_rms_norm2d_prim_func"]

