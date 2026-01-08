def make_layer_norm_residual2d_prim_func(*, n: int = 64, eps: float = 1e-5, threads: int = 128):
    """
    Fused residual + LayerNorm over last dim:

      z[m,n] = inp[m,n] + residual[m,n]
      mean[m] = mean(z[m,:])
      rstd[m] = rsqrt(mean((z-mean)^2) + eps)
      out[m,n] = (z[m,n]-mean[m]) * rstd[m] * weight[n] + bias[n]

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
        residual: T.Tensor((M, N), "float32"),
        weight: T.Tensor((N,), "float32"),
        bias: T.Tensor((N,), "float32"),
        out: T.Tensor((M, N), "float32"),
        mean: T.Tensor((M,), "float32"),
        rstd: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            x = T.alloc_fragment((N,), "float32")
            T.copy(inp[pid_m, 0], x)
            r = T.alloc_fragment((N,), "float32")
            T.copy(residual[pid_m, 0], r)
            w = T.alloc_fragment((N,), "float32")
            T.copy(weight[0], w)
            b = T.alloc_fragment((N,), "float32")
            T.copy(bias[0], b)

            z = T.alloc_fragment((N,), "float32")
            for i in T.serial(N):
                z[i] = x[i] + r[i]

            sm = T.alloc_fragment((1,), "float32")
            T.reduce_sum(z, sm, dim=0)
            m0 = sm[0] * invN
            mean[pid_m] = m0

            sq = T.alloc_fragment((N,), "float32")
            for i in T.serial(N):
                d = z[i] - m0
                sq[i] = d * d
            sm2 = T.alloc_fragment((1,), "float32")
            T.reduce_sum(sq, sm2, dim=0)
            var = sm2[0] * invN
            r0 = T.rsqrt(var + eps_c)
            rstd[pid_m] = r0

            out_row = T.alloc_fragment((N,), "float32")
            for i in T.serial(N):
                out_row[i] = (z[i] - m0) * r0 * w[i] + b[i]
            T.copy(out_row, out[pid_m, 0])

    return main


__all__ = ["make_layer_norm_residual2d_prim_func"]

