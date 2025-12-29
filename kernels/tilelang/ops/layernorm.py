def make_layer_norm_persistent_prim_func(*, n: int = 16, threads: int = 128):
    """
    LayerNorm over the last axis (N), with explicit Mean/Rstd outputs.
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        in_ptr: T.Tensor((M, N), "float32"),
        out_ptr: T.Tensor((M, N), "float32"),
        weight_ptr: T.Tensor((N,), "float32"),
        bias_ptr: T.Tensor((N,), "float32"),
        out_mean_ptr: T.Tensor((M,), "float32"),
        out_rstd_ptr: T.Tensor((M,), "float32"),
    ):
        eps = T.float32(1e-5)
        with T.Kernel(M, threads=threads) as (pid,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(in_ptr[pid, 0], row)
            w = T.alloc_fragment((N,), "float32")
            T.copy(weight_ptr[0], w)
            b = T.alloc_fragment((N,), "float32")
            T.copy(bias_ptr[0], b)

            s = T.alloc_fragment((1,), "float32")
            T.reduce_sum(row, s, dim=0)
            mean = s[0] / T.float32(N)
            out_mean_ptr[pid] = mean

            diff = T.alloc_fragment((N,), "float32")
            sq = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                d = row[r] - mean
                diff[r] = d
                sq[r] = d * d

            ss = T.alloc_fragment((1,), "float32")
            T.reduce_sum(sq, ss, dim=0)
            var = ss[0] / T.float32(N)
            rstd = T.rsqrt(var + eps)
            out_rstd_ptr[pid] = rstd

            out_row = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                out_row[r] = diff[r] * rstd * w[r] + b[r]
            T.copy(out_row, out_ptr[pid, 0])

    return main


__all__ = ["make_layer_norm_persistent_prim_func"]
