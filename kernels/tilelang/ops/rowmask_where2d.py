def make_rowmask_where2d_prim_func(*, n: int = 64, threads: int = 128):
    """
    Row-mask where:
      out[m, n] = inp[m, n] if row_mask[m] else 0

    Keep N static and M dynamic.
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        row_mask: T.Tensor((M,), "bool"),
        out: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(inp[pid_m, 0], row)
            m = row_mask[pid_m]
            mf = T.cast(m, "float32")
            out_row = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                out_row[r] = row[r] * mf
            T.copy(out_row, out[pid_m, 0])

    return main


__all__ = ["make_rowmask_where2d_prim_func"]
