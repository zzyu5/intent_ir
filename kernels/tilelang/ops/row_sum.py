def make_row_sum_prim_func(*, n: int = 16, threads: int = 128):
    """
    Row-wise sum reduction:
      out[m] = sum(inp[m, :])

    Keep N static and M dynamic.
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        out: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(inp[pid_m, 0], row)
            sm = T.alloc_fragment((1,), "float32")
            T.reduce_sum(row, sm, dim=0)
            out[pid_m] = sm[0]

    return main


__all__ = ["make_row_sum_prim_func"]

