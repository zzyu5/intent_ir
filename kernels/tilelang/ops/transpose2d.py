def make_transpose2d_prim_func(*, n: int = 16, threads: int = 128):
    """
    2D transpose: out[n, m] = inp[m, n]

    Keep N static and M dynamic, to mirror the "row-wise" scheduling patterns used
    by our other tilelang kernels.
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        out: T.Tensor((N, M), "float32"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(inp[pid_m, 0], row)
            for r in T.serial(N):
                out[r, pid_m] = row[r]

    return main


__all__ = ["make_transpose2d_prim_func"]

