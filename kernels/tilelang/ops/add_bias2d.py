def make_add_bias2d_prim_func(*, n: int = 16, threads: int = 128):
    """
    Elementwise add with 1D bias broadcast:
      out[m, n] = inp[m, n] + bias[n]

    Keep N static and M dynamic.
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        bias: T.Tensor((N,), "float32"),
        out: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(inp[pid_m, 0], row)
            b = T.alloc_fragment((N,), "float32")
            T.copy(bias[0], b)
            out_row = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                out_row[r] = row[r] + b[r]
            T.copy(out_row, out[pid_m, 0])

    return main


__all__ = ["make_add_bias2d_prim_func"]

