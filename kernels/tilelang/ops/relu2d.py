def make_relu2d_prim_func(*, n: int = 16, threads: int = 128):
    """
    Elementwise ReLU: out = max(inp, 0) for 2D float32 tensors.

    Keep N static and M dynamic (like other TileLang micro-kernels in this repo).
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        out: T.Tensor((M, N), "float32"),
    ):
        zero = T.float32(0.0)
        with T.Kernel(M, threads=threads) as (pid_m,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(inp[pid_m, 0], row)
            out_row = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                out_row[r] = T.max(row[r], zero)
            T.copy(out_row, out[pid_m, 0])

    return main


__all__ = ["make_relu2d_prim_func"]

