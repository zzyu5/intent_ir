def make_clamp2d_prim_func(*, n: int = 16, lo: float = -0.5, hi: float = 0.5, threads: int = 128):
    """
    Elementwise clamp: out = min(max(inp, lo), hi) for 2D float32 tensors.

    Keep N static and M dynamic. `lo/hi` are modeled as scalar constants.
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)
    lo_c = T.float32(float(lo))
    hi_c = T.float32(float(hi))

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        out: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(inp[pid_m, 0], row)
            out_row = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                y0 = T.max(row[r], lo_c)
                y1 = T.min(y0, hi_c)
                out_row[r] = y1
            T.copy(out_row, out[pid_m, 0])

    return main


__all__ = ["make_clamp2d_prim_func"]
