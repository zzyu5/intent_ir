def make_masked_softmax2d_prim_func(*, n: int = 64, threads: int = 128):
    """
    Masked softmax over the last axis (2D).

    - inp: [M, N] float32
    - mask: [N] bool (True = keep, False = masked to a large negative number)
    - out: [M, N] float32

    Keep N static and M dynamic (TileLang-friendly).
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        mask: T.Tensor((N,), "bool"),
        out: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(inp[pid_m, 0], row)

            masked_row = T.alloc_fragment((N,), "float32")
            neg = T.float32(-1.0e9)
            one = T.float32(1.0)
            for r in T.serial(N):
                mf = T.cast(mask[r], "float32")
                masked_row[r] = row[r] * mf + neg * (one - mf)

            mx = T.alloc_fragment((1,), "float32")
            T.reduce_max(masked_row, mx, dim=0)

            exp_row = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                exp_row[r] = T.exp(masked_row[r] - mx[0])

            sm = T.alloc_fragment((1,), "float32")
            T.reduce_sum(exp_row, sm, dim=0)

            out_row = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                out_row[r] = exp_row[r] / sm[0]
            T.copy(out_row, out[pid_m, 0])

    return main


__all__ = ["make_masked_softmax2d_prim_func"]
