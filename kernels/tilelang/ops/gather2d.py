def make_gather2d_prim_func(*, n: int = 64, l: int = 256, threads: int = 128):
    """
    2D gather with irregular indexing.

    inp: [M, N] float32
    row_idx: [L] int32
    col_idx: [L] int32
    out: [L] float32 where out[i] = inp[row_idx[i], col_idx[i]]

    Keep N/L static and M dynamic (coverage kernel).
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)
    L = int(l)

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        row_idx: T.Tensor((L,), "int32"),
        col_idx: T.Tensor((L,), "int32"),
        out: T.Tensor((L,), "float32"),
    ):
        with T.Kernel(L, threads=threads) as (pid,):
            r = row_idx[pid]
            c = col_idx[pid]
            out[pid] = inp[r, c]

    return main


__all__ = ["make_gather2d_prim_func"]
