def make_grouped_row_sum2d_prim_func(*, n: int = 64, group_size: int = 4, threads: int = 128):
    """
    Grouped row reduction over the last axis.

    inp: [M, N] float32
    out: [M, G] float32, where G = N / group_size

    Keep N static and M dynamic.
    """
    if int(group_size) <= 0:
        raise ValueError("group_size must be positive")
    if int(n) % int(group_size) != 0:
        raise ValueError("n must be divisible by group_size")

    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)
    GS = int(group_size)
    G = int(N // GS)

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        out: T.Tensor((M, G), "float32"),
    ):
        with T.Kernel(G, M, threads=threads) as (pid_g, pid_m):
            vals = T.alloc_fragment((GS,), "float32")
            T.copy(inp[pid_m, pid_g * GS], vals)
            sm = T.alloc_fragment((1,), "float32")
            T.reduce_sum(vals, sm, dim=0)
            out[pid_m, pid_g] = sm[0]

    return main


__all__ = ["make_grouped_row_sum2d_prim_func"]
