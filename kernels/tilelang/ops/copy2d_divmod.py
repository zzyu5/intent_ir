def make_copy2d_divmod_prim_func(*, n: int = 16, block_n: int = 16, threads: int = 128):
    """
    2D copy kernel, but with a 1D program id that is explicitly decomposed via div/mod:
      pid -> (pid_m, pid_nb) using grid_n = ceildiv(N, block_n)

    This exists to exercise (//, %) witnesses in downstream analysis.
    Keep N static and M dynamic.
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)
    BN = int(block_n)
    grid_n = (N + BN - 1) // BN

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        out: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M * grid_n, threads=threads) as (pid,):
            pid_m = pid // grid_n
            pid_nb = pid % grid_n
            frag = T.alloc_fragment((BN,), "float32")
            T.copy(inp[pid_m, pid_nb * BN], frag)
            T.copy(frag, out[pid_m, pid_nb * BN])

    return main


__all__ = ["make_copy2d_divmod_prim_func"]

