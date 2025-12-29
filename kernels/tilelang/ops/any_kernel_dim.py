def make_any_kernel_dim_prim_func(*, n: int = 16, threads: int = 128):
    """
    Row-wise `any` reduction: out[m] = any(inp[m, :]).

    This mirrors the spirit of Triton's `any_kernel_dim` (reduce over the last axis).
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        inp: T.Tensor((M, N), "float32"),
        out: T.Tensor((M,), "bool"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(inp[pid_m, 0], row)
            mx = T.alloc_fragment((1,), "float32")
            T.reduce_max(row, mx, dim=0)
            out[pid_m] = mx[0] != T.float32(0.0)

    return main


__all__ = ["make_any_kernel_dim_prim_func"]
