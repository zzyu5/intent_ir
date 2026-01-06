def make_add2d_prim_func(*, n: int = 16, threads: int = 128):
    """
    Elementwise add: C = A + B for 2D tensors.

    We keep N static (for stable tile shapes) and M dynamic (to exercise shape binding).
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            a = T.alloc_fragment((N,), "float32")
            b = T.alloc_fragment((N,), "float32")
            T.copy(A[pid_m, 0], a)
            T.copy(B[pid_m, 0], b)
            c = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                c[r] = a[r] + b[r]
            T.copy(c, C[pid_m, 0])

    return main


__all__ = ["make_add2d_prim_func"]

