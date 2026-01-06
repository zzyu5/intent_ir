def make_where2d_prim_func(*, n: int = 16, threads: int = 128):
    """
    Elementwise where on 2D float32 tensors:
      out[m, n] = where(A[m, n] > B[m, n], A[m, n], B[m, n])

    Keep N static and M dynamic.
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
            out = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                out[r] = T.if_then_else(a[r] > b[r], a[r], b[r])
            T.copy(out, C[pid_m, 0])

    return main


__all__ = ["make_where2d_prim_func"]
