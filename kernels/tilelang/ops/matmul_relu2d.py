def make_matmul_relu2d_prim_func(
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_stages: int = 2,
    threads: int = 128,
):
    """
    GEMM(+ReLU) micro-kernel in float32, used for expanded coverage.

    This mirrors `make_gemm_relu_prim_func` but pins dtype to float32 to keep the
    RVV backend path simple (f32 end-to-end).
    """
    import tilelang.language as T  # noqa: PLC0415

    M = T.dynamic("M")
    N = T.dynamic("N")
    K = T.dynamic("K")

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), "float32")
            B_shared = T.alloc_shared((block_k, block_n), "float32")
            C_local = T.alloc_fragment((block_m, block_n), "float32")
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(A[by * block_m, ko * block_k], A_shared)
                T.copy(B[ko * block_k, bx * block_n], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            zero = T.float32(0.0)
            for i, j in T.Parallel(block_m, block_n):
                C_local[i, j] = T.max(C_local[i, j], zero)
            T.copy(C_local, C[by * block_m, bx * block_n])

    return main


__all__ = ["make_matmul_relu2d_prim_func"]

