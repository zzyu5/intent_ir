def make_gemm_relu_prim_func(
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 32,
    num_stages: int = 3,
    threads: int = 128,
    in_dtype: str = "float16",
    out_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Real TileLang GEMM(+ReLU) kernel.

    Notes:
    - We keep a first-class `PrimFunc` because Task4 (facts/certificate) needs to parse TIR.
    - The runtime pipeline will compile/execute this PrimFunc on CUDA to produce baseline IO.
    """
    import tilelang.language as T  # noqa: PLC0415

    M = T.dynamic("M")
    N = T.dynamic("N")
    K = T.dynamic("K")

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), in_dtype)
            B_shared = T.alloc_shared((block_k, block_n), in_dtype)
            C_local = T.alloc_fragment((block_m, block_n), accum_dtype)
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(A[by * block_m, ko * block_k], A_shared)
                T.copy(B[ko * block_k, bx * block_n], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_m, block_n):
                C_local[i, j] = T.max(C_local[i, j], 0)
            T.copy(C_local, C[by * block_m, bx * block_n])

    return main


__all__ = ["make_gemm_relu_prim_func"]
