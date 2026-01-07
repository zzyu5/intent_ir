def make_matmul_fused_epilogue2d_prim_func(
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_stages: int = 2,
    threads: int = 128,
):
    """
    GEMM + fused epilogue with broadcast + mask (where semantics):

      mm = A @ B
      tmp = mm + bias[None, :]
      C[m,n] = tmp if (row_mask[m] & col_mask[n]) else 0
    """
    import tilelang.language as T  # noqa: PLC0415

    M = T.dynamic("M")
    N = T.dynamic("N")
    K = T.dynamic("K")

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        bias: T.Tensor((N,), "float32"),
        row_mask: T.Tensor((M,), "bool"),
        col_mask: T.Tensor((N,), "bool"),
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

            bias_local = T.alloc_fragment((block_n,), "float32")
            T.copy(bias[bx * block_n], bias_local)
            row_local = T.alloc_fragment((block_m,), "bool")
            col_local = T.alloc_fragment((block_n,), "bool")
            T.copy(row_mask[by * block_m], row_local)
            T.copy(col_mask[bx * block_n], col_local)

            for i, j in T.Parallel(block_m, block_n):
                cond = row_local[i] & col_local[j]
                mf = T.cast(cond, "float32")
                C_local[i, j] = (C_local[i, j] + bias_local[j]) * mf
            T.copy(C_local, C[by * block_m, bx * block_n])

    return main


__all__ = ["make_matmul_fused_epilogue2d_prim_func"]
