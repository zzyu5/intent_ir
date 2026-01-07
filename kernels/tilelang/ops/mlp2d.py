def make_mlp2d_prim_func(
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    block_h: int = 64,
    num_stages: int = 2,
    threads: int = 128,
):
    """
    Fused 2-layer MLP (float32):
      H = relu(A @ W1 + b1)
      C = H @ W2 + b2

    A:  [M, K]
    W1: [K, H]
    b1: [H]
    W2: [H, N]
    b2: [N]
    C:  [M, N]
    """
    import tilelang.language as T  # noqa: PLC0415

    M = T.dynamic("M")
    N = T.dynamic("N")
    K = T.dynamic("K")
    H = T.dynamic("H")

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float32"),
        W1: T.Tensor((K, H), "float32"),
        b1: T.Tensor((H,), "float32"),
        W2: T.Tensor((H, N), "float32"),
        b2: T.Tensor((N,), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            C_local = T.alloc_fragment((block_m, block_n), "float32")
            T.clear(C_local)
            A_shared = T.alloc_shared((block_m, block_k), "float32")
            W1_shared = T.alloc_shared((block_k, block_h), "float32")
            H_shared = T.alloc_shared((block_m, block_h), "float32")
            W2_shared = T.alloc_shared((block_h, block_n), "float32")
            H_local = T.alloc_fragment((block_m, block_h), "float32")
            b1_local = T.alloc_fragment((block_h,), "float32")
            zero = T.float32(0.0)

            # Iterate H in blocks; accumulate into C_local.
            for ho in T.serial(T.ceildiv(H, block_h)):
                # Hidden = A @ W1 + b1, then ReLU.
                T.clear(H_local)
                for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                    T.copy(A[by * block_m, ko * block_k], A_shared)
                    T.copy(W1[ko * block_k, ho * block_h], W1_shared)
                    T.gemm(A_shared, W1_shared, H_local)

                T.copy(b1[ho * block_h], b1_local)
                for i, j in T.Parallel(block_m, block_h):
                    H_local[i, j] = T.max(H_local[i, j] + b1_local[j], zero)
                T.copy(H_local, H_shared)

                # Accumulate C += Hidden @ W2 for this H block.
                T.copy(W2[ho * block_h, bx * block_n], W2_shared)
                T.gemm(H_shared, W2_shared, C_local)

            b2_local = T.alloc_fragment((block_n,), "float32")
            T.copy(b2[bx * block_n], b2_local)
            for i, j in T.Parallel(block_m, block_n):
                C_local[i, j] = C_local[i, j] + b2_local[j]
            T.copy(C_local, C[by * block_m, bx * block_n])

    return main


__all__ = ["make_mlp2d_prim_func"]
