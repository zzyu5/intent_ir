def make_group_norm_kernel_prim_func(*, c: int = 64, hw: int = 16, num_groups: int = 4, threads: int = 128):
    """
    GroupNorm forward (N, C, HW) with Mean/Rstd outputs (N, G).
    """
    import tilelang.language as T

    N = T.dynamic("N")
    C = int(c)
    HW = int(hw)
    G = int(num_groups)
    if G <= 0 or C % G != 0:
        raise ValueError(f"invalid groupnorm config: C={C} num_groups={G}")
    group_size = C // G

    @T.prim_func
    def main(
        X: T.Tensor((N, C, HW), "float32"),
        Y: T.Tensor((N, C, HW), "float32"),
        W: T.Tensor((C,), "float32"),
        B: T.Tensor((C,), "float32"),
        Mean: T.Tensor((N, G), "float32"),
        Rstd: T.Tensor((N, G), "float32"),
    ):
        eps = T.float32(1e-5)
        denom = T.float32(group_size * HW)
        with T.Kernel(N, G, threads=threads) as (pid_n, pid_g):
            n = pid_n
            g = pid_g
            c0 = g * group_size

            x_tile = T.alloc_fragment((group_size, HW), "float32")
            T.copy(X[n, c0, 0], x_tile)
            w_tile = T.alloc_fragment((group_size,), "float32")
            T.copy(W[c0], w_tile)
            b_tile = T.alloc_fragment((group_size,), "float32")
            T.copy(B[c0], b_tile)

            # sum(x)
            sum_hw = T.alloc_fragment((group_size,), "float32")
            T.reduce_sum(x_tile, sum_hw, dim=1)
            sum_all = T.alloc_fragment((1,), "float32")
            T.reduce_sum(sum_hw, sum_all, dim=0)
            mean = sum_all[0] / denom

            # sum(x^2)
            x2_tile = T.alloc_fragment((group_size, HW), "float32")
            for ci0 in T.serial(group_size):
                for hi0 in T.serial(HW):
                    v = x_tile[ci0, hi0]
                    x2_tile[ci0, hi0] = v * v
            sum2_hw = T.alloc_fragment((group_size,), "float32")
            T.reduce_sum(x2_tile, sum2_hw, dim=1)
            sum2_all = T.alloc_fragment((1,), "float32")
            T.reduce_sum(sum2_hw, sum2_all, dim=0)
            mean2 = sum2_all[0] / denom
            var = mean2 - mean * mean
            rstd = T.rsqrt(var + eps)

            Mean[n, g] = mean
            Rstd[n, g] = rstd

            for ci1 in T.serial(group_size):
                for hi1 in T.serial(HW):
                    Y[n, c0 + ci1, hi1] = (x_tile[ci1, hi1] - mean) * rstd * w_tile[ci1] + b_tile[ci1]

    return main

__all__ = ["make_group_norm_kernel_prim_func"]
