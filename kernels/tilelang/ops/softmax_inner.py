def make_softmax_inner_prim_func(*, n: int = 16, threads: int = 128):
    """
    Softmax over the last axis, with auxiliary outputs (row_max/row_sum) to make
    reductions explicit in the extracted evidence/contract.
    """
    import tilelang.language as T

    M = T.dynamic("M")
    N = int(n)

    @T.prim_func
    def main(
        input_ptr: T.Tensor((M, N), "float32"),
        output_ptr: T.Tensor((M, N), "float32"),
        row_max_ptr: T.Tensor((M,), "float32"),
        row_sum_ptr: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=threads) as (pid_m,):
            row = T.alloc_fragment((N,), "float32")
            T.copy(input_ptr[pid_m, 0], row)

            mx = T.alloc_fragment((1,), "float32")
            T.reduce_max(row, mx, dim=0)
            row_max_ptr[pid_m] = mx[0]

            exp_row = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                exp_row[r] = T.exp(row[r] - mx[0])

            sm = T.alloc_fragment((1,), "float32")
            T.reduce_sum(exp_row, sm, dim=0)
            row_sum_ptr[pid_m] = sm[0]

            out_row = T.alloc_fragment((N,), "float32")
            for r in T.serial(N):
                out_row[r] = exp_row[r] / sm[0]
            T.copy(out_row, output_ptr[pid_m, 0])

    return main


__all__ = ["make_softmax_inner_prim_func"]
