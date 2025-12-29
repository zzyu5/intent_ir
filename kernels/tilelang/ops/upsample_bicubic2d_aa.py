def make_upsample_bicubic2d_aa_prim_func(*, threads: int = 128):
    """
    A small TileLang PrimFunc for an upsample-like kernel.

    The TileLang frontend uses this PrimFunc only as "evidence" (anchors/accesses).
    The semantic diff uses the numpy/PyTorch reference + IntentIR interpreter.
    """
    import tilelang.language as T

    # Keep the PrimFunc tiny and deterministic.
    N = 1
    C = 1
    IH = 4
    IW = 4
    OH = 8
    OW = 8

    @T.prim_func
    def main(I: T.Tensor((N, C, IH, IW), "float32"), O: T.Tensor((N, C, OH, OW), "float32")):
        with T.Kernel(OH, OW, threads=threads) as (pid_oh, pid_ow):
            oh = pid_oh
            ow = pid_ow
            ih = oh // 2
            iw = ow // 2
            O[0, 0, oh, ow] = I[0, 0, ih, iw]

    return main


__all__ = ["make_upsample_bicubic2d_aa_prim_func"]
