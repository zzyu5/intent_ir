import triton.language as tl


class _TLExtraShim:
    def rsqrt(self, x):
        return tl.math.rsqrt(x)


tl_extra_shim = _TLExtraShim()
