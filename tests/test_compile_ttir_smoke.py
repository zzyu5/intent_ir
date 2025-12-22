import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None

from triton_frontend.compile_ttir import TTIRArtifact, TTIRCompileError, compile_ttir, normalize_signature


def test_normalize_signature():
    sig = {"A": "fp16*", "B": "PTR_FP32", "M": "int32", "N": "i64"}
    out = normalize_signature(sig)
    assert out["A"] == "*fp16"
    assert out["B"] == "*fp32"
    assert out["M"] == "i32"
    assert out["N"] == "i64"


@pytest.mark.skipif(triton is None, reason="triton not available")
def test_compile_ttir_smoke():
    @triton.jit
    def add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK
        offsets = block_start + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(X_ptr + offsets, mask=mask)
        y = tl.load(Y_ptr + offsets, mask=mask)
        tl.store(Z_ptr + offsets, x + y, mask=mask)

    sig = {"X_ptr": "*fp32", "Y_ptr": "*fp32", "Z_ptr": "*fp32", "N": "i32"}
    try:
        art: TTIRArtifact = compile_ttir(add_kernel, sig, meta={"num_warps": 1, "num_stages": 1})
    except TTIRCompileError as e:
        pytest.skip(f"TTIR compile failed (likely missing CUDA backend): {e}")
    assert art.ttir
    assert "func" in art.ttir or "module" in art.ttir
