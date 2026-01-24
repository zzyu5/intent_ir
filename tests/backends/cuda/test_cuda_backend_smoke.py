from __future__ import annotations

import shutil

import numpy as np
import pytest

try:
    import torch
except Exception:
    torch = None

from intent_ir.ir import IntentFunction


def _cuda_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available (torch extension build)")
def test_cuda_backend_matmul_matches_numpy():
    from backends.cuda.codegen.intentir_to_cuda import lower_intent_to_cuda_kernel
    from backends.cuda.runtime import run_cuda_kernel

    intent = IntentFunction.from_json_dict(
        {
            "name": "mm_cuda_smoke",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "matmul", "inputs": ["A", "B"], "output": "C", "attrs": {}}],
            "outputs": ["C"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_m": 8, "tile_n": 16, "tile_k": 16, "parallel_axes": ["M", "N"]},
        }
    )

    M, N, K = 32, 48, 16
    bindings = {"M": M, "N": N, "K": K}
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings=bindings)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    ref = A @ B

    out = run_cuda_kernel(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
        launch=lowered.launch,
        bindings=lowered.bindings,
        inputs_np={"A": A, "B": B},
        output_names=lowered.output_names,
    )
    got = out["C"]
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

