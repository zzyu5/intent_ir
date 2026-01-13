from __future__ import annotations

import os
import shutil

import pytest

try:
    import torch
except Exception:
    torch = None

from verify.gen_cases import TestCase
from pipeline.cuda.core import default_kernel_specs


def _cuda_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_free_mem_mb() -> int:
    if torch is None:
        return 0
    try:
        if not torch.cuda.is_available():
            return 0
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        free, _total = torch.cuda.mem_get_info()
        return int(free // (1024 * 1024))
    except Exception:
        return 0


def _cuda_min_free_mem_mb() -> int:
    # Default to disabled. Tiny smoke tests can run even when the GPU is mostly busy.
    try:
        return max(0, int(os.getenv("INTENTIR_CUDA_MIN_FREE_MB", "0")))
    except Exception:
        return 0


_MIN_FREE_MB = _cuda_min_free_mem_mb()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.skipif((_MIN_FREE_MB > 0) and (_cuda_free_mem_mb() < _MIN_FREE_MB), reason=f"CUDA free memory too low (<{_MIN_FREE_MB} MiB)")
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available (torch extension build)")
def test_cuda_vec_add_baseline_smoke():
    spec = next(s for s in default_kernel_specs() if s.name == "vec_add")
    case = TestCase(shapes=dict(spec.canonical_shapes), dtypes={}, seed=0)
    io = spec.runner(case)
    assert "A" in io and "B" in io and "C" in io
    assert io["A"].shape == io["B"].shape == io["C"].shape
