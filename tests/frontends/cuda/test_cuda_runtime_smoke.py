from __future__ import annotations

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


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(_cuda_free_mem_mb() < 1024, reason="CUDA free memory too low (<1024 MiB)")
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available (torch extension build)")
def test_cuda_vec_add_baseline_smoke():
    spec = next(s for s in default_kernel_specs() if s.name == "vec_add")
    case = TestCase(shapes=dict(spec.canonical_shapes), dtypes={}, seed=0)
    io = spec.runner(case)
    assert "A" in io and "B" in io and "C" in io
    assert io["A"].shape == io["B"].shape == io["C"].shape
