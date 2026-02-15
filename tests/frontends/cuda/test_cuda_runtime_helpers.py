from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frontends.cuda.runtime import _tensor_to_numpy


def test_tensor_to_numpy_preserves_float32() -> None:
    t = torch.tensor([1.5, -2.0], dtype=torch.float32)
    arr = _tensor_to_numpy(t)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, np.asarray([1.5, -2.0], dtype=np.float32))


def test_tensor_to_numpy_bf16_promotes_to_float32() -> None:
    t = torch.tensor([1.25, -3.5], dtype=torch.bfloat16)
    arr = _tensor_to_numpy(t)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, t.to(torch.float32).numpy(), atol=1e-6, rtol=0.0)
