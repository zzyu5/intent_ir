from __future__ import annotations

import pytest


pytestmark = pytest.mark.skip(reason="legacy CUDA cpp_codegen smoke path removed in strict MLIR hard-cut")


def test_cuda_backend_legacy_smoke_removed() -> None:
    assert True
