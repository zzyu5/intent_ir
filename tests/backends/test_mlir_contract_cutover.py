from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_cuda_pipeline_driver_does_not_use_mlir_bridge() -> None:
    src = _read("backends/cuda/pipeline/driver.py")
    assert "mlir_bridge" not in src
    assert "resolve_intent_payload_with_meta" not in src


def test_rvv_pipeline_driver_does_not_use_mlir_bridge() -> None:
    src = _read("backends/spmd_rvv/pipeline/driver.py")
    assert "mlir_bridge" not in src
    assert "resolve_intent_payload_with_meta" not in src
