from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

try:
    import torch
except Exception:
    torch = None

from pipeline.tilelang.core import default_kernel_specs, run_pipeline_for_spec


def _cuda_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.skipif(importlib.util.find_spec("tilelang") is None, reason="tilelang not available")
@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_tilelang_mvp_pipeline(tmp_path: Path):
    specs = default_kernel_specs()
    assert len(specs) == 6, f"expected 6 tilelang kernels, got {len(specs)}"
    for spec in specs:
        # Unit tests keep TileLang pipeline deterministic (no LLM / no CUDA runtime)
        # so CI remains lightweight and key-free.
        report = run_pipeline_for_spec(
            spec,
            out_dir=tmp_path,
            cases_limit=4,
            stage_c=False,
            mutation_kill=False,
            use_llm=False,
            use_tilelang_runtime=False,
        )
        assert report["frontend"] == "tilelang"
        assert report.get("diff", {}).get("ok") is True, f"{spec.name} diff failed: {report.get('diff')}"
