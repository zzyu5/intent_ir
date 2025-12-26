from __future__ import annotations

from pathlib import Path

from pipeline.tilelang.core import default_kernel_specs, run_pipeline_for_spec


def test_tilelang_mvp_pipeline(tmp_path: Path):
    specs = default_kernel_specs()
    assert specs, "expected at least one tilelang KernelSpec"
    report = run_pipeline_for_spec(specs[0], out_dir=tmp_path, cases_limit=4)
    assert report["frontend"] == "tilelang"
    assert report.get("diff", {}).get("ok") is True

