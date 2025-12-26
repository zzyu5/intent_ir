from __future__ import annotations

from pathlib import Path

from pipeline.tilelang.core import default_kernel_specs, run_pipeline_for_spec


def test_tilelang_mvp_pipeline(tmp_path: Path):
    specs = default_kernel_specs()
    assert len(specs) == 6, f"expected 6 tilelang kernels, got {len(specs)}"
    for spec in specs:
        report = run_pipeline_for_spec(spec, out_dir=tmp_path, cases_limit=4, stage_c=False, mutation_kill=False)
        assert report["frontend"] == "tilelang"
        assert report.get("diff", {}).get("ok") is True, f"{spec.name} diff failed: {report.get('diff')}"
