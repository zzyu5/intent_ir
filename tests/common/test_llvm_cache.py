from __future__ import annotations

import os
from pathlib import Path

from pipeline.common import llvm_cache


def _write_llvm(path: Path, *, triple: str = "nvptx64-nvidia-cuda") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"; ModuleID = 'x'\nsource_filename = \"x\"\ntarget triple = \"{triple}\"\n\ndefine void @k() {{ ret void }}\n",
        encoding="utf-8",
    )


def test_discover_cached_downstream_llvm_prefers_current_out_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(llvm_cache, "ROOT", tmp_path)
    llvm_cache._reset_llvm_cache_index_for_tests()
    current_out = tmp_path / "run" / "pipeline_reports"
    local = current_out / "prod_dim2d.intentir.intentdialect.downstream_cuda_llvm.mlir"
    _write_llvm(local)

    resolved = llvm_cache.discover_cached_downstream_llvm_module_path(
        spec_name="prod_dim2d",
        llvm_pipeline="downstream_cuda_llvm",
        current_out_dir=current_out,
    )
    assert resolved == str(local)


def test_discover_cached_downstream_llvm_uses_latest_indexed_artifact(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(llvm_cache, "ROOT", tmp_path)
    llvm_cache._reset_llvm_cache_index_for_tests()

    older = (
        tmp_path
        / "artifacts"
        / "flaggems_matrix"
        / "daily"
        / "20260225"
        / "full196_old"
        / "pipeline_reports"
        / "prod_dim2d.intentir.intentdialect.downstream_cuda_llvm.mlir"
    )
    newer = (
        tmp_path
        / "artifacts"
        / "flaggems_matrix"
        / "daily"
        / "20260226"
        / "full196_new"
        / "pipeline_reports"
        / "prod_dim2d.intentir.intentdialect.downstream_cuda_llvm.mlir"
    )
    _write_llvm(older)
    _write_llvm(newer)
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    resolved = llvm_cache.discover_cached_downstream_llvm_module_path(
        spec_name="prod_dim2d",
        llvm_pipeline="downstream_cuda_llvm",
        current_out_dir=(tmp_path / "empty"),
    )
    assert resolved == str(newer)
