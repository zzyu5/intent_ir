from __future__ import annotations

from pathlib import Path

from frontends.triton.adapter import TritonAdapter
from pipeline.interfaces import KernelDescriptor


def test_triton_adapter_persists_ttgir_ptx_llir_and_cubin(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    dump_dir = tmp_path / "dump"
    cache_dir = tmp_path / "cache"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    kernel_dir = dump_dir / "ABC"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "flash_attention2d_kernel.ttir").write_text("module {}", encoding="utf-8")
    (kernel_dir / "flash_attention2d_kernel.ttgir").write_text("#blocked = #ttg.blocked<{}>", encoding="utf-8")
    (kernel_dir / "flash_attention2d_kernel.ptx").write_text(".visible .entry flash_attention2d_kernel() {}", encoding="utf-8")
    (kernel_dir / "flash_attention2d_kernel.llir").write_text("target triple = \"nvptx64-nvidia-cuda\"\n", encoding="utf-8")
    (kernel_dir / "flash_attention2d_kernel.cubin").write_bytes(b"\x7fELFfake")

    desc = KernelDescriptor(
        schema_version="kernel_desc_v1.0",
        name="flash_attention2d",
        frontend="triton",
        source_kind="source",
        source_text="def flash_attention2d(): pass",
    )
    desc.meta["artifact_dir"] = str(artifact_dir)
    desc.meta["triton_dump_dir"] = str(dump_dir)
    desc.meta["triton_cache_dir"] = str(cache_dir)

    adapter = TritonAdapter()
    out = adapter.ensure_artifacts(desc, kernel=object())

    assert Path(str(out.artifacts.ttir_path)).is_file()
    assert Path(str(out.artifacts.ttgir_path)).is_file()
    assert Path(str((out.artifacts.extra or {}).get("ptx_path"))).is_file()
    assert Path(str((out.artifacts.extra or {}).get("llvm_ir_path"))).is_file()
    assert Path(str((out.artifacts.extra or {}).get("cubin_path"))).is_file()
    assert str(out.meta.get("ptx_original_path") or "").endswith(".ptx")
    assert str(out.meta.get("llvm_ir_original_path") or "").endswith(".llir")
    assert str(out.meta.get("cubin_original_path") or "").endswith(".cubin")
