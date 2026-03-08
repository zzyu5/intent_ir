from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "intentir" / "provision_mlir_toolchain.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("intentir_provision_mlir_toolchain", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_official_release_version_candidates_descend() -> None:
    mod = _load_module()
    versions = mod._official_release_version_candidates(20)
    assert versions[0] == "20.1.9"
    assert "20.1.0" in versions
    assert versions[-1] == "20.0.0"


def test_official_prebuilt_url_candidates_include_linux_x64() -> None:
    mod = _load_module()
    urls = mod._official_prebuilt_url_candidates("20.1.0")
    assert urls[0].endswith("/LLVM-20.1.0-Linux-X64.tar.xz")
    assert any("clang+llvm-20.1.0-x86_64" in url for url in urls)


def test_version_candidates_for_source_use_external_search_window() -> None:
    mod = _load_module()
    assert mod._version_candidates_for_source("official_prebuilt", 14) == [20, 19, 18]
    assert mod._version_candidates_for_source("official_prebuilt", 19) == [19]
    assert mod._version_candidates_for_source("apt", 15) == [15]
