from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
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


def test_resolve_tool_paths_uses_companion_fallback(tmp_path) -> None:
    mod = _load_module()
    toolchain_root = tmp_path / "toolchains"
    source_root = toolchain_root / "LLVM-20.1.8-Linux-X64"
    current_root = toolchain_root / "mlir-14"
    source_bin = source_root / "bin"
    current_bin = current_root / "bin"
    source_bin.mkdir(parents=True)
    current_bin.mkdir(parents=True)

    def _write_exec(path: Path, body: str) -> None:
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)

    _write_exec(source_bin / "llc", "#!/usr/bin/env bash\nif [[ \"$1\" == \"-march=nvptx64\" && \"$2\" == \"-mcpu=help\" ]]; then echo sm_120; exit 0; fi\necho 'LLVM 20'\n")
    _write_exec(current_bin / "mlir-opt", "#!/usr/bin/env bash\necho 'LLVM 14'\n")
    _write_exec(current_bin / "mlir-translate", "#!/usr/bin/env bash\necho 'LLVM 14'\n")
    _write_exec(current_bin / "llvm-as", "#!/usr/bin/env bash\necho 'LLVM 14'\n")
    _write_exec(current_bin / "opt", "#!/usr/bin/env bash\necho 'LLVM 14'\n")

    current_link = toolchain_root / "mlir-current"
    current_link.symlink_to("mlir-14")
    resolved, origins = mod._resolve_tool_paths(
        source_root=source_root,
        toolchain_root=toolchain_root,
        current_link=current_link,
        version=20,
        require_cuda_sm="sm_120",
    )
    assert resolved["llc"] == source_bin / "llc"
    assert resolved["mlir-opt"] == current_bin / "mlir-opt"
    assert resolved["mlir-translate"] == current_bin / "mlir-translate"
    assert resolved["llvm-as"] == current_bin / "llvm-as"
    assert resolved["opt"] == current_bin / "opt"
    assert origins == {
        "llc": "source",
        "mlir-opt": "fallback",
        "mlir-translate": "fallback",
        "llvm-as": "fallback",
        "opt": "fallback",
    }
