from __future__ import annotations

import os
import time
from pathlib import Path

from backends.common.cpp_build import (
    resolve_binary_path,
    resolve_build_root,
    sources_newer_than_binary,
    stable_source_tag,
)


def test_stable_source_tag_is_deterministic(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    t0 = stable_source_tag(src)
    t1 = stable_source_tag(src)
    assert t0 == t1
    assert len(t0) == 10


def test_resolve_build_root_honors_override_env(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    override = tmp_path / "build_override"
    monkeypatch.setenv("INTENTIR_TEST_CPP_BUILD_DIR", str(override))
    got = resolve_build_root(src, env_var="INTENTIR_TEST_CPP_BUILD_DIR", namespace="unit_test_cpp")
    assert got == override.resolve()


def test_resolve_build_root_default_under_xdg_cache(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    xdg = tmp_path / "xdg_cache"
    monkeypatch.delenv("INTENTIR_TEST_CPP_BUILD_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    got = resolve_build_root(src, env_var="INTENTIR_TEST_CPP_BUILD_DIR", namespace="unit_test_cpp")
    assert str(got).startswith(str((xdg / "intentir" / "unit_test_cpp").resolve()))


def test_resolve_binary_path_uses_build_type_subdir(tmp_path: Path) -> None:
    build_root = tmp_path / "build_root"
    got = resolve_binary_path(build_root, build_type="Release", binary_name="intentir_codegen")
    assert got == build_root / "release" / "intentir_codegen"


def test_sources_newer_than_binary_when_binary_missing(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "a.cpp").write_text("int x = 1;\n", encoding="utf-8")
    build_dir = tmp_path / "build"
    bin_path = build_dir / "release" / "intentir_codegen"
    assert sources_newer_than_binary(src, build_dir=build_dir, binary_path=bin_path) is True


def test_sources_newer_than_binary_false_when_sources_older(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    build_dir = tmp_path / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    src_file = src / "a.cpp"
    src_file.write_text("int x = 1;\n", encoding="utf-8")
    bin_path = build_dir / "intentir_codegen"
    bin_path.write_text("bin", encoding="utf-8")
    now = time.time()
    os.utime(src_file, (now - 10.0, now - 10.0))
    os.utime(bin_path, (now, now))
    assert sources_newer_than_binary(src, build_dir=build_dir, binary_path=bin_path) is False
