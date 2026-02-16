from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Iterable

_DEFAULT_SOURCE_SUFFIXES = {
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".cmake",
    ".inc",
    ".inl",
}


def stable_source_tag(source_dir: Path) -> str:
    return hashlib.sha1(str(source_dir.resolve()).encode("utf-8")).hexdigest()[:10]


def resolve_build_root(
    source_dir: Path,
    *,
    env_var: str,
    namespace: str,
) -> Path:
    override = os.getenv(str(env_var))
    if override:
        return Path(str(override)).expanduser().resolve()
    cache_root = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser()
    return (cache_root / "intentir" / str(namespace) / stable_source_tag(source_dir)).resolve()


def resolve_binary_path(
    build_root: Path,
    *,
    build_type: str,
    binary_name: str,
) -> Path:
    return Path(build_root) / str(build_type).lower() / str(binary_name)


def sources_newer_than_binary(
    source_dir: Path,
    *,
    build_dir: Path,
    binary_path: Path,
    source_suffixes: Iterable[str] | None = None,
) -> bool:
    if not binary_path.exists():
        return True
    try:
        bin_mtime = binary_path.stat().st_mtime
    except FileNotFoundError:
        return True

    suffixes = set(str(s) for s in (source_suffixes or _DEFAULT_SOURCE_SUFFIXES))
    for p in source_dir.rglob("*"):
        if build_dir in p.parents:
            continue
        if not p.is_file():
            continue
        if p.name == "CMakeLists.txt" or p.suffix in suffixes:
            try:
                if p.stat().st_mtime > bin_mtime:
                    return True
            except FileNotFoundError:
                continue
    return False


def ensure_cmake_binary_built(
    *,
    source_dir: Path,
    build_dir: Path,
    binary_path: Path,
    build_type: str = "Release",
    label: str = "cpp codegen",
    source_suffixes: Iterable[str] | None = None,
) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    if not sources_newer_than_binary(
        source_dir,
        build_dir=build_dir,
        binary_path=binary_path,
        source_suffixes=source_suffixes,
    ):
        return binary_path

    cfg = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    res = subprocess.run(cfg, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{label}: cmake configure failed:\n{res.stderr or res.stdout}")

    build = ["cmake", "--build", str(build_dir), "-j"]
    res = subprocess.run(build, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{label}: cmake build failed:\n{res.stderr or res.stdout}")

    if not binary_path.exists():
        raise RuntimeError(f"{label}: build succeeded but binary not found at {binary_path}")
    return binary_path
