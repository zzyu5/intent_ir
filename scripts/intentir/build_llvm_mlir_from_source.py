#!/usr/bin/env python3
"""
Build and install an LLVM/MLIR toolchain from source (no sudo).

Why:
- Ubuntu mlir-14/mlir-15 packages are often missing newer dialects/passes needed
  for IntentIR's RVV real-MLIR lowering (e.g., RISC-V / riscv_vector).

Outputs (under artifacts/toolchains):
- llvm-<tag>/ (install prefix)
- llvm-<tag>/toolchain_manifest.json
- mlir-current -> llvm-<tag> (stable symlink, unless disabled)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOOLCHAIN_ROOT = ROOT / "artifacts" / "toolchains"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=(str(cwd) if cwd is not None else None), capture_output=True, text=True)


def _require_ok(p: subprocess.CompletedProcess[str], *, step: str) -> None:
    if p.returncode == 0:
        return
    out = str(p.stdout or "").strip()
    err = str(p.stderr or "").strip()
    detail = "\n".join([x for x in [out, err] if x])
    raise RuntimeError(f"{step} failed (rc={p.returncode}){(': ' + detail) if detail else ''}")


def _symlink_force(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(str(src), str(dst.parent))
    dst.symlink_to(rel)


def _tool_version(path: Path) -> str:
    try:
        p = _run([str(path), "--version"])
    except Exception:
        return ""
    if p.returncode != 0:
        return ""
    line = str((p.stdout or p.stderr or "").splitlines()[0]).strip()
    return line


def _help_contains(path: Path, *, pattern: str) -> bool:
    try:
        p = _run([str(path), "--help"])
    except Exception:
        return False
    if p.returncode != 0:
        return False
    s = (p.stdout or "") + "\n" + (p.stderr or "")
    return pattern.lower() in s.lower()


def _clone_or_update_repo(*, tag: str, src_dir: Path, force: bool) -> None:
    url = "https://github.com/llvm/llvm-project.git"
    if src_dir.exists():
        if not force:
            # Allow reuse if the repo already exists; ensure tag is present.
            p = _run(["git", "-C", str(src_dir), "rev-parse", "--is-inside-work-tree"])
            _require_ok(p, step="git rev-parse")
            return
        shutil.rmtree(src_dir)
    src_dir.parent.mkdir(parents=True, exist_ok=True)
    # Shallow clone the tag for speed and disk. If the tag isn't fetchable
    # shallowly (rare), users can rerun with --no-shallow.
    p = _run(["git", "clone", "--depth", "1", "--branch", str(tag), url, str(src_dir)])
    _require_ok(p, step=f"git clone {tag}")


def _configure(
    *,
    src_dir: Path,
    build_dir: Path,
    prefix: Path,
    tag: str,
    cmake_build_type: str,
    enable_assertions: bool,
    projects: str,
    targets: str,
    generator: str,
) -> dict[str, Any]:
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.parent.mkdir(parents=True, exist_ok=True)

    # Keep configuration lean to avoid missing-dev-package failures.
    cmake_args = [
        "cmake",
        "-S",
        str(src_dir / "llvm"),
        "-B",
        str(build_dir),
        "-G",
        str(generator),
        f"-DCMAKE_BUILD_TYPE={cmake_build_type}",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        f"-DLLVM_ENABLE_PROJECTS={projects}",
        f"-DLLVM_TARGETS_TO_BUILD={targets}",
        f"-DLLVM_ENABLE_ASSERTIONS={'ON' if enable_assertions else 'OFF'}",
        "-DLLVM_ENABLE_TERMINFO=OFF",
        "-DLLVM_ENABLE_ZLIB=OFF",
        "-DLLVM_ENABLE_ZSTD=OFF",
        "-DLLVM_ENABLE_LIBXML2=OFF",
    ]
    p = _run(cmake_args, cwd=ROOT)
    _require_ok(p, step=f"cmake configure ({tag})")
    return {"cmake_args": list(cmake_args), "stdout": p.stdout, "stderr": p.stderr}


def _build_and_install(*, build_dir: Path, jobs: int) -> dict[str, Any]:
    cmd = ["cmake", "--build", str(build_dir), "--target", "install"]
    if jobs > 0:
        cmd.extend(["--", f"-j{int(jobs)}"])
    p = subprocess.run(cmd, cwd=str(ROOT))
    if p.returncode != 0:
        raise RuntimeError(f"build/install failed rc={p.returncode}")
    return {"cmd": list(cmd), "rc": int(p.returncode)}


def _verify(prefix: Path) -> dict[str, Any]:
    bin_dir = prefix / "bin"
    tools = {
        "mlir-opt": bin_dir / "mlir-opt",
        "mlir-translate": bin_dir / "mlir-translate",
        "llvm-as": bin_dir / "llvm-as",
        "opt": bin_dir / "opt",
        "llc": bin_dir / "llc",
        "clang": bin_dir / "clang",
    }
    missing = [k for k, p in tools.items() if not p.is_file()]
    ok = not missing
    dialect_probe = {}
    mlir_opt = tools["mlir-opt"]
    if mlir_opt.is_file():
        dialect_probe = {
            "has_riscv": _help_contains(mlir_opt, pattern="riscv"),
            "has_riscv_vector": _help_contains(mlir_opt, pattern="riscv_vector"),
        }
    versions = {k: _tool_version(p) for k, p in tools.items() if p.is_file()}
    return {"ok": bool(ok), "missing_tools": missing, "versions": versions, "dialect_probe": dialect_probe}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvm-tag", default="llvmorg-20.1.0")
    ap.add_argument("--toolchain-root", type=Path, default=DEFAULT_TOOLCHAIN_ROOT)
    ap.add_argument("--prefix", type=Path, default=None)
    ap.add_argument("--src-dir", type=Path, default=None)
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8)))
    ap.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use-current-link", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--clean-build-dir", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--generator", default="Ninja")
    ap.add_argument("--cmake-build-type", default="Release")
    ap.add_argument("--enable-assertions", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--projects", default="mlir;clang")
    ap.add_argument("--targets", default="X86;NVPTX;RISCV")
    args = ap.parse_args()

    tag = str(args.llvm_tag).strip()
    if not tag:
        raise SystemExit("--llvm-tag is empty")

    toolchain_root = Path(args.toolchain_root)
    prefix = Path(args.prefix) if args.prefix is not None else (toolchain_root / f"llvm-{tag}")
    src_dir = Path(args.src_dir) if args.src_dir is not None else (toolchain_root / "src" / f"llvm-project-{tag}")
    build_dir = (
        Path(args.build_dir) if args.build_dir is not None else (toolchain_root / "build" / f"llvm-project-{tag}")
    )

    if prefix.exists():
        if not bool(args.force):
            raise SystemExit(f"prefix already exists: {prefix} (use --force to replace)")
        shutil.rmtree(prefix)

    _clone_or_update_repo(tag=tag, src_dir=src_dir, force=bool(args.force))
    cfg = _configure(
        src_dir=src_dir,
        build_dir=build_dir,
        prefix=prefix,
        tag=tag,
        cmake_build_type=str(args.cmake_build_type),
        enable_assertions=bool(args.enable_assertions),
        projects=str(args.projects),
        targets=str(args.targets),
        generator=str(args.generator),
    )
    _build_and_install(build_dir=build_dir, jobs=int(args.jobs))
    verify = _verify(prefix)

    if bool(args.use_current_link):
        _symlink_force(toolchain_root / "mlir-current", prefix)

    manifest = {
        "schema_version": "intentir_llvm_mlir_toolchain_manifest_v1",
        "generated_at_utc": _utc_now_iso(),
        "llvm_tag": tag,
        "toolchain_root": str(toolchain_root),
        "prefix": str(prefix),
        "src_dir": str(src_dir),
        "build_dir": str(build_dir),
        "configure": {"generator": str(args.generator), **cfg},
        "build": {"jobs": int(args.jobs), "cmake_build_type": str(args.cmake_build_type)},
        "verify": dict(verify),
        "next_steps": [
            "python scripts/intentir.py mlir check",
            "python -c 'from intent_ir.mlir import detect_mlir_toolchain; import json; print(json.dumps(detect_mlir_toolchain(), indent=2))'",
        ],
    }
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "toolchain_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if bool(args.clean_build_dir):
        try:
            shutil.rmtree(build_dir)
        except Exception:
            pass

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

