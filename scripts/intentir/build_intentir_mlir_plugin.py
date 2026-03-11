#!/usr/bin/env python3
"""
Build the out-of-tree MLIR pass plugin used by the "cpp_plugin" compiler stack.

Outputs:
- artifacts/mlir_plugins/intentir/libIntentIRPasses.so
- artifacts/mlir_plugins/intentir/plugin_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intent_ir.mlir.toolchain import detect_mlir_toolchain


DEFAULT_TOOLCHAIN_PREFIX = ROOT / "artifacts" / "toolchains" / "mlir-current"
PLUGIN_SRC = ROOT / "compiler" / "intentir_mlir_plugin"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "mlir_plugins" / "intentir"
DEFAULT_BUILD_DIR = ROOT / "artifacts" / "mlir_plugins" / "_build" / "intentir"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class CmdResult:
    cmd: list[str]
    rc: int
    stdout: str
    stderr: str


def _run(cmd: list[str], *, cwd: Path | None = None) -> CmdResult:
    p = subprocess.run(cmd, cwd=(str(cwd) if cwd is not None else None), capture_output=True, text=True)
    return CmdResult(cmd=list(cmd), rc=int(p.returncode), stdout=str(p.stdout or ""), stderr=str(p.stderr or ""))


def _require_ok(res: CmdResult, *, step: str) -> None:
    if res.rc == 0:
        return
    detail = "\n".join([x for x in [res.stdout.strip(), res.stderr.strip()] if x])
    raise RuntimeError(f"{step} failed (rc={res.rc}){(': ' + detail) if detail else ''}")


def _tool_version(path: Path) -> str:
    try:
        p = subprocess.run([str(path), "--version"], capture_output=True, text=True)
    except Exception:
        return ""
    if p.returncode != 0:
        return ""
    line = str((p.stdout or p.stderr or "").splitlines()[0]).strip()
    return line


def _git_head() -> str:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True)
    except Exception:
        return ""
    if p.returncode != 0:
        return ""
    return str(p.stdout).strip()


def _resolve_generator(raw: str) -> str:
    generator = str(raw or "").strip() or "Ninja"
    if generator.lower() == "ninja" and shutil.which("ninja") is None:
        return "Unix Makefiles"
    return generator

def _resolve_default_toolchain_prefix() -> Path:
    report = detect_mlir_toolchain()
    selected = str(report.get("selected_prefix") or "").strip()
    if selected:
        path = Path(selected)
        if path.is_dir():
            return path
    llc = dict(((report.get("tools") or {}).get("llc") or {}))
    llc_path = str(llc.get("path") or "").strip()
    if llc_path:
        p = Path(llc_path)
        for candidate in (p.parent.parent, p.parent.parent.parent):
            if (candidate / "lib" / "cmake" / "mlir").is_dir() and (candidate / "lib" / "cmake" / "llvm").is_dir():
                return candidate
    return DEFAULT_TOOLCHAIN_PREFIX


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolchain-prefix", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    ap.add_argument("--generator", default="Ninja")
    ap.add_argument("--build-type", default="Release")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8)))
    ap.add_argument("--clean", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    toolchain = Path(args.toolchain_prefix) if args.toolchain_prefix is not None else _resolve_default_toolchain_prefix()
    if not toolchain.is_dir():
        raise SystemExit(f"toolchain prefix not found: {toolchain}")
    if not PLUGIN_SRC.is_dir():
        raise SystemExit(f"plugin source dir not found: {PLUGIN_SRC}")

    mlir_dir = toolchain / "lib" / "cmake" / "mlir"
    llvm_dir = toolchain / "lib" / "cmake" / "llvm"
    if not mlir_dir.is_dir():
        raise SystemExit(f"MLIR cmake dir not found: {mlir_dir}")
    if not llvm_dir.is_dir():
        raise SystemExit(f"LLVM cmake dir not found: {llvm_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir = Path(args.build_dir)
    if bool(args.clean) and build_dir.exists():
        import shutil

        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    generator = _resolve_generator(str(args.generator))

    cfg_cmd = [
        "cmake",
        "-S",
        str(PLUGIN_SRC),
        "-B",
        str(build_dir),
        "-G",
        str(generator),
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        f"-DMLIR_DIR={mlir_dir}",
        f"-DLLVM_DIR={llvm_dir}",
    ]
    cfg = _run(cfg_cmd, cwd=ROOT)
    _require_ok(cfg, step="cmake configure (intentir mlir plugin)")

    build_cmd = ["cmake", "--build", str(build_dir)]
    if int(args.jobs) > 0:
        build_cmd.extend(["--", f"-j{int(args.jobs)}"])
    build = _run(build_cmd, cwd=ROOT)
    _require_ok(build, step="cmake build (intentir mlir plugin)")

    # Locate the built shared library (name depends on platform/toolchain).
    candidates = [
        build_dir / "libIntentIRPasses.so",
        build_dir / "IntentIRPasses.so",
        build_dir / "libIntentIRPasses.dylib",
    ]
    lib_path = next((p for p in candidates if p.is_file()), None)
    if lib_path is None:
        # Fallback: search build dir for a likely output.
        hits = list(build_dir.glob("**/libIntentIRPasses.so"))
        lib_path = hits[0] if hits else None
    if lib_path is None:
        raise SystemExit(f"built plugin library not found under {build_dir}")

    out_lib = out_dir / "libIntentIRPasses.so"
    out_lib.write_bytes(lib_path.read_bytes())

    mlir_opt = toolchain / "bin" / "mlir-opt"
    manifest: dict[str, Any] = {
        "schema_version": "intentir_mlir_pass_plugin_manifest_v1",
        "generated_at": _utc_now_iso(),
        "repo_head": _git_head(),
        "toolchain_prefix": str(toolchain),
        "generator": str(generator),
        "mlir_opt_version": _tool_version(mlir_opt) if mlir_opt.is_file() else "",
        "plugin_src": str(PLUGIN_SRC),
        "plugin_path": str(out_lib),
        "cmake": {"configure_cmd": cfg.cmd, "build_cmd": build.cmd},
    }
    (out_dir / "plugin_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[intentir] built MLIR pass plugin: {out_lib}")
    print(f"[intentir] manifest: {out_dir / 'plugin_manifest.json'}")


if __name__ == "__main__":
    main()
