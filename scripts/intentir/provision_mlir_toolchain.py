#!/usr/bin/env python3
"""
Provision a repository-local MLIR toolchain without sudo.

Default strategy:
1) `apt download mlir-<version>-tools llvm-<version>`
2) `dpkg-deb -x` into `artifacts/toolchains/mlir-<version>`
3) create stable symlink `artifacts/toolchains/mlir-current`
4) write env helper script + json summary
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import tarfile
import urllib.request
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOOLCHAIN_ROOT = ROOT / "artifacts" / "toolchains"


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=(str(cwd) if cwd is not None else None), capture_output=True, text=True)


def _require_ok(p: subprocess.CompletedProcess[str], *, step: str) -> None:
    if p.returncode == 0:
        return
    out = str(p.stdout or "").strip()
    err = str(p.stderr or "").strip()
    detail = "\n".join([x for x in [out, err] if x])
    raise RuntimeError(f"{step} failed (rc={p.returncode}){(': ' + detail) if detail else ''}")


def _find_tool(root: Path, *, tool: str, version: int) -> Path:
    candidates = [
        root / "bin" / tool,
        root / "usr" / "bin" / tool,
        root / "usr" / "bin" / f"{tool}-{version}",
        root / "usr" / "lib" / f"llvm-{version}" / "bin" / tool,
        root / "lib" / f"llvm-{version}" / "bin" / tool,
    ]
    for p in candidates:
        if p.is_file() and os.access(str(p), os.X_OK):
            return p
    raise FileNotFoundError(
        f"cannot find executable `{tool}` under {root}. checked: {', '.join(str(x) for x in candidates)}"
    )


def _find_tool_in_roots(roots: list[Path], *, tool: str, version: int) -> Path:
    checked: list[str] = []
    for root in list(roots or []):
        try:
            return _find_tool(root, tool=tool, version=version)
        except FileNotFoundError as exc:
            checked.append(str(exc))
            continue
    raise FileNotFoundError(
        f"cannot find executable `{tool}` in companion roots: {', '.join(str(x) for x in roots)}"
        + (f" checked={checked}" if checked else "")
    )


def _normalize_sm(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if s.startswith("sm_"):
        digits = "".join(ch for ch in s[3:] if ch.isdigit())
    elif s.startswith("sm"):
        digits = "".join(ch for ch in s[2:] if ch.isdigit())
    elif s.startswith("compute_"):
        digits = "".join(ch for ch in s[len("compute_") :] if ch.isdigit())
    else:
        digits = "".join(ch for ch in s if ch.isdigit())
    return f"sm_{digits}" if digits else ""


def _llc_supported_sms(llc_path: Path) -> list[str]:
    try:
        cp = _run([str(llc_path), "-march=nvptx64", "-mcpu=help"])
        text = f"{cp.stdout or ''}\n{cp.stderr or ''}"
    except Exception:
        return []
    sms = sorted(
        {
            _normalize_sm(token)
            for token in __import__("re").findall(r"\bsm_[0-9]{2,3}\b", text)
            if _normalize_sm(token)
        },
        key=lambda x: int("".join(ch for ch in x if ch.isdigit()) or "-1"),
    )
    return list(sms)


def _supports_required_sm(llc_path: Path, require_sm: str) -> bool:
    req = _normalize_sm(require_sm)
    if not req:
        return True
    return req in set(_llc_supported_sms(llc_path))


def _effective_sm(llc_path: Path, require_sm: str) -> str:
    req = _normalize_sm(require_sm)
    supported = _llc_supported_sms(llc_path)
    if req and req in supported:
        return req
    return (supported[-1] if supported else req)


def _official_release_version_candidates(major: int) -> list[str]:
    out: list[str] = []
    for minor in (1, 0):
        for patch in range(9, -1, -1):
            out.append(f"{int(major)}.{int(minor)}.{int(patch)}")
    return out


def _official_prebuilt_url_candidates(version: str) -> list[str]:
    ver = str(version).strip()
    machine = platform.machine().strip().lower()
    if machine not in {"x86_64", "amd64"}:
        raise RuntimeError(f"official_prebuilt currently only supports x86_64 host, got {machine!r}")
    base = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{ver}"
    return [
        f"{base}/LLVM-{ver}-Linux-X64.tar.xz",
        f"{base}/clang+llvm-{ver}-x86_64-unknown-linux-gnu.tar.xz",
        f"{base}/clang+llvm-{ver}-x86_64-linux-gnu-ubuntu-22.04.tar.xz",
        f"{base}/clang+llvm-{ver}-x86_64-linux-gnu-ubuntu-20.04.tar.xz",
    ]


def _url_exists(url: str) -> bool:
    try:
        req = urllib.request.Request(str(url), method="HEAD", headers={"User-Agent": "intentir-provision/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return int(getattr(resp, "status", 200)) < 400
    except Exception:
        return False


def _download_url(url: str, *, out_path: Path) -> None:
    req = urllib.request.Request(str(url), headers={"User-Agent": "intentir-provision/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, out_path.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def _extract_tar_xz(archive: Path, *, prefix: Path) -> None:
    with tarfile.open(str(archive), mode="r:xz") as tf:
        members = tf.getmembers()
        top_names = {m.name.split("/", 1)[0] for m in members if m.name}
        tf.extractall(path=str(prefix.parent))
    if len(top_names) != 1:
        raise RuntimeError(f"unexpected archive layout for {archive}: top-level entries={sorted(top_names)}")
    extracted = prefix.parent / next(iter(top_names))
    if prefix.exists():
        if prefix.is_file() or prefix.is_symlink():
            prefix.unlink()
        else:
            shutil.rmtree(prefix)
    extracted.rename(prefix)


def _version_candidates_for_source(source: str, requested_version: int) -> list[int]:
    src = str(source or "").strip().lower()
    if src == "official_prebuilt":
        if int(requested_version) in {18, 19, 20}:
            return [int(requested_version)]
        return [20, 19, 18]
    return [int(requested_version)]


def _existing_official_prefix(toolchain_root: Path, release_version: str) -> Path | None:
    candidate = Path(toolchain_root) / f"LLVM-{str(release_version).strip()}-Linux-X64"
    return candidate if candidate.is_dir() else None


def _companion_roots(*, toolchain_root: Path, current_link: Path, exclude: list[Path]) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for p in [current_link, *sorted(Path(toolchain_root).glob("mlir-*")), *sorted(Path(toolchain_root).glob("LLVM-*"))]:
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        if not resolved.exists():
            continue
        if any(resolved == ex for ex in exclude):
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        roots.append(resolved)
    return roots


def _resolve_tool_paths(
    *,
    source_root: Path,
    toolchain_root: Path,
    current_link: Path,
    version: int,
    require_cuda_sm: str,
) -> tuple[dict[str, Path], dict[str, str]]:
    resolved: dict[str, Path] = {}
    origin: dict[str, str] = {}
    llc = _find_tool(source_root, tool="llc", version=version)
    if require_cuda_sm and (not _supports_required_sm(llc, require_cuda_sm)):
        raise RuntimeError(
            f"toolchain root {source_root} does not support required CUDA SM {require_cuda_sm}; supported={_llc_supported_sms(llc)}"
        )
    resolved["llc"] = llc
    origin["llc"] = "source"
    companion_roots = _companion_roots(
        toolchain_root=toolchain_root,
        current_link=current_link,
        exclude=[source_root],
    )
    for tool_name in ("mlir-opt", "mlir-translate", "llvm-as", "opt"):
        tool_path = _find_tool_in_roots(companion_roots, tool=tool_name, version=version)
        resolved[tool_name] = tool_path
        origin[tool_name] = "fallback"
    return resolved, origin


def _symlink_force(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(str(src), str(dst.parent))
    dst.symlink_to(rel)


def _write_env_file(
    *,
    env_file: Path,
    toolchain_root: Path,
    mlir_opt: Path,
    mlir_translate: Path,
    llvm_as: str,
    llvm_opt: str,
    llc: str,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by scripts/intentir/provision_mlir_toolchain.py",
        f"export INTENTIR_MLIR_TOOLCHAIN_ROOT='{toolchain_root}'",
        f"export INTENTIR_MLIR_OPT='{mlir_opt}'",
        f"export INTENTIR_MLIR_TRANSLATE='{mlir_translate}'",
    ]
    if llvm_as:
        lines.append(f"export INTENTIR_LLVM_AS='{llvm_as}'")
    if llvm_opt:
        lines.append(f"export INTENTIR_LLVM_OPT='{llvm_opt}'")
    if llc:
        lines.append(f"export INTENTIR_LLC='{llc}'")
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env_file.chmod(0o755)


def _tool_version(path: Path) -> str:
    p = _run([str(path), "--version"])
    if p.returncode != 0:
        return ""
    line = str((p.stdout or p.stderr or "").splitlines()[0]).strip()
    return line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=14, help="Ubuntu MLIR package major version (default: 14).")
    ap.add_argument("--source", choices=["apt", "official_prebuilt"], default="apt")
    ap.add_argument("--require-cuda-sm", default="", help="Require llc NVPTX support for this SM (e.g. sm_120).")
    ap.add_argument(
        "--prefix",
        type=Path,
        default=None,
        help="Install prefix (default: artifacts/toolchains/mlir-<version>).",
    )
    ap.add_argument(
        "--toolchain-root",
        type=Path,
        default=DEFAULT_TOOLCHAIN_ROOT,
        help="Toolchain root dir (default: artifacts/toolchains).",
    )
    ap.add_argument(
        "--current-link",
        type=Path,
        default=None,
        help="Symlink path to active toolchain (default: <toolchain-root>/mlir-current).",
    )
    ap.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Output env helper (default: <toolchain-root>/mlir-current/env.sh).",
    )
    ap.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--use-current-link",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Maintain <toolchain-root>/mlir-current symlink (default: true).",
    )
    ap.add_argument("--out", type=Path, default=None, help="Optional json summary path.")
    args = ap.parse_args()

    version = int(args.version)
    source = str(args.source).strip().lower()
    require_cuda_sm = _normalize_sm(str(args.require_cuda_sm or ""))
    toolchain_root = Path(args.toolchain_root)
    prefix = Path(args.prefix) if args.prefix is not None else (toolchain_root / f"mlir-{version}")
    current_link = Path(args.current_link) if args.current_link is not None else (toolchain_root / "mlir-current")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    if prefix.exists():
        if not bool(args.force):
            raise SystemExit(f"target prefix already exists: {prefix} (use --force to replace)")
        if prefix.is_symlink() or prefix.is_file():
            prefix.unlink()
        else:
            shutil.rmtree(prefix)

    selected_version = int(version)
    selected_release_version = ""
    selected_url = ""
    selected_source_root: Path | None = None
    packages: list[str] = []
    for candidate_version in _version_candidates_for_source(source, version):
        selected_version = int(candidate_version)
        candidate_prefix = Path(args.prefix) if args.prefix is not None else (toolchain_root / f"mlir-{selected_version}")
        candidate_prefix.parent.mkdir(parents=True, exist_ok=True)

        if source == "apt":
            pkg_mlir = f"mlir-{selected_version}-tools"
            pkg_llvm = f"llvm-{selected_version}"
            packages = [pkg_mlir, pkg_llvm]
            with tempfile.TemporaryDirectory(prefix="intentir_mlir_pkg_") as td:
                tmp = Path(td)
                for pkg in packages:
                    p_dl = _run(["apt", "download", pkg], cwd=tmp)
                    _require_ok(p_dl, step=f"apt download {pkg}")
                    debs = sorted(tmp.glob(f"{pkg}_*_amd64.deb"))
                    if not debs:
                        raise RuntimeError(f"download succeeded but no .deb matched {pkg}_*_amd64.deb")
                    deb = debs[-1]
                    p_x = _run(["dpkg-deb", "-x", str(deb), str(candidate_prefix)])
                    _require_ok(p_x, step=f"dpkg-deb -x ({pkg})")
            selected_source_root = candidate_prefix
        else:
            packages = []
            selected_url = ""
            for release_version in _official_release_version_candidates(selected_version):
                existing = _existing_official_prefix(toolchain_root, release_version)
                if existing is not None:
                    selected_release_version = str(release_version)
                    selected_source_root = existing
                    break
                with tempfile.TemporaryDirectory(prefix="intentir_mlir_prebuilt_") as td:
                    tmp = Path(td)
                    archive = tmp / f"llvm-{selected_version}.tar.xz"
                    for url in _official_prebuilt_url_candidates(release_version):
                        if not _url_exists(url):
                            continue
                        selected_release_version = str(release_version)
                        selected_url = str(url)
                        extracted_prefix = toolchain_root / f"LLVM-{selected_release_version}-Linux-X64"
                        _download_url(selected_url, out_path=archive)
                        _extract_tar_xz(archive, prefix=extracted_prefix)
                        selected_source_root = extracted_prefix
                        break
                    if selected_source_root is not None:
                        break
                if selected_source_root is not None:
                    break
            if selected_source_root is None:
                if source == "official_prebuilt" and candidate_version == _version_candidates_for_source(source, version)[-1]:
                    raise RuntimeError(f"no official prebuilt archive found for LLVM {selected_version}")
                continue

        resolved_tools, tool_origins = _resolve_tool_paths(
            source_root=Path(selected_source_root),
            toolchain_root=toolchain_root,
            current_link=current_link,
            version=selected_version,
            require_cuda_sm=require_cuda_sm,
        )
        mlir_opt = resolved_tools["mlir-opt"]
        mlir_translate = resolved_tools["mlir-translate"]
        llvm_as = resolved_tools["llvm-as"]
        llvm_opt = resolved_tools["opt"]
        llc = resolved_tools["llc"]
        prefix = candidate_prefix
        break
    else:
        raise RuntimeError(f"unable to provision a {source} toolchain satisfying require-sm={require_cuda_sm or '(none)'}")

    # Provide stable non-versioned entrypoints under <prefix>/bin for discovery.
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _symlink_force(bin_dir / "mlir-opt", mlir_opt)
    _symlink_force(bin_dir / "mlir-translate", mlir_translate)
    _symlink_force(bin_dir / "llvm-as", llvm_as)
    _symlink_force(bin_dir / "opt", llvm_opt)
    _symlink_force(bin_dir / "llc", llc)

    if bool(args.use_current_link):
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        current_link.parent.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(str(prefix), str(current_link.parent))
        current_link.symlink_to(rel)

    env_file = Path(args.env_file) if args.env_file is not None else (current_link / "env.sh")
    _write_env_file(
        env_file=env_file,
        toolchain_root=(current_link if bool(args.use_current_link) else prefix),
        mlir_opt=(bin_dir / "mlir-opt"),
        mlir_translate=(bin_dir / "mlir-translate"),
        llvm_as=str(bin_dir / "llvm-as"),
        llvm_opt=str(bin_dir / "opt"),
        llc=str(bin_dir / "llc"),
    )

    summary: dict[str, Any] = {
        "schema_version": "intentir_mlir_toolchain_provision_v1",
        "ok": True,
        "version": int(selected_version),
        "release_version": (str(selected_release_version) if selected_release_version else str(selected_version)),
        "source": str(source),
        "source_root": str(selected_source_root or prefix),
        "package": (str(packages[0]) if packages else ""),
        "packages": [str(x) for x in packages],
        "prefix": str(prefix),
        "current_link": (str(current_link) if bool(args.use_current_link) else ""),
        "env_file": str(env_file),
        "requested_cuda_sm": str(require_cuda_sm),
        "effective_cuda_sm": _effective_sm(bin_dir / "llc", require_cuda_sm),
        "supported_sms": _llc_supported_sms(bin_dir / "llc"),
        "downleveled": (bool(require_cuda_sm) and not _supports_required_sm(bin_dir / "llc", require_cuda_sm)),
        "download_url": str(selected_url),
        "tool_origins": {str(k): str(v) for k, v in dict(tool_origins or {}).items()},
        "tools": {
            "mlir-opt": {"path": str(bin_dir / "mlir-opt"), "version": _tool_version(bin_dir / "mlir-opt")},
            "mlir-translate": {
                "path": str(bin_dir / "mlir-translate"),
                "version": _tool_version(bin_dir / "mlir-translate"),
            },
            "llvm-as": {"path": str(bin_dir / "llvm-as"), "version": _tool_version(bin_dir / "llvm-as")},
            "opt": {"path": str(bin_dir / "opt"), "version": _tool_version(bin_dir / "opt")},
            "llc": {"path": str(bin_dir / "llc"), "version": _tool_version(bin_dir / "llc")},
        },
        "next_steps": [
            f"source {env_file}",
            "python scripts/intentir.py mlir check",
        ],
    }

    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[intentir] wrote {out}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
