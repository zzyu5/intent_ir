"""
CUDA compilation helpers (Task 3.3).

MVP goal:
- take a CUDA kernel translation unit (kernel-only)
- compile to PTX via NVCC
- return PTX text + metadata
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CudaCompileResult:
    cu_path: Path
    ptx_path: Path
    ptx_text: str
    arch: str
    nvcc_version: str
    compile_backend: str = "nvcc"


def _normalize_cuda_arch(raw: str, *, default: str = "sm_80") -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return str(default)
    if s.startswith("compute_"):
        digits = "".join(ch for ch in s[len("compute_") :] if ch.isdigit())
        return f"sm_{digits}" if digits else str(default)
    if s.startswith("sm_"):
        digits = "".join(ch for ch in s[len("sm_") :] if ch.isdigit())
        return f"sm_{digits}" if digits else str(default)
    if s.startswith("sm"):
        digits = "".join(ch for ch in s[2:] if ch.isdigit())
        return f"sm_{digits}" if digits else str(default)
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return f"sm_{digits}"
    return str(default)


def _normalize_nvrtc_arch(raw: str, *, default: str = "") -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return str(default)
    if s.startswith("compute_"):
        digits = "".join(ch for ch in s[len("compute_") :] if ch.isdigit())
        return f"compute_{digits}" if digits else str(default)
    if s.startswith("sm_"):
        digits = "".join(ch for ch in s[len("sm_") :] if ch.isdigit())
        return f"compute_{digits}" if digits else str(default)
    if s.startswith("sm"):
        digits = "".join(ch for ch in s[2:] if ch.isdigit())
        return f"compute_{digits}" if digits else str(default)
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"compute_{digits}" if digits else str(default)


def _nvrtc_arch_for_sm(sm_arch: str) -> str:
    sm = _normalize_cuda_arch(sm_arch, default="")
    digits = "".join(ch for ch in str(sm) if ch.isdigit())
    if not digits:
        return ""
    try:
        sm_num = int(digits)
    except Exception:
        return f"compute_{digits}"
    if sm_num >= 120:
        compat = _normalize_nvrtc_arch(
            os.getenv("INTENTIR_CUDA_NVRTC_COMPAT_ARCH", "compute_90"),
            default="compute_90",
        )
        return compat or "compute_90"
    return f"compute_{digits}"


def _detect_arch() -> str:
    env_arch = _normalize_cuda_arch(os.getenv("INTENTIR_CUDA_SM", ""), default="")
    if env_arch:
        return env_arch
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"sm_{int(major)}{int(minor)}"
    except Exception:
        pass
    # Conservative default (Ampere).
    return "sm_80"


def _nvcc_version() -> str:
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.STDOUT)
        # Keep the first line as a stable summary.
        return out.strip().splitlines()[-1] if out.strip().splitlines() else "nvcc"
    except Exception:
        return "nvcc"


def compile_cuda_to_ptx(
    cuda_src: str,
    *,
    kernel_name: str,
    out_dir: Path,
    arch: Optional[str] = None,
    opt_level: str = "O0",
    include_dirs: Optional[list[Path]] = None,
    extra_cuda_cflags: Optional[list[str]] = None,
) -> CudaCompileResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arch = _normalize_cuda_arch(str(arch or _detect_arch()))
    opt = str(opt_level).upper()
    if opt not in {"O0", "O1", "O2", "O3"}:
        raise ValueError(f"unsupported opt_level: {opt_level}")

    cu_path = out_dir / f"{kernel_name}.cu"
    ptx_path = out_dir / f"{kernel_name}.ptx"
    cu_path.write_text(str(cuda_src), encoding="utf-8")

    mode = str(os.getenv("INTENTIR_CUDA_PTX_COMPILER", "auto") or "").strip().lower() or "auto"
    if mode not in {"auto", "nvcc", "nvrtc"}:
        raise ValueError(f"unsupported INTENTIR_CUDA_PTX_COMPILER={mode!r}; expected auto/nvcc/nvrtc")

    errors: list[str] = []
    has_nvcc = bool(shutil.which("nvcc"))
    if mode in {"auto", "nvcc"}:
        if not has_nvcc:
            if mode == "nvcc":
                raise RuntimeError("nvcc requested but not found on PATH")
        else:
            cmd = [
                "nvcc",
                f"-{opt}",
                "--std=c++17",
                "--ptx",
                "-lineinfo",
                f"-arch={arch}",
            ]
            if include_dirs:
                for d in include_dirs:
                    cmd.extend(["-I", str(Path(d))])
            if extra_cuda_cflags:
                cmd.extend([str(x) for x in extra_cuda_cflags if str(x).strip()])
            cmd += [
                str(cu_path),
                "-o",
                str(ptx_path),
            ]
            proc = subprocess.run(cmd, text=True, capture_output=True)
            if proc.returncode == 0:
                return CudaCompileResult(
                    cu_path=cu_path,
                    ptx_path=ptx_path,
                    ptx_text=ptx_path.read_text(encoding="utf-8"),
                    arch=arch,
                    nvcc_version=_nvcc_version(),
                    compile_backend="nvcc",
                )
            errors.append(f"nvcc rc={proc.returncode}: {proc.stderr or proc.stdout}")
            if mode == "nvcc":
                raise RuntimeError(f"nvcc failed (rc={proc.returncode}):\n{proc.stderr}\n{proc.stdout}")

    if mode in {"auto", "nvrtc"}:
        try:
            from frontends.cuda.runtime import compile_cuda_src_to_ptx  # noqa: PLC0415

            nvrtc_arch = _nvrtc_arch_for_sm(str(arch))
            prev_nvrtc_arch = os.getenv("INTENTIR_CUDA_NVRTC_ARCH")
            try:
                if not (prev_nvrtc_arch and str(prev_nvrtc_arch).strip()) and nvrtc_arch:
                    os.environ["INTENTIR_CUDA_NVRTC_ARCH"] = nvrtc_arch
                ptx_bytes = compile_cuda_src_to_ptx(
                    kernel_name=kernel_name,
                    cuda_src=str(cuda_src),
                    extra_cuda_cflags=extra_cuda_cflags,
                    include_dirs=include_dirs,
                )
            finally:
                if prev_nvrtc_arch is None:
                    os.environ.pop("INTENTIR_CUDA_NVRTC_ARCH", None)
                else:
                    os.environ["INTENTIR_CUDA_NVRTC_ARCH"] = str(prev_nvrtc_arch)
            ptx_text = bytes(ptx_bytes).decode("utf-8", errors="ignore").rstrip("\x00")
            ptx_path.write_text(ptx_text, encoding="utf-8")
            return CudaCompileResult(
                cu_path=cu_path,
                ptx_path=ptx_path,
                ptx_text=ptx_text,
                arch=str(nvrtc_arch or arch),
                nvcc_version="nvrtc",
                compile_backend="nvrtc",
            )
        except Exception as e:
            errors.append(f"nvrtc: {type(e).__name__}: {e}")
            if mode == "nvrtc":
                raise RuntimeError(f"nvrtc compile failed: {type(e).__name__}: {e}") from e

    detail = " | ".join(str(x) for x in errors if str(x).strip())
    raise RuntimeError(f"CUDA->PTX compile failed (mode={mode}): {detail or 'unknown error'}")


__all__ = ["CudaCompileResult", "compile_cuda_to_ptx"]
