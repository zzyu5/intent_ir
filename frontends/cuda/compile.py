"""
CUDA compilation helpers (Task 3.3).

MVP goal:
- take a CUDA kernel translation unit (kernel-only)
- compile to PTX via NVCC
- return PTX text + metadata
"""

from __future__ import annotations

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


def _detect_arch() -> str:
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
    arch = str(arch or _detect_arch())
    opt = str(opt_level).upper()
    if opt not in {"O0", "O1", "O2", "O3"}:
        raise ValueError(f"unsupported opt_level: {opt_level}")

    cu_path = out_dir / f"{kernel_name}.cu"
    ptx_path = out_dir / f"{kernel_name}.ptx"
    cu_path.write_text(str(cuda_src), encoding="utf-8")

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
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed (rc={proc.returncode}):\n{proc.stderr}\n{proc.stdout}")

    return CudaCompileResult(
        cu_path=cu_path,
        ptx_path=ptx_path,
        ptx_text=ptx_path.read_text(encoding="utf-8"),
        arch=arch,
        nvcc_version=_nvcc_version(),
    )


__all__ = ["CudaCompileResult", "compile_cuda_to_ptx"]
