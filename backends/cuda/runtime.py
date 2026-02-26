"""
CUDA backend runtime helpers (MVP).

We reuse the Torch CUDA extension runner from `frontends/cuda/runtime.py` to keep
the backend dependency-light (torch + nvcc only).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from frontends.cuda.runtime import (  # noqa: F401
    CudaLaunch,
    CudaRuntimeError,
    compile_cuda_extension,
    compile_cuda_src_to_ptx,
    cuda_extension_cache_info,
    load_cuda_ptx_module,
    run_cuda_kernel_io,
    run_cuda_kernel_io_ptx,
)


def run_cuda_kernel(
    *,
    kernel_name: str,
    cuda_src: str,
    io_spec: Dict[str, Any],
    launch: CudaLaunch,
    bindings: Dict[str, Any],
    inputs_np: Dict[str, np.ndarray],
    output_names: Iterable[str],
    extra_cuda_cflags: Optional[Iterable[str]] = None,
    compiled_module: Any = None,
) -> Dict[str, np.ndarray]:
    """
    Thin wrapper around `frontends.cuda.runtime.run_cuda_kernel_io`.
    """
    return run_cuda_kernel_io(
        kernel_name=kernel_name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        bindings=bindings,
        inputs_np=inputs_np,
        output_names=output_names,
        extra_cuda_cflags=extra_cuda_cflags,
        compiled_module=compiled_module,
    )


def compile_cuda_ptx(
    *,
    kernel_name: str,
    cuda_src: str,
    extra_cuda_cflags: Optional[Iterable[str]] = None,
) -> bytes:
    return compile_cuda_src_to_ptx(
        kernel_name=kernel_name,
        cuda_src=cuda_src,
        extra_cuda_cflags=extra_cuda_cflags,
    )


def compile_cuda_ptx_nvcc(
    *,
    kernel_name: str,
    cuda_src: str,
    extra_cuda_cflags: Optional[Iterable[str]] = None,
) -> bytes:
    """
    Compile CUDA source to PTX bytes via nvcc (toolchain path).
    """
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise CudaRuntimeError("nvcc not found for compile_cuda_ptx_nvcc")
    arch, _lto_arch = _cuda_arch_tokens()
    runtime_inc = (Path(__file__).resolve().parent / "runtime").resolve()
    if not runtime_inc.is_dir():
        raise CudaRuntimeError(f"cuda runtime include dir missing for nvcc ptx compile: {runtime_inc}")
    flags = [str(x) for x in (extra_cuda_cflags or []) if str(x).strip()]
    with tempfile.TemporaryDirectory(prefix=f"intentir_cuda_nvcc_ptx_{kernel_name}_") as td:
        td_path = Path(td)
        cu_path = td_path / f"{kernel_name}.cu"
        ptx_path = td_path / f"{kernel_name}.ptx"
        cu_path.write_text(str(cuda_src or ""), encoding="utf-8")
        cmd = [
            str(nvcc),
            "-ptx",
            f"-arch={arch}",
            "-O3",
            f"-I{runtime_inc}",
            str(cu_path),
            "-o",
            str(ptx_path),
            *flags,
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if int(cp.returncode) != 0:
            raise CudaRuntimeError(
                f"nvcc ptx compile failed rc={cp.returncode}: {cp.stderr or cp.stdout}"
            )
        if not ptx_path.is_file():
            raise CudaRuntimeError(f"nvcc ptx compile produced no output: {ptx_path}")
        return bytes(ptx_path.read_bytes())


def _cuda_arch_tokens() -> Tuple[str, str]:
    sm_raw = str(os.getenv("INTENTIR_CUDA_SM", "")).strip().lower()
    if sm_raw.startswith("sm_"):
        sm_digits = "".join(ch for ch in sm_raw[3:] if ch.isdigit())
        sm_arch = sm_raw if sm_digits else "sm_80"
        if not sm_digits:
            sm_digits = "80"
    elif sm_raw.isdigit():
        sm_digits = sm_raw
        sm_arch = f"sm_{sm_raw}"
    else:
        sm_digits = "80"
        sm_arch = "sm_80"
    return sm_arch, f"lto_{sm_digits}"


def compile_cuda_ptx_nvcc_dlto(
    *,
    kernel_name: str,
    cuda_src: str,
    extra_cuda_cflags: Optional[Iterable[str]] = None,
) -> tuple[bytes, bytes | None]:
    """
    Compile CUDA source to PTX via NVCC LTO flow and return PTX bytes plus optional LTOIR bytes.

    This path uses:
      1) `nvcc -dc -arch=lto_XX`
      2) `nvcc -dlto -dlink -ptx -arch=sm_XX`
    """
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise CudaRuntimeError("nvcc not found for compile_cuda_ptx_nvcc_dlto")
    sm_arch, lto_arch = _cuda_arch_tokens()
    runtime_inc = (Path(__file__).resolve().parent / "runtime").resolve()
    if not runtime_inc.is_dir():
        raise CudaRuntimeError(f"cuda runtime include dir missing for nvcc dlto ptx compile: {runtime_inc}")
    flags = [str(x) for x in (extra_cuda_cflags or []) if str(x).strip()]
    with tempfile.TemporaryDirectory(prefix=f"intentir_cuda_nvcc_dlto_ptx_{kernel_name}_") as td:
        td_path = Path(td)
        cu_path = td_path / f"{kernel_name}.cu"
        obj_path = td_path / f"{kernel_name}.lto.o"
        ptx_path = td_path / f"{kernel_name}.ptx"
        cu_path.write_text(str(cuda_src or ""), encoding="utf-8")
        cmd_compile = [
            str(nvcc),
            "-dc",
            f"-arch={lto_arch}",
            "-O3",
            f"-I{runtime_inc}",
            "--keep",
            "--keep-dir",
            str(td_path),
            str(cu_path),
            "-o",
            str(obj_path),
            *flags,
        ]
        cp_compile = subprocess.run(cmd_compile, capture_output=True, text=True)
        if int(cp_compile.returncode) != 0:
            raise CudaRuntimeError(
                f"nvcc dlto compile failed rc={cp_compile.returncode}: {cp_compile.stderr or cp_compile.stdout}"
            )
        cmd_dlink = [
            str(nvcc),
            "-dlto",
            f"-arch={sm_arch}",
            "-dlink",
            "-ptx",
            str(obj_path),
            "-o",
            str(ptx_path),
            *flags,
        ]
        cp_dlink = subprocess.run(cmd_dlink, capture_output=True, text=True)
        if int(cp_dlink.returncode) != 0:
            raise CudaRuntimeError(
                f"nvcc dlto dlink->ptx failed rc={cp_dlink.returncode}: {cp_dlink.stderr or cp_dlink.stdout}"
            )
        if not ptx_path.is_file():
            raise CudaRuntimeError(f"nvcc dlto ptx compile produced no output: {ptx_path}")
        ltoir_path = None
        ltoir_candidates = sorted(td_path.glob("*.ltoir"))
        if ltoir_candidates:
            ltoir_path = ltoir_candidates[0]
        ltoir_bytes = bytes(ltoir_path.read_bytes()) if (ltoir_path is not None and ltoir_path.is_file()) else None
        return bytes(ptx_path.read_bytes()), ltoir_bytes


def run_cuda_kernel_ptx(
    *,
    kernel_name: str,
    ptx: bytes | str,
    io_spec: Dict[str, Any],
    launch: CudaLaunch,
    bindings: Dict[str, Any],
    inputs_np: Dict[str, np.ndarray],
    output_names: Iterable[str],
    compiled_module: Any = None,
) -> Dict[str, np.ndarray]:
    return run_cuda_kernel_io_ptx(
        kernel_name=kernel_name,
        ptx=ptx,
        io_spec=io_spec,
        launch=launch,
        bindings=bindings,
        inputs_np=inputs_np,
        output_names=output_names,
        compiled_module=compiled_module,
    )


__all__ = [
    "CudaLaunch",
    "CudaRuntimeError",
    "compile_cuda_extension",
    "compile_cuda_ptx",
    "compile_cuda_ptx_nvcc",
    "compile_cuda_ptx_nvcc_dlto",
    "cuda_extension_cache_info",
    "load_cuda_ptx_module",
    "run_cuda_kernel",
    "run_cuda_kernel_ptx",
]
