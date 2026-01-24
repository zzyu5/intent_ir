"""
CUDA baseline runtime runner (MVP).

We implement baseline execution via a tiny Torch CUDA extension compiled at
runtime (torch.utils.cpp_extension.load_inline). This keeps dependencies to
only torch+nvcc (CuPy/cuda-python not required).
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


class CudaRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class CudaLaunch:
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    shared_mem: int = 0


def _torch() -> Any:
    import torch  # noqa: PLC0415

    return torch


def _cuda_free_mem_mb() -> int:
    """
    Best-effort free CUDA memory query.

    This can fail if the CUDA context cannot be created (e.g., GPU OOM); in that
    case we return 0 so callers can surface a clearer error early.
    """
    torch = _torch()
    try:
        if not torch.cuda.is_available():
            return 0
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        free, _total = torch.cuda.mem_get_info()
        return int(free // (1024 * 1024))
    except Exception:
        return 0


def _min_free_mem_mb() -> int:
    import os  # noqa: PLC0415

    # Default to disabled. Some environments run large GPU workloads (e.g. vLLM)
    # but still have enough headroom for tiny baseline kernels.
    raw = os.getenv("INTENTIR_CUDA_MIN_FREE_MB", "0")
    try:
        v = int(raw)
    except Exception:
        v = 0
    return max(0, v)


def _dtype_to_torch(dt: str):
    torch = _torch()
    s = str(dt)
    if s == "f16":
        return torch.float16
    if s == "f32":
        return torch.float32
    if s == "i8":
        return torch.int8
    if s == "i16":
        return torch.int16
    if s == "i32":
        return torch.int32
    if s == "i64":
        return torch.int64
    if s == "u8":
        return torch.uint8
    if s == "bool":
        return torch.bool
    raise CudaRuntimeError(f"unsupported dtype for CUDA runtime: {dt}")


def _c_type(dt: str) -> str:
    s = str(dt)
    if s == "f16":
        return "__half"
    if s == "f32":
        return "float"
    if s == "i8":
        return "int8_t"
    if s == "i16":
        return "int16_t"
    if s == "i32":
        return "int"
    if s == "i64":
        return "int64_t"
    if s == "u8":
        return "uint8_t"
    if s == "bool":
        return "bool"
    raise CudaRuntimeError(f"unsupported dtype for CUDA runtime: {dt}")


def _hash_src(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def _default_torch_ext_root() -> Path:
    """
    Default Torch extension build root under the repo.

    Some sandboxed environments forbid writing to `~/.cache/torch_extensions`.
    Keeping build outputs under `artifacts/` also makes runs reproducible.
    """
    root = Path(__file__).resolve().parents[2]
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    return root / "artifacts" / "torch_extensions" / py_tag


def _torch_ext_build_dir(name: str) -> Path:
    base = os.getenv("INTENTIR_TORCH_EXT_DIR")
    if base:
        return Path(base) / str(name)
    return _default_torch_ext_root() / str(name)


def _build_extension_src(cuda_src: str, *, kernel_name: str, io_spec: Dict[str, Any]) -> str:
    arg_names = io_spec.get("arg_names") if isinstance(io_spec.get("arg_names"), list) else []
    arg_names = [str(x) for x in arg_names]
    tensors = io_spec.get("tensors") if isinstance(io_spec.get("tensors"), dict) else {}
    scalars = io_spec.get("scalars") if isinstance(io_spec.get("scalars"), dict) else {}

    # Build launch() signature: tensors as torch::Tensor, scalars as int64_t.
    sig_args: list[str] = []
    call_args: list[str] = []
    checks: list[str] = []
    ptr_decls: list[str] = []

    for name in arg_names:
        if name in tensors:
            spec = tensors[name] if isinstance(tensors.get(name), dict) else {}
            dt = str(spec.get("dtype") or "f32")
            sig_args.append(f"torch::Tensor {name}")
            checks += [
                f"TORCH_CHECK({name}.is_cuda(), \"{name} must be CUDA tensor\");",
                f"TORCH_CHECK({name}.is_contiguous(), \"{name} must be contiguous\");",
            ]
            # dtype check (minimal)
            if dt == "f32":
                checks.append(f"TORCH_CHECK({name}.scalar_type() == at::kFloat, \"{name} must be float32\");")
            elif dt == "f16":
                checks.append(f"TORCH_CHECK({name}.scalar_type() == at::kHalf, \"{name} must be float16\");")
            elif dt == "u8":
                checks.append(f"TORCH_CHECK({name}.scalar_type() == at::kByte, \"{name} must be uint8\");")
            elif dt == "i8":
                checks.append(f"TORCH_CHECK({name}.scalar_type() == at::kChar, \"{name} must be int8\");")
            elif dt == "i16":
                checks.append(f"TORCH_CHECK({name}.scalar_type() == at::kShort, \"{name} must be int16\");")
            elif dt == "i32":
                checks.append(f"TORCH_CHECK({name}.scalar_type() == at::kInt, \"{name} must be int32\");")
            elif dt == "i64":
                checks.append(f"TORCH_CHECK({name}.scalar_type() == at::kLong, \"{name} must be int64\");")
            elif dt == "bool":
                checks.append(f"TORCH_CHECK({name}.scalar_type() == at::kBool, \"{name} must be bool\");")
            cty = _c_type(dt)
            ptr_decls.append(f"auto {name}_ptr = ({cty}*){name}.data_ptr();")
            call_args.append(f"{name}_ptr")
        elif name in scalars:
            dt = str(scalars[name])
            if dt == "f32":
                sig_args.append(f"double {name}")
                call_args.append(f"(float){name}")
            else:
                sig_args.append(f"int64_t {name}")
                call_args.append(f"({ _c_type(dt) }){name}")
        else:
            # Unknown arg: treat as int64 scalar.
            sig_args.append(f"int64_t {name}")
            call_args.append(f"(int64_t){name}")

    sig_args += [
        "int64_t grid_x",
        "int64_t grid_y",
        "int64_t grid_z",
        "int64_t block_x",
        "int64_t block_y",
        "int64_t block_z",
        "int64_t shared_mem",
    ]
    dim = "dim3((unsigned)grid_x,(unsigned)grid_y,(unsigned)grid_z)"
    bdim = "dim3((unsigned)block_x,(unsigned)block_y,(unsigned)block_z)"

    src = f"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

{cuda_src}

static void launch({", ".join(sig_args)}) {{
  {" ".join(checks)}
  {" ".join(ptr_decls)}
  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();
  {kernel_name}<<<{dim}, {bdim}, (size_t)shared_mem, stream>>>({", ".join(call_args)});
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("launch", &launch, "Launch CUDA kernel");
}}
""".lstrip()
    return src


@lru_cache(maxsize=32)
def _load_ext_cached(name: str, cuda_src: str, extra_cuda_cflags: Tuple[str, ...]) -> Any:
    torch = _torch()
    from torch.utils.cpp_extension import load_inline  # noqa: PLC0415

    build_dir = _torch_ext_build_dir(name)
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=None,
        with_cuda=True,
        extra_cuda_cflags=["--std=c++17", *list(extra_cuda_cflags)],
        extra_cflags=["-std=c++17"],
        build_directory=str(build_dir),
        verbose=False,
    )


def compile_cuda_extension(
    *,
    kernel_name: str,
    cuda_src: str,
    io_spec: Dict[str, Any],
    extra_cuda_cflags: Optional[Iterable[str]] = None,
) -> Any:
    """
    Compile (or load from cache) a tiny Torch extension that exposes `launch(...)`.
    """
    flags = tuple(str(x) for x in (extra_cuda_cflags or []) if str(x).strip())
    h = _hash_src(cuda_src + "\nFLAGS:" + " ".join(flags))
    mod_name = f"intentir_cuda_{kernel_name}_{h}"
    full_src = _build_extension_src(cuda_src, kernel_name=kernel_name, io_spec=io_spec)
    return _load_ext_cached(mod_name, full_src, flags)


def run_cuda_kernel_io(
    *,
    kernel_name: str,
    cuda_src: str,
    io_spec: Dict[str, Any],
    launch: CudaLaunch,
    bindings: Dict[str, Any],
    inputs_np: Dict[str, np.ndarray],
    output_names: Iterable[str],
    device: str = "cuda",
    extra_cuda_cflags: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Execute a CUDA kernel, returning a numpy IO dict (inputs + outputs).
    """
    torch = _torch()
    if not torch.cuda.is_available():
        raise CudaRuntimeError("torch.cuda is not available; cannot run CUDA baseline")

    min_free = _min_free_mem_mb()
    if min_free > 0:
        free_mb = _cuda_free_mem_mb()
        if free_mb < min_free:
            raise CudaRuntimeError(
                f"CUDA free memory too low ({free_mb} MiB < {min_free} MiB). "
                "Free GPU memory (stop GPU-heavy processes) or set INTENTIR_CUDA_MIN_FREE_MB=0 to bypass."
            )

    tensors = io_spec.get("tensors") if isinstance(io_spec.get("tensors"), dict) else {}
    scalars = io_spec.get("scalars") if isinstance(io_spec.get("scalars"), dict) else {}
    arg_names = io_spec.get("arg_names") if isinstance(io_spec.get("arg_names"), list) else []
    arg_names = [str(x) for x in arg_names]
    out_set = {str(x) for x in output_names}

    # Build torch args in kernel param order.
    args: list[Any] = []
    outputs_torch: Dict[str, Any] = {}
    try:
        for name in arg_names:
            if name in tensors:
                spec = tensors[name] if isinstance(tensors.get(name), dict) else {}
                dt = str(spec.get("dtype") or "f32")
                shape_tpl = spec.get("shape") if isinstance(spec.get("shape"), list) else None
                if name in out_set:
                    if not shape_tpl:
                        raise CudaRuntimeError(f"missing output tensor shape for {name} in io_spec")
                    shape = tuple(int(bindings[str(d)]) if isinstance(d, str) else int(d) for d in shape_tpl)
                    t = torch.empty(shape, device=device, dtype=_dtype_to_torch(dt))
                    outputs_torch[name] = t
                    args.append(t)
                else:
                    if name not in inputs_np:
                        raise CudaRuntimeError(f"missing input {name} for CUDA baseline; have keys={sorted(inputs_np.keys())}")
                    arr = np.asarray(inputs_np[name])
                    t = torch.from_numpy(arr).to(device=device)
                    if t.dtype != _dtype_to_torch(dt):
                        t = t.to(dtype=_dtype_to_torch(dt))
                    args.append(t.contiguous())
            elif name in scalars:
                if name not in bindings:
                    raise CudaRuntimeError(f"missing scalar binding {name}; have {sorted(bindings.keys())}")
                dt = str(scalars[name])
                if dt == "f32":
                    args.append(float(bindings[name]))
                else:
                    args.append(int(bindings[name]))
            else:
                if name not in bindings:
                    raise CudaRuntimeError(f"missing binding for unknown arg {name}")
                args.append(int(bindings[name]))
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            free_mb = _cuda_free_mem_mb()
            raise CudaRuntimeError(f"CUDA OOM during input/output allocation (free≈{free_mb} MiB): {msg}") from e
        raise

    # Append launch dims
    gx, gy, gz = (int(x) for x in launch.grid)
    bx, by, bz = (int(x) for x in launch.block)
    args += [gx, gy, gz, bx, by, bz, int(launch.shared_mem)]

    mod = compile_cuda_extension(kernel_name=kernel_name, cuda_src=cuda_src, io_spec=io_spec, extra_cuda_cflags=extra_cuda_cflags)
    try:
        mod.launch(*args)
        torch.cuda.synchronize()
    except Exception as e:
        raise CudaRuntimeError(f"CUDA kernel launch failed: {type(e).__name__}: {e}") from e

    out: Dict[str, np.ndarray] = {k: np.asarray(v) for k, v in inputs_np.items()}
    for name, t in outputs_torch.items():
        out[name] = t.detach().cpu().numpy()
    return out


__all__ = ["CudaLaunch", "CudaRuntimeError", "compile_cuda_extension", "run_cuda_kernel_io"]
