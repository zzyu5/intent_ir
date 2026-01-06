"""
TileLang runtime helpers used by the pipeline.

Goals:
- Execute real TileLang kernels (PrimFunc) via tilelang.compile on CUDA.
- Provide a reference runner compatible with verify.diff_runner (returns numpy IO).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


class TileLangRuntimeError(RuntimeError):
    pass


def _fp8_shim_path() -> Path:
    return Path(__file__).resolve().parent / "shims" / "fp8_e8m0_shim.h"


@dataclass(frozen=True)
class BufferParam:
    name: str
    dtype: str
    shape: Tuple[Any, ...]  # tvm.tir.PrimExpr / IntImm / Var


def _torch_dtype_from_tvm(dtype: str):
    import torch

    dt = str(dtype)
    if dt in {"float16", "fp16"}:
        return torch.float16
    if dt in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dt in {"float32", "fp32"}:
        return torch.float32
    if dt in {"float64", "fp64"}:
        return torch.float64
    if dt in {"int8"}:
        return torch.int8
    if dt in {"int16"}:
        return torch.int16
    if dt in {"int32"}:
        return torch.int32
    if dt in {"int64"}:
        return torch.int64
    if dt in {"uint8"}:
        return torch.uint8
    if dt in {"uint16"}:
        return torch.uint16
    if dt in {"uint32"}:
        return torch.uint32
    if dt in {"uint64"}:
        return torch.uint64
    if dt in {"bool"}:
        return torch.bool
    raise TileLangRuntimeError(f"unsupported tvm dtype for torch interop: {dtype}")


def _resolve_dim(expr: Any, bindings: Dict[str, int]) -> int:
    # Import TVM lazily to keep this module import-light for non-tilelang users.
    from tilelang import tvm  # noqa: PLC0415
    from tvm import tir  # noqa: PLC0415

    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    if isinstance(expr, tir.Var):
        name = str(expr.name)
        if name not in bindings:
            raise TileLangRuntimeError(f"unbound dynamic dim {name} in TileLang buffer shape; bindings={sorted(bindings.keys())}")
        return int(bindings[name])
    # Fall back to best-effort int conversion (covers PrimExpr literals).
    try:
        return int(expr)  # type: ignore[arg-type]
    except Exception as e:
        raise TileLangRuntimeError(f"cannot resolve TileLang dim expr={expr!r}: {e}") from e


def _buffer_params_from_prim_func(prim_func: Any) -> List[BufferParam]:
    from tvm import tir  # noqa: PLC0415

    if not isinstance(prim_func, tir.PrimFunc):
        raise TileLangRuntimeError(f"expected tvm.tir.PrimFunc, got {type(prim_func)}")
    out: List[BufferParam] = []
    for p in list(prim_func.params):
        if p in prim_func.buffer_map:
            buf = prim_func.buffer_map[p]
            out.append(BufferParam(name=str(buf.name), dtype=str(buf.dtype), shape=tuple(buf.shape)))
    return out


def infer_written_global_buffers(prim_func: Any) -> List[str]:
    """
    Return the list of global buffer names that are written by the PrimFunc.
    """
    import tilelang  # noqa: PLC0415
    from tilelang import tvm  # noqa: PLC0415
    from tvm import tir  # noqa: PLC0415

    if not isinstance(prim_func, tir.PrimFunc):
        raise TileLangRuntimeError(f"expected tvm.tir.PrimFunc, got {type(prim_func)}")
    buffers = {str(buf.name): buf for buf in prim_func.buffer_map.values()}
    written: set[str] = set()

    def visit(node: Any) -> None:
        nonlocal written
        if isinstance(node, tir.BufferStore):
            buf = node.buffer
            name = str(buf.name)
            # Only treat global buffers as outputs (ignore shared/local/fragment).
            scope = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
            if scope == "global":
                written.add(name)
        if isinstance(node, tir.Call):
            op_name = getattr(getattr(node, "op", None), "name", "")
            # Also detect tl.tileop.copy(dst_region, ...) stores to global.
            if op_name == "tl.tileop.copy" and len(node.args) >= 2:
                dst = node.args[1]
                try:
                    if isinstance(dst, tir.Call) and getattr(dst.op, "name", "") == "tl.tileop.region":
                        bl = dst.args[0]
                        if isinstance(bl, tir.BufferLoad):
                            buf = bl.buffer
                            scope = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
                            if scope == "global":
                                written.add(str(buf.name))
                except Exception:
                    pass

    tir.stmt_functor.post_order_visit(prim_func.body, visit)
    # Keep stable order aligned with prim_func param order when possible.
    ordered = [p.name for p in _buffer_params_from_prim_func(prim_func) if p.name in written]
    # Add any remaining (should be rare).
    for n in sorted(written):
        if n not in ordered:
            ordered.append(n)
    return ordered


_PRIMFUNC_REGISTRY: Dict[int, Any] = {}


@lru_cache(maxsize=64)
def _compile_cached(fid: int, flags_key: Tuple[str, ...], target: str, backend: str):
    """
    Process-wide cache for TileLang compilation.

    Note: `tilelang.compile` already has an internal cache, but we keep this
    wrapper to avoid repeated Python-level compilation calls (and the noisy
    "kernel_cache" warnings) when the pipeline re-runs.
    """
    import tilelang  # noqa: PLC0415

    pf = _PRIMFUNC_REGISTRY[fid]
    return tilelang.compile(pf, target=target, execution_backend=backend, compile_flags=list(flags_key))


def compile_tilelang_kernel(
    prim_func: Any,
    *,
    target: str = "cuda",
    execution_backend: str = "tvm_ffi",
    extra_compile_flags: Optional[Iterable[str]] = None,
):
    """
    Compile a TileLang PrimFunc into a callable kernel (tilelang.JITKernel).
    """
    import tilelang  # noqa: PLC0415

    flags: List[str] = []
    if extra_compile_flags:
        flags.extend([str(x) for x in extra_compile_flags])

    # tilelang.compile uses its own cache; additionally memoize per-process to avoid
    # repeated Python-level work (and repeated cache warnings).
    fid = id(prim_func)
    _PRIMFUNC_REGISTRY[fid] = prim_func
    key = tuple(flags)
    try:
        return _compile_cached(fid, key, str(target), str(execution_backend))
    except Exception as e:
        # CUDA 12.8+ defines __nv_fp8_e8m0 in <cuda_fp8.h>. Older toolchains (and
        # some TileLang template bundles) may reference it without having the type.
        # Retry with a small shim header only when compilation errors mention it.
        msg = str(e)
        if ("__nv_fp8_e8m0" not in msg) and ("fp8_e8m0" not in msg):
            raise
        shim = _fp8_shim_path()
        flags2 = ["-include", str(shim)] + list(flags)
        key2 = tuple(flags2)
        return _compile_cached(fid, key2, str(target), str(execution_backend))


def run_tilelang_kernel_io(
    prim_func: Any,
    *,
    bindings: Dict[str, int],
    inputs_np: Dict[str, np.ndarray],
    output_names: Optional[Iterable[str]] = None,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Execute a TileLang PrimFunc with numpy inputs, return a numpy IO dict.

    - inputs_np should contain numpy arrays for all input buffers.
    - Outputs are allocated based on PrimFunc buffer shapes and `bindings`.
    """
    import torch

    kernel = compile_tilelang_kernel(prim_func)
    buf_params = _buffer_params_from_prim_func(prim_func)
    written = set(output_names or infer_written_global_buffers(prim_func))

    args: List[Any] = []
    outputs_torch: Dict[str, torch.Tensor] = {}
    for bp in buf_params:
        if bp.name in written:
            shape = tuple(_resolve_dim(d, bindings) for d in bp.shape)
            t = torch.empty(shape, device=device, dtype=_torch_dtype_from_tvm(bp.dtype))
            outputs_torch[bp.name] = t
            args.append(t)
        else:
            if bp.name not in inputs_np:
                raise TileLangRuntimeError(f"missing input buffer {bp.name} for TileLang run; have keys={sorted(inputs_np.keys())}")
            arr = np.asarray(inputs_np[bp.name])
            t = torch.from_numpy(arr).to(device=device)
            args.append(t)

    # Execute kernel
    kernel(*args)
    torch.cuda.synchronize()

    # Return IO dict (inputs as provided; outputs from device)
    out: Dict[str, np.ndarray] = {k: np.asarray(v) for k, v in inputs_np.items()}
    for name, t in outputs_torch.items():
        out[name] = t.detach().cpu().numpy()
    return out


__all__ = [
    "TileLangRuntimeError",
    "infer_written_global_buffers",
    "compile_tilelang_kernel",
    "run_tilelang_kernel_io",
]
