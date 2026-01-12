"""
TileLang -> CUDA export utilities.

Use-case: treat TileLang as a kernel generator for the CUDA frontend pipeline.
We export:
- CUDA C kernel source (device code, includes tl_templates/* headers)
- PTX text (for CUDA facts extraction / certificate)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class TileLangCudaExport:
    cuda_src: str
    ptx_text: str
    entry_name: str
    include_dirs: List[Path]
    cuda_path: Path
    ptx_path: Path


_PTX_ENTRY_RE = re.compile(r"^\s*\.visible\s+\.entry\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")


def tilelang_include_dirs() -> List[Path]:
    """
    Return include roots needed to compile TileLang-exported CUDA sources.

    TileLang emits includes like:
      #include <tl_templates/cuda/gemm.h>
    which live under `tilelang/src/`.
    """
    try:
        import tilelang  # noqa: PLC0415

        root = Path(tilelang.__file__).resolve().parent
        inc = root / "src"
        if inc.is_dir():
            return [inc]
    except Exception:
        pass
    return []


def export_tilelang_cuda(
    prim_func: Any,
    *,
    out_dir: Path,
    stem: str,
) -> TileLangCudaExport:
    """
    Compile `prim_func` with TileLang and export CUDA kernel source + PTX.
    """
    from frontends.tilelang.runtime import compile_tilelang_kernel  # noqa: PLC0415

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel = compile_tilelang_kernel(prim_func)
    cuda_src = str(kernel.get_kernel_source())
    ptx_text = str(kernel._get_ptx())  # type: ignore[attr-defined]

    entry_name = ""
    for ln in ptx_text.splitlines():
        m = _PTX_ENTRY_RE.match(ln)
        if m:
            entry_name = str(m.group("name"))
            break
    if not entry_name:
        raise RuntimeError("cannot find .visible .entry in exported PTX")

    cuda_path = out_dir / f"{stem}.tilelang_kernel.cu"
    ptx_path = out_dir / f"{stem}.tilelang.ptx"
    cuda_path.write_text(cuda_src, encoding="utf-8")
    ptx_path.write_text(ptx_text, encoding="utf-8")

    return TileLangCudaExport(
        cuda_src=cuda_src,
        ptx_text=ptx_text,
        entry_name=entry_name,
        include_dirs=tilelang_include_dirs(),
        cuda_path=cuda_path,
        ptx_path=ptx_path,
    )


def _dtype_tvm_to_intentir(dt: str) -> str:
    s = str(dt)
    if s in {"float16", "fp16"}:
        return "f16"
    if s in {"float32", "fp32"}:
        return "f32"
    if s in {"float64", "fp64"}:
        return "f64"
    if s in {"int32"}:
        return "i32"
    if s in {"uint8"}:
        return "u8"
    if s in {"bool"}:
        return "bool"
    # Best-effort passthrough.
    return s


def build_io_spec_from_tilelang_prim_func(prim_func: Any) -> Dict[str, Any]:
    """
    Build a CUDA `io_spec` from a TileLang/TVM PrimFunc:
      - tensors: buffer params (name/dtype/shape)
      - scalars: dynamic dims referenced in shapes (int32)
      - arg_names: buffers first, then dynamic dims (in first-seen order)

    This matches TileLang-exported CUDA kernel signatures (ptrs + int dims).
    """
    try:
        from tvm import tir  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"tvm is required to build io_spec from PrimFunc: {e}") from e

    if not isinstance(prim_func, tir.PrimFunc):
        raise TypeError(f"expected tvm.tir.PrimFunc, got {type(prim_func)}")

    tensors: Dict[str, Dict[str, Any]] = {}
    arg_names: List[str] = []
    dyn: List[str] = []
    seen_dyn: set[str] = set()

    # Buffer params (PrimFunc params are handles; shapes may reference tir.Var dims).
    for p in list(prim_func.params):
        if p not in prim_func.buffer_map:
            continue
        buf = prim_func.buffer_map[p]
        name = str(buf.name)
        arg_names.append(name)
        shape: List[int | str] = []
        for d in list(buf.shape):
            if isinstance(d, tir.IntImm):
                shape.append(int(d.value))
            elif isinstance(d, tir.Var):
                sym = str(d.name)
                shape.append(sym)
                if sym not in seen_dyn:
                    seen_dyn.add(sym)
                    dyn.append(sym)
            else:
                # Fallback: keep as string; runtime bindings must provide it.
                sym = str(d)
                shape.append(sym)
                if sym and sym not in seen_dyn:
                    seen_dyn.add(sym)
                    dyn.append(sym)
        tensors[name] = {"dtype": _dtype_tvm_to_intentir(str(buf.dtype)), "shape": shape}

    scalars = {name: "i32" for name in dyn}
    arg_names = arg_names + dyn

    return {"arg_names": arg_names, "tensors": tensors, "scalars": scalars}


__all__ = ["TileLangCudaExport", "build_io_spec_from_tilelang_prim_func", "export_tilelang_cuda", "tilelang_include_dirs"]

