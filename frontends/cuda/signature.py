"""
CUDA kernel signature parsing helpers.

We use this to reconcile:
- CUDA source / PTX parameter order (often differs from the "semantic" IO spec)
- pointer element types (e.g., TileLang uses `signed char*` for bool buffers)

The output is a "runtime io_spec" that can be used by:
- the PTX parser (correct arg<->param mapping)
- the CUDA baseline runner (correct launch arg order and dtypes)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class CudaSignatureError(RuntimeError):
    pass


@dataclass(frozen=True)
class CudaParam:
    c_type: str
    name: str
    is_pointer: bool


_KERNEL_DECL_RE = re.compile(r"__global__\s+void\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)


def _find_kernel_paren_span(cuda_src: str, *, kernel_name: str) -> Optional[Tuple[int, int]]:
    """
    Return (lparen_index, rparen_index) for the parameter list of `kernel_name`,
    or None if not found.
    """
    text = str(cuda_src)
    # Match either:
    #   extern "C" __global__ void k(...);
    #   __global__ void k(...) { ... }
    pat = re.compile(rf"__global__\s+void\s+{re.escape(str(kernel_name))}\s*\(", re.M)
    m = pat.search(text)
    if not m:
        return None
    lparen = m.end() - 1
    depth = 0
    for i in range(lparen, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return (lparen, i)
    return None


def parse_cuda_kernel_signature(cuda_src: str, *, kernel_name: str) -> List[CudaParam]:
    """
    Parse a CUDA kernel signature and return ordered parameters.

    Best-effort parser that is robust to whitespace/newlines and common qualifiers.
    """
    span = _find_kernel_paren_span(cuda_src, kernel_name=kernel_name)
    if span is None:
        return []
    lparen, rparen = span
    arg_text = str(cuda_src)[lparen + 1 : rparen].strip()
    if not arg_text:
        return []

    # Split on commas at top level (no nested function pointer types in our kernels).
    parts = [p.strip() for p in arg_text.split(",") if p.strip()]
    out: List[CudaParam] = []
    for p in parts:
        # Extract trailing identifier as the name.
        m = re.match(r"^(?P<ty>.*?)(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$", p)
        if not m:
            raise CudaSignatureError(f"cannot parse CUDA param: {p!r}")
        ty = str(m.group("ty")).strip()
        name = str(m.group("name")).strip()
        is_ptr = ("*" in ty) or ("*" in p)
        out.append(CudaParam(c_type=(ty + " ").strip(), name=name, is_pointer=is_ptr))
    return out


def _normalize_c_type(ty: str) -> str:
    s = str(ty)
    # Drop common qualifiers.
    for q in ("__restrict__", "restrict", "const", "volatile"):
        s = s.replace(q, " ")
    # Pointers/references are not part of the base type.
    s = s.replace("*", " ").replace("&", " ")
    s = " ".join(s.split())
    return s.strip()


def _dtype_from_c_type(ty: str) -> Optional[str]:
    """
    Map C/CUDA types to our IO dtype strings.

    Returns None if unknown.
    """
    base = _normalize_c_type(ty).lower()
    if base in {"float"}:
        return "f32"
    if base in {"double"}:
        return "f64"
    if base in {"__half", "half"}:
        return "f16"
    if base in {"int", "int32_t"}:
        return "i32"
    if base in {"int64_t", "long", "long long"}:
        return "i64"
    if base in {"uint8_t", "unsigned char"}:
        return "u8"
    if base in {"int8_t", "signed char"}:
        return "i8"
    if base in {"bool"}:
        return "bool"
    return None


def infer_runtime_io_spec(
    *,
    cuda_src: str,
    kernel_name: str,
    semantic_io_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Infer a "runtime io_spec" from CUDA signature + a semantic io_spec.

    - Keeps shapes from semantic_io_spec where possible.
    - Uses signature order (critical for PTX param mapping and baseline launch).
    - Overrides dtypes based on the actual pointer/scalar C types.
    - Drops semantic args not present in the kernel signature.
    """
    params = parse_cuda_kernel_signature(cuda_src, kernel_name=kernel_name)
    if not params:
        # If we cannot parse, fall back to semantic spec unchanged.
        return dict(semantic_io_spec or {})

    sem_tensors = semantic_io_spec.get("tensors") if isinstance(semantic_io_spec.get("tensors"), dict) else {}
    sem_scalars = semantic_io_spec.get("scalars") if isinstance(semantic_io_spec.get("scalars"), dict) else {}

    rt_tensors: Dict[str, Any] = {}
    rt_scalars: Dict[str, Any] = {}
    rt_arg_names: List[str] = []

    for p in params:
        rt_arg_names.append(p.name)
        inferred = _dtype_from_c_type(p.c_type)
        if p.is_pointer:
            tspec = sem_tensors.get(p.name)
            if not isinstance(tspec, dict):
                if inferred is None:
                    raise CudaSignatureError(f"missing tensor spec for {p.name} and cannot infer dtype from {p.c_type!r}")
                raise CudaSignatureError(f"missing tensor spec for {p.name} in semantic io_spec")
            out_spec = dict(tspec)
            if inferred is not None:
                out_spec["dtype"] = inferred
            rt_tensors[p.name] = out_spec
        else:
            dt = sem_scalars.get(p.name)
            if dt is None:
                if inferred is None:
                    # Default to i32 for unknown scalars (common for dims).
                    inferred = "i32"
                dt = inferred
            rt_scalars[p.name] = str(dt)

    return {"arg_names": rt_arg_names, "tensors": rt_tensors, "scalars": rt_scalars}


__all__ = [
    "CudaSignatureError",
    "CudaParam",
    "infer_runtime_io_spec",
    "parse_cuda_kernel_signature",
]

