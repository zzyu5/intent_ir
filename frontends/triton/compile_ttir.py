"""
Task3: Triton → TTIR (MLIR text) extraction with multi-strategy fallback.

If Triton/CUDA are unavailable, compile_ttir will raise TTIRCompileError with
clear context; tests should skip in that case.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class TTIRCompileError(Exception):
    """Raised when Triton TTIR compilation fails."""


@dataclass
class TTIRArtifact:
    kernel_name: str
    ttir: str
    signature: Dict[str, str]
    meta: Dict[str, Any]


def normalize_signature(sig: Dict[str, str]) -> Dict[str, str]:
    """
    Normalize signature strings to a small canonical set: *fp16, *fp32, i32, i64.
    """
    norm = {}
    for k, v in sig.items():
        if not isinstance(v, str):
            raise TTIRCompileError(f"signature value for {k} must be string")
        lv = v.lower().replace(" ", "")
        if lv in {"*fp16", "fp16*", "ptr_fp16"}:
            norm[k] = "*fp16"
        elif lv in {"*fp32", "fp32*", "ptr_fp32"}:
            norm[k] = "*fp32"
        elif lv in {"i32", "int", "int32"}:
            norm[k] = "i32"
        elif lv in {"i64", "int64", "long"}:
            norm[k] = "i64"
        else:
            norm[k] = v
    return norm


def _strategy_compile_attr(kernel_fn, signature, meta, attempts: List[str]) -> Optional[TTIRArtifact]:
    try:
        import triton
    except Exception as e:  # pragma: no cover - import guard
        attempts.append(f"import triton failed: {e}")
        return None
    # Triton 3.x uses `triton.compile(src, ...)` and no longer accepts
    # `triton.compile(kernel_fn, signature=...)`. Keep this strategy only for
    # older versions and fall back to dump-based extraction otherwise.
    try:
        if "signature" not in inspect.signature(triton.compile).parameters:
            attempts.append("triton.compile has no 'signature' param (new API); skip to dump strategy")
            return None
    except Exception:
        # If introspection fails, still try the call and record errors.
        pass
    result = None
    try:
        result = triton.compile(kernel_fn, signature=signature, **(meta or {}))
    except TypeError as e:
        attempts.append(f"triton.compile failed: {e}")
    except Exception as e:
        attempts.append(f"triton.compile failed: {e}")
    if result is None:
        return None
    asm = getattr(result, "asm", None)
    if isinstance(asm, dict):
        for key in ("ttir", "ttir-ir", "mlir"):
            if key in asm and asm[key]:
                return TTIRArtifact(kernel_name=kernel_fn.__name__, ttir=asm[key], signature=normalize_signature(signature), meta=meta or {})
    attempts.append("triton.compile returned no ttir in asm")
    return None


def _strategy_legacy_signature(kernel_fn, signature, meta, attempts: List[str]) -> Optional[TTIRArtifact]:
    """
    Try the older-style triton.compile signature that accepts signature=str and constants.
    """
    try:
        import triton
    except Exception as e:  # pragma: no cover
        attempts.append(f"import triton failed: {e}")
        return None
    try:
        if "signature" not in inspect.signature(triton.compile).parameters:
            attempts.append("triton.compile has no 'signature' param (new API); skip legacy strategy")
            return None
    except Exception:
        pass
    try:
        compiled = triton.compile(kernel_fn, signature=signature, constants=(meta or {}).get("constants"))
    except TypeError as e:
        attempts.append(f"triton.compile legacy call failed: {e}")
        return None
    except Exception as e:
        attempts.append(f"triton.compile legacy call raised: {e}")
        return None
    asm = getattr(compiled, "asm", None)
    if isinstance(asm, dict):
        for key in ("ttir", "ttir-ir", "mlir"):
            if key in asm and asm[key]:
                norm_sig = normalize_signature(signature) if isinstance(signature, dict) else {"signature": str(signature)}
                return TTIRArtifact(kernel_name=kernel_fn.__name__, ttir=asm[key], signature=norm_sig, meta=meta or {})
    attempts.append("legacy compile returned no ttir in asm")
    return None


def _strategy_dump(kernel_fn, signature, meta, attempts: List[str]) -> Optional[TTIRArtifact]:
    """
    Use TRITON_KERNEL_DUMP to force TTIR emission by actually launching the kernel once.
    If meta includes 'launch', it will be used. Otherwise, we try a best-effort
    auto-launch based on the provided signature (dict form).
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        attempts.append(f"import torch failed: {e}")
        return None
    if not torch.cuda.is_available():  # pragma: no cover - depends on runtime
        attempts.append("CUDA not available (torch.cuda.is_available() is False)")
        return None

    # Isolate dump/cache directories by default to avoid cache hits suppressing dumps.
    dump_dir = Path((meta or {}).get("dump_dir") or tempfile.mkdtemp(prefix="intentir_triton_dump_"))
    cache_dir = Path((meta or {}).get("cache_dir") or tempfile.mkdtemp(prefix="intentir_triton_cache_"))
    dump_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = str(dump_dir)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

    before = set(dump_dir.rglob("*.ttir"))
    launch = (meta or {}).get("launch")
    if launch is None:
        if isinstance(signature, dict):
            launch = _make_auto_launch(kernel_fn, signature, meta=meta, attempts=attempts)
        if launch is None:
            attempts.append("dump strategy requires meta['launch'] or dict signature for auto-launch")
            return None
    try:
        # launch is expected to run the kernel once; env already set globally
        launch()
    except Exception as e:  # pragma: no cover
        attempts.append(f"dump launch failed: {e}")
        return None
    time.sleep(0.5)
    after = set(dump_dir.rglob("*.ttir"))
    new_files = list(after - before)
    candidates = new_files if new_files else list(after)
    if not candidates:
        attempts.append("dump strategy produced no .ttir files")
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    ttir_text = newest.read_text()
    name = getattr(kernel_fn, "__name__", kernel_fn.__class__.__name__)
    # Best-effort cleanup: keep dirs if explicitly provided, otherwise remove.
    if (meta or {}).get("dump_dir") is None:
        try:
            shutil.rmtree(dump_dir, ignore_errors=True)
        except Exception:
            pass
    if (meta or {}).get("cache_dir") is None:
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception:
            pass
    return TTIRArtifact(
        kernel_name=name,
        ttir=ttir_text,
        signature=normalize_signature(signature) if isinstance(signature, dict) else {"signature": str(signature)},
        meta=meta or {},
    )


def _unwrap_kernel(kernel_fn):
    """
    Triton kernels may be wrapped (e.g., Autotuner). We want the underlying JITFunction.
    """
    # JITFunction itself often has `.fn` pointing back to the original Python
    # function; unwrapping that would lose JIT metadata like `arg_names`.
    if hasattr(kernel_fn, "arg_names"):
        return kernel_fn
    if hasattr(kernel_fn, "fn"):
        return getattr(kernel_fn, "fn")
    return kernel_fn


def _make_auto_launch(kernel_fn, signature: Dict[str, str], *, meta: Optional[Dict[str, Any]], attempts: List[str]):
    """
    Best-effort kernel launcher that only aims to trigger compilation + dump.

    The goal is NOT functional correctness, only to obtain TTIR text.
    """
    try:
        import torch
        import triton
    except Exception as e:  # pragma: no cover
        attempts.append(f"auto-launch import failed: {e}")
        return None

    k = _unwrap_kernel(kernel_fn)
    arg_names = getattr(k, "arg_names", None)
    constexpr_ids = set(getattr(k, "constexprs", []) or [])
    if not arg_names:
        attempts.append("auto-launch: kernel has no arg_names; provide meta['launch'] explicitly")
        return None

    # Build runtime positional args in arg order (excluding constexpr params).
    runtime_args: List[Any] = []
    runtime_names: List[str] = []
    constexpr_names: List[str] = []
    for i, name in enumerate(arg_names):
        if i in constexpr_ids:
            constexpr_names.append(name)
        else:
            runtime_names.append(name)

    def _dtype_from_ptr(ty: str):
        t = ty.lower().replace(" ", "")
        if "fp16" in t or "f16" in t:
            return torch.float16
        if "bf16" in t:
            return torch.bfloat16
        if "fp32" in t or "f32" in t or "float" in t:
            return torch.float32
        if "i1" in t or "bool" in t:
            return torch.bool
        if "i8" in t:
            return torch.int8
        if "i32" in t:
            return torch.int32
        return torch.float32

    def _is_ptr_type(ty: str) -> bool:
        t = ty.strip().lower()
        return t.startswith("*") or t.endswith("*") or "ptr" in t

    # Pick small-but-safe defaults.
    n_elems = int((meta or {}).get("auto_n_elems") or 1024)
    device = "cuda"

    # Scalars sometimes represent strides/ld; choose conservative values.
    def _scalar_value(name: str, ty: str):
        t = ty.lower()
        if "f" in t and "i" not in t:
            return float((meta or {}).get(f"auto_{name}", 1.0))
        if name.upper() in {"N", "M", "K", "SIZE", "LEN"}:
            return int((meta or {}).get(f"auto_{name}", n_elems))
        if "stride" in name.lower():
            return 1
        return int((meta or {}).get(f"auto_{name}", 1))

    # Allocate tensors for pointer args.
    for name in runtime_names:
        ty = signature.get(name)
        if ty is None:
            # Fall back to int scalars when no type info is available.
            runtime_args.append(int((meta or {}).get(f"auto_{name}", 1)))
            continue
        if _is_ptr_type(ty):
            dt = _dtype_from_ptr(ty)
            # allocate 1D buffer; most kernels treat pointers as flattened anyway
            buf = torch.empty((n_elems,), device=device, dtype=dt)
            if dt.is_floating_point:
                buf.normal_()
            else:
                buf.zero_()
            runtime_args.append(buf)
        else:
            runtime_args.append(_scalar_value(name, ty))

    # Launch kwargs: meta options + constexpr params.
    launch_kwargs: Dict[str, Any] = {}
    if meta:
        for k2, v2 in meta.items():
            if k2 in {"launch", "grid", "dump_dir", "cache_dir", "auto_n_elems"}:
                continue
            if k2 == "constants" and isinstance(v2, dict):
                launch_kwargs.update(v2)
                continue
            launch_kwargs[k2] = v2

    def _default_constexpr(name: str) -> int:
        up = name.upper()
        if up in {"BLOCK", "BLOCK_SIZE"}:
            return 256
        if up.startswith("BLOCK_"):
            return 16
        return 1

    for cname in constexpr_names:
        if cname not in launch_kwargs:
            launch_kwargs[cname] = int((meta or {}).get(f"auto_{cname}", _default_constexpr(cname)))

    # Default to a tiny grid that still provides up to 3 program_id dimensions.
    grid = (meta or {}).get("grid") or (1, 1, 1)

    def _launch():
        k2 = kernel_fn
        # Autotuner kernels need to be invoked via the wrapper; JITFunction via itself.
        k2[grid](*runtime_args, **launch_kwargs)
        torch.cuda.synchronize()

    return _launch


def compile_ttir(kernel_fn, signature: Dict[str, str] | str, meta: Optional[Dict[str, Any]] = None, warmup: bool = False) -> TTIRArtifact:
    """
    Attempt to compile a Triton kernel and extract TTIR text.

    Parameters
    ----------
    kernel_fn: Triton kernel function object
    signature: dict mapping argument names to types (e.g., "*fp16", "i32") or legacy string form
    meta: optional dict of meta such as num_warps/num_stages/constants
    warmup: unused placeholder for future JIT warmup strategies
    """
    attempts: List[str] = []
    artifact = _strategy_compile_attr(kernel_fn, signature, meta, attempts)
    if artifact:
        return artifact
    artifact = _strategy_legacy_signature(kernel_fn, signature, meta, attempts)
    if artifact:
        return artifact
    artifact = _strategy_dump(kernel_fn, signature, meta, attempts)
    if artifact:
        return artifact
    raise TTIRCompileError("Failed to obtain TTIR. Attempts: " + " | ".join(attempts))


__all__ = ["TTIRCompileError", "TTIRArtifact", "compile_ttir", "normalize_signature"]
