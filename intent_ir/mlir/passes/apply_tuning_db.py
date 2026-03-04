from __future__ import annotations

import os
from typing import Any, Mapping

from intent_ir.mlir.module import IntentMLIRModule
from pipeline.common.tuning_db import (
    load_tuning_db,
    resolve_tuning_entries,
    tuning_db_path_from_env,
)


def _normalize_cuda_arch(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if s.isdigit():
        return f"sm{s}"
    if s.startswith("sm_"):
        s = s[3:]
    elif s.startswith("sm"):
        s = s[2:]
    else:
        return ""
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"sm{digits}" if digits else ""


def _detect_cuda_arch() -> str:
    env = _normalize_cuda_arch(os.getenv("INTENTIR_CUDA_SM", ""))
    if env:
        return env
    try:  # pragma: no cover - depends on CUDA env
        import torch  # type: ignore

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if isinstance(major, int) and isinstance(minor, int) and major > 0 and minor >= 0:
                return f"sm{int(major)}{int(minor)}"
    except Exception:
        pass
    return ""


def _coerce_int_dict(raw: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    if raw is None:
        return out
    try:
        items = dict(raw).items()
    except Exception:
        return out
    for k, v in items:
        key = str(k).strip()
        if not key:
            continue
        try:
            out[key] = int(v)
        except Exception:
            continue
    return out


def apply_tuning_db(module: IntentMLIRModule, *, backend: str | None = None, **_: object) -> IntentMLIRModule:
    """
    Pipeline pass: apply persistent tuning DB bindings (and optional kernel-kind overrides)
    to `module.meta["shape_bindings"]` for CUDA real-MLIR lowering.
    """

    b = str(backend or "").strip().lower()
    if b and not b.startswith("cuda"):
        return module

    out = _clone(module)
    meta = dict(out.meta or {})
    kernel = str(
        meta.get("spec_name")
        or meta.get("kernel")
        or meta.get("kernel_name")
        or ""
    ).strip()
    shape_bindings = _coerce_int_dict(meta.get("shape_bindings"))
    arch = _detect_cuda_arch()
    db_path = tuning_db_path_from_env(backend="cuda")
    db, db_rel_path = load_tuning_db(path=db_path, backend="cuda")

    applied: dict[str, int] = {}
    tuning_source = "none"

    kernel_kind_override = ""
    if kernel and arch:
        entries = db.get((kernel, arch)) or []
        compiler_stack = str(os.getenv("INTENTIR_COMPILER_STACK", "python") or "").strip().lower()
        merged, kernel_kind_override = resolve_tuning_entries(
            entries,
            shape_bindings=shape_bindings,
            compiler_stack=compiler_stack,
        )
        for k, v in dict(merged).items():
            key = str(k).strip()
            if not key:
                continue
            try:
                applied[key] = int(v)
            except Exception:
                continue
    if applied or kernel_kind_override:
        tuning_source = "tuning_db"
        if applied:
            merged = dict(shape_bindings)
            merged.update(dict(applied))
            meta["shape_bindings"] = merged
        if kernel_kind_override:
            meta["intentir_kernel_kind_override"] = str(kernel_kind_override)

    meta["intentir_tuning_source"] = str(tuning_source)
    meta["intentir_tuning_applied"] = dict(applied)
    if arch:
        meta["intentir_tuning_arch"] = str(arch)
    if db_rel_path:
        meta["intentir_tuning_db"] = str(db_rel_path)
    out.meta = meta
    return out


def _clone(module: IntentMLIRModule) -> IntentMLIRModule:
    return IntentMLIRModule(
        module_text=str(module.module_text or ""),
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )


__all__ = ["apply_tuning_db"]
