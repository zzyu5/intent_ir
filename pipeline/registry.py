"""
Frontend adapter registry (frontend_name -> adapter instance).

MVP: a simple in-process dict. This keeps multi-frontend plumbing explicit and
dependency-light; later we can add entrypoints / dynamic discovery.
"""

from __future__ import annotations

import importlib
from typing import Dict

from pipeline.interfaces import FrontendAdapter

_REGISTRY: Dict[str, FrontendAdapter] = {}


def register(adapter: FrontendAdapter) -> None:
    _REGISTRY[adapter.name] = adapter


def get(name: str) -> FrontendAdapter:
    if name not in _REGISTRY:
        # Lazy-load built-in adapters so callers don't need to import a frontend
        # module just to register it.
        _lazy = {
            "triton": "frontends.triton.adapter",
            "tilelang": "frontends.tilelang.adapter",
            "cuda": "frontends.cuda.adapter",
        }
        mod = _lazy.get(name)
        if mod:
            try:
                importlib.import_module(mod)
            except Exception:
                # Keep the original error message below; this makes failures
                # consistent and avoids leaking import internals.
                pass
    if name not in _REGISTRY:
        raise KeyError(f"frontend adapter not registered: {name}")
    return _REGISTRY[name]


__all__ = ["register", "get"]
