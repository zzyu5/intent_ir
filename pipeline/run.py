"""
Generic frontend runner (adapter shell).

PR#1 intentionally keeps this module as a thin orchestration helper that can be
reused by future frontends (TileLang/CUDA) without importing torch/triton.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar

from pipeline import registry
from pipeline.interfaces import KernelDescriptor

T = TypeVar("T")


def run_frontend(frontend: str, kernel: Any, *, artifact_dir: str | None = None) -> KernelDescriptor:
    """
    Build a KernelDescriptor and populate frontend evidence via the adapter.

    This does NOT run LLM/IntentIR/verification; it only collects frontend-side
    artifacts (TTIR/etc) + facts/constraints/certificate/contract.
    """
    adapter = registry.get(frontend)
    desc = adapter.build_descriptor(kernel)
    if artifact_dir is not None:
        desc.meta.setdefault("artifact_dir", artifact_dir)
    desc = adapter.ensure_artifacts(desc, kernel)
    facts = adapter.extract_facts(desc)
    constraints = adapter.extract_constraints(desc, facts)
    cert = adapter.build_certificate(desc, facts, constraints)
    adapter.evaluate_contract(facts, constraints, cert)
    return desc


def process_batch(
    items: Iterable[T],
    process_one: Callable[[T], Dict[str, Any]],
    *,
    fail_fast: bool = False,
    name_fn: Optional[Callable[[T], str]] = None,
    on_error: Optional[Callable[[T, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Best-effort batch runner: one kernel failure should not block the whole batch.

    Returns a list of per-item reports. On exception, returns an error report:
      {"ok": False, "kernel": <name>, "error": {"type", "message", "traceback"}}
    """
    results: List[Dict[str, Any]] = []
    for item in items:
        name = None
        try:
            name = name_fn(item) if name_fn else getattr(item, "name", None)
            name = str(name) if name is not None else str(item)
            report = process_one(item)
            results.append(report if isinstance(report, dict) else {"ok": True, "kernel": name, "result": report})
        except Exception as e:
            if fail_fast:
                raise
            if name is None:
                try:
                    name = name_fn(item) if name_fn else getattr(item, "name", None)
                    name = str(name) if name is not None else str(item)
                except Exception:
                    name = "<unknown>"
            err = {
                "ok": False,
                "kernel": str(name),
                "error": {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()},
            }
            results.append(err)
            if on_error is not None:
                try:
                    on_error(item, err)
                except Exception:
                    pass
    return results


__all__ = ["run_frontend", "process_batch"]
