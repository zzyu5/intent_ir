"""
Generic frontend runner (adapter shell).

PR#1 intentionally keeps this module as a thin orchestration helper that can be
reused by future frontends (TileLang/CUDA) without importing torch/triton.
"""

from __future__ import annotations

from typing import Any

from pipeline import registry
from pipeline.interfaces import KernelDescriptor


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


__all__ = ["run_frontend"]
