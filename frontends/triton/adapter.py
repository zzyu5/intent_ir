"""
Triton FrontendAdapter (PR#2): package Triton-specific extraction behind a stable interface.

This adapter intentionally reuses existing Triton modules (facts/contract/certificate)
and only adds "plumbing" + KernelDescriptor population.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from pipeline import registry
from pipeline.interfaces import KernelDescriptor

from frontends.triton.certificate import build_certificate
from frontends.triton.contract import evaluate_contract
from frontends.triton.dump import find_latest_ttir, prepare_dump_and_cache_dirs
from frontends.triton.facts import TTIRConstraints, TTIRFacts, extract_constraints, extract_facts


def _import_attr(mod_name: str, dotted_attr: str) -> Any:
    mod = __import__(mod_name, fromlist=["dummy"])
    obj: Any = mod
    for part in dotted_attr.split("."):
        obj = getattr(obj, part)
    return obj


def _kernel_obj_from_spec(spec: Any) -> Any:
    """
    Best-effort: recover the Triton kernel object from a KernelSpec-like object.
    """
    mod = __import__(spec.module, fromlist=["dummy"])
    root = str(spec.attr).split(".", 1)[0]
    return getattr(mod, root)


def _constexpr_names(kernel_obj: Any) -> list[str]:
    arg_names = getattr(kernel_obj, "arg_names", None)
    if not isinstance(arg_names, list):
        return []
    # Many Triton kernels are wrapped (Autotuner/Heuristics). The underlying JITFunction
    # typically sits at `.fn` and exposes `.constexprs` indices.
    k2 = getattr(kernel_obj, "fn", kernel_obj)
    constexpr_ids = getattr(k2, "constexprs", None) or []
    out: list[str] = []
    for i in constexpr_ids:
        try:
            idx = int(i)
        except Exception:
            continue
        if 0 <= idx < len(arg_names):
            out.append(str(arg_names[idx]))
    return out


def _safe_versions() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        meta["torch"] = getattr(torch, "__version__", None)
    except Exception:
        meta["torch"] = None
    try:
        import triton

        meta["triton"] = getattr(triton, "__version__", None)
    except Exception:
        meta["triton"] = None
    return meta


class TritonAdapter:
    name = "triton"

    def build_descriptor(self, kernel: Any) -> KernelDescriptor:
        # In our repo, `kernel` is typically `pipeline.triton.core.KernelSpec`.
        spec = kernel
        src = str(_import_attr(spec.module, spec.attr))
        kernel_obj = _kernel_obj_from_spec(spec)
        arg_names = list(getattr(kernel_obj, "arg_names", []) or [])
        constexpr_names = _constexpr_names(kernel_obj)

        desc = KernelDescriptor(
            schema_version="kernel_desc_v1.0",
            name=str(spec.name),
            frontend="triton",
            source_kind="source",
            source_text=src,
        )
        desc.launch = {
            "canonical_shapes": dict(getattr(spec, "canonical_shapes", {}) or {}),
            "vary_axes": list(getattr(spec, "vary_axes", []) or []),
            "exclude_axes": list(getattr(spec, "exclude_axes", []) or []),
            "module": str(getattr(spec, "module", "")),
            "attr": str(getattr(spec, "attr", "")),
        }
        desc.io_spec = {
            "arg_names": arg_names,
            "constexpr_names": constexpr_names,
        }
        desc.meta = _safe_versions()
        return desc

    def ensure_artifacts(self, desc: KernelDescriptor, kernel: Any) -> KernelDescriptor:
        """
        Ensure TTIR artifacts exist and write `<kernel>.descriptor.json` if artifact_dir is provided.
        """
        artifact_dir_raw = desc.meta.get("artifact_dir")
        if not artifact_dir_raw:
            # Allow in-memory usage (no filesystem artifacts).
            return desc
        artifact_dir = Path(str(artifact_dir_raw))

        # Use pipeline-provided dump/cache dirs if present; otherwise create per-kernel dirs.
        dump_dir_raw = desc.meta.get("triton_dump_dir")
        cache_dir_raw = desc.meta.get("triton_cache_dir")
        if dump_dir_raw and cache_dir_raw:
            dump_dir = Path(str(dump_dir_raw))
            cache_dir = Path(str(cache_dir_raw))
            dump_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            # (re)assert env vars for downstream compilation steps.
            import os

            os.environ["TRITON_KERNEL_DUMP"] = "1"
            os.environ["TRITON_DUMP_DIR"] = str(dump_dir)
            os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
            os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
        else:
            dump_dir, cache_dir = prepare_dump_and_cache_dirs(artifact_dir, desc.name, clean=True)
            desc.meta["triton_dump_dir"] = str(dump_dir)
            desc.meta["triton_cache_dir"] = str(cache_dir)

        ttir_path = find_latest_ttir(dump_dir, desc.name)
        # If missing, try a single compile-triggering run (best-effort) for KernelSpec-like inputs.
        if ttir_path is None and hasattr(kernel, "runner") and hasattr(kernel, "canonical_shapes"):
            try:
                from verify.gen_cases import TestCase

                shapes = dict(getattr(kernel, "canonical_shapes", {}) or {})
                norm = getattr(kernel, "normalize_shapes", None)
                if callable(norm):
                    shapes = dict(norm(shapes))
                kernel.runner(TestCase(shapes=shapes, dtypes={}, seed=0))
            except Exception:
                pass
            ttir_path = find_latest_ttir(dump_dir, desc.name)

        if ttir_path and ttir_path.exists():
            ttir_copy = artifact_dir / f"{desc.name}.ttir"
            try:
                ttir_copy.write_text(ttir_path.read_text(), encoding="utf-8")
                desc.artifacts.ttir_path = str(ttir_copy)
                desc.artifacts.ttir_text = None
                desc.meta.setdefault("ttir_original_path", str(ttir_path))
            except Exception:
                # Keep whatever we have; pipeline can still continue without a stable copy.
                desc.meta.setdefault("ttir_original_path", str(ttir_path))

        # Persist descriptor for traceability (MVP).
        try:
            out_path = artifact_dir / f"{desc.name}.descriptor.json"
            out_path.write_text(json.dumps(desc.to_json_dict(), indent=2), encoding="utf-8")
            desc.meta["descriptor_path"] = str(out_path)
        except Exception:
            pass
        return desc

    def _load_ttir_text(self, desc: KernelDescriptor) -> str:
        if desc.artifacts.ttir_text:
            return str(desc.artifacts.ttir_text)
        # Prefer copied path if present.
        if desc.artifacts.ttir_path:
            p = Path(str(desc.artifacts.ttir_path))
            if p.exists():
                return p.read_text(encoding="utf-8")
        # Fall back to original dump path.
        p2 = desc.meta.get("ttir_original_path")
        if p2:
            p = Path(str(p2))
            if p.exists():
                return p.read_text(encoding="utf-8")
        raise FileNotFoundError("TTIR not available in descriptor artifacts/meta")

    def extract_facts(self, desc: KernelDescriptor) -> TTIRFacts:
        ttir = self._load_ttir_text(desc)
        facts = extract_facts(ttir)
        desc.frontend_facts = {
            "op_counts": dict(facts.op_counts),
            "has_dot": bool(facts.has_dot),
            "has_reduce": bool(facts.has_reduce),
            "has_atomic": bool(facts.has_atomic),
            "num_loads": int(len(facts.load_sites)),
            "num_stores": int(len(facts.store_sites)),
            "num_masks": int(len(facts.mask_sites)),
            "raw_summary": dict(facts.raw_summary),
        }
        return facts

    def extract_constraints(self, desc: KernelDescriptor, facts: TTIRFacts) -> TTIRConstraints:
        ttir = self._load_ttir_text(desc)
        constraints = extract_constraints(ttir, facts=facts)
        desc.frontend_constraints = asdict(constraints)
        return constraints

    def build_certificate(self, desc: KernelDescriptor, facts: TTIRFacts, constraints: TTIRConstraints | None = None):
        ttir = self._load_ttir_text(desc)
        return build_certificate(ttir, facts=facts)

    def evaluate_contract(self, facts: TTIRFacts, constraints: TTIRConstraints | None = None, cert: Any | None = None):
        return evaluate_contract(facts, constraints)


# Register a singleton adapter instance for pipeline use.
registry.register(TritonAdapter())


__all__ = ["TritonAdapter"]
