"""
CUDA FrontendAdapter (MVP): package CUDA PTX extraction behind a stable interface.

This adapter:
- builds a KernelDescriptor from a CUDA KernelSpec-like object
- compiles CUDA source to PTX (artifacts)
- extracts facts/constraints/certificate (V2)

The full LLM/IntentIR pipeline is implemented in `pipeline/cuda/core.py`.
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

from .compile import compile_cuda_to_ptx
from .facts import CudaFacts, extract_facts
from .constraints import extract_constraints
from .certificate import build_certificate_v2


def _safe_versions() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch  # noqa: PLC0415

        meta["torch"] = getattr(torch, "__version__", None)
        meta["torch_cuda"] = getattr(torch.version, "cuda", None)
    except Exception:
        meta["torch"] = None
    return meta


class CudaAdapter:
    name = "cuda"

    def build_descriptor(self, kernel: Any) -> KernelDescriptor:
        spec = kernel
        src = ""
        cuda_path = getattr(spec, "cuda_path", None)
        if cuda_path is not None:
            try:
                src = Path(str(cuda_path)).read_text(encoding="utf-8")
            except Exception:
                src = ""
        if not src:
            src = str(getattr(spec, "cuda_src", ""))
        io_spec = dict(getattr(spec, "io_spec", {}) or {})
        desc = KernelDescriptor(
            schema_version="kernel_desc_v1.0",
            name=str(getattr(spec, "name", "cuda_kernel")),
            frontend="cuda",
            source_kind="source",
            source_text=src,
        )
        desc.launch = {
            "canonical_shapes": dict(getattr(spec, "canonical_shapes", {}) or {}),
            "vary_axes": list(getattr(spec, "vary_axes", []) or []),
            "exclude_axes": list(getattr(spec, "exclude_axes", []) or []),
            # CUDA launch config (block dims); grid dims depend on runtime shapes.
            "block": tuple(getattr(spec, "block", (1, 1, 1)) or (1, 1, 1)),
        }
        desc.io_spec = io_spec
        desc.meta = _safe_versions()
        if cuda_path is not None:
            desc.meta["cuda_source_path"] = str(cuda_path)
        return desc

    def ensure_artifacts(self, desc: KernelDescriptor, kernel: Any) -> KernelDescriptor:
        artifact_dir_raw = desc.meta.get("artifact_dir")
        if not artifact_dir_raw:
            return desc
        artifact_dir = Path(str(artifact_dir_raw))
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Prefer a prebuilt PTX snapshot (e.g., from TileLang export), otherwise compile via NVCC.
        ptx_text = getattr(kernel, "ptx_text", None)
        if not isinstance(ptx_text, str) or not ptx_text.strip():
            ptx_text = None
        include_dirs = getattr(kernel, "include_dirs", None)
        if isinstance(include_dirs, list):
            include_dirs = [Path(str(x)) for x in include_dirs]
        else:
            include_dirs = None

        if ptx_text is not None:
            ptx_path = artifact_dir / f"{desc.name}.ptx"
            ptx_path.write_text(ptx_text, encoding="utf-8")
            desc.artifacts.ptx_text = None
            desc.artifacts.extra["ptx_path"] = str(ptx_path)
            desc.meta["ptx_origin"] = str(getattr(kernel, "ptx_origin", "prebuilt"))
        else:
            res = compile_cuda_to_ptx(
                desc.source_text,
                kernel_name=str(desc.name),
                out_dir=artifact_dir,
                opt_level="O0",
                include_dirs=include_dirs,
            )
            desc.artifacts.ptx_text = None
            desc.artifacts.extra["ptx_path"] = str(res.ptx_path)
            desc.meta["nvcc"] = str(res.nvcc_version)
            desc.meta["cuda_arch"] = str(res.arch)

        # Persist descriptor for traceability.
        try:
            out_path = artifact_dir / f"{desc.name}.descriptor.json"
            out_path.write_text(json.dumps(desc.to_json_dict(), indent=2), encoding="utf-8")
            desc.meta["descriptor_path"] = str(out_path)
        except Exception:
            pass
        return desc

    def extract_facts(self, desc: KernelDescriptor) -> CudaFacts:
        facts = extract_facts(desc)
        desc.frontend_facts = {
            "anchors": dict(facts.anchors),
            "num_accesses": int(len(facts.accesses)),
            "schema_version": str(facts.schema_version),
        }
        return facts

    def extract_constraints(self, desc: KernelDescriptor, facts: CudaFacts):
        c = extract_constraints(desc, facts)
        desc.frontend_constraints = asdict(c)
        return c

    def build_certificate(self, desc: KernelDescriptor, facts: CudaFacts, _constraints=None):
        return build_certificate_v2(facts, desc=desc)

    def evaluate_contract(self, _facts: Any, _constraints=None, _cert=None):
        # Contract V2 is evaluated by the pipeline (needs obligations + descriptor).
        return None


registry.register(CudaAdapter())

__all__ = ["CudaAdapter"]
