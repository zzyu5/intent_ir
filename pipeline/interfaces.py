"""
Cross-module interfaces shared by pipeline/frontends/verify/backends.

Keep this module dependency-light (no torch/triton) so it can be imported from
core logic without pulling heavy runtime requirements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol


FrontendName = Literal["triton", "tilelang", "cuda"]


@dataclass
class KernelArtifactBundle:
    """
    Frontend-produced compilation artifacts.

    Text vs path is intentionally flexible: some frontends can emit text directly
    (e.g., TTIR string), others can only point to a dump path.
    """

    ttir_text: Optional[str] = None
    ttir_path: Optional[str] = None
    llvm_ir_text: Optional[str] = None
    ptx_text: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelDescriptor:
    """
    A frontend-agnostic description of a kernel + its extracted evidence.
    """

    schema_version: str
    name: str
    frontend: FrontendName

    # Raw input (source/DSL/IR)
    source_kind: Literal["source", "dsl", "ir"] = "source"
    source_text: str = ""

    # Compile/launch metadata (frontend may fill partially).
    launch: Dict[str, Any] = field(default_factory=dict)

    # I/O specification (symbolic shapes, dtypes, roles).
    io_spec: Dict[str, Any] = field(default_factory=dict)

    # Frontend artifacts (TTIR/PTX/LLVM IR, etc).
    artifacts: KernelArtifactBundle = field(default_factory=KernelArtifactBundle)

    # Structured summaries used by LLM + verify.
    frontend_facts: Dict[str, Any] = field(default_factory=dict)
    frontend_constraints: Dict[str, Any] = field(default_factory=dict)

    # Traceability: versions, git commit, compiler versions, etc.
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FrontendAdapter(Protocol):
    """
    Frontend adapter interface used by generic pipeline orchestration.

    This protocol is intentionally permissive about intermediate types (`facts`,
    `constraints`, `cert`) so each frontend can reuse its existing structures
    while still writing stable summaries into `KernelDescriptor`.
    """

    name: FrontendName

    def build_descriptor(self, kernel: Any) -> KernelDescriptor: ...

    def ensure_artifacts(self, desc: KernelDescriptor, kernel: Any) -> KernelDescriptor: ...

    def extract_facts(self, desc: KernelDescriptor) -> Any: ...

    def extract_constraints(self, desc: KernelDescriptor, facts: Any) -> Any: ...

    def build_certificate(self, desc: KernelDescriptor, facts: Any, constraints: Any | None = None) -> Any: ...

    def evaluate_contract(self, facts: Any, constraints: Any | None = None, cert: Any | None = None) -> Any: ...


@dataclass
class FrontendConstraints:
    """
    Minimal frontend-derived constraints used by Task5 case generation.

    The Triton frontend currently extracts these from TTIR, but other frontends
    (CUDA C / TileLang) can populate the same structure from their own IR dumps.
    """

    needs_mask: bool = False
    suggested_edge_cases: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "FrontendName",
    "KernelArtifactBundle",
    "KernelDescriptor",
    "FrontendAdapter",
    "FrontendConstraints",
]
