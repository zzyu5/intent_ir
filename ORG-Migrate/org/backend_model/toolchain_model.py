from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ToolchainModel:
    source: str
    compiler_stack: str
    requested_sm: str = ""
    effective_sm: str = ""
    downleveled: bool = False
    supported_sms: list[str] = field(default_factory=list)
    mlir_version: str = ""
    llvm_version: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "compiler_stack": str(self.compiler_stack),
            "requested_sm": str(self.requested_sm),
            "effective_sm": str(self.effective_sm),
            "downleveled": bool(self.downleveled),
            "supported_sms": [str(x) for x in list(self.supported_sms or []) if str(x).strip()],
            "mlir_version": str(self.mlir_version),
            "llvm_version": str(self.llvm_version),
        }


@dataclass(frozen=True)
class CompileCheck:
    candidate: str
    kernel_kind: str
    bindings: dict[str, int] = field(default_factory=dict)
    report_path: str = ""
    contract_path: str = ""
    ptx_path: str = ""
    entry: str = ""
    requested_sm: str = ""
    effective_sm: str = ""
    downleveled: bool | None = None
    ok: bool = False
    error: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "candidate": str(self.candidate),
            "kernel_kind": str(self.kernel_kind),
            "bindings": {str(k): int(v) for k, v in dict(self.bindings or {}).items() if str(k).strip()},
            "report_path": str(self.report_path),
            "contract_path": str(self.contract_path),
            "ptx_path": str(self.ptx_path),
            "entry": str(self.entry),
            "requested_sm": str(self.requested_sm),
            "effective_sm": str(self.effective_sm),
            "downleveled": self.downleveled,
            "ok": bool(self.ok),
            "error": str(self.error),
        }


def build_toolchain_model(
    *,
    toolchain_report: Mapping[str, Any] | None,
    contract_exec_meta: Mapping[str, Any] | None,
    compiler_stack: str,
) -> ToolchainModel:
    toolchain = dict(toolchain_report or {})
    tools = dict(toolchain.get("tools") or {})
    llc = dict(tools.get("llc") or {})
    mlir_opt = dict(tools.get("mlir-opt") or {})
    llc_path = str(llc.get("path") or "")
    source = "repo_local" if "/artifacts/toolchains/" in llc_path else ("env_override" if llc_path else "unknown")
    exec_meta = dict(contract_exec_meta or {})
    return ToolchainModel(
        source=source,
        compiler_stack=str(compiler_stack or ""),
        requested_sm=str(exec_meta.get("cuda_requested_sm") or ""),
        effective_sm=str(exec_meta.get("cuda_effective_sm") or ""),
        downleveled=bool(exec_meta.get("cuda_target_downleveled")),
        supported_sms=[str(x) for x in list(exec_meta.get("cuda_supported_sms") or []) if str(x).strip()],
        mlir_version=str(mlir_opt.get("version") or ""),
        llvm_version=str(llc.get("version") or ""),
    )


__all__ = ["ToolchainModel", "CompileCheck", "build_toolchain_model"]
