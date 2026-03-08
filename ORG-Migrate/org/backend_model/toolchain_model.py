from __future__ import annotations

from dataclasses import dataclass, field
import subprocess
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


def _normalize_sm(raw: str) -> str:
    text = str(raw or "").strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    return f"sm_{digits}" if digits else ""


def _llc_supported_sms(llc_path: str) -> list[str]:
    path = str(llc_path or "").strip()
    if not path:
        return []
    try:
        proc = subprocess.run(
            [path, "-march=nvptx64", "-mcpu=help"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []
    text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    sms = sorted(
        {
            _normalize_sm(tok)
            for tok in __import__("re").findall(r"\bsm_[0-9]{2,3}\b", text)
            if _normalize_sm(tok)
        },
        key=lambda x: int("".join(ch for ch in x if ch.isdigit()) or "-1"),
    )
    return list(sms)


def _effective_sm(*, requested_sm: str, supported_sms: list[str]) -> str:
    req = _normalize_sm(requested_sm)
    if req and req in set(supported_sms):
        return req
    return str(supported_sms[-1]) if supported_sms else req


def build_toolchain_model(
    *,
    toolchain_report: Mapping[str, Any] | None,
    contract_exec_meta: Mapping[str, Any] | None,
    compiler_stack: str,
    requested_sm: str = "",
) -> ToolchainModel:
    toolchain = dict(toolchain_report or {})
    tools = dict(toolchain.get("tools") or {})
    llc = dict(tools.get("llc") or {})
    mlir_opt = dict(tools.get("mlir-opt") or {})
    llc_path = str(llc.get("path") or "")
    source = "repo_local" if "/artifacts/toolchains/" in llc_path else ("env_override" if llc_path else "unknown")
    exec_meta = dict(contract_exec_meta or {})
    supported_sms = [str(x) for x in list(exec_meta.get("cuda_supported_sms") or []) if str(x).strip()]
    if not supported_sms:
        supported_sms = _llc_supported_sms(llc_path)
    requested = str(exec_meta.get("cuda_requested_sm") or requested_sm or "")
    effective = str(exec_meta.get("cuda_effective_sm") or _effective_sm(requested_sm=requested, supported_sms=supported_sms) or "")
    requested_norm = _normalize_sm(requested)
    effective_norm = _normalize_sm(effective)
    return ToolchainModel(
        source=source,
        compiler_stack=str(compiler_stack or ""),
        requested_sm=requested_norm,
        effective_sm=effective_norm,
        downleveled=bool(exec_meta.get("cuda_target_downleveled")) or bool(requested_norm and effective_norm and requested_norm != effective_norm),
        supported_sms=[str(x) for x in list(supported_sms or []) if str(x).strip()],
        mlir_version=str(mlir_opt.get("version") or ""),
        llvm_version=str(llc.get("version") or ""),
    )


__all__ = ["ToolchainModel", "CompileCheck", "build_toolchain_model"]
