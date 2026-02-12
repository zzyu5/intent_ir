"""
Backend capability query helpers.

This module is the single place to answer:
- what ops a backend target claims to support
- whether a given IntentIR op set can run on that target
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from backends.cuda.opset import CUDA_SUPPORTED_OPS
from backends.spmd_rvv.opset import SPMD_RVV_SUPPORTED_OPS


BackendTarget = Literal["rvv", "cuda_h100", "cuda_5090d"]

_TARGET_OPSETS: dict[str, set[str]] = {
    "rvv": set(SPMD_RVV_SUPPORTED_OPS),
    "cuda_h100": set(CUDA_SUPPORTED_OPS),
    "cuda_5090d": set(CUDA_SUPPORTED_OPS),
}


@dataclass(frozen=True)
class CapabilityResult:
    target: str
    requested_ops: list[str]
    supported_ops: list[str]
    missing_ops: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_ops

    def to_json_dict(self) -> dict:
        return {
            "target": str(self.target),
            "ok": bool(self.ok),
            "requested_ops": list(self.requested_ops),
            "supported_ops": list(self.supported_ops),
            "missing_ops": list(self.missing_ops),
        }


def supported_ops_for_target(target: str) -> set[str]:
    key = str(target)
    if key not in _TARGET_OPSETS:
        raise ValueError(f"unsupported backend target: {target}")
    return set(_TARGET_OPSETS[key])


def check_target_support(target: str, intent_ops: Iterable[str]) -> CapabilityResult:
    requested = sorted({str(op) for op in intent_ops if isinstance(op, str) and op})
    supported = supported_ops_for_target(target)
    ok_ops = sorted(op for op in requested if op in supported)
    missing = sorted(op for op in requested if op not in supported)
    return CapabilityResult(
        target=str(target),
        requested_ops=requested,
        supported_ops=ok_ops,
        missing_ops=missing,
    )


def check_dual_backend_support(intent_ops: Iterable[str]) -> dict:
    rvv = check_target_support("rvv", intent_ops)
    h100 = check_target_support("cuda_h100", intent_ops)
    g5090 = check_target_support("cuda_5090d", intent_ops)
    return {
        "rvv": rvv.to_json_dict(),
        "cuda_h100": h100.to_json_dict(),
        "cuda_5090d": g5090.to_json_dict(),
    }


__all__ = [
    "BackendTarget",
    "CapabilityResult",
    "supported_ops_for_target",
    "check_target_support",
    "check_dual_backend_support",
]
