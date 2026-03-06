from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareModel:
    arch: str
    shared_mem_kb: int
    register_budget: int
    warp_size: int
    supports_async_copy: bool
    supports_mma: bool
    supports_ldmatrix: bool
    supports_shuffle: bool

    def to_json_dict(self) -> dict[str, int | bool | str]:
        return {
            "arch": str(self.arch),
            "shared_mem_kb": int(self.shared_mem_kb),
            "register_budget": int(self.register_budget),
            "warp_size": int(self.warp_size),
            "supports_async_copy": bool(self.supports_async_copy),
            "supports_mma": bool(self.supports_mma),
            "supports_ldmatrix": bool(self.supports_ldmatrix),
            "supports_shuffle": bool(self.supports_shuffle),
        }


def build_hardware_model(*, target: str, arch: str) -> HardwareModel:
    arch_norm = str(arch or "").strip().lower()
    target_norm = str(target or "").strip().lower()
    if not arch_norm:
        if "5090" in target_norm:
            arch_norm = "sm120"
        elif "h100" in target_norm:
            arch_norm = "sm90"
        else:
            arch_norm = "sm120"

    if arch_norm == "sm90":
        return HardwareModel(
            arch="sm90",
            shared_mem_kb=228,
            register_budget=65536,
            warp_size=32,
            supports_async_copy=True,
            supports_mma=True,
            supports_ldmatrix=True,
            supports_shuffle=True,
        )
    if arch_norm == "sm120":
        return HardwareModel(
            arch="sm120",
            shared_mem_kb=128,
            register_budget=65536,
            warp_size=32,
            supports_async_copy=True,
            supports_mma=True,
            supports_ldmatrix=True,
            supports_shuffle=True,
        )
    return HardwareModel(
        arch=str(arch_norm),
        shared_mem_kb=96,
        register_budget=65536,
        warp_size=32,
        supports_async_copy=False,
        supports_mma=False,
        supports_ldmatrix=False,
        supports_shuffle=True,
    )


__all__ = ["HardwareModel", "build_hardware_model"]
