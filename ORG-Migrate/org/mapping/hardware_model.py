from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareModel:
    arch: str
    arch_cluster: str
    memory_cluster: str
    compute_cluster: str
    pipeline_cluster: str
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
            "arch_cluster": str(self.arch_cluster),
            "memory_cluster": str(self.memory_cluster),
            "compute_cluster": str(self.compute_cluster),
            "pipeline_cluster": str(self.pipeline_cluster),
            "shared_mem_kb": int(self.shared_mem_kb),
            "register_budget": int(self.register_budget),
            "warp_size": int(self.warp_size),
            "supports_async_copy": bool(self.supports_async_copy),
            "supports_mma": bool(self.supports_mma),
            "supports_ldmatrix": bool(self.supports_ldmatrix),
            "supports_shuffle": bool(self.supports_shuffle),
        }


def _cluster_fields(*, shared_mem_kb: int, supports_async_copy: bool, supports_mma: bool) -> tuple[str, str, str, str]:
    if bool(supports_mma) and bool(supports_async_copy) and int(shared_mem_kb) >= 192:
        return ("cuda_tc_large_smem", "large_smem", "tensor_core", "async_pipeline")
    if bool(supports_mma) and bool(supports_async_copy) and int(shared_mem_kb) >= 96:
        return ("cuda_tc_mid_smem", "mid_smem", "tensor_core", "async_pipeline")
    return ("cuda_generic", "generic_memory", ("tensor_core" if supports_mma else "generic_compute"), ("async_pipeline" if supports_async_copy else "sync_pipeline"))


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
        arch_cluster, memory_cluster, compute_cluster, pipeline_cluster = _cluster_fields(
            shared_mem_kb=228,
            supports_async_copy=True,
            supports_mma=True,
        )
        return HardwareModel(
            arch="sm90",
            arch_cluster=arch_cluster,
            memory_cluster=memory_cluster,
            compute_cluster=compute_cluster,
            pipeline_cluster=pipeline_cluster,
            shared_mem_kb=228,
            register_budget=65536,
            warp_size=32,
            supports_async_copy=True,
            supports_mma=True,
            supports_ldmatrix=True,
            supports_shuffle=True,
        )
    if arch_norm == "sm120":
        arch_cluster, memory_cluster, compute_cluster, pipeline_cluster = _cluster_fields(
            shared_mem_kb=128,
            supports_async_copy=True,
            supports_mma=True,
        )
        return HardwareModel(
            arch="sm120",
            arch_cluster=arch_cluster,
            memory_cluster=memory_cluster,
            compute_cluster=compute_cluster,
            pipeline_cluster=pipeline_cluster,
            shared_mem_kb=128,
            register_budget=65536,
            warp_size=32,
            supports_async_copy=True,
            supports_mma=True,
            supports_ldmatrix=True,
            supports_shuffle=True,
        )
    arch_cluster, memory_cluster, compute_cluster, pipeline_cluster = _cluster_fields(
        shared_mem_kb=96,
        supports_async_copy=False,
        supports_mma=False,
    )
    return HardwareModel(
        arch=str(arch_norm),
        arch_cluster=arch_cluster,
        memory_cluster=memory_cluster,
        compute_cluster=compute_cluster,
        pipeline_cluster=pipeline_cluster,
        shared_mem_kb=96,
        register_budget=65536,
        warp_size=32,
        supports_async_copy=False,
        supports_mma=False,
        supports_ldmatrix=False,
        supports_shuffle=True,
    )


__all__ = ["HardwareModel", "build_hardware_model"]
