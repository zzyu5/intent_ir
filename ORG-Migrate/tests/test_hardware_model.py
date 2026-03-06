from __future__ import annotations

from org.mapping.hardware_model import build_hardware_model


def test_build_hardware_model_clusters_h100_like() -> None:
    model = build_hardware_model(target="cuda_h100", arch="sm90")
    assert model.arch_cluster == "cuda_tc_large_smem"
    assert model.memory_cluster == "large_smem"
    assert model.compute_cluster == "tensor_core"
    assert model.pipeline_cluster == "async_pipeline"


def test_build_hardware_model_clusters_5090d_like() -> None:
    model = build_hardware_model(target="cuda_5090d", arch="sm120")
    assert model.arch_cluster == "cuda_tc_mid_smem"
    assert model.memory_cluster == "mid_smem"
    assert model.compute_cluster == "tensor_core"
    assert model.pipeline_cluster == "async_pipeline"


def test_build_hardware_model_clusters_generic() -> None:
    model = build_hardware_model(target="cuda_unknown", arch="sm70_like")
    assert model.arch_cluster == "cuda_generic"
    assert model.memory_cluster == "generic_memory"
