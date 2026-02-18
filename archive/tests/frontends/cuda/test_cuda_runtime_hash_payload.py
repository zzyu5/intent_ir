from __future__ import annotations

from frontends.cuda.runtime import _intentir_cuda_runner_hash_payload, cuda_extension_cache_info


def test_runner_hash_payload_is_stable() -> None:
    p0 = _intentir_cuda_runner_hash_payload()
    p1 = _intentir_cuda_runner_hash_payload()
    assert p0 == p1
    assert isinstance(p0, str)
    assert p0


def test_runner_hash_payload_contains_abi_tag() -> None:
    payload = _intentir_cuda_runner_hash_payload()
    assert "runner_abi=runner_abi_v2" in payload


def test_cuda_extension_cache_info_is_deterministic() -> None:
    info0 = cuda_extension_cache_info(kernel_name="k", cuda_src="__global__ void k(){}")
    info1 = cuda_extension_cache_info(kernel_name="k", cuda_src="__global__ void k(){}")
    assert info0["module_name"] == info1["module_name"]
    assert info0["build_dir"] == info1["build_dir"]
    assert isinstance(info0["artifact_exists"], bool)
