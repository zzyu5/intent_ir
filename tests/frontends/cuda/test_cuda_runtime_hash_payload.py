from __future__ import annotations

from frontends.cuda.runtime import _intentir_cuda_runner_hash_payload


def test_runner_hash_payload_is_stable() -> None:
    p0 = _intentir_cuda_runner_hash_payload()
    p1 = _intentir_cuda_runner_hash_payload()
    assert p0 == p1
    assert isinstance(p0, str)
    assert p0


def test_runner_hash_payload_contains_abi_tag() -> None:
    payload = _intentir_cuda_runner_hash_payload()
    assert "runner_abi=runner_abi_v2" in payload
