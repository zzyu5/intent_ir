from __future__ import annotations

from backends.capability import check_dual_backend_support, check_target_support, supported_ops_for_target


def test_supported_ops_for_target_known_targets() -> None:
    assert "add" in supported_ops_for_target("rvv")
    assert "add" in supported_ops_for_target("cuda_h100")
    assert "add" in supported_ops_for_target("cuda_5090d")


def test_check_target_support_reports_missing() -> None:
    res = check_target_support("rvv", ["add", "nonexistent_op"])
    assert res.ok is False
    assert "add" in res.supported_ops
    assert "nonexistent_op" in res.missing_ops


def test_check_dual_backend_support_shape() -> None:
    out = check_dual_backend_support(["add", "softmax"])
    assert set(out.keys()) == {"rvv", "cuda_h100", "cuda_5090d"}
    assert isinstance(out["rvv"]["ok"], bool)
    assert isinstance(out["cuda_h100"]["ok"], bool)
