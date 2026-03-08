from __future__ import annotations

from org.backend_model import build_toolchain_model


def test_build_toolchain_model_from_contract_exec_meta() -> None:
    model = build_toolchain_model(
        toolchain_report={
            "tools": {
                "llc": {"path": "/tmp/LLVM-20/bin/llc", "version": "LLVM 20.1.8"},
                "mlir-opt": {"path": "/tmp/mlir-opt", "version": "MLIR 14.0.0"},
            }
        },
        contract_exec_meta={
            "cuda_requested_sm": "sm_120",
            "cuda_effective_sm": "sm_86",
            "cuda_target_downleveled": True,
            "cuda_supported_sms": ["sm_80", "sm_86"],
        },
        compiler_stack="python",
    )
    payload = model.to_json_dict()
    assert payload["source"] == "env_override"
    assert payload["requested_sm"] == "sm_120"
    assert payload["effective_sm"] == "sm_86"
    assert payload["downleveled"] is True


def test_build_toolchain_model_infers_sm_from_llc(monkeypatch) -> None:
    import org.backend_model.toolchain_model as tm

    monkeypatch.setattr(tm, "_llc_supported_sms", lambda _path: ["sm_86", "sm_120"])
    model = build_toolchain_model(
        toolchain_report={
            "tools": {
                "llc": {"path": "/tmp/LLVM-20/bin/llc", "version": "LLVM 20.1.8"},
                "mlir-opt": {"path": "/tmp/mlir-opt", "version": "MLIR 14.0.0"},
            }
        },
        contract_exec_meta={},
        compiler_stack="python",
        requested_sm="sm120",
    )
    payload = model.to_json_dict()
    assert payload["requested_sm"] == "sm_120"
    assert payload["effective_sm"] == "sm_120"
    assert payload["downleveled"] is False
    assert payload["supported_sms"] == ["sm_86", "sm_120"]
