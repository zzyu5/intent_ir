from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    script_path = ROOT / "scripts" / "rvv_remote_run.py"
    spec = importlib.util.spec_from_file_location("rvv_remote_run", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_to_raw_bytes_respects_declared_dtype_over_bool_array() -> None:
    mod = _load_module()
    arr_bool = np.array([[True, False], [False, True]], dtype=np.bool_)

    raw_f32 = mod._to_raw_bytes(arr_bool, "f32")
    raw_i32 = mod._to_raw_bytes(arr_bool, "i32")
    raw_bool = mod._to_raw_bytes(arr_bool, "bool")

    # 4 elements -> 16 bytes for f32/i32, 4 bytes for bool(u8) payload.
    assert len(raw_f32) == 16
    assert len(raw_i32) == 16
    assert len(raw_bool) == 4


def test_rvv_remote_run_avoids_direct_intentfunction_json_rehydration() -> None:
    src = (ROOT / "scripts" / "rvv_remote_run.py").read_text(encoding="utf-8")
    assert "IntentFunction.from_json_dict" not in src


def test_lower_intent_to_c_with_files_requires_explicit_compat_for_non_contract(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.delenv("INTENTIR_RVV_REMOTE_ALLOW_COMPAT_C", raising=False)
    try:
        _ = mod.lower_intent_to_c_with_files(
            {"name": "k", "tensors": {}, "ops": [], "outputs": []},
            shape_bindings={},
        )
    except RuntimeError as e:
        msg = str(e)
        assert "strict hard-cut" in msg
        assert "INTENTIR_RVV_REMOTE_ALLOW_COMPAT_C=1" in msg
    else:
        raise AssertionError("expected explicit compat requirement")


def test_lower_intent_to_c_with_files_accepts_contract_without_compat(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.delenv("INTENTIR_RVV_REMOTE_ALLOW_COMPAT_C", raising=False)
    monkeypatch.setattr(mod, "lower_rvv_contract_to_c_src", lambda *_a, **_k: "ok")
    out = mod.lower_intent_to_c_with_files(
        {
            "schema_version": "intent_mlir_backend_contract_v2",
            "backend": "rvv",
            "kernel_name": "k",
            "executable": {"format": "rvv_elf", "path": "x", "entry": "k", "target": "rvv"},
            "artifacts": {},
        },
        shape_bindings={},
    )
    assert out == "ok"


def test_rvv_remote_llvm_compile_uses_two_step_clang_ir_then_link() -> None:
    src = (ROOT / "scripts" / "rvv_remote_run.py").read_text(encoding="utf-8")
    assert "clang -O3 -x ir {q_remote_target} -c -o {q_remote_obj} {q_remote_ll} && " in src
    assert (
        "clang -O3 {q_remote_target} -fopenmp -std=c11 -D_POSIX_C_SOURCE=200809L -I{q_remote_dir} "
        "-o {q_remote_bin} {q_remote_obj} "
    ) in src


def test_rvv_staging_dtype_promotes_half_for_strict_modes() -> None:
    mod = _load_module()
    assert mod._rvv_staging_dtype("f16", execution_mode="remote_llvm") == "f32"
    assert mod._rvv_staging_dtype("bf16", execution_mode="prebuilt_elf") == "f32"
    assert mod._rvv_staging_dtype("f16", execution_mode="compat_c_src") == "f16"
    assert mod._rvv_staging_dtype("f32", execution_mode="remote_llvm") == "f32"


def test_lower_intent_to_c_with_files_requires_explicit_compat_for_non_contract(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.delenv("INTENTIR_RVV_REMOTE_ALLOW_COMPAT_C", raising=False)
    try:
        _ = mod.lower_intent_to_c_with_files(
            {"name": "k", "tensors": {}, "ops": [], "outputs": []},
            shape_bindings={"M": 1},
        )
    except RuntimeError as e:
        assert "explicit compat mode" in str(e)
    else:
        raise AssertionError("expected strict-mode rejection for non-contract payload")


def test_lower_intent_to_c_with_files_accepts_contract_without_compat(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.delenv("INTENTIR_RVV_REMOTE_ALLOW_COMPAT_C", raising=False)
    monkeypatch.setattr(mod, "lower_rvv_contract_to_c_src", lambda *_a, **_k: "ok")
    out = mod.lower_intent_to_c_with_files(
        {
            "schema_version": "intent_mlir_backend_contract_v2",
            "backend": "rvv",
            "kernel_name": "k",
            "executable": {"format": "rvv_elf", "path": "x", "entry": "k", "target": "rvv"},
            "artifacts": {},
        },
        shape_bindings={},
    )
    assert out == "ok"


def _fake_elf(path: Path, *, e_machine: int) -> None:
    data = bytearray(64)
    data[0:4] = b"\x7fELF"
    data[4] = 2  # 64-bit
    data[5] = 1  # little-endian
    struct.pack_into("<H", data, 18, int(e_machine))
    path.write_bytes(bytes(data))


def _write_contract(
    tmp_path: Path,
    *,
    llvm_triple: str,
    elf_machine: int,
) -> tuple[Path, dict]:
    elf_path = tmp_path / "k.elf"
    _fake_elf(elf_path, e_machine=int(elf_machine))
    ll_path = tmp_path / "k.ll"
    ll_path.write_text(
        '; ModuleID = "k"\n'
        f'target triple = "{llvm_triple}"\n'
        "define void @k() { ret void }\n",
        encoding="utf-8",
    )
    contract_path = tmp_path / "k.contract.json"
    payload = {
        "schema_version": "intent_mlir_backend_contract_v2",
        "backend": "rvv",
        "kernel_name": "k",
        "executable": {
            "format": "rvv_elf",
            "path": str(elf_path),
            "entry": "k",
            "target": "rvv",
            "invocation": {},
        },
        "artifacts": {"mlir_module_path": str(ll_path)},
    }
    contract_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return contract_path, payload


def test_rvv_execution_plan_prefers_remote_llvm_when_elf_not_riscv(tmp_path: Path) -> None:
    mod = _load_module()
    contract_path, payload = _write_contract(
        tmp_path,
        llvm_triple="riscv64-unknown-linux-gnu",
        elf_machine=62,  # x86_64
    )
    plan = mod._resolve_rvv_execution_plan(
        contract_payload_json=payload,
        contract_artifact_path=str(contract_path),
        compat_c_allowed=False,
    )
    assert str(plan.get("mode") or "") == "remote_llvm"
    assert "riscv64" in str(plan.get("llvm_triple") or "")


def test_rvv_execution_plan_requires_hardcut_evidence_when_compat_disabled(tmp_path: Path) -> None:
    mod = _load_module()
    contract_path, payload = _write_contract(
        tmp_path,
        llvm_triple="x86_64-pc-linux-gnu",
        elf_machine=62,  # x86_64
    )
    try:
        _ = mod._resolve_rvv_execution_plan(
            contract_payload_json=payload,
            contract_artifact_path=str(contract_path),
            compat_c_allowed=False,
        )
    except RuntimeError as e:
        assert "requires either riscv prebuilt ELF or RVV-target LLVM IR" in str(e)
    else:
        raise AssertionError("expected hard-cut resolution failure")


def test_execution_mode_from_plan_requires_explicit_mode() -> None:
    mod = _load_module()
    try:
        _ = mod._execution_mode_from_plan({})
    except RuntimeError as e:
        assert "missing mode" in str(e)
    else:
        raise AssertionError("expected missing mode failure")


def test_execution_mode_from_plan_accepts_strict_modes() -> None:
    mod = _load_module()
    assert mod._execution_mode_from_plan({"mode": "prebuilt_elf"}) == "prebuilt_elf"
    assert mod._execution_mode_from_plan({"mode": "remote_llvm"}) == "remote_llvm"


def test_rvv_execution_plan_allows_explicit_compat_fallback(tmp_path: Path) -> None:
    mod = _load_module()
    contract_path, payload = _write_contract(
        tmp_path,
        llvm_triple="x86_64-pc-linux-gnu",
        elf_machine=62,  # x86_64
    )
    plan = mod._resolve_rvv_execution_plan(
        contract_payload_json=payload,
        contract_artifact_path=str(contract_path),
        compat_c_allowed=True,
    )
    assert str(plan.get("mode") or "") == "compat_c_src"
