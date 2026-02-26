from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

from intent_ir.ir import IntentFunction
from intent_ir.mlir import detect_mlir_toolchain, run_pipeline, to_intent, to_mlir
from intent_ir.mlir.passes.attach_provider_meta import attach_provider_meta
from intent_ir.mlir.passes.backend_legalize import backend_legalize
from intent_ir.mlir.passes.emit_cuda_contract import build_cuda_contract
from intent_ir.mlir.passes.emit_rvv_contract import build_rvv_contract
from intent_ir.mlir.passes.ensure_llvm_ir_text import ensure_llvm_ir_text

_LOWER_LLVM_PASS = importlib.import_module("intent_ir.mlir.passes.lower_intent_to_llvm_dialect")

ROOT = Path(__file__).resolve().parents[2]


def _sample_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "add2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "z": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "add", "inputs": ["x", "y"], "output": "z", "attrs": {}},
            ],
            "outputs": ["z"],
        }
    )


def test_mlir_roundtrip_from_intent() -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    back = to_intent(module)
    assert back.name == intent.name
    assert back.outputs == intent.outputs
    assert [op.op for op in back.ops] == [op.op for op in intent.ops]


def test_mlir_pass_pipeline_runs(tmp_path: Path) -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    out_mod, trace = run_pipeline(module, "upstream", out_dir=tmp_path)
    assert isinstance(out_mod.module_text, str) and out_mod.module_text
    assert bool(trace.get("ok")) is True
    trace_path = tmp_path / "pass_trace.json"
    assert trace_path.is_file()


def test_backend_legalize_preserves_shape_bindings_meta() -> None:
    module = to_mlir(_sample_intent())
    module.meta = dict(module.meta or {})
    module.meta["shape_bindings"] = {"M": 4, "N": 64}

    out = backend_legalize(module, backend="cuda")
    assert dict(out.meta or {}).get("shape_bindings") == {"M": 4, "N": 64}
    assert str(dict(out.meta or {}).get("backend_target") or "") == "cuda"


def test_attach_provider_meta_preserves_shape_bindings_meta() -> None:
    module = to_mlir(_sample_intent())
    module.meta = dict(module.meta or {})
    module.meta["shape_bindings"] = {"M": 4, "N": 64}

    out = attach_provider_meta(module, provider="flaggems", backend="cuda")
    meta = dict(out.meta or {})
    assert meta.get("shape_bindings") == {"M": 4, "N": 64}
    assert str(meta.get("provider") or "") == "flaggems"
    assert str(meta.get("backend_target") or "") == "cuda"


def test_mlir_midend_pipeline_includes_stats_and_macro_expand(tmp_path: Path) -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    _, trace = run_pipeline(module, "midend", out_dir=tmp_path)
    assert bool(trace.get("ok")) is True
    names = [str(p.get("name") or "") for p in list(trace.get("passes") or [])]
    assert "python:expand_macros_intent" in names
    assert "python:canonicalize_intent" in names
    assert "python:cse_like" in names
    assert isinstance(trace.get("input_stats"), dict)
    assert isinstance(trace.get("output_stats"), dict)


def test_mlir_toolchain_probe_schema() -> None:
    payload = detect_mlir_toolchain()
    assert payload["schema_version"] == "intent_mlir_toolchain_probe_v1"
    assert "tools" in payload
    assert "mlir-opt" in payload["tools"]
    assert "llvm-as" in payload["tools"]
    assert "opt" in payload["tools"]


def test_mlir_required_external_passes_fail_fast_without_toolchain(tmp_path: Path) -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    _, trace = run_pipeline(module, "downstream_cuda_llvm", backend="cuda", out_dir=tmp_path)
    names = [str(p.get("name") or "") for p in list(trace.get("passes") or [])]
    assert "python:lower_intent_to_llvm_dialect" in names
    details = " ".join(str(p.get("detail") or "") for p in list(trace.get("passes") or []))
    if bool(trace.get("ok")):
        assert "python:ensure_llvm_ir_text" in names
        assert "llvm-as:" in names
        assert "opt:-O2" in names
    else:
        assert (
            "unavailable" in details
            or "lower_intent_to_llvm_dialect" in details
            or "ensure_llvm_ir_text" in details
            or "failed" in details
        )


def test_ensure_llvm_ir_text_fails_without_real_llvm_text() -> None:
    module = to_mlir(_sample_intent())
    try:
        _ = ensure_llvm_ir_text(module, backend="cuda")
    except RuntimeError as e:
        assert "stub synthesis disabled" in str(e)
    else:
        raise AssertionError("ensure_llvm_ir_text should fail for non-LLVM payloads")


def test_lower_intent_to_llvm_dialect_cuda_prefers_device_llvm_path(monkeypatch) -> None:
    intent = _sample_intent()
    module = to_mlir(intent)

    monkeypatch.setattr(
        _LOWER_LLVM_PASS,
        "lower_intent_to_cuda_kernel_cpp",
        lambda *_a, **_k: {"kernel_name": "add2d", "cuda_src": 'extern "C" __global__ void add2d(float* out) { out[0] = 1.0f; }'},
    )
    monkeypatch.setattr(
        _LOWER_LLVM_PASS,
        "_compile_cuda_src_to_device_llvm_ir",
        lambda _src, *, kernel_name: (
            '; ModuleID = "cuda"\ntarget triple = "nvptx64-nvidia-cuda"\ndefine void @add2d() { ret void }\n',
            "/fake/clang++",
        ),
    )

    out = _LOWER_LLVM_PASS.lower_intent_to_llvm_dialect(module, backend="cuda")
    assert str(out.meta.get("llvm_dialect_origin") or "") == "lowered_from_intent_cuda_codegen"
    assert str(out.meta.get("llvm_cuda_compiler") or "") == "/fake/clang++"
    assert str(out.meta.get("llvm_cuda_kernel_name") or "") == "add2d"
    assert str(out.meta.get("llvm_target_triple") or "") == "nvptx64-nvidia-cuda"
    assert "define void @add2d" in str(out.module_text)


def test_lower_intent_to_llvm_dialect_cuda_falls_back_to_c_codegen_on_cuda_path_error(monkeypatch) -> None:
    intent = _sample_intent()
    module = to_mlir(intent)

    monkeypatch.setattr(
        _LOWER_LLVM_PASS,
        "lower_intent_to_cuda_kernel_cpp",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("cuda lowering failed")),
    )
    monkeypatch.setattr(
        _LOWER_LLVM_PASS,
        "_compile_c_to_llvm_ir",
        lambda _src: (
            '; ModuleID = "host"\ntarget triple = "x86_64-pc-linux-gnu"\ndefine void @main() { ret void }\n',
            "/fake/clang",
        ),
    )

    out = _LOWER_LLVM_PASS.lower_intent_to_llvm_dialect(module, backend="cuda")
    assert str(out.meta.get("llvm_dialect_origin") or "") == "lowered_from_intent_c_codegen_fallback_for_cuda"
    assert "cuda lowering failed" in str(out.meta.get("llvm_cuda_backend_error") or "")
    assert str(out.meta.get("llvm_target_triple") or "") == "x86_64-pc-linux-gnu"


def test_lower_intent_to_llvm_dialect_rvv_retargets_host_llvm_to_riscv(monkeypatch) -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    host_llvm = (
        '; ModuleID = "host"\n'
        'target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"\n'
        'target triple = "x86_64-pc-linux-gnu"\n'
        "define void @main() #0 { ret void }\n"
        'attributes #0 = { "target-cpu"="x86-64" "target-features"="+cx8,+sse2" "tune-cpu"="generic" }\n'
    )
    monkeypatch.setattr(
        _LOWER_LLVM_PASS,
        "_compile_c_to_llvm_ir",
        lambda _src: (host_llvm, "/fake/clang"),
    )

    out = _LOWER_LLVM_PASS.lower_intent_to_llvm_dialect(module, backend="rvv")
    assert str(out.meta.get("llvm_dialect_origin") or "") == "lowered_from_intent_c_codegen_rvv"
    assert str(out.meta.get("llvm_target_triple") or "") == "riscv64-unknown-linux-gnu"
    assert str(out.meta.get("llvm_rvv_target_triple") or "") == "riscv64-unknown-linux-gnu"
    assert '"target-cpu"="x86-64"' not in str(out.module_text or "")
    assert '"target-features"="+cx8,+sse2"' not in str(out.module_text or "")
    assert 'target triple = "riscv64-unknown-linux-gnu"' in str(out.module_text or "")


def test_cuda_device_llvm_preamble_defines_infinity_and_nan() -> None:
    preamble = str(_LOWER_LLVM_PASS._cuda_device_llvm_preamble())
    assert "#define INFINITY (__builtin_inff())" in preamble
    assert '#define NAN (__builtin_nanf(""))' in preamble
    assert "typedef struct __align__(2) { unsigned short x; } __half;" in preamble
    assert "typedef struct __align__(2) { unsigned short x; } __nv_bfloat16;" in preamble
    assert "__shfl_down_sync(unsigned int mask, long long var" in preamble
    assert "__shfl_sync(unsigned int mask, long long var" in preamble
    assert 'asm volatile("atom.exch.b32 %0, [%1], %2;"' in preamble
    assert "atom.global.exch.b32" not in preamble


def test_cuda_codegen_requires_half_bfloat_stubs() -> None:
    assert _LOWER_LLVM_PASS._cuda_codegen_requires_half_bfloat_stubs('extern "C" __global__ void k() {}') is True
    assert _LOWER_LLVM_PASS._cuda_codegen_requires_half_bfloat_stubs('#include "kernels/reduce.cuh"\n') is False
    assert _LOWER_LLVM_PASS._cuda_codegen_requires_half_bfloat_stubs("#include <cuda_fp16.h>\n") is False


def test_mlir_cuda_contract_emitter_builds_contract() -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    contract = build_cuda_contract(module)
    payload = contract.to_json_dict()
    assert payload["backend"] == "cuda"
    assert payload["kernel_name"] == "add2d"
    assert payload["schema_version"] == "intent_mlir_backend_contract_v2"
    assert "intent_json" not in payload
    assert isinstance(payload.get("executable"), dict)


def test_mlir_rvv_contract_emitter_builds_contract() -> None:
    intent = _sample_intent()
    module = to_mlir(intent)
    contract = build_rvv_contract(module)
    payload = contract.to_json_dict()
    assert payload["backend"] == "rvv"
    assert payload["kernel_name"] == "add2d"
    assert payload["schema_version"] == "intent_mlir_backend_contract_v2"
    assert "intent_json" not in payload
    assert isinstance(payload.get("executable"), dict)


def test_intentir_cli_mlir_check(tmp_path: Path) -> None:
    intent = _sample_intent()
    intent_path = tmp_path / "intent.json"
    out = tmp_path / "check.json"
    intent_path.write_text(json.dumps(intent.to_json_dict(), indent=2), encoding="utf-8")
    p = subprocess.run(
        [
            sys.executable,
            "scripts/intentir.py",
            "mlir",
            "check",
            "--intent-json",
            str(intent_path),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr or p.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert bool(payload.get("roundtrip_ok")) is True
