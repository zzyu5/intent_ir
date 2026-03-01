from __future__ import annotations

import json
from pathlib import Path

from intent_ir.ir import IntentFunction
from intent_ir.mlir import to_mlir
from intent_ir.mlir.module import IntentMLIRModule
import pipeline.mlir_contract_artifacts as mca
from pipeline.mlir_contract_artifacts import emit_backend_contract_artifacts


def _add_intent(name: str = "add2d") -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": name,
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "add", "inputs": ["A", "B"], "output": "C", "attrs": {}}],
            "outputs": ["C"],
        }
    )


def _llvm_module(base_mod: IntentMLIRModule, *, kernel_name: str = "k") -> IntentMLIRModule:
    return IntentMLIRModule(
        module_text='; ModuleID = "intentir"\ndefine void @k() { ret void }\n',
        dialect_version=str(base_mod.dialect_version),
        provenance=dict(base_mod.provenance or {}),
        symbols=[kernel_name],
        meta={"kernel_name": kernel_name},
        intent_json=None,
    )


def test_emit_backend_contract_artifacts_emits_nonmaterialized_midend_and_semantic_downstream(tmp_path: Path) -> None:
    mod = to_mlir(_add_intent("contract_semantic_only"))
    mlir_report: dict[str, object] = {}

    _ = emit_backend_contract_artifacts(
        spec_name="contract_semantic_only",
        out_dir=tmp_path,
        midend_module=mod,
        mlir_report=mlir_report,
        downstream_name="downstream_cuda",
        downstream_module=mod,
        shape_bindings={"M": 4, "N": 4},
    )

    mid_cuda = Path(str(mlir_report["midend_cuda_contract_path"]))
    mid_rvv = Path(str(mlir_report["midend_rvv_contract_path"]))
    down_cuda = Path(str(mlir_report["downstream_cuda_contract_path"]))
    assert mid_cuda.is_file()
    assert mid_rvv.is_file()
    assert down_cuda.is_file()
    assert "downstream_contract_path" not in mlir_report

    mid_cuda_payload = json.loads(mid_cuda.read_text(encoding="utf-8"))
    mid_rvv_payload = json.loads(mid_rvv.read_text(encoding="utf-8"))
    down_cuda_payload = json.loads(down_cuda.read_text(encoding="utf-8"))
    assert str((mid_cuda_payload.get("executable") or {}).get("format") or "") == "cuda_mlir_module"
    assert str((mid_rvv_payload.get("executable") or {}).get("format") or "") == "rvv_mlir_module"
    assert str((down_cuda_payload.get("executable") or {}).get("format") or "") == "cuda_mlir_module"


def test_emit_backend_contract_artifacts_materializes_cuda_executable_from_downstream_llvm(
    monkeypatch, tmp_path: Path
) -> None:
    mid_mod = to_mlir(_add_intent("cuda_contract_exec_llc"))
    llvm_mod = _llvm_module(mid_mod, kernel_name="k")
    mlir_report: dict[str, object] = {}

    monkeypatch.setattr(
        "pipeline.mlir_contract_artifacts.detect_mlir_toolchain",
        lambda: {
            "tools": {
                "llc": {
                    "available": True,
                    "path": "/fake/llc",
                    "version": "llc 19.0.0",
                }
            }
        },
    )

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, capture_output, text):
        _ = capture_output, text
        out_idx = cmd.index("-o") + 1
        Path(str(cmd[out_idx])).write_text("// fake llc ptx\n", encoding="utf-8")
        return _Proc()

    monkeypatch.setattr("pipeline.mlir_contract_artifacts.subprocess.run", _fake_run)

    _ = emit_backend_contract_artifacts(
        spec_name="cuda_contract_exec_llc",
        out_dir=tmp_path,
        midend_module=mid_mod,
        fallback_intent_module=mid_mod,
        mlir_report=mlir_report,
        downstream_name="downstream_cuda",
        downstream_module=mid_mod,
        downstream_llvm_name="downstream_cuda_llvm",
        downstream_llvm_module=llvm_mod,
        shape_bindings={"M": 4, "N": 4},
    )

    llvm_contract_path = Path(str(mlir_report["downstream_cuda_llvm_contract_path"]))
    assert llvm_contract_path.is_file()
    assert str(mlir_report["downstream_contract_path"]) == str(llvm_contract_path)
    assert str(mlir_report["downstream_contract_backend"]) == "cuda"
    payload = json.loads(llvm_contract_path.read_text(encoding="utf-8"))
    exe = dict(payload.get("executable") or {})
    assert str(exe.get("format") or "") == "cuda_ptx"
    inv = dict(exe.get("invocation") or {})
    io_spec = dict(inv.get("io_spec") or {})
    assert [str(x) for x in list(io_spec.get("arg_names") or [])] == ["A", "B", "C"]
    assert [str(x) for x in list(inv.get("output_names") or [])] == ["C"]
    ptx_path = Path(str(exe.get("path") or ""))
    assert ptx_path.is_file()
    assert "fake llc ptx" in ptx_path.read_text(encoding="utf-8")


def test_emit_backend_contract_artifacts_reports_cuda_llvm_error_when_materialization_fails(
    monkeypatch, tmp_path: Path
) -> None:
    mid_mod = to_mlir(_add_intent("cuda_contract_exec_llc_fail"))
    llvm_mod = _llvm_module(mid_mod, kernel_name="k")
    mlir_report: dict[str, object] = {}

    monkeypatch.setattr(
        "pipeline.mlir_contract_artifacts.detect_mlir_toolchain",
        lambda: {
            "tools": {
                "llc": {
                    "available": True,
                    "path": "/fake/llc",
                    "version": "llc 19.0.0",
                }
            }
        },
    )

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 1
            self.stdout = ""
            self.stderr = "llc failed"

    monkeypatch.setattr("pipeline.mlir_contract_artifacts.subprocess.run", lambda *_args, **_kwargs: _Proc())

    _ = emit_backend_contract_artifacts(
        spec_name="cuda_contract_exec_llc_fail",
        out_dir=tmp_path,
        midend_module=mid_mod,
        mlir_report=mlir_report,
        downstream_llvm_name="downstream_cuda_llvm",
        downstream_llvm_module=llvm_mod,
        shape_bindings={"M": 4, "N": 4},
    )

    assert "downstream_cuda_llvm_contract_path" not in mlir_report
    assert "downstream_contract_path" not in mlir_report
    err = str(mlir_report.get("downstream_cuda_llvm_contract_error") or "")
    assert "cuda llvm->ptx materialization failed" in err


def test_emit_backend_contract_artifacts_emits_rvv_downstream_llvm_contract_without_materializing_executable(
    monkeypatch, tmp_path: Path
) -> None:
    mid_mod = to_mlir(_add_intent("rvv_contract_exec_llvm"))
    llvm_mod = _llvm_module(mid_mod, kernel_name="k")
    mlir_report: dict[str, object] = {}

    monkeypatch.setattr(
        "pipeline.mlir_contract_artifacts.detect_mlir_toolchain",
        lambda: {
            "tools": {
                "llc": {"available": True, "path": "/fake/llc", "version": "llc 19.0.0"},
                "clang": {"available": True, "path": "/fake/clang", "version": "clang 19.0.0"},
            }
        },
    )

    _ = emit_backend_contract_artifacts(
        spec_name="rvv_contract_exec_llvm",
        out_dir=tmp_path,
        midend_module=mid_mod,
        mlir_report=mlir_report,
        downstream_llvm_name="downstream_rvv_llvm",
        downstream_llvm_module=llvm_mod,
        shape_bindings={"M": 4, "N": 4},
    )

    llvm_contract_path = Path(str(mlir_report["downstream_rvv_llvm_contract_path"]))
    assert llvm_contract_path.is_file()
    assert str(mlir_report["downstream_contract_path"]) == str(llvm_contract_path)
    assert str(mlir_report["downstream_contract_backend"]) == "rvv"
    payload = json.loads(llvm_contract_path.read_text(encoding="utf-8"))
    exe = dict(payload.get("executable") or {})
    assert str(exe.get("format") or "") == "rvv_mlir_module"
    inv = dict(exe.get("invocation") or {})
    io_spec = dict(inv.get("io_spec") or {})
    assert [str(x) for x in list(io_spec.get("arg_names") or [])] == ["A", "B", "C"]
    assert [str(x) for x in list(inv.get("output_names") or [])] == ["C"]
    module_path = Path(str(exe.get("path") or ""))
    assert module_path.is_file()
    assert "; ModuleID" in module_path.read_text(encoding="utf-8")


def test_compile_llvm_ir_to_elf_uses_host_retarget_fallback_for_rvv_triple(
    monkeypatch, tmp_path: Path
) -> None:
    llvm_ir = (
        '; ModuleID = "k"\n'
        'target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"\n'
        'target triple = "riscv64-unknown-linux-gnu"\n'
        "define void @k() #0 { ret void }\n"
        'attributes #0 = { "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }\n'
    )
    out_path = tmp_path / "k.elf"
    calls: list[list[str]] = []
    clang_calls = 0

    monkeypatch.setattr(
        "pipeline.mlir_contract_artifacts.detect_mlir_toolchain",
        lambda: {
            "tools": {
                "llc": {"available": True, "path": "/fake/llc", "version": "llc 19.0.0"},
                "clang": {"available": True, "path": "/fake/clang", "version": "clang 19.0.0"},
            }
        },
    )

    class _Proc:
        def __init__(self, *, rc: int = 0, out: str = "", err: str = "") -> None:
            self.returncode = int(rc)
            self.stdout = str(out)
            self.stderr = str(err)

    def _fake_run(cmd, capture_output, text):
        nonlocal clang_calls
        _ = capture_output, text
        calls.append([str(x) for x in cmd])
        tool = Path(str(cmd[0])).name
        out_idx = cmd.index("-o") + 1
        out_file = Path(str(cmd[out_idx]))
        if tool == "llc":
            out_file.write_bytes(b"\x7fOBJfake")
            return _Proc()
        if tool == "clang":
            clang_calls += 1
            if clang_calls == 1:
                return _Proc(rc=1, err="link failed")
            out_file.write_bytes(b"\x7fELFfake")
            return _Proc()
        raise AssertionError(f"unexpected tool: {tool}")

    monkeypatch.setattr("pipeline.mlir_contract_artifacts.subprocess.run", _fake_run)

    fp = mca._compile_llvm_ir_to_elf(llvm_ir_text=llvm_ir, out_path=out_path)
    assert out_path.is_file()
    assert out_path.read_bytes().startswith(b"\x7fELF")
    assert "host_retarget_fallback:" in str(fp)
    assert clang_calls == 2
    llc_invocations = [c for c in calls if Path(str(c[0])).name == "llc"]
    assert len(llc_invocations) == 2


def test_emit_backend_contract_artifacts_reports_rvv_llvm_error_without_primary_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    mid_mod = to_mlir(_add_intent("rvv_contract_exec_llvm_fail"))
    llvm_mod = _llvm_module(mid_mod, kernel_name="k")
    mlir_report: dict[str, object] = {}

    monkeypatch.setattr(
        "pipeline.mlir_contract_artifacts.detect_mlir_toolchain",
        lambda: {
            "tools": {
                "llc": {"available": True, "path": "/fake/llc", "version": "llc 19.0.0"},
                "clang": {"available": False, "path": "", "version": ""},
            }
        },
    )

    _ = emit_backend_contract_artifacts(
        spec_name="rvv_contract_exec_llvm_fail",
        out_dir=tmp_path,
        midend_module=mid_mod,
        mlir_report=mlir_report,
        downstream_name="downstream_rvv",
        downstream_module=mid_mod,
        downstream_llvm_name="downstream_rvv_llvm",
        downstream_llvm_module=llvm_mod,
        shape_bindings={"M": 4, "N": 4},
    )

    assert "downstream_rvv_contract_path" in mlir_report
    assert "downstream_rvv_llvm_contract_error" not in mlir_report
    assert "downstream_contract_path" in mlir_report
    payload = json.loads(Path(str(mlir_report["downstream_rvv_contract_path"])).read_text(encoding="utf-8"))
    assert str((payload.get("executable") or {}).get("format") or "") == "rvv_mlir_module"


def test_emit_backend_contract_artifacts_records_cuda_llvm_triple_and_origin(monkeypatch, tmp_path: Path) -> None:
    mid_mod = to_mlir(_add_intent("cuda_contract_exec_meta"))
    llvm_mod = _llvm_module(mid_mod, kernel_name="k")
    llvm_mod.meta["llvm_dialect_origin"] = "lowered_from_intent_cuda_codegen"
    llvm_mod.module_text = (
        '; ModuleID = "intentir"\n'
        'target triple = "nvptx64-nvidia-cuda"\n'
        "define void @k() { ret void }\n"
    )
    mlir_report: dict[str, object] = {}

    monkeypatch.setattr(
        "pipeline.mlir_contract_artifacts.detect_mlir_toolchain",
        lambda: {
            "tools": {
                "llc": {
                    "available": True,
                    "path": "/fake/llc",
                    "version": "llc 19.0.0",
                }
            }
        },
    )

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, capture_output, text):
        _ = capture_output, text
        out_idx = cmd.index("-o") + 1
        Path(str(cmd[out_idx])).write_text("// fake llc ptx\n", encoding="utf-8")
        return _Proc()

    monkeypatch.setattr("pipeline.mlir_contract_artifacts.subprocess.run", _fake_run)

    _ = emit_backend_contract_artifacts(
        spec_name="cuda_contract_exec_meta",
        out_dir=tmp_path,
        midend_module=mid_mod,
        mlir_report=mlir_report,
        downstream_llvm_name="downstream_cuda_llvm",
        downstream_llvm_module=llvm_mod,
        shape_bindings={"M": 4, "N": 4},
    )

    payload = json.loads(Path(str(mlir_report["downstream_cuda_llvm_contract_path"])).read_text(encoding="utf-8"))
    artifacts = dict(payload.get("artifacts") or {})
    assert str(artifacts.get("cuda_llvm_target_triple") or "") == "nvptx64-nvidia-cuda"
    assert str(artifacts.get("cuda_llvm_origin") or "") == "lowered_from_intent_cuda_codegen"


def test_emit_backend_contract_artifacts_caps_launch_block_by_ptx_maxntid(monkeypatch, tmp_path: Path) -> None:
    mid_mod = to_mlir(_add_intent("cuda_contract_exec_launch_cap"))
    llvm_mod = _llvm_module(mid_mod, kernel_name="add2d")
    llvm_mod.module_text = (
        '; ModuleID = "intentir"\n'
        'target triple = "nvptx64-nvidia-cuda"\n'
        "define void @add2d() { ret void }\n"
    )
    mlir_report: dict[str, object] = {}

    monkeypatch.setattr(
        "pipeline.mlir_contract_artifacts.detect_mlir_toolchain",
        lambda: {
            "tools": {
                "llc": {
                    "available": True,
                    "path": "/fake/llc",
                    "version": "llc 19.0.0",
                }
            }
        },
    )

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, capture_output, text):
        _ = capture_output, text
        out_idx = cmd.index("-o") + 1
        Path(str(cmd[out_idx])).write_text(
            "\n".join(
                [
                    "// fake llc ptx",
                    ".visible .entry add2d(",
                    ")",
                    ".maxntid 32, 1, 1",
                    "{",
                    "ret;",
                    "}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return _Proc()

    monkeypatch.setattr("pipeline.mlir_contract_artifacts.subprocess.run", _fake_run)

    _ = emit_backend_contract_artifacts(
        spec_name="cuda_contract_exec_launch_cap",
        out_dir=tmp_path,
        midend_module=mid_mod,
        fallback_intent_module=mid_mod,
        mlir_report=mlir_report,
        downstream_llvm_name="downstream_cuda_llvm",
        downstream_llvm_module=llvm_mod,
        shape_bindings={"M": 4, "N": 64},
    )

    payload = json.loads(Path(str(mlir_report["downstream_cuda_llvm_contract_path"])).read_text(encoding="utf-8"))
    launch = dict(payload.get("launch") or {})
    artifacts = dict(payload.get("artifacts") or {})
    # Hard-cut keeps LLVM/PTX metadata only; no runtime fallback markers should appear.
    assert list(launch.get("block") or []) == [32, 1, 1]
    assert list(artifacts.get("cuda_ptx_maxntid") or []) == [32, 1, 1]
    assert str(artifacts.get("runtime_fallback") or "") == ""


def test_compile_llvm_ir_to_cuda_ptx_rewrites_math_intrinsics_for_nvptx(monkeypatch, tmp_path: Path) -> None:
    llvm_ir = (
        '; ModuleID = "intentir"\n'
        'target triple = "nvptx64-nvidia-cuda"\n'
        "define void @k(float %x) {\n"
        "  %e = call float @llvm.exp.f32(float %x)\n"
        "  %l = call float @llvm.log.f32(float %x)\n"
        "  ret void\n"
        "}\n"
        "declare float @llvm.exp.f32(float)\n"
        "declare float @llvm.log.f32(float)\n"
    )
    seen = {"ll_text": ""}

    monkeypatch.setattr(
        "pipeline.mlir_contract_artifacts.detect_mlir_toolchain",
        lambda: {
            "tools": {
                "llc": {
                    "available": True,
                    "path": "/fake/llc",
                    "version": "llc 14.0.0",
                }
            }
        },
    )

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, capture_output, text):
        _ = capture_output, text
        ll_path = Path(str(cmd[-1]))
        seen["ll_text"] = ll_path.read_text(encoding="utf-8")
        out_idx = cmd.index("-o") + 1
        Path(str(cmd[out_idx])).write_text("// fake llc ptx\n", encoding="utf-8")
        return _Proc()

    monkeypatch.setattr("pipeline.mlir_contract_artifacts.subprocess.run", _fake_run)
    out_path = tmp_path / "k.ptx"
    _ = mca._compile_llvm_ir_to_cuda_ptx(llvm_ir_text=llvm_ir, out_path=out_path)

    ll_text = str(seen.get("ll_text") or "")
    assert "@llvm.exp.f32(" not in ll_text
    assert "@llvm.log.f32(" not in ll_text
    assert "@intentir_nvvm_expf_approx(" in ll_text
    assert "@intentir_nvvm_logf_approx(" in ll_text
    assert "define internal float @intentir_nvvm_expf_approx(float %x)" in ll_text
    assert "define internal float @intentir_nvvm_logf_approx(float %x)" in ll_text
    assert "@llvm.nvvm.ex2.approx.f(" in ll_text
    assert "@llvm.nvvm.lg2.approx.f(" in ll_text
    assert out_path.is_file()


def test_compile_llvm_ir_to_cuda_ptx_retargets_host_triple_to_nvptx(monkeypatch, tmp_path: Path) -> None:
    llvm_ir = (
        '; ModuleID = "intentir"\n'
        'target triple = "x86_64-pc-linux-gnu"\n'
        "define void @k() { ret void }\n"
    )
    seen = {"ll_text": ""}

    monkeypatch.setattr(
        "pipeline.mlir_contract_artifacts.detect_mlir_toolchain",
        lambda: {
            "tools": {
                "llc": {
                    "available": True,
                    "path": "/fake/llc",
                    "version": "llc 14.0.0",
                }
            }
        },
    )

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, capture_output, text):
        _ = capture_output, text
        ll_path = Path(str(cmd[-1]))
        seen["ll_text"] = ll_path.read_text(encoding="utf-8")
        out_idx = cmd.index("-o") + 1
        Path(str(cmd[out_idx])).write_text("// fake llc ptx\n", encoding="utf-8")
        return _Proc()

    monkeypatch.setattr("pipeline.mlir_contract_artifacts.subprocess.run", _fake_run)
    out_path = tmp_path / "k.ptx"
    _ = mca._compile_llvm_ir_to_cuda_ptx(llvm_ir_text=llvm_ir, out_path=out_path)

    ll_text = str(seen.get("ll_text") or "")
    assert 'target triple = "nvptx64-nvidia-cuda"' in ll_text
    assert "x86_64-pc-linux-gnu" not in ll_text
    assert out_path.is_file()


def test_compile_llvm_ir_to_cuda_ptx_rewrites_atanf_acosf_libcalls(monkeypatch, tmp_path: Path) -> None:
    llvm_ir = (
        '; ModuleID = "intentir"\n'
        'target triple = "nvptx64-nvidia-cuda"\n'
        "define void @k(float %x) {\n"
        "  %a = call float @acosf(float %x)\n"
        "  %b = call float @atanf(float %x)\n"
        "  ret void\n"
        "}\n"
        "declare dso_local float @acosf(float noundef) local_unnamed_addr #2\n"
        "declare dso_local float @atanf(float noundef) local_unnamed_addr #2\n"
    )
    seen = {"ll_text": ""}

    monkeypatch.setattr(
        "pipeline.mlir_contract_artifacts.detect_mlir_toolchain",
        lambda: {
            "tools": {
                "llc": {
                    "available": True,
                    "path": "/fake/llc",
                    "version": "llc 14.0.0",
                }
            }
        },
    )

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, capture_output, text):
        _ = capture_output, text
        ll_path = Path(str(cmd[-1]))
        seen["ll_text"] = ll_path.read_text(encoding="utf-8")
        out_idx = cmd.index("-o") + 1
        Path(str(cmd[out_idx])).write_text("// fake llc ptx\n", encoding="utf-8")
        return _Proc()

    monkeypatch.setattr("pipeline.mlir_contract_artifacts.subprocess.run", _fake_run)
    out_path = tmp_path / "k.ptx"
    _ = mca._compile_llvm_ir_to_cuda_ptx(llvm_ir_text=llvm_ir, out_path=out_path)

    ll_text = str(seen.get("ll_text") or "")
    assert "@acosf(" not in ll_text
    assert "@atanf(" not in ll_text
    assert "@intentir_nvvm_acosf_approx(" in ll_text
    assert "@intentir_nvvm_atanf_approx(" in ll_text
    assert "define internal float @intentir_nvvm_acosf_approx(float %x)" in ll_text
    assert "define internal float @intentir_nvvm_atanf_approx(float %x)" in ll_text
    assert "@llvm.sqrt.f32(" in ll_text
    assert "@llvm.fabs.f32(" in ll_text
    assert out_path.is_file()


def test_emit_backend_contract_artifacts_keeps_primary_downstream_llvm_contract(
    monkeypatch, tmp_path: Path
) -> None:
    mid_mod = to_mlir(_add_intent("cuda_and_rvv_downstream_llvm"))
    cuda_llvm_mod = _llvm_module(mid_mod, kernel_name="k_cuda")
    rvv_llvm_mod = _llvm_module(mid_mod, kernel_name="k_rvv")
    mlir_report: dict[str, object] = {}

    def _fake_emit_contract(
        *,
        backend: str,
        spec_name: str,
        out_dir: Path,
        module: IntentMLIRModule,  # noqa: ARG001
        suffix: str,
        shape_bindings=None,  # noqa: ANN001, ARG001
        materialize_executable: bool = False,  # noqa: ARG001
        fallback_intent_module=None,  # noqa: ANN001, ARG001
    ):
        payload = {
            "schema_version": "intent_mlir_backend_contract_v2",
            "backend": str(backend),
            "executable": {"format": ("cuda_ptx" if backend == "cuda" else "rvv_elf")},
            "artifacts": {},
        }
        path = Path(out_dir) / f"{spec_name}.intentir.intentdialect.{suffix}.contract.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path), {}

    monkeypatch.setattr("pipeline.mlir_contract_artifacts._emit_contract", _fake_emit_contract)

    _ = emit_backend_contract_artifacts(
        spec_name="cuda_and_rvv_downstream_llvm",
        out_dir=tmp_path,
        midend_module=mid_mod,
        mlir_report=mlir_report,
        downstream_llvm_name="downstream_cuda_llvm",
        downstream_llvm_module=cuda_llvm_mod,
        downstream_llvm_variants=[("downstream_rvv_llvm", rvv_llvm_mod)],
        shape_bindings={"M": 4, "N": 4},
    )

    assert str(mlir_report["downstream_llvm_contract_backend"]) == "cuda"
    assert str(mlir_report["downstream_llvm_contract_path"]) == str(mlir_report["downstream_cuda_llvm_contract_path"])
