from __future__ import annotations

import subprocess
from pathlib import Path

from intent_ir.ir import IntentFunction
from intent_ir.mlir import detect_mlir_toolchain, run_pipeline, to_mlir


def _sample_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "add2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "z": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "add", "inputs": ["x", "y"], "output": "z", "attrs": {}}],
            "outputs": ["z"],
        }
    )


def test_downstream_rvv_std_llvm_emits_riscv_triple(tmp_path: Path) -> None:
    toolchain = detect_mlir_toolchain()
    assert bool(toolchain.get("ok")) is True
    for k in ("mlir-opt", "mlir-translate", "llvm-as", "opt"):
        assert bool(((toolchain.get("tools") or {}).get(k) or {}).get("available")) is True

    mod = to_mlir(_sample_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out, trace = run_pipeline(mod, "downstream_rvv_std_llvm", backend="rvv", out_dir=tmp_path, fail_on_error=True)
    assert bool(trace.get("ok")) is True
    text = str(out.module_text or "")
    assert "target triple = \"riscv64-unknown-linux-gnu\"" in text
    assert str((out.meta or {}).get("llvm_dialect_origin") or "") == "mlir_translate"

    llc = (toolchain.get("tools") or {}).get("llc") or {}
    llc_path = str(llc.get("path") or "").strip()
    if not (bool(llc.get("available")) and llc_path):
        return
    in_ll = tmp_path / "module.ll"
    in_ll.write_text(text, encoding="utf-8")
    out_s = tmp_path / "module.s"
    p = subprocess.run(
        [
            llc_path,
            "-mtriple=riscv64-unknown-linux-gnu",
            "-mattr=+v",
            "-O3",
            "-filetype=asm",
            str(in_ll),
            "-o",
            str(out_s),
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr or p.stdout
    assert "vsetvli" in out_s.read_text(encoding="utf-8")
