from __future__ import annotations

from intent_ir.mlir.module import IntentMLIRModule
from intent_ir.mlir.passes.rewrite_cuda_nv_exp2f_calls import rewrite_cuda_nv_exp2f_calls


def test_rewrite_cuda_nv_exp2f_calls_replaces_callsite() -> None:
    text = """module {\n  llvm.func @__nv_exp2f(f32) -> f32\n  llvm.func @k(%arg0: f32) -> f32 {\n    %0 = llvm.call @__nv_exp2f(%arg0) : (f32) -> f32\n    llvm.return %0 : f32\n  }\n}\n"""
    mod = IntentMLIRModule(module_text=text, dialect_version="std_mlir_v1")
    out = rewrite_cuda_nv_exp2f_calls(mod)
    assert "llvm.call @__nv_exp2f" not in out.module_text
    assert "llvm.intr.exp2(%arg0)" in out.module_text

