from .attach_provider_meta import attach_provider_meta
from .backend_legalize import backend_legalize
from .canonicalize_intent import canonicalize_intent
from .cse_like import cse_like
from .emit_cuda_contract import emit_cuda_contract
from .emit_rvv_contract import emit_rvv_contract
from .ensure_llvm_ir_text import ensure_llvm_ir_text
from .expand_macros_intent import expand_macros_intent
from .extract_gpu_module_llvm import extract_gpu_module_llvm
from .lower_intent_to_cuda_gpu_kernel import lower_intent_to_cuda_gpu_kernel
from .lower_intent_to_llvm_dialect import lower_intent_to_llvm_dialect
from .normalize_symbols import normalize_symbols
from .rewrite_cuda_nv_exp2f_calls import rewrite_cuda_nv_exp2f_calls
from .set_llvm_target_triple import set_llvm_target_triple

PASS_REGISTRY = {
    "normalize_symbols": normalize_symbols,
    "attach_provider_meta": attach_provider_meta,
    "expand_macros_intent": expand_macros_intent,
    "canonicalize_intent": canonicalize_intent,
    "cse_like": cse_like,
    "backend_legalize": backend_legalize,
    "lower_intent_to_cuda_gpu_kernel": lower_intent_to_cuda_gpu_kernel,
    "lower_intent_to_llvm_dialect": lower_intent_to_llvm_dialect,
    "ensure_llvm_ir_text": ensure_llvm_ir_text,
    "set_llvm_target_triple": set_llvm_target_triple,
    "extract_gpu_module_llvm": extract_gpu_module_llvm,
    "rewrite_cuda_nv_exp2f_calls": rewrite_cuda_nv_exp2f_calls,
    "emit_cuda_contract": emit_cuda_contract,
    "emit_rvv_contract": emit_rvv_contract,
}

__all__ = ["PASS_REGISTRY"]
