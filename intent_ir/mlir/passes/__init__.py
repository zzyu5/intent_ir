from .attach_provider_meta import attach_provider_meta
from .backend_legalize import backend_legalize
from .canonicalize_intent import canonicalize_intent
from .cse_like import cse_like
from .emit_cuda_contract import emit_cuda_contract
from .emit_rvv_contract import emit_rvv_contract
from .expand_macros_intent import expand_macros_intent
from .normalize_symbols import normalize_symbols

PASS_REGISTRY = {
    "normalize_symbols": normalize_symbols,
    "attach_provider_meta": attach_provider_meta,
    "expand_macros_intent": expand_macros_intent,
    "canonicalize_intent": canonicalize_intent,
    "cse_like": cse_like,
    "backend_legalize": backend_legalize,
    "emit_cuda_contract": emit_cuda_contract,
    "emit_rvv_contract": emit_rvv_contract,
}

__all__ = ["PASS_REGISTRY"]
