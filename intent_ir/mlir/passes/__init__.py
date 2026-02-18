from .attach_provider_meta import attach_provider_meta
from .canonicalize_intent import canonicalize_intent
from .cse_like import cse_like
from .normalize_symbols import normalize_symbols

PASS_REGISTRY = {
    "normalize_symbols": normalize_symbols,
    "attach_provider_meta": attach_provider_meta,
    "canonicalize_intent": canonicalize_intent,
    "cse_like": cse_like,
}

__all__ = ["PASS_REGISTRY"]

