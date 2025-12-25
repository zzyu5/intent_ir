"""
Macro op lowering infrastructure.

Macro ops are semantic (LLM-friendly) operators that carry a compact but
structured implementation spec. The compiler lowers them into the primitive
IntentIR op set for interpretation and backend codegen.
"""

from .registry import lower_macro_op, supports_macro

__all__ = ["lower_macro_op", "supports_macro"]

