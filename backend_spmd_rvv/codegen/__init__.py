"""
Task6 backend code generation entrypoints.
"""

from .cpp_driver import ensure_cpp_codegen_built, lower_intent_to_c_with_files_cpp  # noqa: F401
from .intentir_to_c import lower_intent_to_c_with_files  # noqa: F401
from .matmul_c import generate_c  # noqa: F401

__all__ = [
    "ensure_cpp_codegen_built",
    "lower_intent_to_c_with_files_cpp",
    "lower_intent_to_c_with_files",
    "generate_c",
]

