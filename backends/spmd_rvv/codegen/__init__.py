"""
Compatibility shim for removed RVV C/C++ codegen path.

Strict hard-cut mode no longer supports generating RVV C source via
`backends.spmd_rvv.codegen`. Import path is kept only to provide clear errors
for stale callers.
"""

from .cpp_driver import ensure_cpp_codegen_built, lower_intent_json_to_c_with_files_cpp, lower_intent_to_c_with_files, lower_intent_to_c_with_files_cpp  # noqa: F401

__all__ = [
    "ensure_cpp_codegen_built",
    "lower_intent_to_c_with_files_cpp",
    "lower_intent_json_to_c_with_files_cpp",
    "lower_intent_to_c_with_files",
]
