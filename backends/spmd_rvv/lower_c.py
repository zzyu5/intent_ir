"""
Deprecated shim (legacy Python backend removed).

Task6 originally used a Python implementation to lower IntentIR ops into a
standalone C program. The backend is now implemented as a C++ tool under
`backends/spmd_rvv/cpp_codegen/` (more compiler-like: IR parsing + lowering in
C++), with Python only used as orchestration.

Use:
  - `backends.spmd_rvv.codegen.intentir_to_c.lower_intent_to_c_with_files`
"""

from __future__ import annotations

from .codegen.intentir_to_c import lower_intent_to_c_with_files

__all__ = ["lower_intent_to_c_with_files"]
