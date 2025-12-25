"""
Frontend implementations.

A frontend is responsible for:
- providing a kernel "source string" for LLM prompting
- providing a runnable baseline implementation (for correctness checking)
- optionally providing a frontend IR dump (e.g., TTIR) and extracting constraints
"""

__all__ = ["triton"]

