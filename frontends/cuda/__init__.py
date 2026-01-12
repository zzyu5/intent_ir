"""
CUDA frontend (WIP).

Implements:
- compile CUDA C++ -> PTX artifacts
- extract CanonicalEvidence from PTX (Tier-A MVP)
- baseline runner via torch CUDA extension
- LLM prompt builder for CUDA source
"""

