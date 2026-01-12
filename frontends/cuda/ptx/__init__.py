"""
PTX parsing utilities for the CUDA frontend (MVP).

The goal is NOT a full PTX interpreter. We only extract enough stable evidence
for Tier-A kernels:
- global ld/st tensor mapping (by param ordinal)
- basic affine-ish index recovery for thread/block based addressing
- simple guard predicate clauses (setp + early-exit branch)
"""

from .parser import parse_ptx_kernel

__all__ = ["parse_ptx_kernel"]

