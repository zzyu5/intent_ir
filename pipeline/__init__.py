"""
Pipeline orchestration (frontend -> IntentIR -> verification -> backend).
"""

# Avoid MKL/OpenMP hard-crashes seen when importing NumPy before Torch in some
# sandbox/CI environments (e.g., "OMP: Error #179: Can't open SHM2 ...").
# Prefer a stable default; callers can override explicitly via env vars.
import os

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

__all__ = ["interfaces", "registry", "run", "triton", "tilelang", "cuda"]
