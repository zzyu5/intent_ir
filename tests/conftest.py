from __future__ import annotations

import os
import sys
from pathlib import Path

# CI/sandbox environments may not provide a functional POSIX shared-memory mount
# (/dev/shm). Some OpenMP runtimes can hard-crash when SHM is unavailable.
# Keep tests deterministic and robust by defaulting to single-threaded BLAS/OMP.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_SHM_DISABLE", "1")

# Ensure repo root is importable for all tests, regardless of nested test layout.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
