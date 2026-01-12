"""
CUDA kernel library (source-only).

These kernels are plain CUDA C++ strings used by the CUDA frontend:
- compile to PTX (artifacts)
- execute for baseline outputs (runtime)
- feed to LLM for IntentIR extraction (Task2)

In `kernels/cuda/ops/`, kernel sources live as real `.cu` files.
"""
