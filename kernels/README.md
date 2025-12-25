# `kernels/`

This directory contains testcase kernels used by the pipeline.

Each frontend should place its kernels under a dedicated subdirectory:

- `kernels/triton/`: Triton Python DSL kernels (current)
- `kernels/cuda_c/`: CUDA C kernels (planned)
- `kernels/tilelang/`: TileLang kernels (planned)

The pipeline selects kernels via `pipeline/core.py` (kernel specs).

