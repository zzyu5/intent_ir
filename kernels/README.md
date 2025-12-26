# `kernels/`

This directory contains testcase kernels used by the pipeline.

Each frontend should place its kernels under a dedicated subdirectory:

- `kernels/triton/`: Triton Python DSL kernels (current)
- `kernels/tilelang/`: TileLang kernels (MVP)
- `kernels/cuda_c/`: CUDA C kernels (planned)

The pipeline selects kernels via `pipeline/<frontend>/core.py` (kernel specs).
