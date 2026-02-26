# `backends/cuda/`

CUDA strict backend components.

## Execution model

- Mainline execution is contract-first and strict:
  - MLIR backend contract with executable payload (`cuda_ptx`/`ptx`)
  - runtime launch through `backends/cuda/pipeline/driver.py`
- Legacy IntentIR->CUDA C/C++ codegen path is removed from the main repo path.

## Directory layout

- `backends/cuda/pipeline/`: strict MLIR backend-contract pipeline and runtime stages
- `backends/cuda/runtime/`: CUDA runtime helpers and kernel launch glue
