# `backends/`

Backends lower IntentIR into executable target code.

Current backend paths:

- `backends/cuda/`: strict MLIR backend-contract path for CUDA execution
- `backends/spmd_rvv/`: strict MLIR backend-contract path for RVV execution

Notes:

- backend implementation stays split by target (CUDA/RVV)
- compatibility C/C++ codegen fallbacks are removed from mainline strict path
