# `tests/mlir/`

MLIR and backend-lowering tests.

This directory covers:

- Python real-MLIR lowering
- C++ plugin fallback/selection
- CUDA/RVV backend contract emission
- PTX / LLVM / launch metadata behavior

The intent is to keep lowering/compiler regressions separate from IntentIR semantic-core tests.
