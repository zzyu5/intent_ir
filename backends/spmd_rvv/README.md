# Task6 Backend (`backends/spmd_rvv`)

This package contains the RVV strict backend components.

**Directory layout**

- `backends/spmd_rvv/analysis/`: cost model, tiling search, hardware profiles
- `backends/spmd_rvv/pipeline/`: strict MLIR backend-contract pipeline and runtime stages
- `backends/spmd_rvv/runtime/`: target-side runtime helpers used by remote execution

**Execution model**

- Mainline execution is contract-first and strict:
  - prebuilt RVV ELF
  - remote LLVM compile on RVV target
- Legacy C/C++ compatibility codegen paths have been removed from the main repo path.
