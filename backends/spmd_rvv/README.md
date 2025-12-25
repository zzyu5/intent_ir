# Task6 Backend (`backends/spmd_rvv`)

This package contains the Task6 backend components.

**Directory layout**

- `backends/spmd_rvv/analysis/`: cost model, tiling search, hardware profiles
- `backends/spmd_rvv/codegen/`: code generation entrypoints
- `backends/spmd_rvv/cpp_codegen/`: C++ host tool that parses IntentIR JSON and emits a standalone C program

**Key API**

- `backends.spmd_rvv.codegen.intentir_to_c.lower_intent_to_c_with_files()`:
  Generates a standalone `main.c` that reads `<tensor>.bin` inputs and compares
  outputs against `<output>_ref.bin`.

The remote runner (`scripts/rvv_remote_run.py`) uses this API and compiles the
generated C on the RVV host.
