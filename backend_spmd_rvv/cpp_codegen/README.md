# IntentIR C++ Codegen (Host Tool)

This directory contains a small host-side C++ codegen tool:

- Parses expanded IntentIR JSON + shape bindings JSON
- Emits a standalone C program that reads `*.bin` inputs and compares `*_ref.bin` outputs

Build:

```bash
cmake -S backend_spmd_rvv/cpp_codegen -B backend_spmd_rvv/cpp_codegen/build -DCMAKE_BUILD_TYPE=Release
cmake --build backend_spmd_rvv/cpp_codegen/build -j
```

Run:

```bash
backend_spmd_rvv/cpp_codegen/build/intentir_codegen \
  --intent /path/to/intent.json \
  --shapes /path/to/shapes.json \
  --atol 1e-3 --rtol 1e-3 > main.c
```

The Python remote runner (`scripts/rvv_remote_run.py`) can auto-build and use this tool.

