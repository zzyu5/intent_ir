# IntentIR C++ Codegen (Host Tool)

This directory contains a small host-side C++ codegen tool:

- Parses expanded IntentIR JSON + shape bindings JSON
- Emits a standalone C program that reads `*.bin` inputs and compares `*_ref.bin` outputs

Build:

```bash
cmake -S backends/spmd_rvv/cpp_codegen -B backends/spmd_rvv/cpp_codegen/build -DCMAKE_BUILD_TYPE=Release
cmake --build backends/spmd_rvv/cpp_codegen/build -j
```

Note: the Python entrypoint (`backends.spmd_rvv.codegen`) auto-builds this tool
into a per-checkout cache dir under `~/.cache/intentir/cpp_codegen/` by default.
You can override with `INTENTIR_CPP_CODEGEN_BUILD_DIR=/path/to/build`.

Run:

```bash
backends/spmd_rvv/cpp_codegen/build/intentir_codegen \
  --intent /path/to/intent.json \
  --shapes /path/to/shapes.json \
  --atol 1e-3 --rtol 1e-3 > main.c
```

The Python remote runner (`scripts/rvv_remote_run.py`) can auto-build and use this tool.

Preferred Python entrypoint:

- `backends.spmd_rvv.codegen.intentir_to_c.lower_intent_to_c_with_files()`
