# IntentIR

IntentIR is a research prototype that extracts **high-level kernel intent** from GPU kernels (currently Triton), validates it against TTIR-derived constraints, and lowers it to backend code (currently RVV C) for end-to-end checking.

## Project Structure

- `intent_ir/`: IntentIR types, JSON spec, LLM extraction + parser, MLIR-like printer, macro system (semantic ops + compiler-style lowering).
- `frontends/triton/`: Triton-specific frontend: TTIR compilation, TTIR facts/constraints, contract + certificate, static validation.
- `verify/`: Task5 verification: case generation, interpreter, diff runner, metamorphic + mutation-kill harness.
- `backends/spmd_rvv/`: Task6 backend: analysis (tiling/cost model) + codegen (IntentIR JSON -> standalone C; remote RVV execution in `scripts/rvv_remote_run.py`).
- `kernels/triton/`: Real Triton kernels used as testcases.
- `scripts/`: User-facing CLI runners (full pipeline, backend smoke, remote RVV).
- `pipeline/`: reusable orchestration library used by `scripts/`.
- `docs/`: Maintainer docs: architecture, environment, testing.
- `doc/`: Design/task notes (vibe-coding docs; not required for running the code).

## Quickstart (Local)

- Environment check: `python scripts/check_env.py`
- Unit tests: `pytest -q`
- Backend codegen (no LLM, no remote): `python scripts/backend_codegen_smoke.py`
- Full pipeline (LLM + Triton launch + TTIR + verify): `python scripts/triton/full_pipeline_verify.py`

## Remote RVV (Task6)

Run a kernel end-to-end and compare remote outputs against the saved Triton baseline:

`python scripts/rvv_remote_run.py --kernel any_kernel_dim --host <host> --user <user> --use-key`

Set the SSH password via `INTENTIR_SSH_PASSWORD` or let it prompt.

## Extensibility

This repo is structured around a frontend/backend split:

- **Frontends** provide: kernel source, baseline runner, (optional) IR dumps (e.g., TTIR), and constraint extraction.
- **IntentIR** is the cross-frontend semantic layer (plus schedule sketch).
- **Backends** lower IntentIR to target code (e.g., RVV C) and can be tested by comparing against the frontend baseline runner.

See `docs/ARCHITECTURE.md` for a concrete, step-by-step view of the pipeline.
For a precise definition of IntentIR op meaning + verification guarantees, see `docs/FORMAL_SEMANTICS.md`.
