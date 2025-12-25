# Testing

## Unit tests (fast)

Run:

- `pytest -q`

These tests cover IntentIR types/parser/printer and the TTIR facts/contract helpers.

## Full pipeline (LLM + Triton + Task4/5)

Run:

- `PYTHONPATH=. python scripts/triton/full_pipeline_verify.py`

Outputs are written under:

- `artifacts/full_pipeline_verify/`

## Backend codegen smoke (local C compile/run)

Run:

- `python scripts/backend_codegen_smoke.py`

This validates “IntentIR ops → generated C → local gcc → compare with baseline .npz”.

## Remote RVV run (end-to-end Task6)

Run:

- `PYTHONPATH=. python scripts/rvv_remote_run.py --kernel <name> --host <host> --user <user>`

The remote executable prints PASS/FAIL and per-output diff stats.
