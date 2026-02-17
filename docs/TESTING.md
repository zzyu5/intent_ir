# Testing

## Unit tests (fast)

Run:

- `pytest -q`

These tests cover IntentIR types/parser/printer and the TTIR facts/contract helpers.

## Full pipeline (LLM + Triton + Task4/5)

Run:

- `python scripts/triton/flaggems_full_pipeline_verify.py --suite smoke`

Outputs are written under:

- `artifacts/flaggems_triton_full_pipeline/`

## Backend codegen smoke (local C compile/run)

Run:

- `python scripts/backend_codegen_smoke.py`

This validates “IntentIR ops → generated C → local gcc → compare with baseline .npz”.

## Remote RVV run (end-to-end Task6)

Run:

- `python scripts/rvv_remote_run.py --kernel <name> --host <host> --user <user> --use-key`

The remote executable prints PASS/FAIL and per-output diff stats.

## Remote RVV suite (6 kernels × frontends)

Run:

- `python scripts/rvv_remote_suite.py --frontend both --host <host> --user <user> --use-key --case-index 0`

## Remote RVV perf suite (optional)

This repo's performance tracking is done via the workflow artifacts (`schedule_profiles.json`, `timing_delta.json`,
`stage_timing_breakdown.json`) produced by nightly and backend compiler lane runs.
