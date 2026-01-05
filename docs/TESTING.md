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

## Remote RVV suite (6 kernels × frontends)

Run:

- `PYTHONPATH=. python scripts/rvv_remote_suite.py --frontend both --host <host> --user <user> --use-key --case-index 0`

## Remote RVV perf suite (optional)

Run:

- `PYTHONPATH=. python scripts/benchmark_suite.py --frontend both --host <host> --user <user> --use-key --bench-iters 50 --bench-warmup 5 --out artifacts/perf_latest.json`
- `python scripts/compare_perf.py --baseline <baseline.json> --current artifacts/perf_latest.json --threshold 0.05`
