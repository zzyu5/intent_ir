# Test Suite Scope

`pytest` in this repo is intentionally a fast gate, not a long-run coverage driver.

What stays in `tests/`:

- Workflow/gate contracts (`ci_gate`, `check_batch_gate`, workflow state/session scripts)
- Provider boundary contracts (Triton provider plugin hooks/boundary)
- IntentIR core correctness (types/interpreter/tolerances/parser)
- Backend compiler stage contracts (CUDA/RVV pipeline and smoke timing schema)

What moved to `archive/tests/`:

- Paper/experiment evidence tests
- Deprecated entrypoint tests
- Redundant historical tests that do not protect active workflow gates

For long runs (full196 category batches, RVV remote, CUDA local), use:

- `python scripts/intentir.py suite --suite flaggems-full196 ...`

Do not use `pytest` as a replacement for workflow long-run evidence.
