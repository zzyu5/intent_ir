# Test Suite Scope

`pytest` in this repo is intentionally a fast gate, not a long-run coverage driver.

## What Are `__pycache__` / `.pyc`

- `__pycache__/` and `.pyc` are Python bytecode cache files generated after importing/running tests.
- They are not source tests and should not be committed.
- Safe cleanup command:
  - `find tests -type d -name '__pycache__' -prune -exec rm -rf {} +`

## Test Catalog

- File-to-purpose mapping lives in `tests/CATALOG.json`.
- Use it when you want to know "which test protects which contract" without reading all files.

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
