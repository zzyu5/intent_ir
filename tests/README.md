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

Top-level layout now splits the active suite by responsibility:

- `tests/core/`: IntentIR semantic core, parser, interpreter, strict policy.
- `tests/mlir/`: Python/C++ MLIR lowering, backend contracts, CUDA/RVV compiler path tests.
- `tests/frontends/`: frontend/provider contracts and workflow-facing adapters.
- `tests/pipeline/`: end-to-end pipeline behavior that is not ORG-specific.
- `ORG-Migrate/tests/`: ORG plugin tests only.

What stays archived under `archive/tests/`:

- Paper/experiment evidence tests
- Deprecated entrypoint tests
- Redundant historical tests that do not protect active workflow gates

For long runs (full196 category batches, RVV remote, CUDA local), use:

- `python scripts/intentir.py suite --suite flaggems-full196 ...`

Do not use `pytest` as a replacement for workflow long-run evidence.
