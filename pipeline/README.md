# `pipeline/`

Shared orchestration logic for end-to-end pipelines.

User-facing entrypoints stay in `scripts/` and should remain thin wrappers.

- `pipeline/interfaces.py`: small shared types (constraints, future frontend/backend protocols).
- `pipeline/registry.py`: frontend adapter registry (`frontend -> adapter`).
- `pipeline/run.py`: generic frontend runner shell (adapter-only).
- `pipeline/triton/core.py`: Triton pipeline runner + kernel specs used by `scripts/triton/full_pipeline_verify.py`.
