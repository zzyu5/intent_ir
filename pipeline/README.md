# `pipeline/`

Shared orchestration logic for end-to-end pipelines.

User-facing entrypoints stay in `scripts/` and should remain thin wrappers.

- `pipeline/interfaces.py`: small shared types (constraints, future frontend/backend protocols).
- `pipeline/registry.py`: frontend adapter registry (`frontend -> adapter`).
- `pipeline/run.py`: generic frontend runner shell (adapter-only).
- `pipeline/triton/core.py`: Triton pipeline runner + kernel specs used by `scripts/triton/flaggems_full_pipeline_verify.py`.
- `pipeline/triton/providers/`: Triton provider plugins (FlagGems lives under `pipeline/triton/providers/flaggems/`).
- `pipeline/triton/flaggems_registry.json`: frozen registry baseline generated from `flag_gems.ops.__all__` (via `scripts/flaggems/generate_registry.py`).
