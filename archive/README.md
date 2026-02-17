# Archive (History-Only)

This directory is for historical materials (papers, experiments, third-party clones, large local datasets).

Rules:
- `archive/` is **not** part of the runtime or CI gate paths.
- Workflow, nightly, and active scripts **must not** depend on anything under `archive/`.
- Large/private content under `archive/` is typically gitignored (see `.gitignore`).

If you need to regenerate artifacts from archived sources, use explicit flags/paths (never implicit defaults).

