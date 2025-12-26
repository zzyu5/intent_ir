# `scripts/` (CLI Entrypoints)

Scripts are intended to be user-facing entrypoints; shared logic should live in
importable modules (e.g., `pipeline/triton/core.py`, or backend/verify packages).

## Key scripts

- `triton/full_pipeline_verify.py`: run the full Task1–5 Triton pipeline for the default kernel set.
  - `--list` lists kernels, `--kernel NAME` runs a subset.
- `tilelang/full_pipeline_verify.py`: run the TileLang MVP pipeline for the default kernel set.
- `pipeline/triton/core.py`: reusable Triton pipeline helper library used by `scripts/triton/full_pipeline_verify.py`.
- `backend_codegen_smoke.py`: validate Task6 backend codegen locally (no LLM, no remote).
- `rvv_remote_run.py`: run Task6 on a remote RVV host and compare with saved baseline.
- `triton/pipeline_report.py`: generate a compact per-kernel report (LLM + TTIR).
- `triton/verify_ops.py`: per-op verification runner (debugging).
