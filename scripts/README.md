# `scripts/` (CLI Entrypoints)

Scripts are intended to be user-facing entrypoints; shared logic should live in
importable modules (e.g., `pipeline/triton/core.py`, or backend/verify packages).

## Key scripts

- `triton/full_pipeline_verify.py`: run the full Task1–5 Triton pipeline.
  - `--provider native|flaggems` chooses Triton provider (`flaggems` is still Triton, not a separate frontend).
  - `--use-llm/--no-use-llm` toggles LLM extraction vs deterministic fallback intents.
  - `--suite smoke|coverage|all`, `--list`, and `--kernel NAME` control kernel selection.
- `tilelang/full_pipeline_verify.py`: run the TileLang MVP pipeline for the default kernel set.
- `pipeline/triton/core.py`: reusable Triton pipeline helper library used by `scripts/triton/full_pipeline_verify.py`.
- `backend_codegen_smoke.py`: validate Task6 backend codegen locally (no LLM, no remote).
  - `--triton-provider native|flaggems` selects Triton artifact directory for backend smoke.
- `rvv_remote_run.py`: run Task6 on a remote RVV host and compare with saved baseline.
  - `--triton-provider native|flaggems` allows direct remote run from FlagGems Triton artifacts.
  - Defaults: `--host 192.168.8.72 --user ubuntu` (overridable by `INTENTIR_RVV_HOST`, `INTENTIR_RVV_USER`).
  - `--profile-ops` emits `INTENTIR_PROFILE {..}` per-op timing JSON (from the RVV program stdout).
- `rvv_remote_suite.py`: run remote RVV tests across kernel suites (user-facing).
  - Supports `--triton-provider native|flaggems`; when using `flaggems`, Triton defaults to the FlagGems kernel suite.
- `benchmark_suite.py`: run remote RVV perf microbenchmarks (ns/iter) for kernel suites.
  - Supports `--triton-provider native|flaggems` and the same RVV host/user defaults as `rvv_remote_run.py`.
  - For richer reports: pass `--profile-ops` (per-op timing) + `--tune-debug` (predicted cost-model debug).
- `analyze_perf.py`: summarize benchmark JSON (predicted vs measured, correlations).
- `compare_perf.py`: compare two perf JSONs and fail on regressions.
- `triton/pipeline_report.py`: generate a compact per-kernel report (LLM + TTIR).
- `triton/verify_ops.py`: per-op verification runner (debugging).

## Experiments

- `experiments/mutation_kill_ablation.py`: summarize mutation-kill outcomes from existing artifacts.
- `experiments/rvv_e2e_case_study.py`: run `rvv_remote_run.py` across the 6-kernel suite and collect a single JSON report.
  - Supports `--triton-provider native|flaggems` and the same RVV host/user defaults.
- `experiments/portability_vs_perf.py`: E5 freeze-vs-retune study on remote RVV.
  - Supports `--triton-provider native|flaggems`; provider also selects the default kernel family.
- `experiments/experiment_a_ai_benchmark.py`: E5 external baseline comparison (AI-Benchmark vs ours).
  - Supports `--triton-provider`; currently requires `native` because AI-Benchmark-equivalent kernels are not yet in the FlagGems spec set.
