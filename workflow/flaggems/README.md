# FlagGems Long-Run Workflow

This directory is the long-running handoff harness for IntentIR x FlagGems.
FlagGems remains a Triton provider (not a separate frontend type).

Script governance is enforced by `scripts/CATALOG.json`.
Always run:

```bash
python scripts/validate_catalog.py
```

Unified user CLI (recommended for manual runs):

```bash
python scripts/intentir.py suite --suite flaggems-full196 --help
python scripts/intentir.py kernel --help
python scripts/intentir.py env
```

## Current Baseline

- Registry truth source: `pipeline/triton/flaggems_registry.json`
- Coverage baseline: `196` semantic ops (`deterministic_forward`)
- Registry snapshot may show `dual_pass=196`, but trusted full196 status is tracked by:
  - `state/current_status.json::full196_last_run`
  - `state/current_status.json::full196_last_ok`
  - `state/current_status.json::coverage_integrity_phase`
- Workflow mode: mixed tracks (`coverage`, `ir_arch`, `backend_compiler`, `workflow`)

## Quick Start

```bash
bash workflow/flaggems/init.sh
```

This will:
- sync mixed-track `state/feature_list.json` from registry + `state/task_templates.json`
- freeze baseline snapshot under `state/baselines/`
- build `state/current_status.json` + `state/session_context.json`
- plan lane batches into:
  - `state/active_batch_coverage.json`
  - `state/active_batch_ir_arch.json`
  - `state/active_batch_backend_compiler.json`
  - `state/active_batch_workflow.json`
- start a lane session (`ir_arch` by default in `init.sh`)

## Session Lifecycle

1. Start lane session:
```bash
python scripts/flaggems/plan_next_batch.py --lane ir_arch --batch-size 8
python scripts/flaggems/start_session.py --lane ir_arch
```
2. Implement one lane batch (mapping/spec/regression or architecture/compiler tasks).
3. End session:
```bash
python scripts/flaggems/end_session.py \
  --lane ir_arch \
  --summary "what was done" \
  --run-summary artifacts/flaggems_matrix/<run>/run_summary.json \
  --status-converged artifacts/flaggems_matrix/<run>/status_converged.json
```
4. Enforce gate:
```bash
python scripts/flaggems/check_batch_gate.py \
  --profile ir_arch \
  --active-batch workflow/flaggems/state/active_batch_ir_arch.json \
  --run-summary artifacts/flaggems_matrix/<run>/run_summary.json \
  --status-converged artifacts/flaggems_matrix/<run>/status_converged.json
```

Lane runner shortcuts:
```bash
python scripts/flaggems/run_ir_arch_batch.py --out-dir artifacts/flaggems_matrix/daily/<YYYYMMDD>/ir_arch_<run>
python scripts/flaggems/run_backend_compiler_batch.py --out-dir artifacts/flaggems_matrix/daily/<YYYYMMDD>/backend_compiler_<run>
```

Backend compiler runner defaults to `workflow/flaggems/state/backend_kernel_manifest.json`
when `--kernel` is not explicitly provided.
Each successful backend compiler batch now emits:
- `stage_timing_breakdown.json`
- `timing_delta.json`
- `schedule_profiles.json`

## Matrix + CI Gate (Coverage Lane)

```bash
python scripts/intentir.py suite --suite flaggems-full196 \
  --out-root artifacts/flaggems_matrix/daily/<YYYYMMDD>/<run_name> \
  --flaggems-path intentir \
  --intentir-mode force_compile \
  --intentir-miss-policy strict \
  --progress-style chunk \
  --pipeline-timeout-sec 1800 \
  --run-rvv-remote \
  --rvv-host 192.168.8.72 \
  --rvv-user ubuntu \
  --rvv-use-key \
  --cuda-runtime-backend nvrtc \
  --family-kernel-chunk-size 12 \
  --resume
```

```bash
python scripts/flaggems/ci_gate.py \
  --profiles coverage \
  --run-summary artifacts/flaggems_matrix/daily/<YYYYMMDD>/<run_name>/run_summary.json \
  --status-converged artifacts/flaggems_matrix/daily/<YYYYMMDD>/<run_name>/status_converged_registry_write.json
```

## Nightly Maintenance

```bash
python scripts/flaggems/nightly_maintenance.py \
  --suite coverage \
  --coverage-mode category_batches \
  --lane coverage \
  --ci-profiles coverage \
  --run-rvv-remote \
  --rvv-host 192.168.8.72 \
  --rvv-user ubuntu \
  --rvv-use-key \
  --cuda-runtime-backend nvrtc
```

GPU perf nightly (CUDA Graph, Triton-native 대비 ratio gate):

```bash
python scripts/flaggems/nightly_maintenance.py \
  --suite gpu_perf \
  --ci-profiles gpu_perf \
  --gpu-perf-threshold 0.80 \
  --perf-warmup 20 \
  --perf-iters 200 \
  --perf-repeats 5 \
  --family-kernel-chunk-size 12 \
  --progress-style chunk \
  --cuda-runtime-backend nvrtc
```

Use `--dry-run` to inspect composed commands without executing.

Scheduler wrapper:

```bash
bash workflow/flaggems/nightly.sh
```

Environment overrides:
- `FLAGGEMS_NIGHTLY_SUITE`
- `FLAGGEMS_NIGHTLY_CASES_LIMIT`
- `FLAGGEMS_NIGHTLY_COVERAGE_MODE` (`category_batches|single_run`)
- `FLAGGEMS_NIGHTLY_RVV_HOST`
- `FLAGGEMS_NIGHTLY_RVV_USER`
- `FLAGGEMS_NIGHTLY_INTENTIR_MISS_POLICY`
- `FLAGGEMS_NIGHTLY_MAX_TOTAL_REGRESSION_PCT`
- `FLAGGEMS_NIGHTLY_MIN_REGRESSION_DELTA_MS`
- `FLAGGEMS_NIGHTLY_MAX_REGRESSION_RATIO`
- `FLAGGEMS_NIGHTLY_LANE`
- `FLAGGEMS_NIGHTLY_CI_PROFILES` (comma-separated)
- `FLAGGEMS_NIGHTLY_GPU_PERF_THRESHOLD` (default `0.80`)
- `FLAGGEMS_NIGHTLY_PERF_WARMUP` (default `20`)
- `FLAGGEMS_NIGHTLY_PERF_ITERS` (default `200`)
- `FLAGGEMS_NIGHTLY_PERF_REPEATS` (default `5`)
- `FLAGGEMS_NIGHTLY_FAMILY_KERNEL_CHUNK_SIZE` (default `12`)
- `FLAGGEMS_NIGHTLY_PROGRESS_STYLE` (`chunk|plain|tqdm|none`)
- `FLAGGEMS_NIGHTLY_GPU_PERF_FAMILIES` (comma-separated family filter for `--suite gpu_perf`)
- `FLAGGEMS_NIGHTLY_RUN_RVV_REMOTE` (`1|0`)
- `FLAGGEMS_NIGHTLY_RVV_USE_KEY` (`1|0`)
- `FLAGGEMS_NIGHTLY_ALLOW_CUDA_SKIP` (`1|0`)
- `FLAGGEMS_NIGHTLY_WRITE_REGISTRY` (`1|0`)
- `FLAGGEMS_NIGHTLY_DRY_RUN` (`1|0`)

Systemd install helper:

```bash
python scripts/flaggems/install_systemd_nightly.py --dry-run
```

## State Files

- `state/feature_list.json`: mixed-track truth (`coverage` + `ir_arch` + `backend_compiler`).
- `state/task_templates.json`: manual non-coverage task templates.
- `state/current_status.json`: single-file current truth snapshot.
- `state/session_context.json`: startup read-order + compact context.
- `state/active_batch_coverage.json`: active coverage lane batch.
- `state/active_batch_ir_arch.json`: active IR architecture lane batch.
- `state/active_batch_backend_compiler.json`: active backend compiler lane batch.
- `state/active_batch_workflow.json`: active workflow-governance lane batch.
- `state/coverage_batches.json`: fixed 7-family coverage batch plan.
- `state/progress_log.jsonl`: append-only session history.
- `state/handoff.md`: latest human-readable handoff.
- `state/baselines/*.json`: frozen baseline snapshots.
- `state/metrics_history.jsonl`: status time series.
- `state/roadmap.json`: milestone tracker.

## Notes

- Coverage source of truth stays `pipeline/triton/flaggems_registry.json`.
- Mixed tracks are workflow scheduling overlays, not registry replacements.
- Registry write-back policy remains: write registry only after gate pass.
- Deprecated scripts are tracked in `scripts/CATALOG.json` and must not be invoked by workflow wrappers.
