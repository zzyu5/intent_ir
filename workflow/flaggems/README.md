# FlagGems Long-Run Workflow

This directory is a session handoff harness for long-running FlagGems work.
FlagGems is integrated as a Triton provider (not a separate frontend type).

## Current Baseline

- Registry truth source: `pipeline/triton/flaggems_registry.json`
- Current coverage target: `196` semantic ops (`deterministic_forward`)
- Current status: `dual_pass=196`, `blocked_ir=0`, `blocked_backend=0`

## Quick Start

```bash
bash workflow/flaggems/init.sh
```

This will:
- sync `state/feature_list.json` from `pipeline/triton/flaggems_registry.json`
- freeze a baseline snapshot under `state/baselines/`
- plan the next active batch into `state/active_batch.json`
- validate session context snapshot (allows empty batch when full coverage is reached)

## Session Lifecycle

1. Start session:
```bash
python scripts/flaggems/plan_next_batch.py --batch-size 10
python scripts/flaggems/start_session.py
```
2. Implement one batch (mapping + spec + regression + convergence).
3. End session:
```bash
python scripts/flaggems/end_session.py \
  --summary "what was done" \
  --run-summary artifacts/flaggems_matrix/<run>/run_summary.json \
  --status-converged artifacts/flaggems_matrix/<run>/status_converged.json
```
4. Enforce hard gate:
```bash
python scripts/flaggems/check_batch_gate.py \
  --run-summary artifacts/flaggems_matrix/<run>/run_summary.json \
  --status-converged artifacts/flaggems_matrix/<run>/status_converged.json
```

## Matrix + Gate (Canonical)

```bash
python scripts/flaggems/run_multibackend_matrix.py \
  --suite coverage \
  --flaggems-path intentir \
  --intentir-mode auto \
  --run-rvv-remote \
  --rvv-host 192.168.8.72 \
  --rvv-user ubuntu \
  --rvv-use-key \
  --cuda-runtime-backend nvrtc \
  --cuda-codegen-mode py \
  --out-dir artifacts/flaggems_matrix/daily/<YYYYMMDD>/<run_name>
```

```bash
python scripts/flaggems/ci_gate.py \
  --run-summary artifacts/flaggems_matrix/daily/<YYYYMMDD>/<run_name>/run_summary.json \
  --status-converged artifacts/flaggems_matrix/daily/<YYYYMMDD>/<run_name>/status_converged_registry_write.json
```

## Nightly Maintenance (Post-Coverage)

When `active_batch` is empty (full coverage reached), use nightly drift checks:

```bash
python scripts/flaggems/nightly_maintenance.py \
  --suite coverage \
  --run-rvv-remote \
  --rvv-host 192.168.8.72 \
  --rvv-user ubuntu \
  --rvv-use-key \
  --cuda-runtime-backend nvrtc \
  --cuda-codegen-mode py
```

Use `--dry-run` to verify command composition without executing pipeline/backend jobs.

Scheduler wrapper (for cron/systemd):

```bash
bash workflow/flaggems/nightly.sh
```

Environment overrides (optional):
- `FLAGGEMS_NIGHTLY_SUITE`
- `FLAGGEMS_NIGHTLY_CASES_LIMIT`
- `FLAGGEMS_NIGHTLY_RVV_HOST`
- `FLAGGEMS_NIGHTLY_RVV_USER`
- `FLAGGEMS_NIGHTLY_RUN_RVV_REMOTE` (`1|0`)
- `FLAGGEMS_NIGHTLY_RVV_USE_KEY` (`1|0`)
- `FLAGGEMS_NIGHTLY_ALLOW_CUDA_SKIP` (`1|0`)
- `FLAGGEMS_NIGHTLY_WRITE_REGISTRY` (`1|0`)
- `FLAGGEMS_NIGHTLY_DRY_RUN` (`1|0`)

Cron example:

```bash
crontab workflow/flaggems/scheduler.crontab.example
```

Systemd timer example (copy and adjust paths/user):
- `workflow/flaggems/flaggems-nightly.service.example`
- `workflow/flaggems/flaggems-nightly.timer.example`

## State Files

- `state/feature_list.json`: registry-derived feature truth for scheduling.
- `state/active_batch.json`: current picked batch.
- `state/progress_log.jsonl`: append-only session log.
- `state/handoff.md`: human-readable latest handoff.
- `state/baselines/*.json`: frozen metric snapshots.
- `state/metrics_history.jsonl`: time-series metrics from registry snapshots.
- `state/roadmap.json`: persistent milestone checklist for long-run handoff.

## Notes

- `plan_next_batch.py` returning `Selected 0 ops` means full coverage is complete; switch to maintenance mode (nightly matrix + CI gate).
- Registry write-back policy remains: only after batch gate passes.
