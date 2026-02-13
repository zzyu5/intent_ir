# FlagGems Long-Run Workflow

This directory is a session handoff harness for long-running FlagGems work.

## Quick Start

```bash
bash workflow/flaggems/init.sh
```

This will:
- sync `state/feature_list.json` from `pipeline/triton/flaggems_registry.json`
- freeze a baseline snapshot under `state/baselines/`
- select the next active batch into `state/active_batch.json`

## Session Lifecycle

1. Start session:
```bash
python scripts/flaggems/start_session.py --batch-size 10
```
2. Implement one batch (mapping + spec + regression + convergence).
3. End session:
```bash
python scripts/flaggems/end_session.py \
  --summary "what was done" \
  --run-summary artifacts/flaggems_matrix/<run>/run_summary.json \
  --status-converged artifacts/flaggems_matrix/<run>/status_converged.json
```

## State Files

- `state/feature_list.json`: registry-derived feature truth for scheduling.
- `state/active_batch.json`: current picked batch.
- `state/progress_log.jsonl`: append-only session log.
- `state/handoff.md`: human-readable latest handoff.
- `state/baselines/*.json`: frozen metric snapshots.

