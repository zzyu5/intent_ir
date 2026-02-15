#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[flaggems:init] pwd: $(pwd)"
echo "[flaggems:init] branch: $(git branch --show-current)"

python scripts/validate_catalog.py
python scripts/flaggems/sync_feature_list_mixed.py --freeze-baseline
python scripts/flaggems/build_workflow_state.py
python scripts/flaggems/plan_next_batch.py --lane coverage --batch-size "${1:-10}"
python scripts/flaggems/plan_next_batch.py --lane ir_arch --batch-size "${2:-8}"
python scripts/flaggems/plan_next_batch.py --lane backend_compiler --batch-size "${3:-8}"
python scripts/flaggems/start_session.py --lane ir_arch --no-require-non-empty

echo "[flaggems:init] done"
