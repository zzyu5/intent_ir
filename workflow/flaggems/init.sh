#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[flaggems:init] pwd: $(pwd)"
echo "[flaggems:init] branch: $(git branch --show-current)"

python scripts/flaggems/sync_feature_list_from_registry.py --freeze-baseline
python scripts/flaggems/plan_next_batch.py --batch-size "${1:-10}"
python scripts/flaggems/start_session.py

echo "[flaggems:init] done"
