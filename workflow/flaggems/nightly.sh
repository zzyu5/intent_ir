#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="$(date -u +%Y%m%d)"
TIME_TAG="$(date -u +%H%M%S)"

SUITE="${FLAGGEMS_NIGHTLY_SUITE:-coverage}"
CASES_LIMIT="${FLAGGEMS_NIGHTLY_CASES_LIMIT:-8}"
RUN_NAME="${FLAGGEMS_NIGHTLY_RUN_NAME:-nightly_${TIME_TAG}}"
OUT_ROOT="${FLAGGEMS_NIGHTLY_OUT_ROOT:-$ROOT_DIR/artifacts/flaggems_matrix/daily}"
RVV_HOST="${FLAGGEMS_NIGHTLY_RVV_HOST:-192.168.8.72}"
RVV_USER="${FLAGGEMS_NIGHTLY_RVV_USER:-ubuntu}"
RVV_PORT="${FLAGGEMS_NIGHTLY_RVV_PORT:-22}"
CUDA_RUNTIME_BACKEND="${FLAGGEMS_NIGHTLY_CUDA_RUNTIME_BACKEND:-nvrtc}"
INTENTIR_MISS_POLICY="${FLAGGEMS_NIGHTLY_INTENTIR_MISS_POLICY:-deterministic}"
MAX_TOTAL_REGRESSION_PCT="${FLAGGEMS_NIGHTLY_MAX_TOTAL_REGRESSION_PCT:-8}"
MIN_REGRESSION_DELTA_MS="${FLAGGEMS_NIGHTLY_MIN_REGRESSION_DELTA_MS:-50}"
MAX_REGRESSION_RATIO="${FLAGGEMS_NIGHTLY_MAX_REGRESSION_RATIO:-0.5}"
LANE="${FLAGGEMS_NIGHTLY_LANE:-coverage}"
CI_PROFILES="${FLAGGEMS_NIGHTLY_CI_PROFILES:-coverage}"
COVERAGE_MODE="${FLAGGEMS_NIGHTLY_COVERAGE_MODE:-category_batches}"
GPU_PERF_THRESHOLD="${FLAGGEMS_NIGHTLY_GPU_PERF_THRESHOLD:-0.80}"
PERF_WARMUP="${FLAGGEMS_NIGHTLY_PERF_WARMUP:-20}"
PERF_ITERS="${FLAGGEMS_NIGHTLY_PERF_ITERS:-200}"
PERF_REPEATS="${FLAGGEMS_NIGHTLY_PERF_REPEATS:-5}"
FAMILY_KERNEL_CHUNK_SIZE="${FLAGGEMS_NIGHTLY_FAMILY_KERNEL_CHUNK_SIZE:-12}"
PROGRESS_STYLE="${FLAGGEMS_NIGHTLY_PROGRESS_STYLE:-chunk}"
GPU_PERF_FAMILIES="${FLAGGEMS_NIGHTLY_GPU_PERF_FAMILIES:-}"

RUN_RVV_REMOTE="${FLAGGEMS_NIGHTLY_RUN_RVV_REMOTE:-1}"
RVV_USE_KEY="${FLAGGEMS_NIGHTLY_RVV_USE_KEY:-1}"
ALLOW_CUDA_SKIP="${FLAGGEMS_NIGHTLY_ALLOW_CUDA_SKIP:-1}"
WRITE_REGISTRY="${FLAGGEMS_NIGHTLY_WRITE_REGISTRY:-0}"
DRY_RUN="${FLAGGEMS_NIGHTLY_DRY_RUN:-0}"

LOG_DIR="${FLAGGEMS_NIGHTLY_LOG_DIR:-$ROOT_DIR/artifacts/flaggems_matrix/nightly_logs}"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/${DATE_TAG}_${RUN_NAME}.log"

echo "[flaggems:nightly] root=${ROOT_DIR}" | tee "$LOG_PATH"
echo "[flaggems:nightly] date=${DATE_TAG} run=${RUN_NAME}" | tee -a "$LOG_PATH"

echo "[flaggems:nightly] validating scripts catalog" | tee -a "$LOG_PATH"
python scripts/validate_catalog.py 2>&1 | tee -a "$LOG_PATH"

CMD=(
  python
  scripts/flaggems/nightly_maintenance.py
  --suite "$SUITE"
  --cases-limit "$CASES_LIMIT"
  --out-root "$OUT_ROOT"
  --date-tag "$DATE_TAG"
  --run-name "$RUN_NAME"
  --rvv-host "$RVV_HOST"
  --rvv-user "$RVV_USER"
  --rvv-port "$RVV_PORT"
  --cuda-runtime-backend "$CUDA_RUNTIME_BACKEND"
  --intentir-miss-policy "$INTENTIR_MISS_POLICY"
  --max-total-regression-pct "$MAX_TOTAL_REGRESSION_PCT"
  --min-regression-delta-ms "$MIN_REGRESSION_DELTA_MS"
  --max-regression-ratio "$MAX_REGRESSION_RATIO"
  --coverage-mode "$COVERAGE_MODE"
  --lane "$LANE"
  --gpu-perf-threshold "$GPU_PERF_THRESHOLD"
  --perf-warmup "$PERF_WARMUP"
  --perf-iters "$PERF_ITERS"
  --perf-repeats "$PERF_REPEATS"
  --family-kernel-chunk-size "$FAMILY_KERNEL_CHUNK_SIZE"
  --progress-style "$PROGRESS_STYLE"
)

if [[ "$RUN_RVV_REMOTE" == "1" ]]; then
  CMD+=(--run-rvv-remote)
else
  CMD+=(--no-run-rvv-remote)
fi

if [[ "$RVV_USE_KEY" == "1" ]]; then
  CMD+=(--rvv-use-key)
else
  CMD+=(--no-rvv-use-key)
fi

if [[ "$ALLOW_CUDA_SKIP" == "1" ]]; then
  CMD+=(--allow-cuda-skip)
else
  CMD+=(--no-allow-cuda-skip)
fi

if [[ "$WRITE_REGISTRY" == "1" ]]; then
  CMD+=(--write-registry)
fi

for profile in ${CI_PROFILES//,/ }; do
  if [[ -n "${profile}" ]]; then
    CMD+=(--ci-profiles "$profile")
  fi
done

for family in ${GPU_PERF_FAMILIES//,/ }; do
  if [[ -n "${family}" ]]; then
    CMD+=(--gpu-perf-family "$family")
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

echo "[flaggems:nightly] cmd: ${CMD[*]}" | tee -a "$LOG_PATH"
"${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"

echo "[flaggems:nightly] done" | tee -a "$LOG_PATH"
