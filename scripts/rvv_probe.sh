#!/usr/bin/env bash
# Simple RVV device probe script: prints JSON profile to stdout.

set -euo pipefail

num_cores=$(nproc 2>/dev/null || echo 1)
freq_ghz=$(python - <<'PY' 2>/dev/null || echo 1.0
import re
freq = 1.0
with open("/proc/cpuinfo") as f:
    for line in f:
        if "cpu MHz" in line:
            try:
                mhz = float(line.strip().split(":")[1])
                freq = mhz / 1000.0
                break
            except Exception:
                pass
print(freq)
PY
)
vlen_bits=$(python - <<'PY' 2>/dev/null || echo 256
import re
vlen = None
with open("/proc/cpuinfo") as f:
    for line in f:
        if "vlenb" in line:
            try:
                bytes_v = int(line.strip().split()[-1])
                vlen = bytes_v * 8
                break
            except Exception:
                pass
print(vlen if vlen else 256)
PY
)

cache_line_bytes=64
l1d_kb=32
l2_kb=512
l3_kb=""

parse_size_kb() {
  # Parse strings like "32K", "512K", "4M" into KB.
  local s="$1"
  s="${s// /}"
  if [[ "$s" =~ ^([0-9]+)K$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  if [[ "$s" =~ ^([0-9]+)M$ ]]; then
    echo "$(( ${BASH_REMATCH[1]} * 1024 ))"
    return
  fi
  if [[ "$s" =~ ^[0-9]+$ ]]; then
    # bytes
    echo "$(( s / 1024 ))"
    return
  fi
  echo ""
}

if [[ -d "/sys/devices/system/cpu/cpu0/cache" ]]; then
  for idx in /sys/devices/system/cpu/cpu0/cache/index*; do
    [[ -d "$idx" ]] || continue
    level=$(cat "$idx/level" 2>/dev/null || echo "")
    type=$(cat "$idx/type" 2>/dev/null || echo "")
    size_str=$(cat "$idx/size" 2>/dev/null || echo "")
    line_sz=$(cat "$idx/coherency_line_size" 2>/dev/null || echo "")
    if [[ -n "$line_sz" ]]; then
      cache_line_bytes="$line_sz"
    fi
    size_kb=$(parse_size_kb "$size_str" || echo "")
    [[ -n "$size_kb" ]] || continue
    if [[ "$level" == "1" && "$type" == "Data" ]]; then
      l1d_kb="$size_kb"
    elif [[ "$level" == "2" && "$type" != "Instruction" ]]; then
      l2_kb="$size_kb"
    elif [[ "$level" == "3" && "$type" != "Instruction" ]]; then
      l3_kb="$size_kb"
    fi
  done
else
  # Fallback: use getconf if sysfs cache info isn't available.
  l1d=$(getconf LEVEL1_DCACHE_SIZE 2>/dev/null || echo 32768)
  l2=$(getconf LEVEL2_CACHE_SIZE 2>/dev/null || echo 524288)
  l1d_kb=$((l1d/1024))
  l2_kb=$((l2/1024))
fi

# Memory bandwidth microbench (very small STREAM-like memcpy benchmark).
mem_bw=12.0
if command -v gcc >/dev/null 2>&1; then
  # Prefer an executable filesystem (some environments mount /tmp as noexec).
  tmp_dir=$(
    mktemp -d "${HOME:-/tmp}/intentir_bw_XXXXXX" 2>/dev/null \
      || mktemp -d "/tmp/intentir_bw_XXXXXX"
  )
  tmp_c="$tmp_dir/bw.c"
  tmp_bin="$tmp_dir/bw"
  cat >"$tmp_c" <<'EOF'
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int main(void) {
  const size_t bytes = (size_t)128 * 1024 * 1024;  // 128MB
  void* a = NULL;
  void* b = NULL;
  if (posix_memalign(&a, 64, bytes) != 0) return 1;
  if (posix_memalign(&b, 64, bytes) != 0) return 1;
  memset(a, 1, bytes);
  memset(b, 0, bytes);

  // warmup
  for (int i = 0; i < 2; ++i) memcpy(b, a, bytes);

  const int iters = 5;
  uint64_t t0 = now_ns();
  for (int i = 0; i < iters; ++i) memcpy(b, a, bytes);
  uint64_t t1 = now_ns();
  double sec = (double)(t1 - t0) / 1e9;
  double gb = (double)bytes * (double)iters / 1e9;
  double gbps = (sec > 0.0) ? (gb / sec) : 0.0;
  printf("%.3f\n", gbps);
  return 0;
}
EOF
  if gcc -O3 -std=c11 -o "$tmp_bin" "$tmp_c" >/dev/null 2>&1 || gcc -O3 -std=c11 -o "$tmp_bin" "$tmp_c" -lrt >/dev/null 2>&1; then
    bw=""
    if out=$("$tmp_bin" 2>/dev/null); then
      bw="$out"
    fi
    if [[ "$bw" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      mem_bw="$bw"
    fi
  fi
  rm -rf "$tmp_dir" >/dev/null 2>&1 || true
fi

cat <<EOF
{
  "num_cores": ${num_cores},
  "rvv_vlen_bits": ${vlen_bits},
  "frequency_ghz": ${freq_ghz},
  "mem_bandwidth_gbps": ${mem_bw},
  "l1d_cache_kb": ${l1d_kb},
  "l2_cache_kb": ${l2_kb},
  "l3_cache_kb": ${l3_kb:-null},
  "cache_line_bytes": ${cache_line_bytes}
}
EOF
