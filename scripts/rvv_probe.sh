#!/usr/bin/env bash
# Simple RVV device probe script: prints JSON profile to stdout.

set -e

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
l1d=$(getconf LEVEL1_DCACHE_SIZE 2>/dev/null || echo 32768)
l2=$(getconf LEVEL2_CACHE_SIZE 2>/dev/null || echo 524288)
mem_bw=12.0

cat <<EOF
{
  "num_cores": ${num_cores},
  "rvv_vlen_bits": ${vlen_bits},
  "frequency_ghz": ${freq_ghz},
  "mem_bandwidth_gbps": ${mem_bw},
  "l1d_cache_kb": $((l1d/1024)),
  "l2_cache_kb": $((l2/1024))
}
EOF
