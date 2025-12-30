"""
Cost-model validation harness (predicted vs measured GFLOPs) on a real RVV host.

Example:
  PYTHONPATH=. python scripts/rvv_cost_model_validate.py --host 192.168.8.149 --user ubuntu --M 256 --N 256 --K 256
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backends.spmd_rvv.experiments.cost_model_validation import (  # noqa: E402
    TileConfig,
    validate_gemm_cost_model_remote,
)


def _parse_tile(s: str) -> TileConfig:
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected tm,tn,tk, got {s!r}")
    tm, tn, tk = (int(parts[0]), int(parts[1]), int(parts[2]))
    return TileConfig(tile_m=tm, tile_n=tn, tile_k=tk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--password", default=None, help="SSH password (prefer env INTENTIR_SSH_PASSWORD or prompt)")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument("--bench-warmup", type=int, default=2)
    ap.add_argument("--tiles-limit", type=int, default=10)
    ap.add_argument("--tile", action="append", default=[], help="repeatable; format: tm,tn,tk (overrides auto tile set)")
    ap.add_argument("--profile", default=None, help="RVV profile name or JSON path (default: probe remote host)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    password = args.password or os.getenv("INTENTIR_SSH_PASSWORD")
    if password is None:
        password = getpass.getpass(f"SSH password for {args.user}@{args.host}: ")

    tiles: List[TileConfig] | None = None
    if args.tile:
        tiles = [_parse_tile(t) for t in args.tile]

    measurements, stats = validate_gemm_cost_model_remote(
        host=str(args.host),
        user=str(args.user),
        password=str(password),
        port=int(args.port),
        M=int(args.M),
        N=int(args.N),
        K=int(args.K),
        bench_iters=int(args.bench_iters),
        bench_warmup=int(args.bench_warmup),
        tiles=tiles,
        tiles_limit=int(args.tiles_limit),
        profile_name_or_path=(str(args.profile) if args.profile else None),
        seed=int(args.seed),
    )

    print("tile_m,tile_n,tile_k,pred_gflops,measured_gflops,ns_per_iter")
    for m in measurements:
        print(
            f"{m.tile.tile_m},{m.tile.tile_n},{m.tile.tile_k},{m.pred_gflops:.6f},"
            f"{'' if m.measured_gflops is None else f'{m.measured_gflops:.6f}'},"
            f"{'' if m.ns_per_iter is None else f'{m.ns_per_iter:.1f}'}"
        )
    print(f"# spearman_r={stats.get('spearman_r')} n={stats.get('n')}")


if __name__ == "__main__":
    main()

