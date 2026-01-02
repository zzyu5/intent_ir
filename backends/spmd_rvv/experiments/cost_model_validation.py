from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import paramiko

from backends.spmd_rvv.analysis.cost_model import GEMMCostModel
from backends.spmd_rvv.analysis.device_query import load_profile, query_remote_device
from backends.spmd_rvv.analysis.hardware_profile import RVVHardwareProfile
from backends.spmd_rvv.codegen.intentir_to_c import lower_intent_to_c_with_files
from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType


@dataclass(frozen=True)
class TileConfig:
    tile_m: int
    tile_n: int
    tile_k: int


@dataclass(frozen=True)
class TileMeasurement:
    tile: TileConfig
    pred_gflops: float
    measured_gflops: Optional[float]
    ns_per_iter: Optional[float]
    bench: Optional[dict]
    run_rc: int


def _rankdata(xs: Sequence[float]) -> List[float]:
    """
    Assign average ranks for ties (1-based ranks).
    """
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i + 1
        while j < len(idx) and xs[idx[j]] == xs[idx[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[idx[k]] = avg
        i = j
    return ranks


def spearman_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(len(rx)))
    denx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    deny = math.sqrt(sum((r - my) ** 2 for r in ry))
    if denx == 0.0 or deny == 0.0:
        return float("nan")
    return num / (denx * deny)


def _rm_layout() -> TensorLayout:
    return TensorLayout(kind="row_major", params={})


def build_gemm_intent(*, M: int, N: int, K: int, tile: TileConfig, vec_width: int) -> IntentFunction:
    rm = _rm_layout()
    tensors = {
        "A": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "K")], layout=rm),
        "B": TensorType(dtype="f32", shape=[Dim("sym", "K"), Dim("sym", "N")], layout=rm),
        "C": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
    }
    ops = [Op(op="matmul", inputs=["A", "B"], output="C", attrs={})]
    schedule = ScheduleSketch(
        tile_m=int(tile.tile_m),
        tile_n=int(tile.tile_n),
        tile_k=int(tile.tile_k),
        vec_width=int(vec_width),
        pipeline_depth=1,
    )
    return IntentFunction(
        name="gemm_cost_model_validation",
        tensors=tensors,
        ops=ops,
        outputs=["C"],
        schedule=schedule,
        axis_roles={"M": "spatial", "N": "spatial", "K": "reduction"},
    )


def _sftp_mkdir_p(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = [p for p in path.split("/") if p]
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def _sftp_write_bytes(sftp: paramiko.SFTPClient, path: str, data: bytes) -> None:
    with sftp.file(path, "wb") as f:
        f.write(data)


def _parse_bench(stdout: str) -> Optional[dict]:
    for ln in str(stdout).splitlines():
        if ln.startswith("INTENTIR_BENCH "):
            try:
                return json.loads(ln[len("INTENTIR_BENCH ") :].strip())
            except Exception:
                return None
    return None


def _gen_tiles(profile: RVVHardwareProfile, *, M: int, N: int, K: int, limit: int) -> List[TileConfig]:
    vec_lanes = max(1, int(profile.rvv_vlen_bits) // 32)
    tm_list = [16, 32, 64, 128, 256]
    tn_list = [vec_lanes, 32, 64, 128, 256]
    tk_list = [16, 32, 64, 128]
    tm_list = [t for t in tm_list if 0 < t <= M] or [max(1, min(16, M))]
    tn_list = [t for t in tn_list if 0 < t <= N] or [max(1, min(vec_lanes, N))]
    tk_list = [t for t in tk_list if 0 < t <= K] or [max(1, min(16, K))]

    tiles: List[TileConfig] = []
    for tm in tm_list:
        for tn in tn_list:
            for tk in tk_list:
                ws_bytes = (tm * tk + tk * tn + tm * tn) * 4
                if ws_bytes <= int(profile.l2_cache_kb) * 1024:
                    tiles.append(TileConfig(tile_m=tm, tile_n=tn, tile_k=tk))
    if not tiles:
        tiles = [TileConfig(tile_m=tm_list[0], tile_n=tn_list[0], tile_k=tk_list[0])]
    # Diverse: sort by (predicted gflops desc) and take top-k, then add a couple of small baselines.
    model = GEMMCostModel(profile, M=M, N=N, K=K)
    tiles_sorted = sorted(tiles, key=lambda t: model.evaluate_tile(t.tile_m, t.tile_n, t.tile_k).gflops, reverse=True)
    out = tiles_sorted[: max(1, limit)]
    # Ensure we include the smallest tile too (for a low-end point).
    smallest = min(tiles, key=lambda t: (t.tile_m * t.tile_n * t.tile_k, t.tile_m, t.tile_n, t.tile_k))
    if smallest not in out:
        out.append(smallest)
    return out[:limit]


def _upload_gemm_io(
    sftp: paramiko.SFTPClient,
    remote_dir: str,
    *,
    A: np.ndarray,
    B: np.ndarray,
    C_ref: np.ndarray,
) -> None:
    _sftp_write_bytes(sftp, f"{remote_dir}/A.bin", np.asarray(A, dtype=np.float32).tobytes(order="C"))
    _sftp_write_bytes(sftp, f"{remote_dir}/B.bin", np.asarray(B, dtype=np.float32).tobytes(order="C"))
    _sftp_write_bytes(sftp, f"{remote_dir}/C_ref.bin", np.asarray(C_ref, dtype=np.float32).tobytes(order="C"))


def validate_gemm_cost_model_remote(
    *,
    host: str,
    user: str,
    password: str | None,
    port: int = 22,
    M: int = 256,
    N: int = 256,
    K: int = 256,
    bench_iters: int = 20,
    bench_warmup: int = 2,
    tiles: Optional[Sequence[TileConfig]] = None,
    tiles_limit: int = 10,
    profile_name_or_path: Optional[str] = None,
    seed: int = 0,
) -> Tuple[List[TileMeasurement], Dict[str, float]]:
    profile = (
        load_profile(profile_name_or_path)
        if profile_name_or_path
        else query_remote_device(host, user=user, password=password, port=port, timeout=30)
    )
    vec_lanes = max(1, int(profile.rvv_vlen_bits) // 32)
    tiles_use = list(tiles) if tiles is not None else _gen_tiles(profile, M=int(M), N=int(N), K=int(K), limit=int(tiles_limit))

    rng = np.random.default_rng(int(seed))
    A = rng.standard_normal((int(M), int(K)), dtype=np.float32)
    B = rng.standard_normal((int(K), int(N)), dtype=np.float32)
    C_ref = A @ B

    model = GEMMCostModel(profile, M=int(M), N=int(N), K=int(K))

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=user, password=password, timeout=30)
    sftp = client.open_sftp()
    remote_base = "/tmp/intentir_cost_model_validate"
    _sftp_mkdir_p(sftp, remote_base)

    measurements: List[TileMeasurement] = []
    try:
        for tile in tiles_use:
            intent = build_gemm_intent(M=int(M), N=int(N), K=int(K), tile=tile, vec_width=vec_lanes)
            shapes = {"M": int(M), "N": int(N), "K": int(K)}
            c_src = lower_intent_to_c_with_files(intent, shape_bindings=shapes, atol=1e-3, rtol=1e-3)

            remote_dir = f"{remote_base}/tm{tile.tile_m}_tn{tile.tile_n}_tk{tile.tile_k}"
            _sftp_mkdir_p(sftp, remote_dir)
            with sftp.file(f"{remote_dir}/main.c", "w") as f:
                f.write(c_src)
            runtime_dir = Path(__file__).resolve().parents[1] / "runtime"
            runtime_h = runtime_dir / "intentir_runtime.h"
            runtime_c = runtime_dir / "intentir_runtime.c"
            driver_h = runtime_dir / "intentir_driver.h"
            driver_c = runtime_dir / "intentir_driver.c"
            ops_h = runtime_dir / "intentir_ops.h"
            ops_c = runtime_dir / "intentir_ops.c"
            if (
                not runtime_h.exists()
                or not runtime_c.exists()
                or not driver_h.exists()
                or not driver_c.exists()
                or not ops_h.exists()
                or not ops_c.exists()
            ):
                raise FileNotFoundError(
                    f"missing RVV runtime: {runtime_h} / {runtime_c} / {driver_h} / {driver_c} / {ops_h} / {ops_c}"
                )
            with sftp.file(f"{remote_dir}/intentir_runtime.h", "w") as f:
                f.write(runtime_h.read_text(encoding="utf-8"))
            with sftp.file(f"{remote_dir}/intentir_runtime.c", "w") as f:
                f.write(runtime_c.read_text(encoding="utf-8"))
            with sftp.file(f"{remote_dir}/intentir_driver.h", "w") as f:
                f.write(driver_h.read_text(encoding="utf-8"))
            with sftp.file(f"{remote_dir}/intentir_driver.c", "w") as f:
                f.write(driver_c.read_text(encoding="utf-8"))
            with sftp.file(f"{remote_dir}/intentir_ops.h", "w") as f:
                f.write(ops_h.read_text(encoding="utf-8"))
            with sftp.file(f"{remote_dir}/intentir_ops.c", "w") as f:
                f.write(ops_c.read_text(encoding="utf-8"))
            _upload_gemm_io(sftp, remote_dir, A=A, B=B, C_ref=C_ref)

            remote_bin = f"{remote_dir}/run"
            compile_cmd = (
                f"gcc -O3 -std=c11 -march=rv64gcv -I{remote_dir} -o {remote_bin} "
                f"{remote_dir}/main.c {remote_dir}/intentir_runtime.c {remote_dir}/intentir_driver.c {remote_dir}/intentir_ops.c -lm -lrt"
            )
            stdin, stdout, stderr = client.exec_command(compile_cmd, timeout=120)
            _ = stdout.read()
            comp_err = stderr.read().decode("utf-8", errors="replace")
            comp_rc = stdout.channel.recv_exit_status()
            if comp_rc != 0:
                raise RuntimeError(f"remote compile failed rc={comp_rc}: {comp_err}")

            run_cmd = (
                f"cd {remote_dir} && INTENTIR_BENCH_ITERS={int(bench_iters)} "
                f"INTENTIR_BENCH_WARMUP={max(0, int(bench_warmup))} {remote_bin}"
            )
            stdin, stdout, stderr = client.exec_command(run_cmd, timeout=300)
            run_out = stdout.read().decode("utf-8", errors="replace")
            run_err = stderr.read().decode("utf-8", errors="replace")
            run_rc = stdout.channel.recv_exit_status()
            if run_rc != 0:
                raise RuntimeError(f"remote run failed rc={run_rc}: {run_err or run_out}")

            bench = _parse_bench(run_out)
            ns_per_iter = None
            measured_gflops = None
            if isinstance(bench, dict):
                try:
                    ns_per_iter = float(bench.get("ns_per_iter"))
                except Exception:
                    ns_per_iter = None
                try:
                    measured_gflops = float(bench.get("matmul_gflops"))
                except Exception:
                    measured_gflops = None

            pred = model.evaluate_tile(int(tile.tile_m), int(tile.tile_n), int(tile.tile_k)).gflops
            measurements.append(
                TileMeasurement(
                    tile=tile,
                    pred_gflops=float(pred),
                    measured_gflops=(float(measured_gflops) if measured_gflops is not None else None),
                    ns_per_iter=(float(ns_per_iter) if ns_per_iter is not None else None),
                    bench=bench,
                    run_rc=int(run_rc),
                )
            )
    finally:
        try:
            sftp.close()
        except Exception:
            pass
        client.close()

    xs = [m.pred_gflops for m in measurements if m.measured_gflops is not None]
    ys = [m.measured_gflops for m in measurements if m.measured_gflops is not None]
    stats = {
        "spearman_r": float(spearman_r(xs, ys)) if len(xs) >= 2 else float("nan"),
        "n": float(len(xs)),
    }
    return measurements, stats


__all__ = [
    "TileConfig",
    "TileMeasurement",
    "spearman_r",
    "validate_gemm_cost_model_remote",
]
