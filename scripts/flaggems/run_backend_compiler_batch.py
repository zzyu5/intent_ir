"""
Run backend_compiler lane batch using matrix smoke path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], *, dry_run: bool, stream_output: bool) -> tuple[int, str, str]:
    if dry_run:
        return 0, "(dry-run)", ""
    if not bool(stream_output):
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        return int(p.returncode), str(p.stdout or ""), str(p.stderr or "")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    merged: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        merged.append(line)
        print(line, end="", flush=True)
    rc = int(proc.wait())
    return rc, "".join(merged), ""


def _default_out_dir() -> Path:
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_name = f"backend_compiler_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    return ROOT / "artifacts" / "flaggems_matrix" / "daily" / date_tag / run_name


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_kernel_manifest(path: Path | None) -> list[str]:
    if path is None:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    payload = _load_json(p)
    raw = payload.get("kernels")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for k in raw:
        name = str(k).strip()
        if name and name not in out:
            out.append(name)
    return out


def _guess_baseline_dir(out_dir: Path) -> Path | None:
    current_status = ROOT / "workflow" / "flaggems" / "state" / "current_status.json"
    if not current_status.is_file():
        return None
    payload = _load_json(current_status)
    run_summary = str((payload.get("latest_artifacts") or {}).get("run_summary") or "").strip()
    if not run_summary:
        return None
    p = Path(run_summary)
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        return None
    baseline_dir = p.parent
    if baseline_dir.resolve() == out_dir.resolve():
        return None
    return baseline_dir


def _coverage_kernel_names(*, flaggems_opset: str, backend_target: str) -> set[str]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from pipeline.triton.providers.flaggems.specs import coverage_flaggems_kernel_specs  # noqa: PLC0415

    specs = coverage_flaggems_kernel_specs(
        flaggems_opset=str(flaggems_opset),
        backend_target=str(backend_target),
    )
    return {str(s.name) for s in specs}


def _select_chunk(*, kernels: list[str], chunk_size: int, chunk_index: int) -> tuple[list[str], dict]:
    total = len(kernels)
    if chunk_size <= 0:
        return list(kernels), {
            "enabled": False,
            "chunk_size": 0,
            "chunk_index": 0,
            "chunk_count": 1,
            "kernel_count_total": total,
            "kernel_count_selected": total,
            "slice_start": 0,
            "slice_end": total,
        }
    if chunk_index < 0:
        raise SystemExit("--chunk-index must be >= 0")
    chunk_count = max(1, int(math.ceil(total / float(chunk_size))))
    if chunk_index >= chunk_count:
        raise SystemExit(
            f"--chunk-index out of range: {chunk_index} (chunk_count={chunk_count}, chunk_size={chunk_size}, kernels={total})"
        )
    start = int(chunk_index * chunk_size)
    end = int(min(total, start + int(chunk_size)))
    selected = list(kernels[start:end])
    return selected, {
        "enabled": True,
        "chunk_size": int(chunk_size),
        "chunk_index": int(chunk_index),
        "chunk_count": int(chunk_count),
        "kernel_count_total": total,
        "kernel_count_selected": len(selected),
        "slice_start": start,
        "slice_end": end,
    }


def _results_by_kernel(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    rows = payload.get("results")
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        kernel = str(row.get("kernel") or "").strip()
        if not kernel:
            continue
        out[kernel] = row
    return out


def _kernel_status_by_spec(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in entries:
        if not isinstance(row, dict):
            continue
        spec = str(row.get("e2e_spec") or "").strip()
        if not spec:
            continue
        out[spec] = row
    return out


def _reason_or_ok(row: dict[str, Any]) -> str:
    reason = str(row.get("reason_code") or "").strip()
    if reason:
        return reason
    ok = row.get("ok")
    if isinstance(ok, bool):
        return "ok" if ok else "fail"
    return "unknown"


def _ms_triplet(row: dict[str, Any]) -> str:
    lower = row.get("lower_ms")
    compile_ms = row.get("compile_ms")
    launch = row.get("launch_ms")
    if not isinstance(lower, (int, float)) and not isinstance(compile_ms, (int, float)) and not isinstance(launch, (int, float)):
        return ""
    l = f"{float(lower):.1f}" if isinstance(lower, (int, float)) else "-"
    c = f"{float(compile_ms):.1f}" if isinstance(compile_ms, (int, float)) else "-"
    r = f"{float(launch):.1f}" if isinstance(launch, (int, float)) else "-"
    return f"{l}/{c}/{r}ms"


def _emit_kernel_progress(out_dir: Path, selected_kernels: list[str]) -> None:
    status_by_spec = _kernel_status_by_spec(out_dir / "status_converged.json")
    rvv_path = out_dir / "rvv_remote.json"
    if not rvv_path.is_file():
        rvv_path = out_dir / "rvv_local.json"
    rvv_by_kernel = _results_by_kernel(rvv_path)
    cuda_by_kernel = _results_by_kernel(out_dir / "cuda_local.json")
    if not status_by_spec and not rvv_by_kernel and not cuda_by_kernel:
        print("[backend-batch] kernel progress unavailable (missing status/rvv/cuda artifacts)", flush=True)
        return
    print("[backend-batch] per-kernel result summary:", flush=True)
    for kernel in selected_kernels:
        entry = status_by_spec.get(str(kernel), {})
        rvv = rvv_by_kernel.get(str(kernel), {})
        cuda = cuda_by_kernel.get(str(kernel), {})
        semantic = str(entry.get("semantic_op") or "-")
        status = str(entry.get("status") or "unknown")
        rvv_reason = _reason_or_ok(rvv) if rvv else "missing"
        cuda_reason = _reason_or_ok(cuda) if cuda else "missing"
        rvv_ms = _ms_triplet(rvv) if rvv else ""
        cuda_ms = _ms_triplet(cuda) if cuda else ""
        line = (
            f"  - {kernel}: semantic={semantic} status={status} "
            f"rvv={rvv_reason}{(' ' + rvv_ms) if rvv_ms else ''} "
            f"cuda={cuda_reason}{(' ' + cuda_ms) if cuda_ms else ''}"
        )
        print(line, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=_default_out_dir())
    ap.add_argument("--suite", choices=["smoke", "coverage", "all"], default="smoke")
    ap.add_argument("--kernel", action="append", default=[])
    ap.add_argument("--flaggems-opset", choices=["deterministic_forward"], default="deterministic_forward")
    ap.add_argument("--backend-target", choices=["rvv", "cuda_h100", "cuda_5090d"], default="rvv")
    ap.add_argument(
        "--kernel-manifest",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "backend_kernel_manifest.json"),
        help="Optional kernel manifest JSON with `kernels: [..]` used when --kernel is omitted.",
    )
    ap.add_argument("--chunk-size", type=int, default=0, help="Optional chunk size for manifest kernels.")
    ap.add_argument("--chunk-index", type=int, default=0, help="Chunk index when --chunk-size > 0.")
    ap.add_argument(
        "--validate-kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate requested kernels against deterministic coverage specs (default: true).",
    )
    ap.add_argument("--run-rvv-remote", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rvv-host", default="192.168.8.72")
    ap.add_argument("--rvv-user", default="ubuntu")
    ap.add_argument("--rvv-use-key", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="auto")
    ap.add_argument(
        "--schedule-profile-tag",
        default="wave4",
        help="Profile tag passed to backend schedule override env (default: wave4).",
    )
    ap.add_argument("--cuda-tile-m", type=int, default=None)
    ap.add_argument("--cuda-tile-n", type=int, default=None)
    ap.add_argument("--cuda-tile-k", type=int, default=None)
    ap.add_argument("--rvv-tile-m", type=int, default=None)
    ap.add_argument("--rvv-tile-n", type=int, default=None)
    ap.add_argument("--rvv-tile-k", type=int, default=None)
    ap.add_argument("--allow-cuda-skip", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--emit-timing-delta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit stage timing delta report against previous backend run.",
    )
    ap.add_argument(
        "--stream-subprocess-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream matrix subprocess logs in real time (default: true).",
    )
    ap.add_argument(
        "--print-kernel-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print one-line status for each selected kernel after run (default: true).",
    )
    ap.add_argument("--timing-baseline-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    kernels = [str(k) for k in list(args.kernel or []) if str(k).strip()]
    manifest_kernels = _load_kernel_manifest(args.kernel_manifest)
    if not kernels:
        kernels = list(manifest_kernels or ["add2d", "mul2d"])
    dedup_kernels: list[str] = []
    for k in kernels:
        if k not in dedup_kernels:
            dedup_kernels.append(k)
    if bool(args.validate_kernels):
        valid = _coverage_kernel_names(
            flaggems_opset=str(args.flaggems_opset),
            backend_target=str(args.backend_target),
        )
        unknown = [k for k in dedup_kernels if k not in valid]
        if unknown:
            raise SystemExit(
                f"unknown kernel(s) for deterministic_forward coverage: {', '.join(sorted(set(unknown)))}"
            )
    selected_kernels, chunk_meta = _select_chunk(
        kernels=dedup_kernels,
        chunk_size=int(args.chunk_size),
        chunk_index=int(args.chunk_index),
    )
    if not selected_kernels:
        raise SystemExit("no kernels selected after chunking")

    cmd = [
        sys.executable,
        "scripts/flaggems/run_multibackend_matrix.py",
        "--suite",
        str(args.suite),
        "--lane",
        "backend_compiler",
        "--flaggems-opset",
        str(args.flaggems_opset),
        "--backend-target",
        str(args.backend_target),
        "--flaggems-path",
        "intentir",
        "--intentir-mode",
        "auto",
        "--out-dir",
        str(args.out_dir),
        "--cuda-runtime-backend",
        str(args.cuda_runtime_backend),
        "--schedule-profile-tag",
        str(args.schedule_profile_tag),
        "--rvv-host",
        str(args.rvv_host),
        "--rvv-user",
        str(args.rvv_user),
    ]
    if args.cuda_tile_m is not None:
        cmd += ["--cuda-tile-m", str(int(args.cuda_tile_m))]
    if args.cuda_tile_n is not None:
        cmd += ["--cuda-tile-n", str(int(args.cuda_tile_n))]
    if args.cuda_tile_k is not None:
        cmd += ["--cuda-tile-k", str(int(args.cuda_tile_k))]
    if args.rvv_tile_m is not None:
        cmd += ["--rvv-tile-m", str(int(args.rvv_tile_m))]
    if args.rvv_tile_n is not None:
        cmd += ["--rvv-tile-n", str(int(args.rvv_tile_n))]
    if args.rvv_tile_k is not None:
        cmd += ["--rvv-tile-k", str(int(args.rvv_tile_k))]
    if bool(args.run_rvv_remote):
        cmd.append("--run-rvv-remote")
    else:
        cmd.append("--no-run-rvv-remote")
    if bool(args.rvv_use_key):
        cmd.append("--rvv-use-key")
    else:
        cmd.append("--no-rvv-use-key")
    if bool(args.allow_cuda_skip):
        cmd.append("--allow-cuda-skip")
    else:
        cmd.append("--no-allow-cuda-skip")
    if bool(args.stream_subprocess_output):
        cmd.append("--stream-subprocess-output")
    else:
        cmd.append("--no-stream-subprocess-output")
    for k in selected_kernels:
        cmd += ["--kernel", str(k)]

    rc, stdout, stderr = _run(
        cmd,
        dry_run=bool(args.dry_run),
        stream_output=bool(args.stream_subprocess_output),
    )
    timing_delta_path = ""
    timing_delta_ok = True
    timing_delta_stderr = ""
    schedule_profiles_path = ""
    schedule_profiles_ok = True
    schedule_profiles_stderr = ""
    stage_timing_breakdown_path = ""
    stage_timing_breakdown_ok = True
    stage_timing_breakdown_stderr = ""
    baseline_dir = Path(args.timing_baseline_dir) if args.timing_baseline_dir is not None else _guess_baseline_dir(Path(args.out_dir))
    if bool(args.emit_timing_delta) and not bool(args.dry_run) and int(rc) == 0:
        current_rvv = Path(args.out_dir) / "rvv_local.json"
        current_cuda = Path(args.out_dir) / "cuda_local.json"
        baseline_rvv = (baseline_dir / "rvv_local.json") if baseline_dir is not None else None
        baseline_cuda = (baseline_dir / "cuda_local.json") if baseline_dir is not None else None
        timing_delta = Path(args.out_dir) / "timing_delta.json"
        delta_cmd = [
            sys.executable,
            "scripts/flaggems/compute_stage_timing_delta.py",
            "--current-rvv",
            str(current_rvv),
            "--current-cuda",
            str(current_cuda),
            "--out",
            str(timing_delta),
        ]
        if baseline_rvv is not None:
            delta_cmd += ["--baseline-rvv", str(baseline_rvv)]
        if baseline_cuda is not None:
            delta_cmd += ["--baseline-cuda", str(baseline_cuda)]
        rc_delta, _out_delta, err_delta = _run(
            delta_cmd,
            dry_run=False,
            stream_output=False,
        )
        timing_delta_ok = int(rc_delta) == 0
        timing_delta_stderr = str(err_delta).strip()
        timing_delta_path = str(timing_delta)
        run_summary_path = Path(args.out_dir) / "run_summary.json"
        if run_summary_path.is_file():
            run_summary = _load_json(run_summary_path)
            stages = [s for s in list(run_summary.get("stages") or []) if isinstance(s, dict)]
            stages = [s for s in stages if str(s.get("stage") or "") != "timing_delta"]
            stages.append(
                {
                    "stage": "timing_delta",
                    "rc": 0 if bool(timing_delta_ok) else 1,
                    "ok": bool(timing_delta_ok),
                    "stdout": f"timing delta generated at {timing_delta_path}" if timing_delta_path else "",
                    "stderr": str(timing_delta_stderr),
                    "cmd": delta_cmd,
                    "json_path": timing_delta_path,
                    "baseline_dir": str(baseline_dir) if baseline_dir is not None else "",
                }
            )
            run_summary["stages"] = stages
            run_summary["ok"] = bool(run_summary.get("ok")) and bool(timing_delta_ok)
            run_summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if not bool(args.dry_run) and int(rc) == 0:
        schedule_profiles = Path(args.out_dir) / "schedule_profiles.json"
        profiles_cmd = [
            sys.executable,
            "scripts/flaggems/export_schedule_profiles.py",
            "--out",
            str(schedule_profiles),
            "--schedule-profile-tag",
            str(args.schedule_profile_tag),
        ]
        if args.cuda_tile_m is not None:
            profiles_cmd += ["--cuda-tile-m", str(int(args.cuda_tile_m))]
        if args.cuda_tile_n is not None:
            profiles_cmd += ["--cuda-tile-n", str(int(args.cuda_tile_n))]
        if args.cuda_tile_k is not None:
            profiles_cmd += ["--cuda-tile-k", str(int(args.cuda_tile_k))]
        if args.rvv_tile_m is not None:
            profiles_cmd += ["--rvv-tile-m", str(int(args.rvv_tile_m))]
        if args.rvv_tile_n is not None:
            profiles_cmd += ["--rvv-tile-n", str(int(args.rvv_tile_n))]
        if args.rvv_tile_k is not None:
            profiles_cmd += ["--rvv-tile-k", str(int(args.rvv_tile_k))]
        rc_profiles, _out_profiles, err_profiles = _run(
            profiles_cmd,
            dry_run=False,
            stream_output=False,
        )
        schedule_profiles_ok = int(rc_profiles) == 0
        schedule_profiles_stderr = str(err_profiles).strip()
        schedule_profiles_path = str(schedule_profiles)
        run_summary_path = Path(args.out_dir) / "run_summary.json"
        if run_summary_path.is_file():
            run_summary = _load_json(run_summary_path)
            stages = [s for s in list(run_summary.get("stages") or []) if isinstance(s, dict)]
            stages = [s for s in stages if str(s.get("stage") or "") != "schedule_profiles"]
            stages.append(
                {
                    "stage": "schedule_profiles",
                    "rc": 0 if bool(schedule_profiles_ok) else 1,
                    "ok": bool(schedule_profiles_ok),
                    "stdout": f"schedule profiles generated at {schedule_profiles_path}" if schedule_profiles_path else "",
                    "stderr": str(schedule_profiles_stderr),
                    "cmd": profiles_cmd,
                    "json_path": schedule_profiles_path,
                }
            )
            run_summary["stages"] = stages
            run_summary["ok"] = bool(run_summary.get("ok")) and bool(schedule_profiles_ok)
            run_summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if not bool(args.dry_run) and int(rc) == 0:
        stage_timing_breakdown = Path(args.out_dir) / "stage_timing_breakdown.json"
        breakdown_cmd = [
            sys.executable,
            "scripts/flaggems/compute_stage_timing_breakdown.py",
            "--rvv-json",
            str(Path(args.out_dir) / "rvv_local.json"),
            "--cuda-json",
            str(Path(args.out_dir) / "cuda_local.json"),
            "--out",
            str(stage_timing_breakdown),
        ]
        rc_breakdown, _out_breakdown, err_breakdown = _run(
            breakdown_cmd,
            dry_run=False,
            stream_output=False,
        )
        stage_timing_breakdown_ok = int(rc_breakdown) == 0
        stage_timing_breakdown_stderr = str(err_breakdown).strip()
        stage_timing_breakdown_path = str(stage_timing_breakdown)
        run_summary_path = Path(args.out_dir) / "run_summary.json"
        if run_summary_path.is_file():
            run_summary = _load_json(run_summary_path)
            stages = [s for s in list(run_summary.get("stages") or []) if isinstance(s, dict)]
            stages = [s for s in stages if str(s.get("stage") or "") != "stage_timing_breakdown"]
            stages.append(
                {
                    "stage": "stage_timing_breakdown",
                    "rc": 0 if bool(stage_timing_breakdown_ok) else 1,
                    "ok": bool(stage_timing_breakdown_ok),
                    "stdout": (
                        f"stage timing breakdown generated at {stage_timing_breakdown_path}"
                        if stage_timing_breakdown_path
                        else ""
                    ),
                    "stderr": str(stage_timing_breakdown_stderr),
                    "cmd": breakdown_cmd,
                    "json_path": stage_timing_breakdown_path,
                }
            )
            run_summary["stages"] = stages
            run_summary["ok"] = bool(run_summary.get("ok")) and bool(stage_timing_breakdown_ok)
            run_summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "ok": bool(rc == 0 and timing_delta_ok and schedule_profiles_ok and stage_timing_breakdown_ok),
        "lane": "backend_compiler",
        "cmd": cmd,
        "out_dir": str(args.out_dir),
        "kernel_manifest": str(args.kernel_manifest),
        "manifest_kernel_count": len(manifest_kernels),
        "kernel_count_requested": len(dedup_kernels),
        "kernel_count_selected": len(selected_kernels),
        "kernels_selected": list(selected_kernels),
        "chunk": dict(chunk_meta),
        "flaggems_opset": str(args.flaggems_opset),
        "backend_target": str(args.backend_target),
        "cuda_runtime_backend": str(args.cuda_runtime_backend),
        "schedule_profile_tag": str(args.schedule_profile_tag),
        "cuda_tile_m": (int(args.cuda_tile_m) if args.cuda_tile_m is not None else None),
        "cuda_tile_n": (int(args.cuda_tile_n) if args.cuda_tile_n is not None else None),
        "cuda_tile_k": (int(args.cuda_tile_k) if args.cuda_tile_k is not None else None),
        "rvv_tile_m": (int(args.rvv_tile_m) if args.rvv_tile_m is not None else None),
        "rvv_tile_n": (int(args.rvv_tile_n) if args.rvv_tile_n is not None else None),
        "rvv_tile_k": (int(args.rvv_tile_k) if args.rvv_tile_k is not None else None),
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "timing_delta": {
            "ok": bool(timing_delta_ok),
            "path": timing_delta_path,
            "baseline_dir": str(baseline_dir) if baseline_dir is not None else "",
            "stderr": timing_delta_stderr,
        },
        "schedule_profiles": {
            "ok": bool(schedule_profiles_ok),
            "path": schedule_profiles_path,
            "stderr": schedule_profiles_stderr,
        },
        "stage_timing_breakdown": {
            "ok": bool(stage_timing_breakdown_ok),
            "path": stage_timing_breakdown_path,
            "stderr": stage_timing_breakdown_stderr,
        },
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batch_summary = out / "backend_compiler_batch_summary.json"
    batch_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"backend_compiler batch summary written: {batch_summary}")
    if bool(args.print_kernel_progress) and not bool(args.dry_run):
        _emit_kernel_progress(out, selected_kernels)
    raise SystemExit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
