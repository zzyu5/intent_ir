"""
Run full coverage in fixed family batches and aggregate to one full196 evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _utc_date_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str], *, stream_output: bool) -> tuple[int, str, str]:
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


def _normalize_family_list(raw: list[str]) -> list[str]:
    out: list[str] = []
    for item in list(raw or []):
        fam = str(item).strip()
        if fam and fam not in out:
            out.append(fam)
    return out


def _family_dir(out_root: Path, family: str) -> Path:
    return out_root / f"family_{str(family)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coverage-batches",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "coverage_batches.json"),
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=(ROOT / "artifacts" / "flaggems_matrix" / "daily" / _utc_date_tag() / "coverage_categories"),
    )
    ap.add_argument("--family", action="append", default=[], help="Optional subset of family names.")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--suite", choices=["coverage"], default="coverage")
    ap.add_argument("--cases-limit", type=int, default=8)
    ap.add_argument("--flaggems-path", choices=["original", "intentir"], default="intentir")
    ap.add_argument("--intentir-mode", choices=["auto", "force_compile", "force_cache"], default="force_compile")
    ap.add_argument("--intentir-miss-policy", choices=["deterministic", "strict"], default="strict")
    ap.add_argument("--flaggems-opset", choices=["deterministic_forward"], default="deterministic_forward")
    ap.add_argument("--backend-target", choices=["rvv", "cuda_h100", "cuda_5090d"], default="rvv")
    ap.add_argument("--run-rvv-remote", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rvv-host", default="192.168.8.72")
    ap.add_argument("--rvv-user", default="ubuntu")
    ap.add_argument("--rvv-port", type=int, default=22)
    ap.add_argument("--rvv-use-key", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow-cuda-skip", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cuda-timeout-sec", type=int, default=120)
    ap.add_argument("--cuda-compile-timeout-sec", type=int, default=120)
    ap.add_argument("--cuda-launch-timeout-sec", type=int, default=120)
    ap.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="nvrtc")
    ap.add_argument("--write-registry", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--stream-subprocess-output", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--aggregate", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    payload = _load_json(args.coverage_batches)
    family_order = [str(x).strip() for x in list(payload.get("family_order") or []) if str(x).strip()]
    by_family = {
        str(b.get("family") or "").strip(): b
        for b in list(payload.get("batches") or [])
        if isinstance(b, dict) and str(b.get("family") or "").strip()
    }
    requested_families = _normalize_family_list(list(args.family or []))
    if requested_families:
        families = requested_families
    else:
        families = [f for f in family_order if f in by_family]
    if not families:
        raise SystemExit("no coverage families selected")

    unknown = [f for f in families if f not in by_family]
    if unknown:
        raise SystemExit(f"unknown family name(s): {', '.join(unknown)}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    family_rows: list[dict[str, Any]] = []
    total_families = len(families)
    for idx, family in enumerate(families, start=1):
        batch = by_family[family]
        kernels = [str(k).strip() for k in list(batch.get("kernels") or []) if str(k).strip()]
        semantics = [str(s).strip() for s in list(batch.get("semantic_ops") or []) if str(s).strip()]
        if not kernels:
            family_rows.append(
                {
                    "family": family,
                    "ok": False,
                    "rc": 1,
                    "reason": "family has no kernels",
                    "semantic_count": int(len(semantics)),
                    "kernel_count": 0,
                    "out_dir": str(_family_dir(out_root, family)),
                    "skipped": False,
                }
            )
            print(f"[{idx}/{total_families}] family={family} has no kernels; mark failed", flush=True)
            continue

        family_out = _family_dir(out_root, family)
        family_out.mkdir(parents=True, exist_ok=True)
        run_summary_path = family_out / "run_summary.json"
        status_path = family_out / "status_converged.json"

        if bool(args.resume) and run_summary_path.is_file() and status_path.is_file():
            try:
                run_summary_payload = _load_json(run_summary_path)
                resume_ok = bool(run_summary_payload.get("ok"))
            except Exception:
                resume_ok = False
            if resume_ok:
                print(f"[{idx}/{total_families}] SKIP family={family} (resume hit)", flush=True)
                family_rows.append(
                    {
                        "family": family,
                        "ok": True,
                        "rc": 0,
                        "reason": "resume_hit",
                        "semantic_count": int(len(semantics)),
                        "kernel_count": int(len(kernels)),
                        "out_dir": str(family_out),
                        "run_summary_path": str(run_summary_path),
                        "status_converged_path": str(status_path),
                        "skipped": True,
                    }
                )
                continue

        print(
            f"[{idx}/{total_families}] RUN family={family} kernels={len(kernels)} semantics={len(semantics)}",
            flush=True,
        )
        cmd = [
            sys.executable,
            "scripts/flaggems/run_multibackend_matrix.py",
            "--suite",
            str(args.suite),
            "--lane",
            "coverage",
            "--cases-limit",
            str(int(args.cases_limit)),
            "--flaggems-path",
            str(args.flaggems_path),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            str(args.intentir_miss_policy),
            "--flaggems-opset",
            str(args.flaggems_opset),
            "--backend-target",
            str(args.backend_target),
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--cuda-timeout-sec",
            str(int(args.cuda_timeout_sec)),
            "--cuda-compile-timeout-sec",
            str(int(args.cuda_compile_timeout_sec)),
            "--cuda-launch-timeout-sec",
            str(int(args.cuda_launch_timeout_sec)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--out-dir",
            str(family_out),
        ]
        cmd.append("--run-rvv-remote" if bool(args.run_rvv_remote) else "--no-run-rvv-remote")
        cmd.append("--rvv-use-key" if bool(args.rvv_use_key) else "--no-rvv-use-key")
        cmd.append("--allow-cuda-skip" if bool(args.allow_cuda_skip) else "--no-allow-cuda-skip")
        if bool(args.write_registry):
            cmd.append("--write-registry")
        for kernel in kernels:
            cmd += ["--kernel", str(kernel)]

        rc, out, err = _run(cmd, stream_output=bool(args.stream_subprocess_output))
        ok = int(rc) == 0
        family_rows.append(
            {
                "family": family,
                "ok": bool(ok),
                "rc": int(rc),
                "semantic_count": int(len(semantics)),
                "kernel_count": int(len(kernels)),
                "out_dir": str(family_out),
                "run_summary_path": str(run_summary_path),
                "status_converged_path": str(status_path),
                "stdout": str(out).strip(),
                "stderr": str(err).strip(),
                "skipped": False,
            }
        )
        print(
            f"[{idx}/{total_families}] DONE family={family} rc={rc} run_summary={run_summary_path}",
            flush=True,
        )

    runs_payload = {
        "schema_version": "flaggems_coverage_batch_runs_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "coverage_batches_path": str(args.coverage_batches),
        "out_root": str(out_root),
        "families": family_rows,
        "ok": bool(all(bool(r.get("ok")) for r in family_rows)),
    }
    runs_summary_path = out_root / "coverage_batch_runs.json"
    runs_summary_path.write_text(json.dumps(runs_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Coverage batch runs written: {runs_summary_path}", flush=True)

    aggregate_rc = 0
    if bool(args.aggregate):
        aggregate_cmd = [
            sys.executable,
            "scripts/flaggems/aggregate_coverage_batches.py",
            "--coverage-batches",
            str(args.coverage_batches),
            "--runs-root",
            str(out_root),
            "--intentir-mode",
            str(args.intentir_mode),
        ]
        print("[coverage-batches] aggregate full196 evidence", flush=True)
        aggregate_rc, _, _ = _run(aggregate_cmd, stream_output=bool(args.stream_subprocess_output))

    ok = bool(runs_payload["ok"]) and bool(aggregate_rc == 0)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
