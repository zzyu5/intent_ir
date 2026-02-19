"""
Run full coverage in fixed family batches and aggregate to one full196 evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

try:  # Optional dependency; plain progress remains available without tqdm.
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _utc_date_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(
    cmd: list[str],
    *,
    stream_output: bool,
    dry_run: bool,
    heartbeat_label: str = "",
    heartbeat_sec: int = 30,
) -> tuple[int, str, str]:
    if bool(dry_run):
        print(f"[dry-run] {' '.join(cmd)}", flush=True)
        return 0, "(dry-run)", ""

    if not bool(stream_output):
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        next_beat = time.monotonic() + max(5, int(heartbeat_sec))
        while True:
            rc = proc.poll()
            if rc is not None:
                out, err = proc.communicate()
                return int(rc), str(out or ""), str(err or "")
            if heartbeat_label and time.monotonic() >= next_beat:
                print(f"[progress] {heartbeat_label} status=RUNNING", flush=True)
                next_beat = time.monotonic() + max(5, int(heartbeat_sec))
            time.sleep(1.0)

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


def _with_env_prefix(cmd: list[str], env_map: dict[str, str] | None) -> list[str]:
    if not env_map:
        return list(cmd)
    pairs = [f"{k}={v}" for k, v in sorted(env_map.items()) if str(k).strip() and str(v).strip()]
    if not pairs:
        return list(cmd)
    return ["env", *pairs, *list(cmd)]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")


def _normalize_family_list(raw: list[str]) -> list[str]:
    out: list[str] = []
    for item in list(raw or []):
        fam = str(item).strip()
        if fam and fam not in out:
            out.append(fam)
    return out


def _family_dir(out_root: Path, family: str) -> Path:
    return out_root / f"family_{str(family)}"


def _chunk_kernels(kernels: list[str], chunk_size: int) -> list[list[str]]:
    if int(chunk_size) <= 0 or len(kernels) <= int(chunk_size):
        return [list(kernels)]
    out: list[list[str]] = []
    step = int(chunk_size)
    for i in range(0, len(kernels), step):
        out.append(list(kernels[i : i + step]))
    return out


def _entry_status_rank(status: str) -> int:
    rank = {
        "dual_pass": 6,
        "rvv_only": 5,
        "cuda_only": 4,
        "blocked_backend": 3,
        "blocked_ir": 2,
        "unknown": 1,
    }
    return int(rank.get(str(status), 0))


def _entry_quality(entry: dict[str, Any]) -> tuple[int, int, int, int, int]:
    status_rank = _entry_status_rank(str(entry.get("status") or ""))
    artifact_complete = 1 if bool(entry.get("artifact_complete")) else 0
    determinable = 1 if bool(entry.get("determinability")) else 0
    in_scope_alias = 1 if bool(entry.get("in_scope_kernel_alias")) else 0
    reason_code = str(entry.get("reason_code") or "").strip()
    reason_penalty = 0 if reason_code == "provider_report_missing" else 1
    return (status_rank, artifact_complete, determinable, in_scope_alias, reason_penalty)


def _counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for row in list(entries or []):
        c[str(row.get("status") or "unknown")] += 1
    return {k: int(v) for k, v in sorted(c.items(), key=lambda kv: kv[0])}


def _placeholder_entry(*, semantic_op: str, family: str, reason_detail: str) -> dict[str, Any]:
    reason = "pipeline_missing_report"
    return {
        "semantic_op": str(semantic_op),
        "family": str(family),
        "status": "blocked_backend",
        "reason_code": str(reason),
        "status_reason": str(reason),
        "status_reason_detail": str(reason_detail),
        "runtime": {"provider": "missing", "rvv": "unknown", "cuda": "unknown"},
        "runtime_detail": {
            "rvv": {"reason_code": str(reason), "reason_detail": str(reason_detail)},
            "cuda": {"reason_code": str(reason), "reason_detail": str(reason_detail)},
        },
        "compiler_stage": {
            "provider_report": "missing",
            "rvv_result": "missing",
            "cuda_result": "missing",
        },
        "artifact_complete": False,
        "determinability": False,
    }


def _materialize_family_outputs(
    *,
    family: str,
    semantics: list[str],
    kernels: list[str],
    family_out: Path,
    chunk_rows: list[dict[str, Any]],
) -> tuple[bool, Path, Path]:
    semantic_set = set(semantics)
    merged_by_semantic: dict[str, dict[str, Any]] = {}
    all_chunks_ok = True
    for row in list(chunk_rows or []):
        all_chunks_ok = bool(all_chunks_ok and bool(row.get("ok")))
        status_path = Path(str(row.get("status_converged_path") or ""))
        if not status_path.is_file():
            all_chunks_ok = False
            continue
        try:
            payload = _load_json(status_path)
        except Exception:
            all_chunks_ok = False
            continue
        for entry in list(payload.get("entries") or []):
            if not isinstance(entry, dict):
                continue
            sop = str(entry.get("semantic_op") or "").strip()
            if not sop or sop not in semantic_set:
                continue
            prev = merged_by_semantic.get(sop)
            if prev is None or _entry_quality(entry) > _entry_quality(prev):
                merged_by_semantic[sop] = entry

    final_entries: list[dict[str, Any]] = []
    missing_semantics: list[str] = []
    non_dual_semantics: list[str] = []
    for sop in semantics:
        entry = merged_by_semantic.get(sop)
        if entry is None:
            missing_semantics.append(sop)
            entry = _placeholder_entry(
                semantic_op=sop,
                family=family,
                reason_detail=f"missing semantic entry after chunk merge for family={family}",
            )
        if str(entry.get("status") or "") != "dual_pass":
            non_dual_semantics.append(sop)
        final_entries.append(entry)

    family_ok = bool(all_chunks_ok and not missing_semantics and not non_dual_semantics)
    status_path = family_out / "status_converged.json"
    run_summary_path = family_out / "run_summary.json"
    status_payload = {
        "schema_version": "flaggems_status_converged_v3",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope_enabled": False,
        "entries": final_entries,
        "counts_global": _counts(final_entries),
        "counts_scoped": _counts(final_entries),
        "counts_scoped_active": _counts(final_entries),
        "counts_scoped_kernel_alias": _counts(final_entries),
        "global_entries_count": int(len(final_entries)),
        "scoped_entries_count": int(len(final_entries)),
        "scoped_entries_active_count": int(len(final_entries)),
        "scoped_entries_kernel_alias_count": int(len(final_entries)),
    }
    status_path.write_text(json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    run_payload = {
        "ok": bool(family_ok),
        "suite": "coverage",
        "requested_suite": "coverage",
        "kernel_filter": list(kernels),
        "scope_kernels": list(kernels),
        "coverage_mode": "category_batches",
        "full196_evidence_kind": "batch_aggregate",
        "family": str(family),
        "chunk_count": int(len(chunk_rows)),
        "missing_semantics": list(missing_semantics),
        "non_dual_semantics": list(non_dual_semantics),
        "status_converged_path": str(status_path),
        "chunk_runs": [
            {
                "chunk": str(r.get("chunk") or ""),
                "ok": bool(r.get("ok")),
                "rc": int(r.get("rc", 1)),
                "out_dir": str(r.get("out_dir") or ""),
                "run_summary_path": str(r.get("run_summary_path") or ""),
                "status_converged_path": str(r.get("status_converged_path") or ""),
                "kernel_count": int(r.get("kernel_count", 0)),
                "kernels": list(r.get("kernels") or []),
            }
            for r in list(chunk_rows or [])
        ],
    }
    run_summary_path.write_text(json.dumps(run_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return family_ok, run_summary_path, status_path


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
    ap.add_argument(
        "--execution-ir",
        choices=["intent", "mlir"],
        default=(str(os.getenv("INTENTIR_EXECUTION_IR", "mlir")).strip().lower() or "mlir"),
        help="Execution IR mode propagated to matrix sub-runs (default: env INTENTIR_EXECUTION_IR or mlir).",
    )
    ap.add_argument("--flaggems-path", choices=["original", "intentir"], default="intentir")
    ap.add_argument("--intentir-mode", choices=["auto", "force_compile", "force_cache"], default="auto")
    ap.add_argument("--intentir-miss-policy", choices=["deterministic", "strict"], default="strict")
    ap.add_argument("--flaggems-opset", choices=["deterministic_forward"], default="deterministic_forward")
    ap.add_argument("--backend-target", choices=["rvv", "cuda_h100", "cuda_5090d"], default="rvv")
    ap.add_argument(
        "--seed-cache-dir",
        type=Path,
        default=(ROOT / "artifacts" / "flaggems_seed_cache"),
        help="Stable seed cache shared across runs (default: artifacts/flaggems_seed_cache).",
    )
    ap.add_argument("--run-rvv-remote", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--skip-rvv-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rvv_local stage and rely on rvv_remote + cuda_local (default: true).",
    )
    ap.add_argument("--rvv-host", default="192.168.8.72")
    ap.add_argument("--rvv-user", default="ubuntu")
    ap.add_argument("--rvv-port", type=int, default=22)
    ap.add_argument("--rvv-use-key", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow-cuda-skip", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cuda-timeout-sec", type=int, default=120)
    ap.add_argument("--cuda-compile-timeout-sec", type=int, default=120)
    ap.add_argument("--cuda-launch-timeout-sec", type=int, default=120)
    ap.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="nvrtc")
    ap.add_argument(
        "--family-kernel-chunk-size",
        type=int,
        default=12,
        help="Split each family into chunks with at most N kernels (default: 12, <=0 disables).",
    )
    ap.add_argument("--write-registry", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--stream-subprocess-output", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--progress-style",
        choices=["auto", "tqdm", "plain", "none"],
        default="auto",
        help=(
            "Chunk progress style. `auto` selects `tqdm` on interactive terminals, "
            "otherwise `plain`."
        ),
    )
    ap.add_argument("--aggregate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if str(args.execution_ir) not in {"intent", "mlir"}:
        raise SystemExit(f"unsupported --execution-ir={args.execution_ir!r}")

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
    seed_cache_dir = Path(args.seed_cache_dir)
    seed_cache_dir.mkdir(parents=True, exist_ok=True)

    requested_progress_style = str(args.progress_style).strip().lower()
    progress_style = requested_progress_style
    if progress_style == "auto":
        progress_style = "tqdm" if (sys.stdout.isatty() and tqdm is not None) else "plain"
    if progress_style == "tqdm" and tqdm is None:
        print("[coverage-batches] tqdm unavailable; falling back to plain progress output", flush=True)
        progress_style = "plain"
    print(
        "[coverage-batches] progress-style "
        f"requested={requested_progress_style} selected={progress_style} "
        f"stdout_tty={sys.stdout.isatty()} tqdm_available={tqdm is not None}",
        flush=True,
    )

    family_rows: list[dict[str, Any]] = []
    family_plans: list[dict[str, Any]] = []
    total_families = len(families)
    total_chunks = 0
    for family in families:
        batch = by_family[family]
        kernels = [str(k).strip() for k in list(batch.get("kernels") or []) if str(k).strip()]
        semantics = [str(s).strip() for s in list(batch.get("semantic_ops") or []) if str(s).strip()]
        chunks = _chunk_kernels(kernels, int(args.family_kernel_chunk_size)) if kernels else []
        total_chunks += int(len(chunks))
        family_plans.append(
            {
                "family": family,
                "batch": batch,
                "kernels": kernels,
                "semantics": semantics,
                "chunks": chunks,
            }
        )

    progress_bar = None
    if progress_style == "tqdm":
        progress_bar = tqdm(total=int(total_chunks), desc="full196", unit="chunk", dynamic_ncols=True)

    chunk_done = 0
    chunk_failures: list[dict[str, Any]] = []
    for idx, plan in enumerate(family_plans, start=1):
        family = str(plan["family"])
        kernels = list(plan["kernels"])
        semantics = list(plan["semantics"])
        chunks = list(plan["chunks"])
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
                chunk_done += int(len(chunks))
                if progress_bar is not None:
                    progress_bar.set_postfix_str(f"{family} resume")
                    progress_bar.update(int(len(chunks)))
                elif progress_style == "plain":
                    print(
                        f"[progress] chunks {chunk_done}/{total_chunks} family={family} status=RESUME",
                        flush=True,
                    )
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

        chunk_enabled = bool(len(chunks) > 1)
        print(
            f"[{idx}/{total_families}] RUN family={family} kernels={len(kernels)} semantics={len(semantics)} "
            f"chunks={len(chunks)} chunk_size={int(args.family_kernel_chunk_size)}",
            flush=True,
        )

        chunk_rows: list[dict[str, Any]] = []
        for chunk_idx, chunk_kernels in enumerate(chunks, start=1):
            chunk_name = f"chunk_{chunk_idx:03d}"
            chunk_out = (family_out / chunk_name) if chunk_enabled else family_out
            chunk_pipeline_out = chunk_out / "pipeline_reports"
            chunk_out.mkdir(parents=True, exist_ok=True)
            chunk_pipeline_out.mkdir(parents=True, exist_ok=True)
            chunk_run_summary = chunk_out / "run_summary.json"
            chunk_status = chunk_out / "status_converged.json"
            if bool(args.resume) and chunk_run_summary.is_file() and chunk_status.is_file():
                try:
                    chunk_summary_payload = _load_json(chunk_run_summary)
                    chunk_resume_ok = bool(chunk_summary_payload.get("ok"))
                except Exception:
                    chunk_resume_ok = False
                if chunk_resume_ok:
                    print(
                        f"[{idx}/{total_families}] SKIP family={family} {chunk_name} "
                        f"(resume hit kernels={len(chunk_kernels)})",
                        flush=True,
                    )
                    chunk_rows.append(
                        {
                            "family": family,
                            "chunk": chunk_name,
                            "chunk_index": int(chunk_idx),
                            "ok": True,
                            "rc": 0,
                            "reason": "resume_hit",
                            "kernel_count": int(len(chunk_kernels)),
                            "kernels": list(chunk_kernels),
                            "out_dir": str(chunk_out),
                            "run_summary_path": str(chunk_run_summary),
                            "status_converged_path": str(chunk_status),
                            "pipeline_out_dir": str(chunk_pipeline_out),
                            "seed_cache_dir": str(seed_cache_dir),
                            "skipped": True,
                        }
                    )
                    chunk_done += 1
                if progress_bar is not None:
                    progress_bar.set_postfix_str(f"{family} {chunk_idx}/{len(chunks)} resume")
                    progress_bar.update(1)
                if progress_style in {"plain", "tqdm"}:
                    print(
                        f"[progress] chunks {chunk_done}/{total_chunks} family={family} "
                        f"chunk={chunk_idx}/{len(chunks)} status=RESUME",
                        flush=True,
                    )
                    continue

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
                str(chunk_out),
                "--pipeline-out-dir",
                str(chunk_pipeline_out),
                "--seed-cache-dir",
                str(seed_cache_dir),
            ]
            cmd.append("--skip-rvv-local" if bool(args.skip_rvv_local) else "--no-skip-rvv-local")
            cmd.append("--run-rvv-remote" if bool(args.run_rvv_remote) else "--no-run-rvv-remote")
            cmd.append("--rvv-use-key" if bool(args.rvv_use_key) else "--no-rvv-use-key")
            cmd.append("--allow-cuda-skip" if bool(args.allow_cuda_skip) else "--no-allow-cuda-skip")
            if bool(args.write_registry):
                cmd.append("--write-registry")
            for kernel in chunk_kernels:
                cmd += ["--kernel", str(kernel)]
            cmd_run = _with_env_prefix(cmd, {"INTENTIR_EXECUTION_IR": str(args.execution_ir)})

            print(
                f"  - chunk {chunk_idx}/{len(chunks)} family={family} kernels={len(chunk_kernels)}",
                flush=True,
            )
            rc, out, err = _run(
                cmd_run,
                stream_output=bool(args.stream_subprocess_output),
                dry_run=bool(args.dry_run),
                heartbeat_label=(
                    f"chunks {chunk_done + 1}/{total_chunks} family={family} "
                    f"chunk={chunk_idx}/{len(chunks)}"
                ),
                heartbeat_sec=30,
            )
            chunk_log_dir = chunk_out / "logs"
            stdout_path = chunk_log_dir / "matrix.stdout.log"
            stderr_path = chunk_log_dir / "matrix.stderr.log"
            if str(out):
                _write_text(stdout_path, str(out))
            if str(err):
                _write_text(stderr_path, str(err))
            chunk_ok = int(rc) == 0
            chunk_rows.append(
                {
                    "family": family,
                    "chunk": chunk_name,
                    "chunk_index": int(chunk_idx),
                    "ok": bool(chunk_ok),
                    "rc": int(rc),
                    "kernel_count": int(len(chunk_kernels)),
                    "kernels": list(chunk_kernels),
                    "out_dir": str(chunk_out),
                    "run_summary_path": str(chunk_run_summary),
                    "status_converged_path": str(chunk_status),
                    "stdout_path": str(stdout_path) if str(out) else "",
                    "stderr_path": str(stderr_path) if str(err) else "",
                    "skipped": False,
                    "cmd": cmd_run,
                    "pipeline_out_dir": str(chunk_pipeline_out),
                    "seed_cache_dir": str(seed_cache_dir),
                    "execution_ir": str(args.execution_ir),
                }
            )
            chunk_done += 1
            if progress_bar is not None:
                progress_bar.set_postfix_str(f"{family} {chunk_idx}/{len(chunks)} {'ok' if chunk_ok else 'fail'}")
                progress_bar.update(1)
            if progress_style in {"plain", "tqdm"}:
                print(
                    f"[progress] chunks {chunk_done}/{total_chunks} family={family} "
                    f"chunk={chunk_idx}/{len(chunks)} status={'OK' if chunk_ok else 'FAIL'}",
                    flush=True,
                )
            if not chunk_ok:
                chunk_failures.append(
                    {
                        "family": family,
                        "chunk": chunk_name,
                        "chunk_index": int(chunk_idx),
                        "kernel_count": int(len(chunk_kernels)),
                        "kernels": list(chunk_kernels),
                        "rc": int(rc),
                        "run_summary_path": str(chunk_run_summary),
                        "status_converged_path": str(chunk_status),
                        "stdout_path": str(stdout_path) if str(out) else "",
                        "stderr_path": str(stderr_path) if str(err) else "",
                    }
                )

        # Always materialize a family-level summary/status from chunk outputs.
        # This keeps family runs scoped to the selected semantics even when
        # chunk_count == 1 (no chunk subdirectory), avoiding misleading 196-op
        # full-scope converge payloads for partial family reruns.
        if not bool(args.dry_run):
            family_ok, run_summary_path, status_path = _materialize_family_outputs(
                family=family,
                semantics=semantics,
                kernels=kernels,
                family_out=family_out,
                chunk_rows=chunk_rows,
            )
        else:
            family_ok = bool(all(bool(r.get("ok")) for r in chunk_rows))
            if chunk_rows:
                run_summary_path = Path(str(chunk_rows[-1].get("run_summary_path") or run_summary_path))
                status_path = Path(str(chunk_rows[-1].get("status_converged_path") or status_path))

        family_rows.append(
            {
                "family": family,
                "ok": bool(family_ok),
                "rc": int(0 if family_ok else 1),
                "semantic_count": int(len(semantics)),
                "kernel_count": int(len(kernels)),
                "chunk_count": int(len(chunks)),
                "chunk_size": int(args.family_kernel_chunk_size),
                "chunk_enabled": bool(chunk_enabled),
                "out_dir": str(family_out),
                "run_summary_path": str(run_summary_path),
                "status_converged_path": str(status_path),
                "chunks": chunk_rows,
                "skipped": bool(len(chunk_rows) > 0 and all(bool(r.get("skipped")) for r in chunk_rows)),
                "seed_cache_dir": str(seed_cache_dir),
                "execution_ir": str(args.execution_ir),
            }
        )
        print(
            f"[{idx}/{total_families}] DONE family={family} ok={family_ok} "
            f"chunks={len(chunks)} run_summary={run_summary_path}",
            flush=True,
        )

    if progress_bar is not None:
        progress_bar.close()

    failures_path = out_root / "errors.json"
    failure_payload = {
        "schema_version": "flaggems_coverage_batch_errors_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "error_count": int(len(chunk_failures)),
        "errors": list(chunk_failures),
    }
    failures_path.write_text(json.dumps(failure_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    runs_payload = {
        "schema_version": "flaggems_coverage_batch_runs_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "coverage_batches_path": str(args.coverage_batches),
        "out_root": str(out_root),
        "seed_cache_dir": str(seed_cache_dir),
        "execution_ir": str(args.execution_ir),
        "families": family_rows,
        "errors_path": str(failures_path),
        "error_count": int(len(chunk_failures)),
        "ok": bool(all(bool(r.get("ok")) for r in family_rows)),
    }
    runs_summary_path = out_root / "coverage_batch_runs.json"
    runs_summary_path.write_text(json.dumps(runs_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Coverage batch runs written: {runs_summary_path}", flush=True)
    if int(len(chunk_failures)) > 0:
        print(f"Coverage batch errors written: {failures_path} (count={len(chunk_failures)})", flush=True)

    aggregate_rc = 0
    if bool(args.aggregate) and (not bool(args.dry_run)):
        aggregate_cmd = [
            sys.executable,
            "scripts/flaggems/aggregate_coverage_batches.py",
            "--coverage-batches",
            str(args.coverage_batches),
            "--runs-root",
            str(out_root),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            str(args.intentir_miss_policy),
            "--execution-ir",
            str(args.execution_ir),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--family-kernel-chunk-size",
            str(int(args.family_kernel_chunk_size)),
        ]
        aggregate_cmd.append("--run-rvv-remote" if bool(args.run_rvv_remote) else "--no-run-rvv-remote")
        print("[coverage-batches] aggregate full196 evidence", flush=True)
        aggregate_rc, _, _ = _run(
            aggregate_cmd,
            stream_output=bool(args.stream_subprocess_output),
            dry_run=False,
        )

    ok = bool(runs_payload["ok"]) and bool(aggregate_rc == 0)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
