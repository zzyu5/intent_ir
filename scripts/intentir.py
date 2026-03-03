#!/usr/bin/env python3
"""
Unified IntentIR user-facing CLI.

This provides one stable entrypoint for:
- suite-level runs (full/category/smoke)
- single-kernel runs
- environment inspection
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import os
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: Sequence[str], *, stream: bool, dry_run: bool, env_overrides: dict[str, str] | None = None) -> int:
    pretty = " ".join(str(x) for x in cmd)
    print(f"[intentir] $ {pretty}", flush=True)
    if dry_run:
        return 0
    env = None
    if env_overrides:
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in env_overrides.items()})
    if stream:
        p = subprocess.run(list(cmd), cwd=str(ROOT), env=env)
    else:
        p = subprocess.run(list(cmd), cwd=str(ROOT), capture_output=True, text=True, env=env)
        if p.stdout:
            print(p.stdout, end="")
        if p.stderr:
            print(p.stderr, file=sys.stderr, end="")
    return int(p.returncode)


def _python_cmd(script: str, *extra: str) -> list[str]:
    return [sys.executable, script, *extra]


def _execution_ir_default() -> str:
    return "mlir"


def _infer_full196_artifact_dir() -> str:
    status_path = ROOT / "workflow" / "flaggems" / "state" / "current_status.json"
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    raw = str(payload.get("full196_last_run") or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    try:
        rp = p.resolve()
    except Exception:
        rp = p
    if rp.is_file():
        return str(rp.parent)
    return ""


def _utc_ymd() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d")


def _normalize_cuda_arch(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if s.isdigit():
        return f"sm{s}"
    if s.startswith("sm_"):
        s = s[3:]
    elif s.startswith("sm"):
        s = s[2:]
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"sm{digits}" if digits else ""


def _detect_cuda_arch() -> str:
    env = _normalize_cuda_arch(os.getenv("INTENTIR_CUDA_SM", ""))
    if env:
        return env
    try:  # pragma: no cover - depends on CUDA env
        import torch  # type: ignore

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if isinstance(major, int) and isinstance(minor, int):
                return f"sm{int(major)}{int(minor)}"
    except Exception:
        pass
    return ""


def _default_cuda_wave() -> str:
    env = str(os.getenv("INTENTIR_CUDA_REAL_MLIR_WAVE", "")).strip().lower()
    if env:
        return env
    state = ROOT / "workflow" / "flaggems" / "state"
    best_n = None
    if state.is_dir():
        for p in state.glob("cuda_real_mlir_wave*_kernels.json"):
            m = re.match(r"cuda_real_mlir_wave(\d+)_kernels\.json$", p.name)
            if not m:
                continue
            try:
                n = int(m.group(1))
            except Exception:
                continue
            if best_n is None or n > best_n:
                best_n = n
    return f"wave{best_n}" if best_n is not None else ""


def _parse_candidate(raw: str) -> tuple[str, dict[str, int]]:
    """
    Syntax:
      <kernel_kind>
      <kernel_kind>:K=V,A=B
    """

    s = str(raw or "").strip()
    if not s:
        raise ValueError("empty candidate")
    if ":" in s:
        kind, rest = s.split(":", 1)
    else:
        kind, rest = s, ""
    kernel_kind = str(kind).strip()
    if not kernel_kind:
        raise ValueError(f"invalid candidate={raw!r}: empty kernel_kind")
    bindings: dict[str, int] = {}
    rest = str(rest).strip()
    if rest:
        for part in rest.split(","):
            item = str(part).strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"invalid candidate binding {item!r} (expected K=V)")
            k, v = item.split("=", 1)
            key = str(k).strip()
            val_raw = str(v).strip()
            if not key:
                raise ValueError(f"invalid candidate binding {item!r} (empty key)")
            try:
                bindings[key] = int(val_raw, 0)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"invalid candidate binding {item!r} (non-int value)") from e
    return kernel_kind, bindings


def _candidate_id(kernel_kind: str, bindings: dict[str, int], *, idx: int) -> str:
    flat = ",".join(f"{k}={v}" for k, v in sorted(bindings.items()))
    key = f"{kernel_kind}|{flat}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", kernel_kind).strip("_")
    return f"{idx:02d}_{slug}_{h}"


def _cmd_tune(args: argparse.Namespace) -> int:
    backend_target = str(args.backend_target).strip()
    if not backend_target.startswith("cuda"):
        raise SystemExit("--backend-target must be cuda_* for tune")
    kernel = str(args.kernel).strip()
    if not kernel:
        raise SystemExit("--kernel is required")

    arch = _normalize_cuda_arch(str(args.arch or "")) or _detect_cuda_arch()
    if not arch:
        raise SystemExit("unable to detect CUDA arch (set INTENTIR_CUDA_SM=sm_XX or pass --arch)")
    evidence_mode = str(args.evidence_mode).strip().lower() or "off"
    if evidence_mode not in {"on", "off"}:
        raise SystemExit(f"invalid --evidence-mode={evidence_mode!r} (expected on/off)")

    out_root = Path(args.out_root) if args.out_root else (ROOT / "artifacts" / "tuning_runs" / _utc_ymd() / f"{kernel}_{arch}")
    out_root.mkdir(parents=True, exist_ok=True)

    wave = _default_cuda_wave()
    if not wave:
        raise SystemExit("missing INTENTIR_CUDA_REAL_MLIR_WAVE and no wave files found under workflow/flaggems/state/")

    raw_candidates = list(args.candidate or [])
    if not raw_candidates:
        raise SystemExit("--candidate is required (repeatable)")
    candidates: list[tuple[str, dict[str, int]]] = []
    for raw in raw_candidates:
        candidates.append(_parse_candidate(str(raw)))

    base_env = {
        "INTENTIR_REAL_MLIR": "1",
        "INTENTIR_FALLBACK_POLICY": "strict",
        "INTENTIR_CUDA_REQUIRE_LLVM_PTX": "1",
        "INTENTIR_EVIDENCE_MODE": str(evidence_mode),
        "INTENTIR_CUDA_REAL_MLIR_WAVE": str(wave),
    }
    # Preserve user-provided SM string (e.g. sm_120a) if present; otherwise set a normalized one.
    if not str(os.getenv("INTENTIR_CUDA_SM", "")).strip():
        base_env["INTENTIR_CUDA_SM"] = str(arch)

    results: list[dict[str, object]] = []
    for idx, (kernel_kind, bindings) in enumerate(candidates):
        cand_id = _candidate_id(kernel_kind, bindings, idx=idx)
        cand_dir = out_root / cand_id
        cov_dir = cand_dir / "coverage"
        perf_dir = cand_dir / "perf"
        cand_db = cand_dir / "tuning.jsonl"
        cand_dir.mkdir(parents=True, exist_ok=True)
        cand_db.write_text(
            json.dumps(
                {
                    "schema_version": "intentir_tuning_db_entry_v1",
                    "backend": "cuda",
                    "kernel": str(kernel),
                    "arch": str(arch),
                    "bindings": dict(bindings),
                    "kernel_kind": str(kernel_kind),
                    "note": "autotune_candidate",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        env = dict(base_env)
        env["INTENTIR_CUDA_TUNING_DB"] = str(cand_db)

        rc_cov = _run(
            _python_cmd(
                "scripts/intentir.py",
                "suite",
                "--suite",
                "flaggems-coverage-single",
                "--backend-target",
                str(backend_target),
                "--kernel",
                str(kernel),
                "--cases-limit",
                str(int(args.cases_limit)),
                "--out-root",
                str(cov_dir),
                "--cuda-runtime-backend",
                str(args.cuda_runtime_backend),
                "--stream",
            ),
            stream=True,
            dry_run=bool(args.dry_run),
            env_overrides=env,
        )
        if rc_cov != 0:
            results.append(
                {
                    "candidate": cand_id,
                    "kernel_kind": str(kernel_kind),
                    "bindings": dict(bindings),
                    "coverage_rc": int(rc_cov),
                    "perf_rc": None,
                    "ratio": None,
                    "error": f"coverage_rc={rc_cov}",
                    "coverage_dir": str(cov_dir),
                    "perf_dir": str(perf_dir),
                }
            )
            continue

        rc_perf = _run(
            _python_cmd(
                "scripts/intentir.py",
                "suite",
                "--suite",
                "gpu-perf-triton-native",
                "--backend-target",
                str(backend_target),
                "--kernel",
                str(kernel),
                "--out-root",
                str(perf_dir),
                "--gpu-perf-intent-artifact-dir",
                str(cov_dir),
                "--perf-warmup",
                str(int(args.perf_warmup)),
                "--perf-iters",
                str(int(args.perf_iters)),
                "--perf-repeats",
                str(int(args.perf_repeats)),
                "--cuda-runtime-backend",
                str(args.cuda_runtime_backend),
                "--stream",
            ),
            stream=True,
            dry_run=bool(args.dry_run),
            env_overrides=env,
        )
        ratio = None
        if rc_perf == 0 and not bool(args.dry_run):
            try:
                graph = json.loads((perf_dir / "gpu_perf_graph.json").read_text(encoding="utf-8"))
                entries = list(graph.get("entries") or [])
                for e in entries:
                    if str((e or {}).get("kernel") or "") == str(kernel):
                        ratio = (e or {}).get("ratio")
                        break
            except Exception:
                ratio = None
        results.append(
            {
                "candidate": cand_id,
                "kernel_kind": str(kernel_kind),
                "bindings": dict(bindings),
                "coverage_rc": int(rc_cov),
                "perf_rc": int(rc_perf),
                "ratio": ratio,
                "coverage_dir": str(cov_dir),
                "perf_dir": str(perf_dir),
            }
        )

    summary = {
        "schema_version": "intentir_cuda_autotune_summary_v1",
        "kernel": str(kernel),
        "backend_target": str(backend_target),
        "arch": str(arch),
        "wave": str(wave),
        "evidence_mode": str(evidence_mode),
        "candidates": results,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    best = None
    best_ratio = None
    for r in results:
        try:
            ratio = r.get("ratio")
            if not isinstance(ratio, (int, float)):
                continue
            if best is None or best_ratio is None or float(ratio) > float(best_ratio):
                best = r
                best_ratio = float(ratio)
        except Exception:
            continue

    if best is None:
        print(f"[intentir][tune] no passing candidates for kernel={kernel} arch={arch}", file=sys.stderr, flush=True)
        return 2

    recommended = {
        "schema_version": "intentir_tuning_db_entry_v1",
        "backend": "cuda",
        "kernel": str(kernel),
        "arch": str(arch),
        "bindings": dict(best.get("bindings") or {}),
        "kernel_kind": str(best.get("kernel_kind") or ""),
        "note": f"autotune_best ratio={best_ratio}",
    }
    rec_line = json.dumps(recommended, ensure_ascii=False)
    (out_root / "recommended.jsonl").write_text(rec_line + "\n", encoding="utf-8")
    print("[intentir][tune] recommended tuning_db entry:", flush=True)
    print(rec_line, flush=True)
    return 0


def _cmd_suite(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root) if args.out_root else (ROOT / "artifacts" / "intentir_suite")
    out_root.mkdir(parents=True, exist_ok=True)
    env_overrides = {"INTENTIR_EXECUTION_IR": str(args.execution_ir)}
    progress_file = Path(args.progress_file) if args.progress_file else (out_root / "chunk_progress.json")

    if args.suite == "flaggems-full196":
        rc = _run(
            _python_cmd("scripts/flaggems/build_coverage_batches.py"),
            stream=bool(args.stream),
            dry_run=bool(args.dry_run),
            env_overrides=env_overrides,
        )
        if rc != 0:
            return rc
        cmd = _python_cmd(
            "scripts/flaggems/run_coverage_batches.py",
            "--out-root",
            str(out_root),
            "--cases-limit",
            str(int(args.cases_limit)),
            "--execution-ir",
            str(args.execution_ir),
            "--flaggems-path",
            str(args.flaggems_path),
            "--backend-target",
            str(args.backend_target),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            str(args.intentir_miss_policy),
            "--pipeline-timeout-sec",
            str(int(args.pipeline_timeout_sec)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--family-kernel-chunk-size",
            str(int(args.family_kernel_chunk_size)),
            "--progress-style",
            str(args.progress_style),
            "--progress-file",
            str(progress_file),
            "--stream-subprocess-output"
            if bool(args.stream_subprocess_detail)
            else "--no-stream-subprocess-output",
            "--allow-cuda-skip" if args.allow_cuda_skip else "--no-allow-cuda-skip",
            "--run-rvv-remote" if args.run_rvv_remote else "--no-run-rvv-remote",
            "--skip-rvv-local" if args.skip_rvv_local else "--no-skip-rvv-local",
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--rvv-remote-timeout-sec",
            str(int(args.rvv_remote_timeout_sec)),
            "--rvv-use-key" if args.rvv_use_key else "--no-rvv-use-key",
            "--resume" if args.resume else "--no-resume",
        )
        if bool(args.write_registry):
            cmd.append("--write-registry")
        for fam in list(args.family or []):
            cmd.extend(["--family", str(fam)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    if args.suite == "gpu-perf-graph":
        rc = _run(
            _python_cmd("scripts/flaggems/build_coverage_batches.py"),
            stream=bool(args.stream),
            dry_run=bool(args.dry_run),
            env_overrides=env_overrides,
        )
        if rc != 0:
            return rc
        cmd = _python_cmd(
            "scripts/flaggems/run_gpu_perf_graph.py",
            "--out-root",
            str(out_root),
            "--family-kernel-chunk-size",
            str(int(args.family_kernel_chunk_size)),
            "--threshold",
            str(float(args.gpu_perf_threshold)),
            "--p50-threshold",
            str(float(args.gpu_perf_p50_threshold)),
            "--warmup",
            str(int(args.perf_warmup)),
            "--iters",
            str(int(args.perf_iters)),
            "--repeats",
            str(int(args.perf_repeats)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--progress-style",
            str(args.progress_style),
            "--progress-file",
            str(progress_file),
            "--resume" if args.resume else "--no-resume",
            "--stream" if args.stream else "--no-stream",
        )
        intent_artifact_dir = str(getattr(args, "gpu_perf_intent_artifact_dir", "") or "").strip()
        if not intent_artifact_dir:
            intent_artifact_dir = _infer_full196_artifact_dir()
        if intent_artifact_dir:
            cmd.extend(["--intent-artifact-dir", str(intent_artifact_dir)])
        if str(args.gpu_perf_policy_json).strip():
            cmd.extend(["--policy-json", str(args.gpu_perf_policy_json)])
        for kernel in list(args.gpu_perf_gate_exclude_kernel or []):
            if str(kernel).strip():
                cmd.extend(["--gate-exclude-kernel", str(kernel)])
        for kernel in list(args.kernel or []):
            if str(kernel).strip():
                cmd.extend(["--kernel", str(kernel)])
        for fam in list(args.family or []):
            cmd.extend(["--family", str(fam)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    if args.suite == "gpu-perf-triton-native":
        cmd = _python_cmd(
            "scripts/flaggems/run_gpu_perf_graph.py",
            "--kernel-source",
            "triton_native",
            "--monitor-only",
            "--out-root",
            str(out_root),
            "--family-kernel-chunk-size",
            str(int(args.family_kernel_chunk_size)),
            "--threshold",
            str(float(args.gpu_perf_threshold)),
            "--p50-threshold",
            str(float(args.gpu_perf_p50_threshold)),
            "--warmup",
            str(int(args.perf_warmup)),
            "--iters",
            str(int(args.perf_iters)),
            "--repeats",
            str(int(args.perf_repeats)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--progress-style",
            str(args.progress_style),
            "--progress-file",
            str(progress_file),
            "--resume" if args.resume else "--no-resume",
            "--stream" if args.stream else "--no-stream",
        )
        intent_artifact_dir = str(getattr(args, "gpu_perf_intent_artifact_dir", "") or "").strip()
        if intent_artifact_dir:
            cmd.extend(["--intent-artifact-dir", str(intent_artifact_dir)])
        if str(args.gpu_perf_policy_json).strip():
            cmd.extend(["--policy-json", str(args.gpu_perf_policy_json)])
        for kernel in list(args.gpu_perf_gate_exclude_kernel or []):
            if str(kernel).strip():
                cmd.extend(["--gate-exclude-kernel", str(kernel)])
        for kernel in list(args.kernel or []):
            if str(kernel).strip():
                cmd.extend(["--kernel", str(kernel)])
        for fam in list(args.family or []):
            cmd.extend(["--family", str(fam)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    if args.suite == "gpu-perf-wave-allowlist":
        rc = _run(
            _python_cmd("scripts/flaggems/build_coverage_batches.py"),
            stream=bool(args.stream),
            dry_run=bool(args.dry_run),
            env_overrides=env_overrides,
        )
        if rc != 0:
            return rc
        wave = str(os.getenv("INTENTIR_CUDA_REAL_MLIR_WAVE", "wave9")).strip() or "wave9"
        allowlist_json = ROOT / "workflow" / "flaggems" / "state" / f"cuda_real_mlir_{wave}_kernels.json"
        if not allowlist_json.is_file():
            allowlist_json = ROOT / "workflow" / "flaggems" / "state" / "cuda_real_mlir_wave9_kernels.json"
        cmd = _python_cmd(
            "scripts/flaggems/run_gpu_perf_graph.py",
            "--kernel-source",
            "coverage_batches",
            "--kernel-allowlist-json",
            str(allowlist_json),
            "--bench-mode",
            "eager",
            "--monitor-only",
            "--out-root",
            str(out_root),
            "--family-kernel-chunk-size",
            str(int(args.family_kernel_chunk_size)),
            "--threshold",
            str(float(args.gpu_perf_threshold)),
            "--p50-threshold",
            str(float(args.gpu_perf_p50_threshold)),
            "--warmup",
            str(int(args.perf_warmup)),
            "--iters",
            str(int(args.perf_iters)),
            "--repeats",
            str(int(args.perf_repeats)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--progress-style",
            str(args.progress_style),
            "--progress-file",
            str(progress_file),
            "--resume" if args.resume else "--no-resume",
            "--stream" if args.stream else "--no-stream",
        )
        intent_artifact_dir = str(getattr(args, "gpu_perf_intent_artifact_dir", "") or "").strip()
        if not intent_artifact_dir:
            intent_artifact_dir = _infer_full196_artifact_dir()
        if intent_artifact_dir:
            cmd.extend(["--intent-artifact-dir", str(intent_artifact_dir)])
        if str(args.gpu_perf_policy_json).strip():
            cmd.extend(["--policy-json", str(args.gpu_perf_policy_json)])
        for kernel in list(args.gpu_perf_gate_exclude_kernel or []):
            if str(kernel).strip():
                cmd.extend(["--gate-exclude-kernel", str(kernel)])
        for fam in list(args.family or []):
            cmd.extend(["--family", str(fam)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    if args.suite == "triton-smoke":
        cmd = _python_cmd(
            "scripts/flaggems/run_multibackend_matrix.py",
            "--suite",
            "smoke",
            "--out-dir",
            str(out_root),
            "--cases-limit",
            str(int(args.cases_limit)),
            "--backend-target",
            str(args.backend_target),
            "--flaggems-path",
            str(args.flaggems_path),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            str(args.intentir_miss_policy),
            "--pipeline-timeout-sec",
            str(int(args.pipeline_timeout_sec)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--allow-cuda-skip" if args.allow_cuda_skip else "--no-allow-cuda-skip",
            "--run-rvv-remote" if args.run_rvv_remote else "--no-run-rvv-remote",
            "--skip-rvv-local" if args.skip_rvv_local else "--no-skip-rvv-local",
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--rvv-remote-timeout-sec",
            str(int(args.rvv_remote_timeout_sec)),
            "--rvv-use-key" if args.rvv_use_key else "--no-rvv-use-key",
            "--stream-subprocess-output" if args.stream else "--no-stream-subprocess-output",
        )
        if bool(args.write_registry):
            cmd.append("--write-registry")
        for kernel in list(args.kernel or []):
            cmd.extend(["--kernel", str(kernel)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    if args.suite == "flaggems-coverage-single":
        cmd = _python_cmd(
            "scripts/flaggems/run_multibackend_matrix.py",
            "--suite",
            "coverage",
            "--out-dir",
            str(out_root),
            "--cases-limit",
            str(int(args.cases_limit)),
            "--backend-target",
            str(args.backend_target),
            "--flaggems-path",
            str(args.flaggems_path),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            str(args.intentir_miss_policy),
            "--pipeline-timeout-sec",
            str(int(args.pipeline_timeout_sec)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--allow-cuda-skip" if args.allow_cuda_skip else "--no-allow-cuda-skip",
            "--run-rvv-remote" if args.run_rvv_remote else "--no-run-rvv-remote",
            "--skip-rvv-local" if args.skip_rvv_local else "--no-skip-rvv-local",
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--rvv-remote-timeout-sec",
            str(int(args.rvv_remote_timeout_sec)),
            "--rvv-use-key" if args.rvv_use_key else "--no-rvv-use-key",
            "--stream-subprocess-output" if args.stream else "--no-stream-subprocess-output",
        )
        if bool(args.write_registry):
            cmd.append("--write-registry")
        for kernel in list(args.kernel or []):
            cmd.extend(["--kernel", str(kernel)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    if args.suite == "tilelang-smoke":
        cmd = _python_cmd(
            "scripts/tilelang/full_pipeline_verify.py",
            "--out-dir",
            str(out_root),
            "--cases-limit",
            str(int(args.cases_limit)),
        )
        for kernel in list(args.kernel or []):
            cmd.extend(["--kernel", str(kernel)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    if args.suite == "cuda-smoke":
        cmd = _python_cmd(
            "scripts/cuda/full_pipeline_verify.py",
            "--out-dir",
            str(out_root),
            "--cases-limit",
            str(int(args.cases_limit)),
        )
        for kernel in list(args.kernel or []):
            cmd.extend(["--kernel", str(kernel)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    print(f"unsupported suite: {args.suite}", file=sys.stderr)
    return 2


def _cmd_kernel(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "intentir_kernel")
    out_dir.mkdir(parents=True, exist_ok=True)
    env_overrides = {"INTENTIR_EXECUTION_IR": str(args.execution_ir)}

    if args.frontend == "triton":
        cmd = _python_cmd(
            "scripts/flaggems/run_multibackend_matrix.py",
            "--suite",
            "smoke",
            "--kernel",
            str(args.kernel),
            "--out-dir",
            str(out_dir),
            "--cases-limit",
            str(int(args.cases_limit)),
            "--backend-target",
            str(args.backend_target),
            "--flaggems-path",
            str(args.flaggems_path),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            str(args.intentir_miss_policy),
            "--pipeline-timeout-sec",
            str(int(args.pipeline_timeout_sec)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--allow-cuda-skip" if args.allow_cuda_skip else "--no-allow-cuda-skip",
            "--run-rvv-remote" if args.run_rvv_remote else "--no-run-rvv-remote",
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--rvv-remote-timeout-sec",
            str(int(args.rvv_remote_timeout_sec)),
            "--rvv-use-key" if args.rvv_use_key else "--no-rvv-use-key",
            "--stream-subprocess-output" if args.stream else "--no-stream-subprocess-output",
        )
        if str(args.provider) == "native":
            cmd.extend(["--flaggems-path", "original"])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    if args.frontend == "tilelang":
        cmd = _python_cmd(
            "scripts/tilelang/full_pipeline_verify.py",
            "--kernel",
            str(args.kernel),
            "--out-dir",
            str(out_dir),
            "--cases-limit",
            str(int(args.cases_limit)),
        )
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    if args.frontend == "cuda":
        cmd = _python_cmd(
            "scripts/cuda/full_pipeline_verify.py",
            "--kernel",
            str(args.kernel),
            "--out-dir",
            str(out_dir),
            "--cases-limit",
            str(int(args.cases_limit)),
        )
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    print(f"unsupported frontend: {args.frontend}", file=sys.stderr)
    return 2


def _import_version(name: str) -> tuple[str, str]:
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        return ("missing", str(e))
    version = str(getattr(mod, "__version__", "unknown"))
    location = str(getattr(mod, "__file__", "built-in"))
    return (version, location)


def _cmd_env(_: argparse.Namespace) -> int:
    print(f"repo_root: {ROOT}")
    print(f"python: {sys.version.splitlines()[0]}")
    for name in ("torch", "triton", "tilelang", "flag_gems"):
        ver, loc = _import_version(name)
        print(f"{name}: version={ver} location={loc}")
    try:
        import torch  # type: ignore

        print(f"torch.cuda.is_available: {bool(torch.cuda.is_available())}")
        if bool(torch.cuda.is_available()):
            print(f"torch.cuda.device_count: {int(torch.cuda.device_count())}")
            print(f"torch.cuda.device_name[0]: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"cuda_probe_error: {e}")
    print(f"rvv_default_host: {os.getenv('INTENTIR_RVV_HOST', '192.168.8.72')}")
    print(f"rvv_default_user: {os.getenv('INTENTIR_RVV_USER', 'ubuntu')}")
    print(f"intentir.execution_ir: {os.getenv('INTENTIR_EXECUTION_IR', 'mlir')}")
    return 0


def _cmd_mlir_check(args: argparse.Namespace) -> int:
    from intent_ir.mlir import detect_mlir_toolchain, to_mlir, to_intent  # noqa: PLC0415
    from intent_ir.ir import IntentFunction  # noqa: PLC0415

    toolchain = detect_mlir_toolchain()
    missing = [str(x) for x in list(toolchain.get("missing_required_tools") or []) if str(x).strip()]
    report: dict[str, object] = {"toolchain": toolchain}
    report["toolchain_required_ok"] = bool(toolchain.get("ok"))
    report["toolchain_required_missing"] = missing
    if missing:
        report["toolchain_install_hint"] = (
            "Install MLIR/LLVM tools so these binaries are on PATH: "
            + ", ".join(missing)
        )
    if args.intent_json is not None:
        payload = _load_intent_payload(Path(args.intent_json))
        intent = IntentFunction.from_json_dict(payload)
        module = to_mlir(intent)
        _ = to_intent(module)
        report["roundtrip_ok"] = True
        report["intent_name"] = str(intent.name)
        report["symbol_count"] = len(list(module.symbols or []))
    out = Path(args.out) if args.out else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[intentir] wrote {out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_mlir_pass(args: argparse.Namespace) -> int:
    from intent_ir.mlir import run_pipeline, to_mlir  # noqa: PLC0415
    from intent_ir.ir import IntentFunction  # noqa: PLC0415

    payload = _load_intent_payload(Path(args.intent_json))
    intent = IntentFunction.from_json_dict(payload)
    module = to_mlir(intent)
    shape_bindings = _parse_shape_bindings(getattr(args, "shape", None))
    if shape_bindings:
        module.meta = dict(module.meta or {})
        module.meta["shape_bindings"] = dict(shape_bindings)
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "intentir_mlir_pass")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_module, trace = run_pipeline(
        module,
        str(args.pipeline),
        backend=(str(args.backend) if args.backend else None),
        out_dir=out_dir,
        fail_on_error=bool(args.fail_on_error),
    )
    out_mlir = out_dir / "module.mlir"
    out_mlir.write_text(out_module.module_text, encoding="utf-8")
    out_trace = out_dir / "pass_trace.json"
    if "pass_trace_path" not in trace:
        out_trace.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[intentir] mlir module: {out_mlir}")
    print(f"[intentir] pass trace: {out_trace}")
    return 0 if bool(trace.get("ok")) else 1


def _load_intent_payload(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "intent" in payload and isinstance(payload.get("intent"), dict):
        payload = payload["intent"]
    if not isinstance(payload, dict):
        raise ValueError(f"invalid intent payload at {path}: expected JSON object")
    return payload


def _parse_shape_bindings(raw: list[str] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in list(raw or []):
        s = str(item or "").strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"invalid --shape {s!r} (expected KEY=INT)")
        k, v = s.split("=", 1)
        key = str(k).strip()
        if not key:
            raise ValueError(f"invalid --shape {s!r} (empty key)")
        try:
            out[key] = int(str(v).strip())
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"invalid --shape {s!r} (bad int): {type(e).__name__}: {e}") from e
    return out


def _cmd_mlir_emit_llvm(args: argparse.Namespace) -> int:
    from intent_ir.mlir import detect_mlir_toolchain, run_pipeline, to_mlir  # noqa: PLC0415
    from intent_ir.ir import IntentFunction  # noqa: PLC0415

    payload = _load_intent_payload(Path(args.intent_json))
    intent = IntentFunction.from_json_dict(payload)
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "intentir_mlir_emit_llvm")
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = str(args.backend)
    pipeline_choice = str(getattr(args, "pipeline", "auto") or "auto").strip()
    if pipeline_choice != "auto":
        pipeline = pipeline_choice
    else:
        pipeline = "downstream_cuda_llvm" if backend == "cuda" else "downstream_rvv_llvm"
    module = to_mlir(intent)
    shape_bindings = _parse_shape_bindings(getattr(args, "shape", None))
    if shape_bindings:
        module.meta = dict(module.meta or {})
        module.meta["shape_bindings"] = dict(shape_bindings)

    def _attach_shapes(mod):  # noqa: ANN001
        if not shape_bindings:
            return mod
        try:
            mod.meta = dict(getattr(mod, "meta", None) or {})
            mod.meta["shape_bindings"] = dict(shape_bindings)
        except Exception:
            pass
        return mod

    module = _attach_shapes(module)
    upstream_mod, upstream_trace = run_pipeline(
        module,
        "upstream",
        backend=backend,
        out_dir=(out_dir / "upstream"),
        fail_on_error=bool(args.fail_on_error),
    )
    upstream_mod = _attach_shapes(upstream_mod)
    midend_mod, midend_trace = run_pipeline(
        upstream_mod,
        "midend",
        backend=backend,
        out_dir=(out_dir / "midend"),
        fail_on_error=bool(args.fail_on_error),
    )
    midend_mod = _attach_shapes(midend_mod)
    downstream_mod, downstream_trace = run_pipeline(
        midend_mod,
        pipeline,
        backend=backend,
        out_dir=(out_dir / pipeline),
        fail_on_error=bool(args.fail_on_error),
    )
    llvm_ir_path = out_dir / f"{intent.name}.intentdialect.{pipeline}.ll"
    llvm_ir_path.write_text(str(downstream_mod.module_text or ""), encoding="utf-8")

    toolchain = detect_mlir_toolchain()
    tools = dict(toolchain.get("tools") or {})
    llvm_as_tool = str((tools.get("llvm-as") or {}).get("path") or "").strip()
    llvm_bc_path = out_dir / f"{intent.name}.intentdialect.{pipeline}.bc"
    llvm_bc_ok = False
    llvm_bc_detail = ""
    if bool(args.emit_bc):
        if not llvm_as_tool:
            llvm_bc_detail = "llvm-as unavailable"
        else:
            p = subprocess.run(
                [llvm_as_tool, str(llvm_ir_path), "-o", str(llvm_bc_path)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            llvm_bc_ok = bool(p.returncode == 0 and llvm_bc_path.is_file())
            llvm_bc_detail = (str(p.stderr or p.stdout).strip() if p.returncode != 0 else "ok")
    else:
        llvm_bc_detail = "disabled_by_flag"

    trace_ok = bool(upstream_trace.get("ok")) and bool(midend_trace.get("ok")) and bool(downstream_trace.get("ok"))
    ok = bool(trace_ok and (not bool(args.emit_bc) or llvm_bc_ok))
    report: dict[str, object] = {
        "schema_version": "intentir_mlir_emit_llvm_v1",
        "ok": bool(ok),
        "backend": str(backend),
        "pipeline": str(pipeline),
        "intent_name": str(intent.name),
        "toolchain": toolchain,
        "llvm_ir_path": str(llvm_ir_path),
        "llvm_emit_ok": bool(trace_ok),
        "llvm_bc_path": (str(llvm_bc_path) if bool(llvm_bc_ok) else ""),
        "llvm_bc_ok": bool(llvm_bc_ok),
        "llvm_bc_detail": str(llvm_bc_detail),
        "upstream_trace_path": str((upstream_trace.get("pass_trace_path") or "")),
        "midend_trace_path": str((midend_trace.get("pass_trace_path") or "")),
        "downstream_trace_path": str((downstream_trace.get("pass_trace_path") or "")),
    }
    summary_path = Path(args.out) if args.out else (out_dir / "llvm_emit_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[intentir] llvm ir: {llvm_ir_path}")
    print(f"[intentir] summary: {summary_path}")
    if bool(args.emit_bc):
        print(f"[intentir] llvm bc: {llvm_bc_path if llvm_bc_ok else '(failed)'}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if bool(ok) else 1


def _cmd_mlir_provision_toolchain(args: argparse.Namespace) -> int:
    cmd = _python_cmd(
        "scripts/intentir/provision_mlir_toolchain.py",
        "--version",
        str(int(args.version)),
    )
    if args.prefix is not None:
        cmd.extend(["--prefix", str(args.prefix)])
    if args.toolchain_root is not None:
        cmd.extend(["--toolchain-root", str(args.toolchain_root)])
    if args.current_link is not None:
        cmd.extend(["--current-link", str(args.current_link)])
    if args.env_file is not None:
        cmd.extend(["--env-file", str(args.env_file)])
    cmd.append("--force" if bool(args.force) else "--no-force")
    cmd.append("--use-current-link" if bool(args.use_current_link) else "--no-use-current-link")
    if args.out is not None:
        cmd.extend(["--out", str(args.out)])
    return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run))


def _cmd_tilelang_export_cuda_snapshots(args: argparse.Namespace) -> int:
    cmd = _python_cmd(
        "scripts/tilelang/export_cuda_snapshots.py",
        "--out-dir",
        str(args.out_dir),
    )
    if bool(args.refresh):
        cmd.append("--refresh")
    for kernel in list(args.kernel or []):
        cmd.extend(["--kernel", str(kernel)])
    return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run))


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="IntentIR unified CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    suite = sub.add_parser("suite", help="Run suite-level verification")
    suite.add_argument(
        "--suite",
        choices=[
            "flaggems-full196",
            "gpu-perf-graph",
            "gpu-perf-triton-native",
            "gpu-perf-wave-allowlist",
            "flaggems-coverage-single",
            "triton-smoke",
            "tilelang-smoke",
            "cuda-smoke",
        ],
        required=True,
    )
    suite.add_argument("--out-root", default=None)
    suite.add_argument("--family", action="append", default=[])
    suite.add_argument("--kernel", action="append", default=[])
    suite.add_argument("--cases-limit", type=int, default=8)
    suite.add_argument("--execution-ir", choices=["mlir"], default=_execution_ir_default())
    suite.add_argument("--family-kernel-chunk-size", type=int, default=12)
    suite.add_argument("--gpu-perf-threshold", type=float, default=0.80)
    suite.add_argument("--gpu-perf-p50-threshold", type=float, default=0.90)
    suite.add_argument("--perf-warmup", type=int, default=20)
    suite.add_argument("--perf-iters", type=int, default=200)
    suite.add_argument("--perf-repeats", type=int, default=5)
    suite.add_argument(
        "--gpu-perf-policy-json",
        default="workflow/flaggems/state/gpu_perf_policy.json",
        help="Optional gpu perf policy JSON path passed to run_gpu_perf_graph.py.",
    )
    suite.add_argument(
        "--gpu-perf-gate-exclude-kernel",
        action="append",
        default=[],
        help="Kernel alias to exclude from gpu_perf gate denominator (repeatable).",
    )
    suite.add_argument(
        "--gpu-perf-intent-artifact-dir",
        default="",
        help="Optional artifact root passed to run_gpu_perf_graph.py --intent-artifact-dir. "
        "If empty, auto-infers from workflow state full196_last_run.",
    )
    suite.add_argument(
        "--pipeline-timeout-sec",
        type=int,
        default=0,
        help="Pipeline stage timeout (seconds) for matrix runners; 0 disables.",
    )
    suite.add_argument("--progress-style", choices=["auto", "tqdm", "plain", "chunk", "none"], default="chunk")
    suite.add_argument(
        "--stream-subprocess-detail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward verbose per-kernel backend logs from internal matrix runners (default: false).",
    )
    suite.add_argument(
        "--progress-file",
        default=None,
        help="Optional chunk-progress JSON path (default: <out-root>/chunk_progress.json).",
    )
    suite.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--write-registry", action=argparse.BooleanOptionalAction, default=False)
    suite.add_argument("--flaggems-path", choices=["intentir", "original"], default="intentir")
    suite.add_argument("--backend-target", choices=["rvv", "cuda_h100", "cuda_5090d"], default="rvv")
    suite.add_argument("--intentir-mode", choices=["auto", "force_compile", "force_cache"], default="auto")
    suite.add_argument("--intentir-miss-policy", choices=["deterministic", "strict"], default="strict")
    suite.add_argument("--run-rvv-remote", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--skip-rvv-local", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--rvv-host", default="192.168.8.72")
    suite.add_argument("--rvv-user", default="ubuntu")
    suite.add_argument("--rvv-port", type=int, default=22)
    suite.add_argument(
        "--rvv-remote-timeout-sec",
        type=int,
        default=600,
        help="Timeout (seconds) for rvv_remote stage in matrix runs (0 disables).",
    )
    suite.add_argument("--rvv-use-key", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--allow-cuda-skip", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="nvrtc")
    suite.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    suite.set_defaults(func=_cmd_suite)

    kernel = sub.add_parser("kernel", help="Run one kernel")
    kernel.add_argument("--frontend", choices=["triton", "tilelang", "cuda"], required=True)
    kernel.add_argument("--provider", choices=["native", "flaggems"], default="flaggems")
    kernel.add_argument("--kernel", required=True)
    kernel.add_argument("--out-dir", default=None)
    kernel.add_argument("--cases-limit", type=int, default=8)
    kernel.add_argument(
        "--pipeline-timeout-sec",
        type=int,
        default=0,
        help="Pipeline stage timeout (seconds) for matrix runners; 0 disables.",
    )
    kernel.add_argument("--execution-ir", choices=["mlir"], default=_execution_ir_default())
    kernel.add_argument("--flaggems-path", choices=["intentir", "original"], default="intentir")
    kernel.add_argument("--backend-target", choices=["rvv", "cuda_h100", "cuda_5090d"], default="rvv")
    kernel.add_argument("--intentir-mode", choices=["auto", "force_compile", "force_cache"], default="auto")
    kernel.add_argument("--intentir-miss-policy", choices=["deterministic", "strict"], default="strict")
    kernel.add_argument("--run-rvv-remote", action=argparse.BooleanOptionalAction, default=True)
    kernel.add_argument("--rvv-host", default="192.168.8.72")
    kernel.add_argument("--rvv-user", default="ubuntu")
    kernel.add_argument("--rvv-port", type=int, default=22)
    kernel.add_argument(
        "--rvv-remote-timeout-sec",
        type=int,
        default=600,
        help="Timeout (seconds) for rvv_remote stage in matrix runs (0 disables).",
    )
    kernel.add_argument("--rvv-use-key", action=argparse.BooleanOptionalAction, default=True)
    kernel.add_argument("--allow-cuda-skip", action=argparse.BooleanOptionalAction, default=True)
    kernel.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="nvrtc")
    kernel.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    kernel.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    kernel.set_defaults(func=_cmd_kernel)

    tune = sub.add_parser("tune", help="Measured autotune for one CUDA kernel")
    tune.add_argument("--backend-target", choices=["cuda_h100", "cuda_5090d"], required=True)
    tune.add_argument("--kernel", required=True)
    tune.add_argument("--candidate", action="append", default=[], help="Candidate: KIND[:K=V,A=B] (repeatable)")
    tune.add_argument("--arch", default="", help="Optional arch key override (e.g. sm120). Defaults to detect.")
    tune.add_argument("--out-root", default=None)
    tune.add_argument("--cases-limit", type=int, default=1)
    tune.add_argument("--perf-warmup", type=int, default=10)
    tune.add_argument("--perf-iters", type=int, default=50)
    tune.add_argument("--perf-repeats", type=int, default=3)
    tune.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="nvrtc")
    tune.add_argument("--evidence-mode", choices=["on", "off"], default="off")
    tune.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    tune.set_defaults(func=_cmd_tune)

    env = sub.add_parser("env", help="Show environment and dependency status")
    env.set_defaults(func=_cmd_env)

    tilelang = sub.add_parser("tilelang", help="TileLang helper commands")
    tilelang_sub = tilelang.add_subparsers(dest="tilelang_cmd", required=True)
    tilelang_export = tilelang_sub.add_parser("export-cuda-snapshots", help="Export TileLang kernels to CUDA snapshots")
    tilelang_export.add_argument("--out-dir", default=str(ROOT / "kernels" / "cuda" / "ops" / "snapshots"))
    tilelang_export.add_argument("--kernel", action="append", default=[])
    tilelang_export.add_argument("--refresh", action=argparse.BooleanOptionalAction, default=False)
    tilelang_export.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    tilelang_export.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    tilelang_export.set_defaults(func=_cmd_tilelang_export_cuda_snapshots)

    mlir = sub.add_parser("mlir", help="MLIR migration helpers")
    mlir_sub = mlir.add_subparsers(dest="mlir_cmd", required=True)

    mlir_check = mlir_sub.add_parser("check", help="Probe MLIR toolchain and optional Intent roundtrip")
    mlir_check.add_argument("--intent-json", default=None, help="Path to intent JSON (or report with `intent` field)")
    mlir_check.add_argument("--out", default=None, help="Optional JSON output path")
    mlir_check.set_defaults(func=_cmd_mlir_check)

    mlir_pass = mlir_sub.add_parser("pass", help="Run MLIR pipeline on one intent payload")
    mlir_pass.add_argument("--intent-json", required=True, help="Path to intent JSON (or report with `intent` field)")
    mlir_pass.add_argument(
        "--pipeline",
        required=True,
        choices=[
            "upstream",
            "midend",
            "downstream_cuda",
            "downstream_rvv",
            "downstream_cuda_llvm",
            "downstream_rvv_llvm",
            "upstream_std",
            "midend_std",
            "downstream_rvv_std_llvm",
        ],
    )
    mlir_pass.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Override a shape symbol binding (repeatable), e.g. --shape M=256",
    )
    mlir_pass.add_argument("--backend", default=None, help="Optional backend hint (cuda/rvv)")
    mlir_pass.add_argument("--out-dir", default=None, help="Output directory for module + pass trace")
    mlir_pass.add_argument("--fail-on-error", action=argparse.BooleanOptionalAction, default=False)
    mlir_pass.set_defaults(func=_cmd_mlir_pass)

    mlir_emit = mlir_sub.add_parser("emit-llvm", help="Emit LLVM IR (.ll) and optional bitcode (.bc)")
    mlir_emit.add_argument("--intent-json", required=True, help="Path to intent JSON (or report with `intent` field)")
    mlir_emit.add_argument("--backend", choices=["cuda", "rvv"], default="cuda")
    mlir_emit.add_argument(
        "--pipeline",
        choices=[
            "auto",
            "downstream_cuda_llvm",
            "downstream_rvv_llvm",
            "downstream_rvv_std_llvm",
        ],
        default="auto",
        help="Downstream LLVM pipeline selector for emit-llvm. "
        "Use downstream_rvv_std_llvm for real-MLIR RVV proof runs.",
    )
    mlir_emit.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Override a shape symbol binding (repeatable), e.g. --shape M=256",
    )
    mlir_emit.add_argument("--out-dir", default=None, help="Output directory for LLVM artifacts")
    mlir_emit.add_argument("--emit-bc", action=argparse.BooleanOptionalAction, default=True)
    mlir_emit.add_argument("--fail-on-error", action=argparse.BooleanOptionalAction, default=False)
    mlir_emit.add_argument("--out", default=None, help="Optional summary JSON path")
    mlir_emit.set_defaults(func=_cmd_mlir_emit_llvm)

    mlir_provision = mlir_sub.add_parser(
        "provision-toolchain",
        help="Provision repository-local MLIR toolchain (apt download + extract, no sudo required)",
    )
    mlir_provision.add_argument("--version", type=int, default=14)
    mlir_provision.add_argument("--prefix", default=None)
    mlir_provision.add_argument("--toolchain-root", default=None)
    mlir_provision.add_argument("--current-link", default=None)
    mlir_provision.add_argument("--env-file", default=None)
    mlir_provision.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    mlir_provision.add_argument("--use-current-link", action=argparse.BooleanOptionalAction, default=True)
    mlir_provision.add_argument("--out", default=None)
    mlir_provision.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    mlir_provision.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    mlir_provision.set_defaults(func=_cmd_mlir_provision_toolchain)

    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()
    rc = int(args.func(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
