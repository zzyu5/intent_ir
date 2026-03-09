from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "ORG-Migrate") not in sys.path:
    sys.path.insert(0, str(ROOT / "ORG-Migrate"))

from pipeline.common.tuning_db import load_tuning_db_jsonl, resolve_tuning_db_path, resolve_tuning_entries  # noqa: E402


def _candidate_line(kernel_kind: str, bindings: dict[str, int]) -> str:
    flat = ",".join(f"{k}={int(v)}" for k, v in sorted(bindings.items()))
    return f"{kernel_kind}:{flat}" if flat else str(kernel_kind)


def _coerce_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _parse_candidate_line(candidate: str) -> tuple[str, dict[str, int]]:
    raw = str(candidate or "").strip()
    if not raw:
        return "", {}
    if ":" not in raw:
        return raw, {}
    kernel_kind, flat = raw.split(":", 1)
    bindings: dict[str, int] = {}
    for chunk in flat.split(","):
        item = str(chunk).strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = str(key).strip()
        if not key:
            continue
        try:
            bindings[key] = int(value)
        except Exception:
            continue
    return str(kernel_kind).strip(), bindings


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _toolchain_env_from_report(report: dict[str, Any]) -> dict[str, str]:
    org = dict(report.get("org") or {})
    mlir = dict(report.get("mlir") or {})
    toolchain_model = dict(org.get("toolchain_model") or {})
    env: dict[str, str] = {}
    requires_real_mlir = bool(
        toolchain_model.get("requires_real_mlir")
        or mlir.get("real_mlir_enabled")
        or str(toolchain_model.get("llvm_pipeline") or mlir.get("llvm_pipeline") or "").strip().lower() in {
            "downstream_cuda_std_llvm",
            "downstream_rvv_std_llvm",
            "downstream_cuda_std_cpp_llvm",
            "downstream_rvv_std_llvm_cpp",
        }
    )
    if requires_real_mlir:
        env["INTENTIR_REAL_MLIR"] = "1"
    cuda_wave = str(toolchain_model.get("cuda_real_mlir_wave") or mlir.get("cuda_real_mlir_wave") or "").strip().lower()
    if cuda_wave:
        env["INTENTIR_CUDA_REAL_MLIR_WAVE"] = cuda_wave
    rvv_wave = str(toolchain_model.get("rvv_real_mlir_wave") or mlir.get("rvv_real_mlir_wave") or "").strip().lower()
    if rvv_wave:
        env["INTENTIR_RVV_REAL_MLIR_WAVE"] = rvv_wave
    return env


def _infer_compiler_stack_from_candidate_file(path: Path) -> str:
    try:
        first = str(path.read_text(encoding="utf-8").splitlines()[0] if path.is_file() else "")
    except Exception:
        return ""
    line = str(first or "").strip()
    if "compiler_stack=" not in line:
        return ""
    suffix = line.split("compiler_stack=", 1)[1]
    token = str(suffix.split()[0] if suffix.split() else "").strip()
    return token


def _resolve_source_candidate(
    *,
    kernel: str,
    shape_bindings: dict[str, int],
    compiler_stack: str,
    source_arch: str,
    db_path: Path | None,
    plan: dict[str, Any],
) -> str:
    source = dict(plan.get("source_oracle") or {})
    kind = str(source.get("kernel_kind") or "").strip()
    bindings = {str(k): int(v) for k, v in dict(source.get("bindings") or {}).items() if str(k).strip()}
    if kind:
        return _candidate_line(kind, bindings)

    if not source_arch:
        return ""
    db_file = resolve_tuning_db_path(path=db_path, backend="cuda")
    if db_file is None or not Path(db_file).is_file():
        return ""
    db = load_tuning_db_jsonl(path=Path(db_file), backend="cuda")
    entries = db.get((str(kernel), str(source_arch))) or []
    merged, kk = resolve_tuning_entries(entries, shape_bindings=shape_bindings, compiler_stack=str(compiler_stack))
    kind = str(kk or "").strip()
    bindings = {str(k): int(v) for k, v in dict(merged or {}).items() if str(k).strip()}
    return (_candidate_line(kind, bindings) if kind else "")


def _resolve_target_oracle_candidate(
    *,
    kernel: str,
    shape_bindings: dict[str, int],
    compiler_stack: str,
    target_arch: str,
    db_path: Path | None,
) -> str:
    if not target_arch:
        return ""
    db_file = resolve_tuning_db_path(path=db_path, backend="cuda")
    if db_file is None or not Path(db_file).is_file():
        return ""
    db = load_tuning_db_jsonl(path=Path(db_file), backend="cuda")
    entries = db.get((str(kernel), str(target_arch))) or []
    merged, kk = resolve_tuning_entries(entries, shape_bindings=shape_bindings, compiler_stack=str(compiler_stack))
    kind = str(kk or "").strip()
    bindings = {str(k): int(v) for k, v in dict(merged or {}).items() if str(k).strip()}
    return (_candidate_line(kind, bindings) if kind else "")


def _run_tune(
    *,
    kernel: str,
    backend_target: str,
    out_root: Path,
    candidate_file: Path | None = None,
    candidate: str | None = None,
    cases_limit: int,
    perf_warmup: int,
    perf_iters: int,
    perf_repeats: int,
    cuda_runtime_backend: str,
    compiler_stack: str,
    compiler_cpp_wave: str = "",
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "intentir.py"),
        "tune",
        "--backend-target",
        str(backend_target),
        "--kernel",
        str(kernel),
        "--out-root",
        str(out_root),
        "--cases-limit",
        str(int(cases_limit)),
        "--perf-warmup",
        str(int(perf_warmup)),
        "--perf-iters",
        str(int(perf_iters)),
        "--perf-repeats",
        str(int(perf_repeats)),
        "--cuda-runtime-backend",
        str(cuda_runtime_backend),
    ]
    effective_candidate_file = candidate_file
    if effective_candidate_file is None and candidate is not None:
        effective_candidate_file = out_root / "single_candidate.txt"
        effective_candidate_file.parent.mkdir(parents=True, exist_ok=True)
        effective_candidate_file.write_text(str(candidate).strip() + "\n", encoding="utf-8")
    if effective_candidate_file is not None:
        cmd.extend(["--candidate-file", str(effective_candidate_file)])
    env = dict(os.environ)
    env["INTENTIR_COMPILER_STACK"] = str(compiler_stack or "python")
    if str(compiler_cpp_wave or "").strip():
        env["INTENTIR_COMPILER_CPP_WAVE"] = str(compiler_cpp_wave)
    for key, value in dict(env_overrides or {}).items():
        if str(key).strip() and str(value).strip():
            env[str(key)] = str(value)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    result: dict[str, Any] = {
        "command": cmd,
        "compiler_stack": str(compiler_stack or "python"),
        "compiler_cpp_wave": str(compiler_cpp_wave or ""),
        "env_overrides": {str(k): str(v) for k, v in dict(env_overrides or {}).items() if str(k).strip()},
        "returncode": int(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-10:]),
        "out_root": str(out_root),
    }
    summary_path = out_root / "summary.json"
    recommended_path = out_root / "recommended.jsonl"
    if summary_path.is_file():
        result["summary"] = _load_json(summary_path)
    if recommended_path.is_file():
        lines = [x for x in recommended_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        result["recommended"] = [json.loads(x) for x in lines]
    return result


def _first_candidate_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return {}
    candidates = list(summary.get("candidates") or [])
    if not candidates:
        return {}
    first = dict(candidates[0] or {})
    return {
        "kernel_kind": str(first.get("kernel_kind") or ""),
        "bindings": dict(first.get("bindings") or {}),
        "ratio": first.get("ratio"),
        "coverage_rc": first.get("coverage_rc"),
        "perf_rc": first.get("perf_rc"),
    }


def _candidate_summary_from_line(candidate: str, *, ratio: float | None = None) -> dict[str, Any]:
    kernel_kind, bindings = _parse_candidate_line(candidate)
    return {
        "kernel_kind": str(kernel_kind),
        "bindings": {str(k): int(v) for k, v in dict(bindings or {}).items() if str(k).strip()},
        "ratio": (None if ratio is None else float(ratio)),
        "coverage_rc": None,
        "perf_rc": None,
    }


def _candidate_summaries(result: dict[str, Any]) -> list[dict[str, Any]]:
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return []
    out: list[dict[str, Any]] = []
    for item in list(summary.get("candidates") or []):
        row = dict(item or {})
        out.append(
            {
                "kernel_kind": str(row.get("kernel_kind") or ""),
                "bindings": {str(k): int(v) for k, v in dict(row.get("bindings") or {}).items() if str(k).strip()},
                "ratio": row.get("ratio"),
                "coverage_rc": row.get("coverage_rc"),
                "perf_rc": row.get("perf_rc"),
                "coverage_dir": str(row.get("coverage_dir") or ""),
                "perf_dir": str(row.get("perf_dir") or ""),
                "qps_native": row.get("qps_native"),
                "qps_intentir": row.get("qps_intentir"),
                "latency_native_ms": row.get("latency_native_ms"),
                "latency_intentir_ms": row.get("latency_intentir_ms"),
            }
        )
    return out


def _candidate_with_metrics(item: dict[str, Any]) -> dict[str, Any]:
    row = dict(item or {})
    if row.get("qps_native") is not None and row.get("qps_intentir") is not None:
        return row
    perf_dir = Path(str(row.get("perf_dir") or ""))
    graph_path = perf_dir / "gpu_perf_graph.json"
    if not graph_path.is_file():
        return row
    try:
        obj = _load_json(graph_path)
        first = dict((list(obj.get("entries") or [{}]) or [{}])[0] or {})
    except Exception:
        return row
    for key in ("qps_native", "qps_intentir", "latency_native_ms", "latency_intentir_ms"):
        if row.get(key) is None and first.get(key) is not None:
            row[key] = first.get(key)
    return row


def _candidate_with_contract_meta(item: dict[str, Any]) -> dict[str, Any]:
    row = dict(_candidate_with_metrics(item))
    coverage_dir = Path(str(row.get("coverage_dir") or ""))
    if not coverage_dir.is_dir():
        return row
    report_files = list(coverage_dir.glob("*.json"))
    report_path = next((p for p in report_files if p.suffix == ".json" and p.name.count(".") == 1), None)
    if report_path is None:
        return row
    try:
        obj = _load_json(report_path)
    except Exception:
        return row
    exec_meta = dict(((obj.get("mlir") or {}).get("downstream_cuda_std_llvm_contract_exec_meta") or {}))
    if not exec_meta:
        return row
    for src, dst in (
        ("cuda_requested_sm", "requested_sm"),
        ("cuda_effective_sm", "effective_sm"),
        ("cuda_target_downleveled", "downleveled"),
    ):
        if row.get(dst) is None and exec_meta.get(src) is not None:
            row[dst] = exec_meta.get(src)
    return row


def _best_candidate(result: dict[str, Any], *, metric: str) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_value = float("-inf")
    for item in _candidate_summaries(result):
        row = _candidate_with_contract_meta(item)
        value = row.get(metric)
        if value is None:
            continue
        try:
            fv = float(value)
        except Exception:
            continue
        if not best or fv > best_value:
            best = dict(row)
            best_value = fv
    return best


def _find_candidate_summary(result: dict[str, Any], candidate: str) -> dict[str, Any]:
    kind, bindings = _parse_candidate_line(candidate)
    if not kind:
        return {}
    for item in _candidate_summaries(result):
        row = _candidate_with_contract_meta(item)
        if str(row.get("kernel_kind") or "") != kind:
            continue
        if {str(k): int(v) for k, v in dict(row.get("bindings") or {}).items()} != dict(bindings):
            continue
        return row
    return {}


def _best_ratio(result: dict[str, Any]) -> float | None:
    best = _best_candidate(result, metric="ratio")
    if best:
        return float(best.get("ratio"))
    out_root_local = Path(str(result.get("out_root") or ""))
    run_summaries = list(out_root_local.rglob("run_summary.json")) if out_root_local.is_dir() else []
    fallback = []
    for path in run_summaries:
        try:
            obj = _load_json(path)
        except Exception:
            continue
        for key in ("gpu_perf_min_ratio", "gpu_perf_p50_ratio"):
            val = obj.get(key)
            if val is not None:
                fallback.append(float(val))
    return (max(fallback) if fallback else None)


def _best_qps_intentir(result: dict[str, Any]) -> float | None:
    best = _best_candidate(result, metric="qps_intentir")
    return (float(best.get("qps_intentir")) if best else None)


def _best_qps_native(result: dict[str, Any]) -> float | None:
    best = _best_candidate(result, metric="qps_native")
    return (float(best.get("qps_native")) if best else None)


def _graph_entries(result: dict[str, Any]) -> list[dict[str, Any]]:
    out_root_local = Path(str(result.get("out_root") or ""))
    graph_files = list(out_root_local.rglob("gpu_perf_graph.json")) if out_root_local.is_dir() else []
    entries: list[dict[str, Any]] = []
    for path in graph_files:
        try:
            obj = _load_json(path)
        except Exception:
            continue
        for entry in list(obj.get("entries") or []):
            row = dict(entry or {})
            row["_path"] = str(path)
            entries.append(row)
    return entries


def _first_failure_detail(result: dict[str, Any]) -> dict[str, Any]:
    for first in _graph_entries(result):
        reason_code = str(first.get("reason_code") or "")
        skip_reason = str(first.get("skip_reason") or "")
        if not reason_code and not skip_reason and bool(first.get("ok", True)):
            continue
        return {
            "ok": bool(first.get("ok")) if first.get("ok") is not None else None,
            "reason_code": reason_code,
            "reason_detail": str(first.get("reason_detail") or ""),
            "skip_reason": skip_reason,
            "count_in_denominator": bool(first.get("count_in_denominator")) if first.get("count_in_denominator") is not None else None,
            "path": str(first.get("_path") or ""),
        }
    return {}


def _missing_candidate_outcome(kind: str) -> dict[str, Any]:
    return {
        "status": "candidate_unavailable",
        "best_ratio": None,
        "best_qps_intentir": None,
        "best_qps_native": None,
        "first_candidate": {},
        "candidate_count": 0,
        "returncode": None,
        "failure": {
            "ok": None,
            "reason_code": "candidate_unavailable",
            "reason_detail": f"{kind} candidate unavailable",
            "skip_reason": "candidate_unavailable",
            "count_in_denominator": None,
            "path": "",
        },
    }


def _async_binding_keys(bindings: dict[str, int]) -> list[str]:
    return [str(k) for k, v in dict(bindings or {}).items() if int(v) and str(k).endswith("_ASYNC_COPY")]


def _find_guided_repair(guided_res: dict[str, Any], candidate: str) -> dict[str, Any]:
    kind, bindings = _parse_candidate_line(candidate)
    if not kind:
        return {}
    async_keys = _async_binding_keys(bindings)
    if not async_keys:
        return {}
    normalized = {str(k): int(v) for k, v in dict(bindings).items() if str(k) not in set(async_keys)}
    for guided in _candidate_summaries(guided_res):
        if str(guided.get("kernel_kind") or "") != kind:
            continue
        guided_bindings = {str(k): int(v) for k, v in dict(guided.get("bindings") or {}).items()}
        if guided_bindings != normalized:
            continue
        ratio = guided.get("ratio")
        if ratio is None:
            continue
        return {
            "status": "requires_substitution",
            "reason": "async_binding_removed",
            "dropped_bindings": list(async_keys),
            "repair_candidate": _candidate_line(kind, normalized),
            "repair_ratio": float(ratio),
        }
    return {}


def _repair_metric(row: dict[str, Any]) -> float | None:
    qps = row.get("qps_intentir")
    if qps is not None:
        try:
            return float(qps)
        except Exception:
            pass
    ratio = row.get("ratio")
    if ratio is not None:
        try:
            return float(ratio)
        except Exception:
            return None
    return None


def _find_flash_cluster_repair(
    *,
    guided_res: dict[str, Any],
    candidate: str,
    hardware_cluster: str,
    raw_ratio: float | None = None,
) -> dict[str, Any]:
    kind, bindings = _parse_candidate_line(candidate)
    if str(hardware_cluster) != "cuda_tc_mid_smem":
        return {}
    if kind not in {
        "attn2d_causal_softmax_v6",
        "attn2d_causal_softmax_v7",
        "attn2d_causal_softmax_v8",
        "attn2d_causal_softmax_v9",
    }:
        return {}
    raw_candidate = _candidate_line(kind, bindings)
    block_kv = _coerce_int(bindings.get("ATTN_BLOCK_KV"))
    best: dict[str, Any] | None = None
    best_value = float("-inf")
    for guided in _candidate_summaries(guided_res):
        guided_kind = str(guided.get("kernel_kind") or "")
        if guided_kind not in {
            "attn2d_causal_softmax_v6",
            "attn2d_causal_softmax_v8",
            "attn2d_causal_softmax_v9",
        }:
            continue
        gb = {str(k): int(v) for k, v in dict(guided.get("bindings") or {}).items()}
        if guided.get("ratio") is None and guided.get("qps_intentir") is None:
            continue
        cand_line = _candidate_line(guided_kind, gb)
        if cand_line == raw_candidate:
            continue
        if kind == "attn2d_causal_softmax_v7" and block_kv is not None:
            if int(gb.get("ATTN_BLOCK_KV", -1)) != int(block_kv):
                continue
        metric_value = _repair_metric(guided)
        if metric_value is None:
            continue
        if best is None or metric_value > best_value:
            best = guided
            best_value = metric_value
    if best is None:
        return {}
    repair_ratio = float(best["ratio"])
    if raw_ratio is not None and repair_ratio <= float(raw_ratio) + 0.01:
        return {}
    repair_kind = str(best.get("kernel_kind") or "")
    repair_bindings = dict(best.get("bindings") or {})
    reason = "cluster_variant_shift" if repair_kind != kind else "cluster_param_shift"
    return {
        "status": "requires_substitution",
        "reason": reason,
        "repair_candidate": _candidate_line(repair_kind, repair_bindings),
        "repair_ratio": repair_ratio,
    }


def _find_matmul_cluster_repair(
    *,
    guided_res: dict[str, Any],
    candidate: str,
    hardware_cluster: str,
) -> dict[str, Any]:
    kind, _bindings = _parse_candidate_line(candidate)
    if kind != "matmul_mma_tf32_v1" or str(hardware_cluster) != "cuda_tc_mid_smem":
        return {}
    best: dict[str, Any] | None = None
    best_value = float("-inf")
    for guided in _candidate_summaries(guided_res):
        guided_kind = str(guided.get("kernel_kind") or "")
        if guided_kind not in {"matmul_tile_v2", "matmul_tile_v1"}:
            continue
        metric_value = _repair_metric(guided)
        if metric_value is None:
            continue
        if best is None or metric_value > best_value:
            best = guided
            best_value = metric_value
    if best is None:
        return {}
    return {
        "status": "requires_substitution",
        "reason": "cluster_variant_shift",
        "repair_candidate": _candidate_line(str(best.get("kernel_kind") or ""), dict(best.get("bindings") or {})),
        "repair_ratio": float(best["ratio"]),
    }


def _find_row_softmax_cluster_repair(
    *,
    guided_res: dict[str, Any],
    candidate: str,
    hardware_cluster: str,
) -> dict[str, Any]:
    kind, _bindings = _parse_candidate_line(candidate)
    if kind != "row_softmax_axis1_triton_v1" or str(hardware_cluster) not in {"cuda_tc_mid_smem", "cuda_generic"}:
        return {}
    best: dict[str, Any] | None = None
    best_value = float("-inf")
    for guided in _candidate_summaries(guided_res):
        guided_kind = str(guided.get("kernel_kind") or "")
        if guided_kind not in {"row_softmax_axis1_v1", "row_softmax_axis1_triton_v1"}:
            continue
        metric_value = _repair_metric(guided)
        if metric_value is None:
            continue
        if best is None or metric_value > best_value:
            best = guided
            best_value = metric_value
    if best is None:
        return {}
    return {
        "status": "requires_substitution",
        "reason": "cluster_variant_shift",
        "repair_candidate": _candidate_line(str(best.get("kernel_kind") or ""), dict(best.get("bindings") or {})),
        "repair_ratio": float(best["ratio"]),
    }


def _best_repair(*repairs: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_ratio = float("-inf")
    for repair in repairs:
        if not repair:
            continue
        ratio = repair.get("repair_ratio")
        if ratio is None:
            if not best:
                best = dict(repair)
            continue
        fr = float(ratio)
        if not best or fr > best_ratio:
            best = dict(repair)
            best_ratio = fr
    return best


def _candidate_origin(*, source_oracle_kind: str, resolved_candidate: str, arch: str, kind: str) -> str:
    if not resolved_candidate:
        return ""
    if kind == "source" and str(source_oracle_kind or "").strip():
        return "plan.source_oracle"
    if str(arch or "").strip():
        return f"tuning_db:{arch}"
    return "derived"


def _make_outcome(result: dict[str, Any]) -> dict[str, Any]:
    if not result:
        return {
            "status": "missing",
            "best_ratio": None,
            "first_candidate": {},
            "candidate_count": 0,
            "returncode": None,
            "failure": {},
        }
    ratio = _best_ratio(result)
    first = _first_candidate_summary(result)
    failure = _first_failure_detail(result)
    summary = result.get("summary")
    candidates = list(summary.get("candidates") or []) if isinstance(summary, dict) else []
    returncode = result.get("returncode")
    if ratio is not None:
        best_ratio_row = _best_candidate(result, metric="ratio")
        best_qps_row = _best_candidate(result, metric="qps_intentir") or dict(best_ratio_row)
        return {
            "status": "ok",
            "best_ratio": float(ratio),
            "first_candidate": first,
            "best_candidate": dict(best_ratio_row),
            "best_qps_candidate": dict(best_qps_row),
            "best_qps_intentir": best_qps_row.get("qps_intentir"),
            "best_qps_native": best_qps_row.get("qps_native"),
            "best_latency_intentir_ms": best_qps_row.get("latency_intentir_ms"),
            "best_latency_native_ms": best_qps_row.get("latency_native_ms"),
            "requested_sm": best_qps_row.get("requested_sm"),
            "effective_sm": best_qps_row.get("effective_sm"),
            "downleveled": best_qps_row.get("downleveled"),
            "candidate_count": len(candidates),
            "returncode": returncode,
            "failure": failure,
        }
    if failure:
        return {
            "status": "failed",
            "best_ratio": None,
            "first_candidate": first,
            "best_candidate": {},
            "best_qps_intentir": None,
            "best_qps_native": None,
            "best_latency_intentir_ms": None,
            "best_latency_native_ms": None,
            "requested_sm": None,
            "effective_sm": None,
            "downleveled": None,
            "candidate_count": len(candidates),
            "returncode": returncode,
            "failure": failure,
        }
    if returncode not in (None, 0):
        return {
            "status": "process_error",
            "best_ratio": None,
            "first_candidate": first,
            "best_candidate": {},
            "best_qps_intentir": None,
            "best_qps_native": None,
            "best_latency_intentir_ms": None,
            "best_latency_native_ms": None,
            "requested_sm": None,
            "effective_sm": None,
            "downleveled": None,
            "candidate_count": len(candidates),
            "returncode": returncode,
            "failure": {
                "ok": False,
                "reason_code": "tune_returncode_nonzero",
                "reason_detail": f"tune returned non-zero exit code: {returncode}",
                "skip_reason": "",
                "count_in_denominator": None,
                "path": "",
            },
        }
    return {
        "status": "missing",
        "best_ratio": None,
        "first_candidate": first,
        "best_candidate": {},
        "best_qps_intentir": None,
        "best_qps_native": None,
        "best_latency_intentir_ms": None,
        "best_latency_native_ms": None,
        "requested_sm": None,
        "effective_sm": None,
        "downleveled": None,
        "candidate_count": len(candidates),
        "returncode": returncode,
        "failure": {},
    }


def _analyze_replay_candidate(
    *,
    label: str,
    candidate: str,
    candidate_origin: str,
    replay_result: dict[str, Any],
    guided_res: dict[str, Any],
    hardware_cluster: str,
) -> dict[str, Any]:
    if not str(candidate or "").strip():
        missing = _missing_candidate_outcome(label)
        return {
            "status": "candidate_unavailable",
            "candidate": "",
            "candidate_origin": "",
            "repair": {},
            "outcome": missing,
        }
    outcome = _make_outcome(replay_result)
    if outcome["status"] == "ok":
        flash_cluster_repair = _find_flash_cluster_repair(
            guided_res=guided_res,
            candidate=candidate,
            hardware_cluster=hardware_cluster,
            raw_ratio=outcome.get("best_ratio"),
        )
        matmul_cluster_repair = _find_matmul_cluster_repair(
            guided_res=guided_res,
            candidate=candidate,
            hardware_cluster=hardware_cluster,
        )
        row_softmax_cluster_repair = _find_row_softmax_cluster_repair(
            guided_res=guided_res,
            candidate=candidate,
            hardware_cluster=hardware_cluster,
        )
        cluster_repair = _best_repair(flash_cluster_repair, matmul_cluster_repair, row_softmax_cluster_repair)
        if cluster_repair:
            return {
                "status": str(cluster_repair.get("status") or "requires_substitution"),
                "candidate": str(candidate),
                "candidate_origin": str(candidate_origin),
                "repair": cluster_repair,
                "outcome": outcome,
            }
        return {
            "status": "replayable",
            "candidate": str(candidate),
            "candidate_origin": str(candidate_origin),
            "repair": {},
            "outcome": outcome,
        }
    repair = _best_repair(
        _find_guided_repair(guided_res, candidate),
        (
            _find_flash_cluster_repair(
                guided_res=guided_res,
                candidate=candidate,
                hardware_cluster=hardware_cluster,
                raw_ratio=outcome.get("best_ratio"),
            )
            if label in {"source_replay", "target_oracle"}
            else {}
        ),
        (
            _find_matmul_cluster_repair(
                guided_res=guided_res,
                candidate=candidate,
                hardware_cluster=hardware_cluster,
            )
            if label in {"source_replay", "target_oracle"}
            else {}
        ),
        (
            _find_row_softmax_cluster_repair(
                guided_res=guided_res,
                candidate=candidate,
                hardware_cluster=hardware_cluster,
            )
            if label in {"source_replay", "target_oracle"}
            else {}
        ),
    )
    if repair:
        return {
            "status": str(repair.get("status") or "requires_substitution"),
            "candidate": str(candidate),
            "candidate_origin": str(candidate_origin),
            "repair": repair,
            "outcome": outcome,
        }
    return {
        "status": str(outcome.get("status") or "missing"),
        "candidate": str(candidate),
        "candidate_origin": str(candidate_origin),
        "repair": {},
        "outcome": outcome,
    }


def _portable_outcome(
    *,
    label: str,
    analysis: dict[str, Any],
    raw_outcome: dict[str, Any],
    guided_res: dict[str, Any],
) -> dict[str, Any]:
    status = str(analysis.get("status") or "")
    candidate = str(analysis.get("candidate") or "")
    candidate_origin = str(analysis.get("candidate_origin") or "")
    if status == "replayable":
        return {
            "status": "raw_replayable",
            "best_ratio": raw_outcome.get("best_ratio"),
            "best_qps_intentir": raw_outcome.get("best_qps_intentir"),
            "best_qps_native": raw_outcome.get("best_qps_native"),
            "best_latency_intentir_ms": raw_outcome.get("best_latency_intentir_ms"),
            "best_latency_native_ms": raw_outcome.get("best_latency_native_ms"),
            "requested_sm": raw_outcome.get("requested_sm"),
            "effective_sm": raw_outcome.get("effective_sm"),
            "downleveled": raw_outcome.get("downleveled"),
            "candidate": candidate,
            "candidate_origin": candidate_origin,
            "reason": "raw_replayable",
            "first_candidate": dict(raw_outcome.get("first_candidate") or {}),
            "repair": {},
        }
    repair = dict(analysis.get("repair") or {})
    repair_candidate = str(repair.get("repair_candidate") or "")
    repair_ratio = repair.get("repair_ratio")
    if repair_candidate and repair_ratio is not None:
        repair_row = _find_candidate_summary(guided_res, repair_candidate)
        return {
            "status": "portable_repair_ok",
            "best_ratio": float(repair_ratio),
            "best_qps_intentir": repair_row.get("qps_intentir"),
            "best_qps_native": repair_row.get("qps_native"),
            "best_latency_intentir_ms": repair_row.get("latency_intentir_ms"),
            "best_latency_native_ms": repair_row.get("latency_native_ms"),
            "requested_sm": repair_row.get("requested_sm"),
            "effective_sm": repair_row.get("effective_sm"),
            "downleveled": repair_row.get("downleveled"),
            "candidate": repair_candidate,
            "candidate_origin": "guided_repair",
            "reason": str(repair.get("reason") or "repair"),
            "first_candidate": _candidate_summary_from_line(repair_candidate, ratio=float(repair_ratio)),
            "repair": repair,
        }
    missing = _missing_candidate_outcome(label)
    return {
        "status": ("candidate_unavailable" if status == "candidate_unavailable" else "portable_missing"),
        "best_ratio": None,
        "best_qps_intentir": None,
        "best_qps_native": None,
        "best_latency_intentir_ms": None,
        "best_latency_native_ms": None,
        "requested_sm": None,
        "effective_sm": None,
        "downleveled": None,
        "candidate": repair_candidate,
        "candidate_origin": ("guided_repair" if repair_candidate else candidate_origin),
        "reason": (str(repair.get("reason") or "") or str((raw_outcome.get("failure") or {}).get("reason_code") or "") or "portable_missing"),
        "first_candidate": {},
        "repair": repair,
        "failure": (dict(raw_outcome.get("failure") or {}) if raw_outcome else dict(missing.get("failure") or {})),
    }


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or float(den) == 0.0:
        return None
    return float(num) / float(den)


def _median_or_none(values: list[float]) -> float | None:
    vals = [float(x) for x in values if x is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def _spread_ratio(values: list[float]) -> float | None:
    vals = [float(x) for x in values if x is not None]
    if len(vals) < 2:
        return None
    lo = min(vals)
    hi = max(vals)
    if lo == 0.0:
        return None
    return float(hi / lo)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare source-oracle replay vs ORG-guided candidates on target GPU.")
    parser.add_argument("--report", required=True, help="Path to <kernel>.json report emitted by full_pipeline_verify.")
    parser.add_argument("--backend-target", required=True, choices=["cuda_h100", "cuda_5090d"])
    parser.add_argument("--source-arch", default="sm90")
    parser.add_argument("--tuning-db", default="", help="Optional tuning_db path (defaults to repo cuda.jsonl).")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--cases-limit", type=int, default=1)
    parser.add_argument("--perf-warmup", type=int, default=1)
    parser.add_argument("--perf-iters", type=int, default=5)
    parser.add_argument("--perf-repeats", type=int, default=1)
    parser.add_argument("--cuda-runtime-backend", default="nvrtc", choices=["auto", "nvcc", "nvrtc"])
    args = parser.parse_args()

    report_path = Path(args.report).resolve()
    report = _load_json(report_path)
    kernel = str(report_path.stem)
    org = dict(report.get("org") or {})
    hardware_model = dict(org.get("hardware_model") or {})
    hardware_cluster = str(hardware_model.get("arch_cluster") or "")
    plan_path = Path(str(org.get("plan_path") or report_path.with_name(f"{kernel}.org_plan.json"))).resolve()
    candidates_txt_path = Path(str(org.get("candidates_txt_path") or report_path.with_name(f"{kernel}.org_candidates.txt"))).resolve()
    if not plan_path.is_file():
        raise SystemExit(f"missing plan file: {plan_path}")
    if not candidates_txt_path.is_file():
        raise SystemExit(f"missing guided candidates file: {candidates_txt_path}")

    plan = _load_json(plan_path)
    source_context = dict((report.get("org_doc") or {}).get("source_context") or {})
    shape_bindings = {str(k): int(v) for k, v in dict((org.get("shape_bindings") or source_context.get("shape_bindings") or {})).items() if str(k).strip()}
    if not shape_bindings:
        shape_bindings = {str(k): int(v) for k, v in dict((report.get("baseline") or {}).get("shapes") or {}).items() if str(k).strip()}
    compiler_stack = str((report.get("org") or {}).get("compiler_stack") or "").strip().lower()
    if not compiler_stack:
        compiler_stack = _infer_compiler_stack_from_candidate_file(candidates_txt_path)
    if not compiler_stack:
        compiler_stack = "python"
    target_arch = str((org.get("arch") or "")).strip()
    db_path = (Path(args.tuning_db).resolve() if str(args.tuning_db).strip() else None)
    source_oracle_kind = str(dict(plan.get("source_oracle") or {}).get("kernel_kind") or "").strip()
    source_compiler_stack = str(dict(plan.get("source_oracle") or {}).get("compiler_stack") or compiler_stack or "python").strip().lower() or "python"
    target_compiler_stack = str(compiler_stack or "python").strip().lower() or "python"
    compiler_cpp_wave = str((report.get("org") or {}).get("compiler_cpp_wave") or os.getenv("INTENTIR_COMPILER_CPP_WAVE", "") or "").strip().lower()
    toolchain_env = _toolchain_env_from_report(report)

    source_candidate = _resolve_source_candidate(
        kernel=kernel,
        shape_bindings=shape_bindings,
        compiler_stack=source_compiler_stack,
        source_arch=str(args.source_arch),
        db_path=db_path,
        plan=plan,
    )
    target_candidate = _resolve_target_oracle_candidate(
        kernel=kernel,
        shape_bindings=shape_bindings,
        compiler_stack=target_compiler_stack,
        target_arch=target_arch,
        db_path=db_path,
    )
    source_candidate_origin = _candidate_origin(
        source_oracle_kind=source_oracle_kind,
        resolved_candidate=source_candidate,
        arch=str(args.source_arch),
        kind="source",
    )
    target_candidate_origin = _candidate_origin(
        source_oracle_kind="",
        resolved_candidate=target_candidate,
        arch=target_arch,
        kind="target",
    )

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    guided_res = _run_tune(
        kernel=kernel,
        backend_target=args.backend_target,
        out_root=out_root / "guided",
        candidate_file=candidates_txt_path,
        cases_limit=args.cases_limit,
        perf_warmup=args.perf_warmup,
        perf_iters=args.perf_iters,
        perf_repeats=args.perf_repeats,
        cuda_runtime_backend=args.cuda_runtime_backend,
        compiler_stack=target_compiler_stack,
        compiler_cpp_wave=compiler_cpp_wave,
        env_overrides=toolchain_env,
    )
    source_res = {}
    if source_candidate:
        source_res = _run_tune(
            kernel=kernel,
            backend_target=args.backend_target,
            out_root=out_root / "source_replay",
            candidate=source_candidate,
            cases_limit=args.cases_limit,
            perf_warmup=args.perf_warmup,
            perf_iters=args.perf_iters,
            perf_repeats=args.perf_repeats,
            cuda_runtime_backend=args.cuda_runtime_backend,
            compiler_stack=source_compiler_stack,
            compiler_cpp_wave=compiler_cpp_wave,
            env_overrides=toolchain_env,
        )
    target_res = {}
    if target_candidate:
        target_res = _run_tune(
            kernel=kernel,
            backend_target=args.backend_target,
            out_root=out_root / "target_oracle",
            candidate=target_candidate,
            cases_limit=args.cases_limit,
            perf_warmup=args.perf_warmup,
            perf_iters=args.perf_iters,
            perf_repeats=args.perf_repeats,
            cuda_runtime_backend=args.cuda_runtime_backend,
            compiler_stack=target_compiler_stack,
            compiler_cpp_wave=compiler_cpp_wave,
            env_overrides=toolchain_env,
        )

    guided_outcome = _make_outcome(guided_res)
    source_outcome = (_make_outcome(source_res) if source_candidate else _missing_candidate_outcome("source_replay"))
    target_outcome = (_make_outcome(target_res) if target_candidate else _missing_candidate_outcome("target_oracle"))
    source_analysis = _analyze_replay_candidate(
        label="source_replay",
        candidate=source_candidate,
        candidate_origin=source_candidate_origin,
        replay_result=source_res,
        guided_res=guided_res,
        hardware_cluster=hardware_cluster,
    )
    target_analysis = _analyze_replay_candidate(
        label="target_oracle",
        candidate=target_candidate,
        candidate_origin=target_candidate_origin,
        replay_result=target_res,
        guided_res=guided_res,
        hardware_cluster=hardware_cluster,
    )
    source_portable = _portable_outcome(label="source_replay", analysis=source_analysis, raw_outcome=source_outcome, guided_res=guided_res)
    target_portable = _portable_outcome(label="target_oracle", analysis=target_analysis, raw_outcome=target_outcome, guided_res=guided_res)

    payload = {
        "kernel": kernel,
        "backend_target": str(args.backend_target),
        "source_arch": str(args.source_arch),
        "target_arch": target_arch,
        "shape_bindings": shape_bindings,
        "compiler_stack": compiler_stack,
        "compiler_cpp_wave": compiler_cpp_wave,
        "toolchain_env": dict(toolchain_env),
        "guided_compiler_stack": target_compiler_stack,
        "source_compiler_stack": source_compiler_stack,
        "target_compiler_stack": target_compiler_stack,
        "evidence_source": dict(org.get("evidence_source") or {}),
        "hardware_model": hardware_model,
        "guided_candidate_file": str(candidates_txt_path),
        "source_candidate": source_candidate,
        "source_candidate_origin": source_candidate_origin,
        "target_candidate": target_candidate,
        "target_candidate_origin": target_candidate_origin,
        "guided": guided_res,
        "source_replay": source_res,
        "target_oracle": target_res,
        "comparisons": {
            "guided_best_ratio": _best_ratio(guided_res),
            "guided_best_qps_intentir": _make_outcome(guided_res).get("best_qps_intentir"),
            "guided_best_qps_native": _make_outcome(guided_res).get("best_qps_native"),
            "guided_requested_sm": guided_outcome.get("requested_sm"),
            "guided_effective_sm": guided_outcome.get("effective_sm"),
            "guided_downleveled": guided_outcome.get("downleveled"),
            "source_replay_raw_ratio": _best_ratio(source_res),
            "source_replay_raw_qps_intentir": _make_outcome(source_res).get("best_qps_intentir"),
            "source_replay_raw_qps_native": _make_outcome(source_res).get("best_qps_native"),
            "source_replay_requested_sm": source_outcome.get("requested_sm"),
            "source_replay_effective_sm": source_outcome.get("effective_sm"),
            "source_replay_downleveled": source_outcome.get("downleveled"),
            "source_replay_portable_ratio": source_portable.get("best_ratio"),
            "source_replay_portable_qps_intentir": source_portable.get("best_qps_intentir"),
            "source_replay_portable_qps_native": source_portable.get("best_qps_native"),
            "source_replay_portable_requested_sm": source_portable.get("requested_sm"),
            "source_replay_portable_effective_sm": source_portable.get("effective_sm"),
            "source_replay_portable_downleveled": source_portable.get("downleveled"),
            "target_oracle_raw_ratio": _best_ratio(target_res),
            "target_oracle_raw_qps_intentir": _make_outcome(target_res).get("best_qps_intentir"),
            "target_oracle_raw_qps_native": _make_outcome(target_res).get("best_qps_native"),
            "target_oracle_requested_sm": target_outcome.get("requested_sm"),
            "target_oracle_effective_sm": target_outcome.get("effective_sm"),
            "target_oracle_downleveled": target_outcome.get("downleveled"),
            "target_oracle_portable_ratio": target_portable.get("best_ratio"),
            "target_oracle_portable_qps_intentir": target_portable.get("best_qps_intentir"),
            "target_oracle_portable_qps_native": target_portable.get("best_qps_native"),
            "target_oracle_portable_requested_sm": target_portable.get("requested_sm"),
            "target_oracle_portable_effective_sm": target_portable.get("effective_sm"),
            "target_oracle_portable_downleveled": target_portable.get("downleveled"),
            "source_replay_best_ratio": _best_ratio(source_res),
            "target_oracle_best_ratio": _best_ratio(target_res),
            "guided_first_candidate": _first_candidate_summary(guided_res),
            "source_replay_first_candidate": _first_candidate_summary(source_res),
            "target_oracle_first_candidate": _first_candidate_summary(target_res),
            "guided_failure": _first_failure_detail(guided_res),
            "source_replay_failure": _first_failure_detail(source_res),
            "target_oracle_failure": _first_failure_detail(target_res),
            "guided_outcome": guided_outcome,
            "source_replay_outcome": source_outcome,
            "target_oracle_outcome": target_outcome,
            "source_replay_analysis": source_analysis,
            "target_oracle_analysis": target_analysis,
            "source_replay_portable_outcome": source_portable,
            "target_oracle_portable_outcome": target_portable,
        },
    }
    gp = payload["comparisons"]["guided_best_ratio"]
    sp_raw = payload["comparisons"]["source_replay_raw_ratio"]
    sp_portable = payload["comparisons"]["source_replay_portable_ratio"]
    tp_raw = payload["comparisons"]["target_oracle_raw_ratio"]
    tp_portable = payload["comparisons"]["target_oracle_portable_ratio"]
    native_qps_values = [
        payload["comparisons"].get("guided_best_qps_native"),
        payload["comparisons"].get("source_replay_raw_qps_native"),
        payload["comparisons"].get("target_oracle_raw_qps_native"),
    ]
    shared_native_qps = _median_or_none(native_qps_values)
    payload["comparisons"]["shared_native_qps"] = shared_native_qps
    payload["comparisons"]["native_qps_spread_ratio"] = _spread_ratio(native_qps_values)
    payload["comparisons"]["guided_shared_native_ratio"] = _safe_ratio(payload["comparisons"].get("guided_best_qps_intentir"), shared_native_qps)
    payload["comparisons"]["source_replay_portable_shared_native_ratio"] = _safe_ratio(
        payload["comparisons"].get("source_replay_portable_qps_intentir"), shared_native_qps
    )
    payload["comparisons"]["target_oracle_portable_shared_native_ratio"] = _safe_ratio(
        payload["comparisons"].get("target_oracle_portable_qps_intentir"), shared_native_qps
    )
    payload["comparisons"]["guided_vs_source_replay_raw"] = _safe_ratio(gp, sp_raw)
    payload["comparisons"]["guided_vs_source_replay_portable"] = _safe_ratio(gp, sp_portable)
    payload["comparisons"]["guided_vs_target_oracle_raw"] = _safe_ratio(gp, tp_raw)
    payload["comparisons"]["guided_vs_portable_target_oracle"] = _safe_ratio(gp, tp_portable)
    payload["comparisons"]["guided_vs_source_replay"] = payload["comparisons"]["guided_vs_source_replay_raw"]
    payload["comparisons"]["guided_vs_target_oracle"] = payload["comparisons"]["guided_vs_target_oracle_raw"]

    out_file = out_root / "comparison.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"kernel: {kernel}",
        f"backend_target: {args.backend_target}",
        f"source_arch: {args.source_arch}",
        f"target_arch: {target_arch}",
        f"compiler_stack: {compiler_stack}",
        f"compiler_cpp_wave: {compiler_cpp_wave}",
        f"guided_compiler_stack: {target_compiler_stack}",
        f"source_compiler_stack: {source_compiler_stack}",
        f"target_compiler_stack: {target_compiler_stack}",
        f"shape_bindings: {json.dumps(shape_bindings, ensure_ascii=False, sort_keys=True)}",
        f"evidence_primary: {str((payload.get('evidence_source') or {}).get('primary') or '')}",
        f"hardware_cluster: {str(hardware_model.get('arch_cluster') or '')}",
        f"guided_best_ratio: {payload['comparisons']['guided_best_ratio']}",
        f"guided_best_qps_intentir: {payload['comparisons']['guided_best_qps_intentir']}",
        f"guided_best_qps_native: {payload['comparisons']['guided_best_qps_native']}",
        f"guided_requested_sm: {payload['comparisons']['guided_requested_sm']}",
        f"guided_effective_sm: {payload['comparisons']['guided_effective_sm']}",
        f"guided_downleveled: {payload['comparisons']['guided_downleveled']}",
        f"source_replay_raw_ratio: {payload['comparisons']['source_replay_raw_ratio']}",
        f"source_replay_raw_qps_intentir: {payload['comparisons']['source_replay_raw_qps_intentir']}",
        f"source_replay_raw_qps_native: {payload['comparisons']['source_replay_raw_qps_native']}",
        f"source_replay_requested_sm: {payload['comparisons']['source_replay_requested_sm']}",
        f"source_replay_effective_sm: {payload['comparisons']['source_replay_effective_sm']}",
        f"source_replay_downleveled: {payload['comparisons']['source_replay_downleveled']}",
        f"source_replay_portable_ratio: {payload['comparisons']['source_replay_portable_ratio']}",
        f"source_replay_portable_qps_intentir: {payload['comparisons']['source_replay_portable_qps_intentir']}",
        f"source_replay_portable_qps_native: {payload['comparisons']['source_replay_portable_qps_native']}",
        f"source_replay_portable_requested_sm: {payload['comparisons']['source_replay_portable_requested_sm']}",
        f"source_replay_portable_effective_sm: {payload['comparisons']['source_replay_portable_effective_sm']}",
        f"source_replay_portable_downleveled: {payload['comparisons']['source_replay_portable_downleveled']}",
        f"target_oracle_raw_ratio: {payload['comparisons']['target_oracle_raw_ratio']}",
        f"target_oracle_raw_qps_intentir: {payload['comparisons']['target_oracle_raw_qps_intentir']}",
        f"target_oracle_raw_qps_native: {payload['comparisons']['target_oracle_raw_qps_native']}",
        f"target_oracle_requested_sm: {payload['comparisons']['target_oracle_requested_sm']}",
        f"target_oracle_effective_sm: {payload['comparisons']['target_oracle_effective_sm']}",
        f"target_oracle_downleveled: {payload['comparisons']['target_oracle_downleveled']}",
        f"target_oracle_portable_ratio: {payload['comparisons']['target_oracle_portable_ratio']}",
        f"target_oracle_portable_qps_intentir: {payload['comparisons']['target_oracle_portable_qps_intentir']}",
        f"target_oracle_portable_qps_native: {payload['comparisons']['target_oracle_portable_qps_native']}",
        f"target_oracle_portable_requested_sm: {payload['comparisons']['target_oracle_portable_requested_sm']}",
        f"target_oracle_portable_effective_sm: {payload['comparisons']['target_oracle_portable_effective_sm']}",
        f"target_oracle_portable_downleveled: {payload['comparisons']['target_oracle_portable_downleveled']}",
        f"shared_native_qps: {payload['comparisons']['shared_native_qps']}",
        f"native_qps_spread_ratio: {payload['comparisons']['native_qps_spread_ratio']}",
        f"guided_shared_native_ratio: {payload['comparisons']['guided_shared_native_ratio']}",
        f"source_replay_portable_shared_native_ratio: {payload['comparisons']['source_replay_portable_shared_native_ratio']}",
        f"target_oracle_portable_shared_native_ratio: {payload['comparisons']['target_oracle_portable_shared_native_ratio']}",
        f"guided_vs_source_replay_raw: {payload['comparisons']['guided_vs_source_replay_raw']}",
        f"guided_vs_source_replay_portable: {payload['comparisons']['guided_vs_source_replay_portable']}",
        f"guided_vs_target_oracle_raw: {payload['comparisons']['guided_vs_target_oracle_raw']}",
        f"guided_vs_portable_target_oracle: {payload['comparisons']['guided_vs_portable_target_oracle']}",
        f"guided_outcome: {payload['comparisons']['guided_outcome']['status']}",
        f"source_replay_outcome: {payload['comparisons']['source_replay_outcome']['status']}",
        f"target_oracle_outcome: {payload['comparisons']['target_oracle_outcome']['status']}",
        f"source_replay_analysis: {payload['comparisons']['source_replay_analysis']['status']}",
        f"target_oracle_analysis: {payload['comparisons']['target_oracle_analysis']['status']}",
        f"source_replay_portable_outcome: {payload['comparisons']['source_replay_portable_outcome']['status']}",
        f"target_oracle_portable_outcome: {payload['comparisons']['target_oracle_portable_outcome']['status']}",
    ]
    fail = payload["comparisons"].get("source_replay_failure") or {}
    if fail:
        lines.append(
            "source_replay_failure: "
            + ", ".join(
                [
                    f"ok={fail.get('ok')}",
                    f"reason_code={fail.get('reason_code')}",
                    f"skip_reason={fail.get('skip_reason')}",
                ]
            )
        )
    fail = payload["comparisons"].get("target_oracle_failure") or {}
    if fail:
        lines.append(
            "target_oracle_failure: "
            + ", ".join(
                [
                    f"ok={fail.get('ok')}",
                    f"reason_code={fail.get('reason_code')}",
                    f"skip_reason={fail.get('skip_reason')}",
                ]
            )
        )
    repair = dict((payload["comparisons"].get("source_replay_analysis") or {}).get("repair") or {})
    if repair:
        lines.append(
            "source_replay_repair: "
            + ", ".join(
                [
                    f"reason={repair.get('reason')}",
                    f"repair_candidate={repair.get('repair_candidate')}",
                    f"repair_ratio={repair.get('repair_ratio')}",
                ]
            )
        )
    portable = dict(payload["comparisons"].get("source_replay_portable_outcome") or {})
    if portable:
        lines.append(
            "source_replay_portable: "
            + ", ".join(
                [
                    f"candidate={portable.get('candidate')}",
                    f"reason={portable.get('reason')}",
                    f"ratio={portable.get('best_ratio')}",
                ]
            )
        )
    repair = dict((payload["comparisons"].get("target_oracle_analysis") or {}).get("repair") or {})
    if repair:
        lines.append(
            "target_oracle_repair: "
            + ", ".join(
                [
                    f"reason={repair.get('reason')}",
                    f"repair_candidate={repair.get('repair_candidate')}",
                    f"repair_ratio={repair.get('repair_ratio')}",
                ]
            )
        )
    portable = dict(payload["comparisons"].get("target_oracle_portable_outcome") or {})
    if portable:
        lines.append(
            "target_oracle_portable: "
            + ", ".join(
                [
                    f"candidate={portable.get('candidate')}",
                    f"reason={portable.get('reason')}",
                    f"ratio={portable.get('best_ratio')}",
                ]
            )
        )
    (out_root / "comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"comparison: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
