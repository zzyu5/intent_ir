from __future__ import annotations

import argparse
import json
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
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    result: dict[str, Any] = {
        "command": cmd,
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
            }
        )
    return out


def _best_ratio(result: dict[str, Any]) -> float | None:
    summary = result.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    candidates = list(summary.get("candidates") or [])
    ratios = [float(c.get("ratio")) for c in candidates if c.get("ratio") is not None]
    if ratios:
        return max(ratios)
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
        return {
            "status": "ok",
            "best_ratio": float(ratio),
            "first_candidate": first,
            "candidate_count": len(candidates),
            "returncode": returncode,
            "failure": failure,
        }
    if failure:
        return {
            "status": "failed",
            "best_ratio": None,
            "first_candidate": first,
            "candidate_count": len(candidates),
            "returncode": returncode,
            "failure": failure,
        }
    if returncode not in (None, 0):
        return {
            "status": "process_error",
            "best_ratio": None,
            "first_candidate": first,
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
        return {
            "status": "replayable",
            "candidate": str(candidate),
            "candidate_origin": str(candidate_origin),
            "repair": {},
            "outcome": outcome,
        }
    repair = _find_guided_repair(guided_res, candidate)
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
    compiler_stack = str((report.get("org") or {}).get("compiler_stack") or "python")
    target_arch = str((org.get("arch") or "")).strip()
    db_path = (Path(args.tuning_db).resolve() if str(args.tuning_db).strip() else None)
    source_oracle_kind = str(dict(plan.get("source_oracle") or {}).get("kernel_kind") or "").strip()

    source_candidate = _resolve_source_candidate(
        kernel=kernel,
        shape_bindings=shape_bindings,
        compiler_stack=compiler_stack,
        source_arch=str(args.source_arch),
        db_path=db_path,
        plan=plan,
    )
    target_candidate = _resolve_target_oracle_candidate(
        kernel=kernel,
        shape_bindings=shape_bindings,
        compiler_stack=compiler_stack,
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
    )
    target_analysis = _analyze_replay_candidate(
        label="target_oracle",
        candidate=target_candidate,
        candidate_origin=target_candidate_origin,
        replay_result=target_res,
        guided_res=guided_res,
    )

    payload = {
        "kernel": kernel,
        "backend_target": str(args.backend_target),
        "source_arch": str(args.source_arch),
        "target_arch": target_arch,
        "shape_bindings": shape_bindings,
        "compiler_stack": compiler_stack,
        "evidence_source": dict(org.get("evidence_source") or {}),
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
        },
    }
    gp = payload["comparisons"]["guided_best_ratio"]
    sp = payload["comparisons"]["source_replay_best_ratio"]
    tp = payload["comparisons"]["target_oracle_best_ratio"]
    payload["comparisons"]["guided_vs_source_replay"] = (None if gp is None or sp is None or sp == 0 else gp / sp)
    payload["comparisons"]["guided_vs_target_oracle"] = (None if gp is None or tp is None or tp == 0 else gp / tp)

    out_file = out_root / "comparison.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"kernel: {kernel}",
        f"backend_target: {args.backend_target}",
        f"source_arch: {args.source_arch}",
        f"target_arch: {target_arch}",
        f"compiler_stack: {compiler_stack}",
        f"shape_bindings: {json.dumps(shape_bindings, ensure_ascii=False, sort_keys=True)}",
        f"evidence_primary: {str((payload.get('evidence_source') or {}).get('primary') or '')}",
        f"guided_best_ratio: {payload['comparisons']['guided_best_ratio']}",
        f"source_replay_best_ratio: {payload['comparisons']['source_replay_best_ratio']}",
        f"target_oracle_best_ratio: {payload['comparisons']['target_oracle_best_ratio']}",
        f"guided_vs_source_replay: {payload['comparisons']['guided_vs_source_replay']}",
        f"guided_vs_target_oracle: {payload['comparisons']['guided_vs_target_oracle']}",
        f"guided_outcome: {payload['comparisons']['guided_outcome']['status']}",
        f"source_replay_outcome: {payload['comparisons']['source_replay_outcome']['status']}",
        f"target_oracle_outcome: {payload['comparisons']['target_oracle_outcome']['status']}",
        f"source_replay_analysis: {payload['comparisons']['source_replay_analysis']['status']}",
        f"target_oracle_analysis: {payload['comparisons']['target_oracle_analysis']['status']}",
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
    (out_root / "comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"comparison: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
