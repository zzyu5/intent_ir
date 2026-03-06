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

    def _best_ratio(result: dict[str, Any]) -> float | None:
        summary = result.get("summary")
        if not isinstance(summary, dict):
            summary = {}
        candidates = list(summary.get("candidates") or [])
        ratios = [float(c.get("ratio")) for c in candidates if c.get("ratio") is not None]
        if ratios:
            return max(ratios)
        run_summaries = []
        out_root_local = Path(str(result.get("out_root") or ""))
        if out_root_local.is_dir():
            run_summaries = list(out_root_local.rglob("run_summary.json"))
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

    payload = {
        "kernel": kernel,
        "backend_target": str(args.backend_target),
        "source_arch": str(args.source_arch),
        "target_arch": target_arch,
        "guided_candidate_file": str(candidates_txt_path),
        "source_candidate": source_candidate,
        "target_candidate": target_candidate,
        "guided": guided_res,
        "source_replay": source_res,
        "target_oracle": target_res,
        "comparisons": {
            "guided_best_ratio": _best_ratio(guided_res),
            "source_replay_best_ratio": _best_ratio(source_res),
            "target_oracle_best_ratio": _best_ratio(target_res),
        },
    }
    gp = payload["comparisons"]["guided_best_ratio"]
    sp = payload["comparisons"]["source_replay_best_ratio"]
    tp = payload["comparisons"]["target_oracle_best_ratio"]
    payload["comparisons"]["guided_vs_source_replay"] = (None if gp is None or sp is None or sp == 0 else gp / sp)
    payload["comparisons"]["guided_vs_target_oracle"] = (None if gp is None or tp is None or tp == 0 else gp / tp)

    out_file = out_root / "comparison.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"comparison: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
