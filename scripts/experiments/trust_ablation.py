"""
Paper experiment (E2): trust ablation for mutation-kill.

Goal:
  Compare falsification power with/without static validation (certificate/obligations).

We evaluate 3 variants:
  - full:      static_validate(intent, cert_v2)
  - generic:   static_validate(intent, empty_cert)    (no certificate evidence)
  - diff_only: no static validation (dynamic diff + metamorphic only)

This script intentionally reuses existing pipeline artifacts under:
  - artifacts/full_pipeline_verify/         (frontend=triton)
  - artifacts/tilelang_full_pipeline/       (frontend=tilelang)
  - artifacts/cuda_full_pipeline/           (frontend=cuda)

If a report JSON is missing (or --refresh-artifacts is set), it will run the
full per-kernel pipeline to regenerate the report first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontends.common.certificate_v2 import SemanticCertificateV2
from frontends.common.static_validate import StaticValidationResult, static_validate
from intent_ir.ir import IntentFunction
from verify.gen_cases import TestCase
from verify.mutation import MutationReport, run_mutation_kill


DEFAULT6 = [
    "any_kernel_dim",
    "group_norm_kernel",
    "_attn_fwd",
    "softmax_inner",
    "layer_norm_persistent",
    "upsample_bicubic2d_aa",
]


def _artifact_dir(frontend: str) -> Path:
    if frontend == "triton":
        return ROOT / "artifacts" / "full_pipeline_verify"
    if frontend == "tilelang":
        return ROOT / "artifacts" / "tilelang_full_pipeline"
    if frontend == "cuda":
        return ROOT / "artifacts" / "cuda_full_pipeline"
    raise ValueError(f"unknown frontend: {frontend}")


def _spec_from_pipeline(frontend: str, kernel: str) -> Any:
    """
    Recover the kernel spec object used by the frontend adapter.

    We reuse pipeline's coverage suites so we don't hard-code kernel locations here.
    """
    if frontend == "triton":
        from pipeline.triton.core import coverage_kernel_specs  # noqa: PLC0415

        for s in coverage_kernel_specs():
            if getattr(s, "name", None) == kernel:
                return s
        raise KeyError(kernel)
    if frontend == "tilelang":
        from pipeline.tilelang.core import coverage_kernel_specs  # noqa: PLC0415

        for s in coverage_kernel_specs():
            if getattr(s, "name", None) == kernel:
                return s
        raise KeyError(kernel)
    if frontend == "cuda":
        from pipeline.cuda.core import coverage_kernel_specs  # noqa: PLC0415

        for s in coverage_kernel_specs():
            if getattr(s, "name", None) == kernel:
                return s
        raise KeyError(kernel)
    raise ValueError(f"unknown frontend: {frontend}")


def _run_pipeline(frontend: str, kernel: str, *, out_dir: Path, cases_limit: int) -> Dict[str, Any]:
    spec = _spec_from_pipeline(frontend, kernel)
    if frontend == "triton":
        from pipeline.triton.core import run_pipeline_for_spec  # noqa: PLC0415

        return run_pipeline_for_spec(spec, out_dir=out_dir, cases_limit=cases_limit)  # type: ignore[no-any-return]
    if frontend == "tilelang":
        from pipeline.tilelang.core import run_pipeline_for_spec  # noqa: PLC0415

        return run_pipeline_for_spec(spec, out_dir=out_dir, cases_limit=cases_limit)  # type: ignore[no-any-return]
    if frontend == "cuda":
        from pipeline.cuda.core import run_pipeline_for_spec  # noqa: PLC0415

        return run_pipeline_for_spec(spec, out_dir=out_dir, cases_limit=cases_limit)  # type: ignore[no-any-return]
    raise ValueError(f"unknown frontend: {frontend}")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cert_v2_from_json(d: Dict[str, Any]) -> SemanticCertificateV2:
    return SemanticCertificateV2(
        schema_version=str(d.get("schema_version") or "cert_v2.0"),
        semantic_facts=(d.get("semantic_facts") if isinstance(d.get("semantic_facts"), dict) else {}),
        schedule_hints=(d.get("schedule_hints") if isinstance(d.get("schedule_hints"), dict) else {}),
        meta=(d.get("meta") if isinstance(d.get("meta"), dict) else {}),
    ).canonicalize()


def _cases_from_report(report: Dict[str, Any], *, limit: int) -> List[TestCase]:
    cases = report.get("cases") or {}
    in_contract = cases.get("in_contract") if isinstance(cases, dict) else None
    if not isinstance(in_contract, list):
        return []
    out: List[TestCase] = []
    for i, c in enumerate(in_contract[: max(0, int(limit))]):
        if isinstance(c, dict):
            out.append(TestCase(shapes={str(k): int(v) for k, v in c.items()}, dtypes={}, seed=int(i)))
    return out


def _tolerances_from_report(report: Dict[str, Any]) -> Tuple[float, float]:
    tol = report.get("tolerances") or {}
    if not isinstance(tol, dict):
        return 1e-3, 1e-3
    try:
        atol = float(tol.get("atol", 1e-3))
        rtol = float(tol.get("rtol", 1e-3))
        return atol, rtol
    except Exception:
        return 1e-3, 1e-3


def _summarize_cex(mr: MutationReport) -> Dict[str, Any]:
    """
    Summarize time-to-counterexample info for mutants killed by Stage B.
    """
    cases: List[int] = []
    times: List[float] = []
    for o in mr.outcomes:
        if o.killed_by != "B_diff":
            continue
        cc = getattr(o, "cases_checked", None)
        ts = getattr(o, "time_s", None)
        if isinstance(cc, int):
            cases.append(int(cc))
        if isinstance(ts, (int, float)):
            times.append(float(ts))
    cases.sort()
    times.sort()
    p50_cases = cases[len(cases) // 2] if cases else None
    p50_time = times[len(times) // 2] if times else None
    return {
        "n": int(len(cases)),
        "avg_cases": (float(sum(cases)) / float(len(cases)) if cases else None),
        "p50_cases": p50_cases,
        "avg_time_s": (float(sum(times)) / float(len(times)) if times else None),
        "p50_time_s": p50_time,
    }


@dataclass(frozen=True)
class Mode:
    name: str
    static_validate_fn: Optional[Callable[[IntentFunction], StaticValidationResult]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang", "cuda"], default="triton")
    ap.add_argument("--suite", choices=["default6", "coverage"], default="default6")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; restrict to kernel name(s)")
    ap.add_argument("--refresh-artifacts", action="store_true", help="regenerate per-kernel pipeline report before ablation")
    ap.add_argument("--cases-limit", type=int, default=8, help="used only when regenerating pipeline artifacts")
    ap.add_argument("--diff-cases", type=int, default=2, help="how many in-contract cases to use for mutation stage-B diff")
    ap.add_argument("--n-mutants", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bounded", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "experiments" / "e2_trust_ablation_latest.json"))
    args = ap.parse_args()

    frontend = str(args.frontend)
    out_dir = _artifact_dir(frontend)
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(args.suite) == "coverage":
        # Keep suite definition consistent with pipeline core.
        if frontend == "triton":
            from pipeline.triton.core import coverage_kernel_specs  # noqa: PLC0415

            kernels = [s.name for s in coverage_kernel_specs()]
        elif frontend == "tilelang":
            from pipeline.tilelang.core import coverage_kernel_specs  # noqa: PLC0415

            kernels = [s.name for s in coverage_kernel_specs()]
        else:
            from pipeline.cuda.core import coverage_kernel_specs  # noqa: PLC0415

            kernels = [s.name for s in coverage_kernel_specs()]
    else:
        kernels = list(DEFAULT6)

    wanted = {str(x) for x in (args.kernel or []) if str(x).strip()}
    if wanted:
        kernels = [k for k in kernels if k in wanted]

    modes: List[Mode] = []
    empty_cert = SemanticCertificateV2()
    modes.append(Mode(name="full", static_validate_fn=None))  # filled per-kernel (captures cert)
    modes.append(Mode(name="generic", static_validate_fn=(lambda m, _c=empty_cert: static_validate(m, _c))))
    modes.append(Mode(name="diff_only", static_validate_fn=None))

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for k in kernels:
        report_path = out_dir / f"{k}.json"
        if args.refresh_artifacts or not report_path.exists():
            _run_pipeline(frontend, k, out_dir=out_dir, cases_limit=int(args.cases_limit))
        if not report_path.exists():
            results.append({"kernel": k, "status": "MISSING_REPORT"})
            continue

        report = _read_json(report_path)
        if not isinstance(report.get("diff"), dict) or not bool((report.get("diff") or {}).get("ok", False)):
            results.append({"kernel": k, "status": "BASELINE_DIFF_NOT_OK"})
            continue
        intent_json = report.get("intent")
        cert_json = report.get("certificate_v2")
        if not isinstance(intent_json, dict) or not isinstance(cert_json, dict):
            results.append({"kernel": k, "status": "MISSING_INTENT_OR_CERT"})
            continue

        try:
            intent = IntentFunction.from_json_dict(intent_json)
        except Exception as e:
            results.append({"kernel": k, "status": "INTENT_PARSE_ERROR", "error": f"{type(e).__name__}: {e}"})
            continue
        cert_v2 = _cert_v2_from_json(cert_json)

        spec = _spec_from_pipeline(frontend, k)
        diff_cases = _cases_from_report(report, limit=int(args.diff_cases))
        if not diff_cases:
            results.append({"kernel": k, "status": "NO_CASES"})
            continue
        base_case = diff_cases[0]
        atol, rtol = _tolerances_from_report(report)

        per_mode: Dict[str, Any] = {}
        for m in modes:
            if m.name == "full":
                static_fn = lambda x, _c=cert_v2: static_validate(x, _c)  # noqa: E731
            elif m.name == "diff_only":
                static_fn = None
            else:
                static_fn = m.static_validate_fn
            rep = run_mutation_kill(
                k,
                intent=intent,
                run_ref_fn=spec.runner,
                diff_cases=diff_cases,
                metamorphic_base_case=base_case,
                static_validate_fn=static_fn,
                n_mutants=int(args.n_mutants),
                seed=int(args.seed),
                atol=float(atol),
                rtol=float(rtol),
                include_bounded=(not bool(args.no_bounded)),
                diff_stop_on_first_fail=True,
            )
            per_mode[m.name] = {
                "kill_rate": float(rep.kill_rate),
                "total": int(rep.total),
                "killed": int(rep.killed),
                "survived": int(rep.survived),
                "killed_by_stage": dict(rep.killed_by_stage),
                "time_to_counterexample": _summarize_cex(rep),
            }
        results.append({"kernel": k, "status": "OK", "modes": per_mode})
        print(f"[trust_ablation:{frontend}:{k}] OK", flush=True)

    out: Dict[str, Any] = {
        "experiment": "E2_trust_ablation",
        "frontend": frontend,
        "suite": str(args.suite),
        "kernels": list(kernels),
        "n_mutants": int(args.n_mutants),
        "diff_cases": int(args.diff_cases),
        "seed": int(args.seed),
        "elapsed_s": float(time.time() - t0),
        "results": results,
    }
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
