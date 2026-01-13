"""
Paper utility (P3): LLM regression suite (no diff/remote).

Goal:
  Batch-run LLM -> IntentIR -> (macro expand) -> static_validate
  across a kernel suite, and summarize failure categories / anchor tiers.

Notes:
- This script intentionally does NOT run Task5 dynamic diff or remote RVV.
- It consumes existing stage-4 artifacts (descriptor/certificate_v2) produced by
  `scripts/full_pipeline_verify.py` for each frontend.
- Real LLM calls are rate-limited by providers; use `--rpm` to avoid throttling.
  Cache is enabled by default via `intent_ir.llm.llm_client`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from frontends.common.certificate_v2 import SemanticCertificateV2
from frontends.common.static_validate import static_validate
from intent_ir.llm.llm_hub import LLMIntentHub
from intent_ir.macros import expand_macros
from intent_ir.macros.macro_spec import enrich_intent_macros
from pipeline.interfaces import KernelArtifactBundle, KernelDescriptor


ROOT = Path(__file__).resolve().parents[2]

def _kernels_from_pipeline(frontend: str, suite: str) -> List[str]:
    """
    Single source of truth for kernel suites: reuse pipeline core spec lists.

    We avoid hard-coding kernel names here so the regression suite stays in sync
    with `scripts/full_pipeline_verify.py --suite ...`.
    """
    if suite not in {"smoke", "coverage", "all"}:
        raise ValueError(f"unknown suite: {suite}")
    if frontend == "triton":
        from pipeline.triton.core import coverage_kernel_specs, default_kernel_specs

        specs_fn = default_kernel_specs if suite == "smoke" else coverage_kernel_specs
        return [s.name for s in specs_fn()]
    if frontend == "tilelang":
        from pipeline.tilelang.core import coverage_kernel_specs, default_kernel_specs

        specs_fn = default_kernel_specs if suite == "smoke" else coverage_kernel_specs
        return [s.name for s in specs_fn()]
    if frontend == "cuda":
        from pipeline.cuda.core import coverage_kernel_specs, default_kernel_specs

        specs_fn = default_kernel_specs if suite == "smoke" else coverage_kernel_specs
        return [s.name for s in specs_fn()]
    raise ValueError(f"unknown frontend: {frontend}")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _descriptor_from_json(d: Dict[str, Any]) -> KernelDescriptor:
    art = d.get("artifacts") if isinstance(d.get("artifacts"), dict) else {}
    artifacts = KernelArtifactBundle(
        ttir_text=art.get("ttir_text"),
        ttir_path=art.get("ttir_path"),
        llvm_ir_text=art.get("llvm_ir_text"),
        ptx_text=art.get("ptx_text"),
        extra=(art.get("extra") if isinstance(art.get("extra"), dict) else {}),
    )
    return KernelDescriptor(
        schema_version=str(d.get("schema_version") or "kernel_desc_v1.0"),
        name=str(d.get("name") or "kernel"),
        frontend=str(d.get("frontend") or "triton"),  # type: ignore[arg-type]
        source_kind=str(d.get("source_kind") or "source"),  # type: ignore[arg-type]
        source_text=str(d.get("source_text") or ""),
        launch=(d.get("launch") if isinstance(d.get("launch"), dict) else {}),
        io_spec=(d.get("io_spec") if isinstance(d.get("io_spec"), dict) else {}),
        artifacts=artifacts,
        frontend_facts=(d.get("frontend_facts") if isinstance(d.get("frontend_facts"), dict) else {}),
        frontend_constraints=(d.get("frontend_constraints") if isinstance(d.get("frontend_constraints"), dict) else {}),
        meta=(d.get("meta") if isinstance(d.get("meta"), dict) else {}),
    )


def _cert_v2_from_json(d: Dict[str, Any]) -> SemanticCertificateV2:
    return SemanticCertificateV2(
        schema_version=str(d.get("schema_version") or "cert_v2.0"),
        semantic_facts=(d.get("semantic_facts") if isinstance(d.get("semantic_facts"), dict) else {}),
        schedule_hints=(d.get("schedule_hints") if isinstance(d.get("schedule_hints"), dict) else {}),
        meta=(d.get("meta") if isinstance(d.get("meta"), dict) else {}),
    ).canonicalize()


def _anchor_tier(anchors: Dict[str, Any]) -> str:
    if bool(anchors.get("has_dot")):
        return "A_dot"
    if bool(anchors.get("has_reduce")):
        return "B_reduce"
    if bool(anchors.get("has_copy")):
        return "C_copy"
    return "D_none"


def _anchor_score(anchors: Dict[str, Any]) -> int:
    score = 0
    if bool(anchors.get("has_dot")):
        score += 3
    if bool(anchors.get("has_reduce")):
        score += 2
    if bool(anchors.get("has_copy")):
        score += 1
    return int(score)


def _contract_level(cert_v2: SemanticCertificateV2) -> str:
    meta = getattr(cert_v2, "meta", None)
    if isinstance(meta, dict):
        c = meta.get("contract")
        if isinstance(c, dict) and isinstance(c.get("level"), str):
            return str(c.get("level"))
    return "N/A"


def _merge_feedback(feedback: List[str], reasons: List[str]) -> List[str]:
    out = list(feedback)
    for r in reasons:
        s = str(r).strip()
        if s and s not in out:
            out.append(s)
    return out


@dataclass(frozen=True)
class KernelResult:
    frontend: str
    kernel: str
    contract: str
    anchor_tier: str
    anchor_score: int
    ok: bool
    category: str
    reasons: List[str]
    llm: Dict[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "frontend": str(self.frontend),
            "kernel": str(self.kernel),
            "contract": str(self.contract),
            "anchor_tier": str(self.anchor_tier),
            "anchor_score": int(self.anchor_score),
            "ok": bool(self.ok),
            "category": str(self.category),
            "reasons": list(self.reasons),
            "llm": dict(self.llm),
        }


def _summarize(results: List[KernelResult]) -> Dict[str, Any]:
    failures: Dict[str, int] = {}
    tiers: Dict[str, Dict[str, int]] = {}
    for r in results:
        failures[r.category] = failures.get(r.category, 0) + 1
        tiers.setdefault(r.anchor_tier, {"n": 0, "ok": 0})
        tiers[r.anchor_tier]["n"] += 1
        if r.ok:
            tiers[r.anchor_tier]["ok"] += 1

    curve: List[Dict[str, Any]] = []
    ok_prefix = 0
    for i, r in enumerate(sorted(results, key=lambda x: (-int(x.anchor_score), str(x.kernel))), start=1):
        ok_prefix += 1 if r.ok else 0
        curve.append({"k": int(i), "ok": int(ok_prefix), "ok_rate": float(ok_prefix / i)})

    return {"n": len(results), "failures": failures, "tiers": tiers, "coverage_curve": curve}


def _artifact_dirs(frontend: str, *, triton_dir: Path, tilelang_dir: Path, cuda_dir: Path) -> Path:
    if frontend == "triton":
        return triton_dir
    if frontend == "tilelang":
        return tilelang_dir
    if frontend == "cuda":
        return cuda_dir
    raise ValueError(f"unknown frontend: {frontend}")


def _maybe_sleep_rate_limit(*, rpm: int, last_api_s: float | None) -> None:
    if rpm <= 0 or last_api_s is None:
        return
    min_interval = 60.0 / float(rpm)
    dt = time.time() - float(last_api_s)
    if dt < min_interval:
        time.sleep(min_interval - dt)


def run_one(
    *,
    hub: LLMIntentHub,
    frontend: str,
    kernel: str,
    artifacts_dir: Path,
    model: Optional[str],
    repair_rounds: int,
    rpm: int,
    last_api_s: float | None,
) -> Tuple[KernelResult, float | None]:
    desc_path = artifacts_dir / f"{kernel}.descriptor.json"
    cert_path = artifacts_dir / f"{kernel}.certificate_v2.json"

    if not desc_path.exists() or not cert_path.exists():
        missing = []
        if not desc_path.exists():
            missing.append(str(desc_path))
        if not cert_path.exists():
            missing.append(str(cert_path))
        r = KernelResult(
            frontend=frontend,
            kernel=kernel,
            contract="N/A",
            anchor_tier="D_none",
            anchor_score=0,
            ok=False,
            category="artifact_missing",
            reasons=[f"missing: {p}" for p in missing],
            llm={},
        )
        return r, last_api_s

    desc = _descriptor_from_json(_load_json(desc_path))
    cert_v2 = _cert_v2_from_json(_load_json(cert_path))
    anchors = (cert_v2.semantic_facts or {}).get("anchors") if isinstance(cert_v2.semantic_facts, dict) else {}
    anchors = dict(anchors) if isinstance(anchors, dict) else {}

    contract = _contract_level(cert_v2)
    tier = _anchor_tier(anchors)
    score = _anchor_score(anchors)

    feedback: List[str] = []
    llm_meta: Dict[str, Any] = {}
    best_ok = False
    best_reasons: List[str] = []
    best_category = "unknown"

    for round_id in range(max(1, int(repair_rounds) + 1)):
        _maybe_sleep_rate_limit(rpm=rpm, last_api_s=last_api_s)
        try:
            cand = hub.lift(desc, feedback=feedback, model=model)
        except Exception as e:
            best_ok = False
            best_category = f"llm_error:{type(e).__name__}"
            best_reasons = [str(e)]
            break

        trace = dict(cand.llm_trace or {})
        et = trace.get("extract_trace") if isinstance(trace.get("extract_trace"), dict) else {}
        chosen = et.get("chosen") if isinstance(et, dict) else {}
        cache_hit = bool(chosen.get("cache_hit")) if isinstance(chosen, dict) else False
        if not cache_hit:
            last_api_s = time.time()
        llm_meta = {
            "round": int(round_id),
            "prompt_hash": trace.get("prompt_hash"),
            "chosen": dict(chosen) if isinstance(chosen, dict) else {},
        }

        try:
            enrich_intent_macros(cand.intent)
        except Exception:
            # Macro enrichment is best-effort; do not fail regression on it.
            pass

        try:
            expanded = expand_macros(cand.intent)
            intent_for_validate = expanded
        except Exception as e:
            best_ok = False
            best_category = f"macro_expand_error:{type(e).__name__}"
            best_reasons = [str(e)]
            break

        try:
            sv = static_validate(intent_for_validate, cert_v2)
        except Exception as e:
            best_ok = False
            best_category = f"static_validate_error:{type(e).__name__}"
            best_reasons = [str(e)]
            break

        if bool(sv.ok):
            best_ok = True
            best_category = "ok"
            best_reasons = []
            break

        best_ok = False
        best_category = "static_validate_fail"
        best_reasons = list(sv.reasons)
        feedback = _merge_feedback(feedback, list(sv.reasons))

    r = KernelResult(
        frontend=frontend,
        kernel=kernel,
        contract=contract,
        anchor_tier=tier,
        anchor_score=score,
        ok=bool(best_ok),
        category=str(best_category),
        reasons=list(best_reasons),
        llm=llm_meta,
    )
    return r, last_api_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang", "cuda", "both", "all"], default="both")
    ap.add_argument("--suite", choices=["smoke", "coverage", "all"], default="coverage")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable: restrict to kernel name(s)")
    ap.add_argument("--triton-dir", default=str(ROOT / "artifacts" / "full_pipeline_verify"))
    ap.add_argument("--tilelang-dir", default=str(ROOT / "artifacts" / "tilelang_full_pipeline"))
    ap.add_argument("--cuda-dir", default=str(ROOT / "artifacts" / "cuda_full_pipeline"))
    ap.add_argument("--model", default=None, help="override LLM model name (provider config key)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--attempts", type=int, default=2, help="max LLM attempts inside the hub")
    ap.add_argument("--parse-retries", type=int, default=2, help="max JSON parse retries inside the hub")
    ap.add_argument("--repair-rounds", type=int, default=1, help="how many static-validate repair rounds (0 disables)")
    ap.add_argument("--rpm", type=int, default=5, help="rate limit (calls per minute) for cache-miss LLM calls; 0 disables")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "llm_regression_suite_latest.json"))
    args = ap.parse_args()

    triton_dir = Path(str(args.triton_dir))
    tilelang_dir = Path(str(args.tilelang_dir))
    cuda_dir = Path(str(args.cuda_dir))

    wanted = set(str(x) for x in (args.kernel or []) if str(x).strip())
    if str(args.frontend) == "both":
        frontends = ["triton", "tilelang"]
    elif str(args.frontend) == "all":
        frontends = ["triton", "tilelang", "cuda"]
    else:
        frontends = [str(args.frontend)]
    kernels_by_fe: Dict[str, List[str]] = {}
    for fe in frontends:
        ks = _kernels_from_pipeline(str(fe), str(args.suite))
        if wanted:
            ks = [k for k in ks if k in wanted]
        kernels_by_fe[str(fe)] = list(ks)

    hub = LLMIntentHub(timeout_s=int(args.timeout), max_attempts=int(args.attempts), max_parse_retries=int(args.parse_retries))

    results: List[KernelResult] = []
    last_api_s: float | None = None
    for fe in frontends:
        art_dir = _artifact_dirs(fe, triton_dir=triton_dir, tilelang_dir=tilelang_dir, cuda_dir=cuda_dir)
        for k in kernels_by_fe.get(str(fe), []):
            r, last_api_s = run_one(
                hub=hub,
                frontend=str(fe),
                kernel=str(k),
                artifacts_dir=art_dir,
                model=(str(args.model) if args.model else None),
                repair_rounds=int(args.repair_rounds),
                rpm=int(args.rpm),
                last_api_s=last_api_s,
            )
            results.append(r)
            status = "OK" if r.ok else "FAIL"
            print(f"[{fe}:{k}] {status} ({r.category})", flush=True)

    all_kernels: List[str] = []
    for fe in frontends:
        all_kernels.extend(list(kernels_by_fe.get(str(fe), [])))
    all_kernels = sorted(set(all_kernels))

    out: Dict[str, Any] = {
        "suite": str(args.suite),
        "frontends": list(frontends),
        "kernels": list(all_kernels),
        "kernels_by_frontend": dict(kernels_by_fe),
        "results": [r.to_json_dict() for r in results],
        "summary": {
            "by_frontend": {
                fe: _summarize([r for r in results if r.frontend == fe]) for fe in frontends
            }
        },
    }
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
