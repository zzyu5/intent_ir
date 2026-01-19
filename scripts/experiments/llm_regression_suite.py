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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontends.common.certificate_v2 import SemanticCertificateV2
from frontends.common.contract_v2 import evaluate_contract_v2
from frontends.common.obligations import evaluate_obligations
from frontends.common.static_validate import static_validate
from intent_ir.ir import IntentFunction
from intent_ir.llm.llm_hub import LLMIntentHub
from intent_ir.macros import expand_macros
from intent_ir.macros.macro_spec import enrich_intent_macros
from pipeline.interfaces import KernelArtifactBundle, KernelDescriptor


def _spec_from_pipeline(frontend: str, kernel: str) -> Any:
    """
    Recover the kernel spec object used by the frontend adapter.

    We intentionally use the *coverage* suite as the lookup table because it
    includes the smoke kernels (see `scripts/full_pipeline_verify.py`).
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


def _prepare_frontend_artifacts(*, frontend: str, kernel: str, artifacts_dir: Path) -> None:
    """
    Stage-4-only artifact preparation for the regression suite.

    Unlike `scripts/full_pipeline_verify.py`, we do NOT run:
      - baseline launch (Task3)
      - Task5 diff / remote RVV

    We only materialize what the LLM regression suite needs:
      - `<kernel>.descriptor.json`
      - `<kernel>.certificate_v2.json` (+ contract summary in cert.meta)
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    spec = _spec_from_pipeline(str(frontend), str(kernel))
    # CUDA frontend: requires pre-generated snapshots under kernels/cuda/ops/.
    # If missing, run `PYTHONPATH=. python scripts/tilelang/export_cuda_snapshots.py`.

    # Build descriptor + frontend artifacts via adapter (TTIR/PTX/TVM JSON snapshots).
    from pipeline import registry as pipeline_registry  # noqa: PLC0415

    adapter = pipeline_registry.get(str(frontend))
    desc = adapter.build_descriptor(spec)
    desc.meta["artifact_dir"] = str(artifacts_dir)
    if str(frontend) == "cuda":
        ptx_entry = getattr(spec, "ptx_entry", None)
        if isinstance(ptx_entry, str) and ptx_entry.strip():
            desc.meta["ptx_entry"] = str(ptx_entry)
    desc = adapter.ensure_artifacts(desc, spec)

    # Facts/constraints + CertificateV2.
    facts = adapter.extract_facts(desc)
    constraints = adapter.extract_constraints(desc, facts)

    cert_v2: SemanticCertificateV2
    if str(frontend) == "triton":
        # Triton adapter keeps legacy v1 cert builder for backward compat; use
        # the v2 builder directly (matches pipeline/triton/core.py Stage 4).
        from frontends.triton.certificate import build_certificate_v2  # noqa: PLC0415

        ttir_text = None
        try:
            if desc.artifacts.ttir_path:
                ttir_text = Path(str(desc.artifacts.ttir_path)).read_text(encoding="utf-8")
        except Exception:
            ttir_text = None
        if not ttir_text:
            raise FileNotFoundError("TTIR not available for Triton kernel (missing dump/copy?)")
        cert_v2 = build_certificate_v2(ttir_text, desc=desc, facts=facts)
    else:
        cert = adapter.build_certificate(desc, facts, constraints)
        if not isinstance(cert, SemanticCertificateV2):
            raise TypeError(f"{frontend} adapter returned non-CertificateV2: {type(cert).__name__}")
        cert_v2 = cert

    obligations = evaluate_obligations(desc, cert_v2)
    cert_v2.semantic_facts["obligations"] = [o.to_json_dict() for o in obligations]
    contract = evaluate_contract_v2(desc, cert_v2, obligations, constraints=constraints)
    cert_v2.meta = dict(getattr(cert_v2, "meta", {}) or {})
    cert_v2.meta["contract"] = {"level": str(contract.level), "reasons": list(contract.reasons), "assumptions": list(contract.assumptions)}

    # Persist the artifacts this suite consumes.
    (artifacts_dir / f"{kernel}.descriptor.json").write_text(json.dumps(desc.to_json_dict(), indent=2), encoding="utf-8")
    (artifacts_dir / f"{kernel}.certificate_v2.json").write_text(json.dumps(cert_v2.to_json_dict(), indent=2), encoding="utf-8")


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
    predicted_semantic: str
    predicted_axis_roles: Dict[str, str]
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
            "predicted": {
                "semantic_class": str(self.predicted_semantic),
                "axis_roles": dict(self.predicted_axis_roles),
            },
            "ok": bool(self.ok),
            "category": str(self.category),
            "reasons": list(self.reasons),
            "llm": dict(self.llm),
        }


def _summarize(results: List[KernelResult]) -> Dict[str, Any]:
    failures: Dict[str, int] = {}
    tiers: Dict[str, Dict[str, int]] = {}
    ok_n = 0
    rounds_used: List[int] = []
    api_calls: List[int] = []
    cache_hits: List[int] = []
    cache_misses: List[int] = []
    for r in results:
        failures[r.category] = failures.get(r.category, 0) + 1
        tiers.setdefault(r.anchor_tier, {"n": 0, "ok": 0})
        tiers[r.anchor_tier]["n"] += 1
        if r.ok:
            tiers[r.anchor_tier]["ok"] += 1
            ok_n += 1
        st = r.llm.get("stats") if isinstance(r.llm, dict) else None
        if isinstance(st, dict):
            if isinstance(st.get("rounds_used"), int):
                rounds_used.append(int(st.get("rounds_used")))
            if isinstance(st.get("api_calls"), int):
                api_calls.append(int(st.get("api_calls")))
            if isinstance(st.get("cache_hits"), int):
                cache_hits.append(int(st.get("cache_hits")))
            if isinstance(st.get("cache_misses"), int):
                cache_misses.append(int(st.get("cache_misses")))

    curve: List[Dict[str, Any]] = []
    ok_prefix = 0
    for i, r in enumerate(sorted(results, key=lambda x: (-int(x.anchor_score), str(x.kernel))), start=1):
        ok_prefix += 1 if r.ok else 0
        curve.append({"k": int(i), "ok": int(ok_prefix), "ok_rate": float(ok_prefix / i)})

    def _p50(xs: List[int]) -> int | None:
        if not xs:
            return None
        ys = sorted(int(x) for x in xs)
        return int(ys[len(ys) // 2])

    return {
        "n": len(results),
        "ok": int(ok_n),
        "ok_rate": (float(ok_n) / float(len(results)) if results else 0.0),
        "failures": failures,
        "tiers": tiers,
        "llm_cost": {
            "rounds_used_avg": (float(sum(rounds_used)) / float(len(rounds_used)) if rounds_used else None),
            "rounds_used_p50": _p50(rounds_used),
            "api_calls_total": int(sum(api_calls)) if api_calls else 0,
            "api_calls_avg": (float(sum(api_calls)) / float(len(api_calls)) if api_calls else None),
            "cache_hits_total": int(sum(cache_hits)) if cache_hits else 0,
            "cache_misses_total": int(sum(cache_misses)) if cache_misses else 0,
        },
        "coverage_curve": curve,
    }


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


def _semantic_class_from_intent(intent: IntentFunction) -> str:
    """
    Coarse semantic class for E1 human-label evaluation.
    Kept intentionally simple and stable across frontends.
    """
    ops = [op.op for op in intent.ops]
    ops_set = set(ops)
    matmul_n = ops.count("matmul")
    has_reduce = any(str(o).startswith("reduce") for o in ops_set)
    has_exp = "exp" in ops_set
    has_rsqrt = "rsqrt" in ops_set
    if "upsample_bicubic2d_aa" in ops_set:
        return "upsample"
    # Attention patterns: matmul + softmax + matmul (or explicit attention macro).
    if "flash_attention" in ops_set or "attention" in ops_set:
        return "attention"
    if matmul_n >= 2 and ("softmax" in ops_set or (has_exp and has_reduce)):
        return "attention"
    if "softmax" in ops_set:
        return "softmax"
    if any(o in ops_set for o in ["layer_norm", "group_norm", "rms_norm", "batch_norm", "layernorm", "groupnorm"]):
        return "norm"
    # Derived norm pattern: reduce + rsqrt (mean/var/rstd style).
    if has_rsqrt and has_reduce:
        return "norm"
    # Derived softmax pattern: exp + reduce, but no matmul-major attention structure.
    if has_exp and has_reduce and matmul_n < 2 and (not has_rsqrt):
        return "softmax"
    if "gather" in ops_set:
        return "gather"
    if "transpose" in ops_set:
        return "transpose"
    if "matmul" in ops_set:
        return "matmul"
    if has_reduce:
        return "reduce"
    if ("copy" in ops_set or "identity" in ops_set) and ops_set.issubset({"copy", "identity"}):
        return "copy"
    # Elementwise / fused epilogue bucket.
    if ops_set.intersection(
        {
            "add",
            "mul",
            "div",
            "relu",
            "exp",
            "max",
            "min",
            "where",
            "clip",
            "cast",
            "floor",
            "sigmoid",
            "tanh",
            "gelu",
            "ne",
        }
    ):
        return "elementwise"
    # Fallback: remaining graphs are treated as elementwise rather than "unknown"
    # (e.g., external benchmark kernels with uncommon op names).
    return "elementwise" if ops else "unknown"


def _load_labels(path: Path) -> Dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise TypeError("labels must be a JSON object")
    return d


def _eval_labels(results: List[KernelResult], labels: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate E1 accuracy against a (human) label set.

    Labels format:
      {
        "<kernel_name>": {"semantic_class": "...", "axis_roles": {"N": "batch", ...}},
        ...
      }
    """
    sem_total = 0
    sem_ok = 0
    sem_by_class: Dict[str, Dict[str, int]] = {}
    axis_total = 0
    axis_ok = 0
    axis_exact_total = 0
    axis_exact_ok = 0

    for r in results:
        lab = labels.get(r.kernel)
        if not isinstance(lab, dict):
            continue
        # semantic class
        sc = lab.get("semantic_class")
        if isinstance(sc, str) and sc.strip():
            sem_total += 1
            sem_by_class.setdefault(sc, {"n": 0, "ok": 0})
            sem_by_class[sc]["n"] += 1
            if str(r.predicted_semantic) == sc:
                sem_ok += 1
                sem_by_class[sc]["ok"] += 1
        # axis roles (recall on labeled axes)
        ar = lab.get("axis_roles")
        if isinstance(ar, dict) and ar:
            axis_exact_total += 1
            all_ok = True
            for ax, role in ar.items():
                if not isinstance(ax, str) or not isinstance(role, str):
                    continue
                axis_total += 1
                if r.predicted_axis_roles.get(ax) == role:
                    axis_ok += 1
                else:
                    all_ok = False
            if all_ok:
                axis_exact_ok += 1

    return {
        "semantic_class": {
            "n": int(sem_total),
            "ok": int(sem_ok),
            "acc": (float(sem_ok) / float(sem_total) if sem_total else None),
            "by_class": {
                k: {
                    "n": int(v["n"]),
                    "ok": int(v["ok"]),
                    "acc": (float(v["ok"]) / float(v["n"]) if v["n"] else None),
                }
                for k, v in sem_by_class.items()
            },
        },
        "axis_roles": {
            "n_axes": int(axis_total),
            "ok_axes": int(axis_ok),
            "recall": (float(axis_ok) / float(axis_total) if axis_total else None),
            "n_kernels": int(axis_exact_total),
            "exact_ok": int(axis_exact_ok),
            "exact_rate": (float(axis_exact_ok) / float(axis_exact_total) if axis_exact_total else None),
        },
    }


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
        # Regression suite is expected to be runnable "from scratch". Prepare
        # frontend-side artifacts on demand (Stage 4 only).
        try:
            _prepare_frontend_artifacts(frontend=frontend, kernel=kernel, artifacts_dir=artifacts_dir)
        except Exception as e:
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
                predicted_semantic="unknown",
                predicted_axis_roles={},
                ok=False,
                category=f"artifact_prepare_error:{type(e).__name__}",
                reasons=[str(e)] + ([f"missing: {p}" for p in missing] if missing else []),
                llm={},
            )
            return r, last_api_s

    desc = _descriptor_from_json(_load_json(desc_path))
    cert_v2 = _cert_v2_from_json(_load_json(cert_path))

    # Ensure descriptor carries Stage-4 facts/constraints. Full pipeline reports
    # historically wrote `<kernel>.descriptor.json` before facts extraction,
    # causing cache-misses and poorer prompts in the regression suite.
    needs_refresh = (not bool(desc.frontend_facts)) or (not bool(desc.frontend_constraints))
    if needs_refresh:
        try:
            _prepare_frontend_artifacts(frontend=frontend, kernel=kernel, artifacts_dir=artifacts_dir)
            desc = _descriptor_from_json(_load_json(desc_path))
            cert_v2 = _cert_v2_from_json(_load_json(cert_path))
        except Exception:
            # Best-effort: keep whatever we loaded.
            pass
    anchors = (cert_v2.semantic_facts or {}).get("anchors") if isinstance(cert_v2.semantic_facts, dict) else {}
    anchors = dict(anchors) if isinstance(anchors, dict) else {}

    contract = _contract_level(cert_v2)
    tier = _anchor_tier(anchors)
    score = _anchor_score(anchors)

    feedback: List[str] = []
    llm_rounds: List[Dict[str, Any]] = []
    llm_meta: Dict[str, Any] = {}
    api_calls = 0
    cache_hits = 0
    cache_misses = 0
    rounds_used = 0
    ok_round: int | None = None
    best_ok = False
    best_reasons: List[str] = []
    best_category = "unknown"
    best_semantic = "unknown"
    best_axis_roles: Dict[str, str] = {}

    for round_id in range(max(1, int(repair_rounds) + 1)):
        rounds_used += 1
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
        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1
            api_calls += 1
        if not cache_hit:
            last_api_s = time.time()
        llm_meta = {"prompt_hash": trace.get("prompt_hash"), "chosen": dict(chosen) if isinstance(chosen, dict) else {}}

        try:
            enrich_intent_macros(cand.intent)
        except Exception:
            # Macro enrichment is best-effort; do not fail regression on it.
            pass
        try:
            # E1 semantic labels should be evaluated on the macro-level intent
            # (before macro expansion lowers into primitive ops).
            best_semantic = _semantic_class_from_intent(cand.intent)
            best_axis_roles = dict(getattr(cand.intent, "axis_roles", {}) or {})
        except Exception:
            best_semantic = "unknown"
            best_axis_roles = {}

        try:
            expanded = expand_macros(cand.intent)
            intent_for_validate = expanded
        except Exception as e:
            try:
                best_semantic = _semantic_class_from_intent(cand.intent)
                best_axis_roles = dict(getattr(cand.intent, "axis_roles", {}) or {})
            except Exception:
                pass
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

        llm_rounds.append(
            {
                "round": int(round_id),
                "feedback_n": int(len(feedback)),
                "prompt_hash": trace.get("prompt_hash"),
                "chosen": dict(chosen) if isinstance(chosen, dict) else {},
                "static_ok": bool(sv.ok),
                "static_reasons": list(sv.reasons),
            }
        )

        if bool(sv.ok):
            best_ok = True
            best_category = "ok"
            best_reasons = []
            ok_round = int(round_id)
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
        predicted_semantic=str(best_semantic),
        predicted_axis_roles=dict(best_axis_roles),
        ok=bool(best_ok),
        category=str(best_category),
        reasons=list(best_reasons),
        llm={
            "stats": {
                "rounds_used": int(rounds_used),
                "ok_round": int(ok_round) if ok_round is not None else None,
                "api_calls": int(api_calls),
                "cache_hits": int(cache_hits),
                "cache_misses": int(cache_misses),
            },
            "rounds": list(llm_rounds),
            **(dict(llm_meta) if isinstance(llm_meta, dict) else {}),
        },
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
    ap.add_argument("--labels", default=None, help="optional: JSON labels file for E1 accuracy scoring")
    ap.add_argument(
        "--compare-one-shot",
        action="store_true",
        help="run one-shot (repair_rounds=0) and feedback-loop (repair_rounds=N) back-to-back and report delta",
    )
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

    if bool(args.compare_one_shot) and int(args.repair_rounds) <= 0:
        raise SystemExit("--compare-one-shot requires --repair-rounds >= 1")

    hub = LLMIntentHub(timeout_s=int(args.timeout), max_attempts=int(args.attempts), max_parse_retries=int(args.parse_retries))

    variants: List[Tuple[str, int]] = [("feedback", int(args.repair_rounds))]
    if bool(args.compare_one_shot):
        variants = [("one_shot", 0), ("feedback", int(args.repair_rounds))]

    results_by_variant: Dict[str, List[KernelResult]] = {name: [] for name, _ in variants}
    last_api_s: float | None = None
    for v_name, v_rounds in variants:
        for fe in frontends:
            art_dir = _artifact_dirs(fe, triton_dir=triton_dir, tilelang_dir=tilelang_dir, cuda_dir=cuda_dir)
            for k in kernels_by_fe.get(str(fe), []):
                # If one-shot already succeeded, feedback-loop will succeed too (same initial prompt),
                # so we can reuse the result without re-running.
                if v_name == "feedback" and bool(args.compare_one_shot):
                    prev = next((x for x in results_by_variant["one_shot"] if x.frontend == str(fe) and x.kernel == str(k)), None)
                    if prev is not None and prev.ok:
                        results_by_variant[v_name].append(prev)
                        print(f"[{v_name}:{fe}:{k}] OK (reused one_shot)", flush=True)
                        continue

                r, last_api_s = run_one(
                    hub=hub,
                    frontend=str(fe),
                    kernel=str(k),
                    artifacts_dir=art_dir,
                    model=(str(args.model) if args.model else None),
                    repair_rounds=int(v_rounds),
                    rpm=int(args.rpm),
                    last_api_s=last_api_s,
                )
                results_by_variant[v_name].append(r)
                status = "OK" if r.ok else "FAIL"
                print(f"[{v_name}:{fe}:{k}] {status} ({r.category})", flush=True)

    all_kernels: List[str] = []
    for fe in frontends:
        all_kernels.extend(list(kernels_by_fe.get(str(fe), [])))
    all_kernels = sorted(set(all_kernels))

    out: Dict[str, Any] = {
        "suite": str(args.suite),
        "frontends": list(frontends),
        "kernels": list(all_kernels),
        "kernels_by_frontend": dict(kernels_by_fe),
        "variants": {
            name: {
                "repair_rounds": int(v_rounds),
                "results": [r.to_json_dict() for r in results_by_variant.get(name, [])],
                "summary": {"by_frontend": {fe: _summarize([r for r in results_by_variant.get(name, []) if r.frontend == fe]) for fe in frontends}},
            }
            for name, v_rounds in variants
        },
    }
    if args.labels:
        labels_path = Path(str(args.labels))
        labels = _load_labels(labels_path)
        out["labels"] = {"path": str(labels_path)}
        out["label_eval"] = {
            name: {"by_frontend": {fe: _eval_labels([r for r in results_by_variant.get(name, []) if r.frontend == fe], labels) for fe in frontends}}
            for name, _ in variants
        }
    if bool(args.compare_one_shot):
        # Convenience: compute success-rate delta for paper tables.
        a = results_by_variant.get("one_shot", [])
        b = results_by_variant.get("feedback", [])
        a_ok = sum(1 for x in a if x.ok)
        b_ok = sum(1 for x in b if x.ok)
        out["delta"] = {
            "one_shot_ok": int(a_ok),
            "feedback_ok": int(b_ok),
            "n": int(len(a)),
            "ok_rate_one_shot": (float(a_ok) / float(len(a)) if a else 0.0),
            "ok_rate_feedback": (float(b_ok) / float(len(b)) if b else 0.0),
            "ok_rate_gain": (float(b_ok) / float(len(b)) - float(a_ok) / float(len(a)) if a else 0.0),
        }
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
