"""
Paper experiment (E4): cross-frontend consistency check.

Goal:
  Quantify whether we recover the *same Intent* across different frontends for
  kernels that are intended to be semantically equivalent.

This script compares the recovered IntentIR (A/B layers) between frontends
using a normalization that:
  - ignores schedule/meta
  - renames SSA values deterministically (so naming differences do not count)
  - compares ops + tensor signatures + axis_roles

It consumes existing full pipeline reports:
  - artifacts/full_pipeline_verify/*.json
  - artifacts/tilelang_full_pipeline/*.json
  - artifacts/cuda_full_pipeline/*.json

If a report is missing (or --refresh-artifacts is set), it will run the
frontend pipeline for that kernel first.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from frontends.common.certificate_v2 import SemanticCertificateV2
from intent_ir.ir import IntentFunction, Op, TensorType


ROOT = Path(__file__).resolve().parents[2]


def _artifact_dir(frontend: str) -> Path:
    if frontend == "triton":
        return ROOT / "artifacts" / "full_pipeline_verify"
    if frontend == "tilelang":
        return ROOT / "artifacts" / "tilelang_full_pipeline"
    if frontend == "cuda":
        return ROOT / "artifacts" / "cuda_full_pipeline"
    raise ValueError(f"unknown frontend: {frontend}")


def _spec_from_pipeline(frontend: str, kernel: str) -> Any:
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


def _anchors_from_cert(cert: SemanticCertificateV2) -> Dict[str, Any]:
    anchors = (cert.semantic_facts or {}).get("anchors") if isinstance(cert.semantic_facts, dict) else {}
    return dict(anchors) if isinstance(anchors, dict) else {}


def _anchor_tier(anchors: Dict[str, Any]) -> str:
    if bool(anchors.get("has_dot")):
        return "A_dot"
    if bool(anchors.get("has_reduce")):
        return "B_reduce"
    if bool(anchors.get("has_copy")):
        return "C_copy"
    return "D_none"


def _tensor_sig(t: TensorType) -> Dict[str, Any]:
    return {
        "dtype": str(t.dtype),
        "shape": [f"{d.kind}:{d.value}" for d in list(t.shape)],
        "layout": (t.layout.name if hasattr(t.layout, "name") else str(t.layout)),
    }


def _canonicalize_intent(intent: IntentFunction) -> Dict[str, Any]:
    """
    Produce a backend-agnostic signature for comparing intent graphs.

    Key normalization:
      - ignore schedule/meta (C layer)
      - rename values deterministically so SSA naming differences don't matter
    """
    produced = {op.output for op in intent.ops if op.output}
    used: List[str] = []
    for op in intent.ops:
        used.extend(list(op.inputs))
    external_inputs = [n for n in used if (n in intent.tensors and n not in produced)]

    # Deterministic renaming:
    # - external inputs: sorted by tensor signature, then name
    # - op outputs: in program order (intent.ops order)
    def in_key(n: str) -> Tuple[str, int, str]:
        t = intent.tensors.get(n)
        if t is None:
            return ("", 0, n)
        sig = _tensor_sig(t)
        return (f"{sig['dtype']}|{sig['layout']}|{','.join(sig['shape'])}", len(sig["shape"]), n)

    ext_sorted = sorted(set(external_inputs), key=in_key)
    name_map: Dict[str, str] = {}
    for i, n in enumerate(ext_sorted):
        name_map[n] = f"in{i}"

    for i, op in enumerate(intent.ops):
        name_map[op.output] = f"v{i}"

    def canon_name(n: str) -> str:
        return name_map.get(n, n)

    # Canonical tensors: only those referenced (external inputs + produced + outputs), under renamed keys.
    touched = set(ext_sorted) | {op.output for op in intent.ops} | set(intent.outputs)
    tensors = {
        canon_name(k): _tensor_sig(v)
        for k, v in intent.tensors.items()
        if k in touched
    }

    def canon_attrs(op: Op) -> Dict[str, Any]:
        attrs = dict(op.attrs or {})
        # Drop schedule-ish or trace-ish fields if they ever leak in attrs.
        for k in ["tile_m", "tile_n", "tile_k", "vec_width", "pipeline_depth", "trace", "debug"]:
            attrs.pop(k, None)
        # Canonicalize list/dict ordering.
        out: Dict[str, Any] = {}
        for k in sorted(attrs.keys()):
            v = attrs[k]
            if isinstance(v, dict):
                out[str(k)] = {str(kk): v[kk] for kk in sorted(v.keys(), key=lambda x: str(x))}
            elif isinstance(v, list):
                out[str(k)] = list(v)
            else:
                out[str(k)] = v
        return out

    ops = [
        {
            "op": str(op.op),
            "inputs": [canon_name(x) for x in list(op.inputs)],
            "output": canon_name(op.output),
            "attrs": canon_attrs(op),
        }
        for op in intent.ops
    ]

    axis_roles = {str(k): str(v) for k, v in sorted((intent.axis_roles or {}).items(), key=lambda kv: str(kv[0]))}
    parallel_axes = sorted(list(intent.parallel_axes or []), key=lambda x: str(x))
    outputs = [canon_name(x) for x in list(intent.outputs)]

    return {
        "tensors": tensors,
        "ops": ops,
        "outputs": outputs,
        "axis_roles": axis_roles,
        "parallel_axes": parallel_axes,
    }


def _diff_reasons(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if a.get("ops") != b.get("ops"):
        reasons.append("ops_mismatch")
    if a.get("tensors") != b.get("tensors"):
        reasons.append("tensors_mismatch")
    if a.get("outputs") != b.get("outputs"):
        reasons.append("outputs_mismatch")
    if a.get("axis_roles") != b.get("axis_roles"):
        reasons.append("axis_roles_mismatch")
    if a.get("parallel_axes") != b.get("parallel_axes"):
        reasons.append("parallel_axes_mismatch")
    if not reasons:
        reasons.append("unknown_mismatch")
    return reasons


@dataclass(frozen=True)
class PairResult:
    kernel: str
    ok: bool
    anchor_tier: str
    reasons: List[str]
    sigs: Dict[str, Any]


def _kernels_from_pipeline(frontend: str) -> List[str]:
    if frontend == "triton":
        from pipeline.triton.core import coverage_kernel_specs  # noqa: PLC0415

        return [s.name for s in coverage_kernel_specs()]
    if frontend == "tilelang":
        from pipeline.tilelang.core import coverage_kernel_specs  # noqa: PLC0415

        return [s.name for s in coverage_kernel_specs()]
    if frontend == "cuda":
        from pipeline.cuda.core import coverage_kernel_specs  # noqa: PLC0415

        return [s.name for s in coverage_kernel_specs()]
    raise ValueError(f"unknown frontend: {frontend}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", action="append", default=[], help="repeatable; default compares triton+tilelang")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; restrict to kernel name(s)")
    ap.add_argument("--refresh-artifacts", action="store_true")
    ap.add_argument("--cases-limit", type=int, default=4, help="used only when regenerating pipeline artifacts")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "experiments" / "e4_cross_frontend_consistency_latest.json"))
    args = ap.parse_args()

    frontends = [str(x) for x in (args.frontend or []) if str(x).strip()]
    if not frontends:
        frontends = ["triton", "tilelang"]
    if len(frontends) < 2:
        raise SystemExit("need at least 2 --frontend values (e.g., --frontend triton --frontend tilelang)")

    wanted = {str(x) for x in (args.kernel or []) if str(x).strip()}

    # Candidate kernels = intersection of pipeline coverage suites.
    sets = [set(_kernels_from_pipeline(fe)) for fe in frontends]
    kernels = sorted(set.intersection(*sets)) if sets else []
    if wanted:
        kernels = [k for k in kernels if k in wanted]

    results: List[PairResult] = []
    for k in kernels:
        per_fe: Dict[str, Dict[str, Any]] = {}
        tier = "D_none"
        ok = True
        reasons: List[str] = []

        # Load per-frontend report + intent signature.
        for fe in frontends:
            out_dir = _artifact_dir(fe)
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / f"{k}.json"
            if args.refresh_artifacts or not report_path.exists():
                _run_pipeline(fe, k, out_dir=out_dir, cases_limit=int(args.cases_limit))
            if not report_path.exists():
                ok = False
                reasons.append(f"missing_report:{fe}")
                continue
            rep = _read_json(report_path)
            intent_json = rep.get("intent_expanded") or rep.get("intent")
            cert_json = rep.get("certificate_v2")
            if not isinstance(intent_json, dict) or not isinstance(cert_json, dict):
                ok = False
                reasons.append(f"missing_intent_or_cert:{fe}")
                continue
            try:
                intent = IntentFunction.from_json_dict(intent_json)
            except Exception as e:
                ok = False
                reasons.append(f"intent_parse_error:{fe}:{type(e).__name__}")
                continue
            cert_v2 = _cert_v2_from_json(cert_json)
            anchors = _anchors_from_cert(cert_v2)
            tier = _anchor_tier(anchors)
            per_fe[fe] = {"contract": (rep.get("contract") or {}).get("level"), "sig": _canonicalize_intent(intent)}

        # Compare signatures (ignore schedule/meta).
        base_fe = frontends[0]
        base_sig = per_fe.get(base_fe, {}).get("sig")
        if not isinstance(base_sig, dict):
            ok = False
            reasons.append("missing_base_signature")
        else:
            for fe in frontends[1:]:
                sig = per_fe.get(fe, {}).get("sig")
                if not isinstance(sig, dict):
                    ok = False
                    continue
                if sig != base_sig:
                    ok = False
                    reasons.extend([f"{fe}:{r}" for r in _diff_reasons(base_sig, sig)])

        results.append(PairResult(kernel=k, ok=bool(ok), anchor_tier=str(tier), reasons=sorted(set(reasons)), sigs=per_fe))
        status = "OK" if ok else "FAIL"
        print(f"[E4:{'+'.join(frontends)}:{k}] {status}", flush=True)

    # Summaries
    by_tier: Dict[str, Dict[str, int]] = {}
    by_reason: Dict[str, int] = {}
    for r in results:
        by_tier.setdefault(r.anchor_tier, {"n": 0, "ok": 0})
        by_tier[r.anchor_tier]["n"] += 1
        if r.ok:
            by_tier[r.anchor_tier]["ok"] += 1
        for rr in r.reasons:
            by_reason[rr] = by_reason.get(rr, 0) + 1

    out = {
        "experiment": "E4_cross_frontend_consistency",
        "frontends": list(frontends),
        "kernels": list(kernels),
        "summary": {
            "n": int(len(results)),
            "ok": int(sum(1 for r in results if r.ok)),
            "ok_rate": (float(sum(1 for r in results if r.ok)) / float(len(results)) if results else 0.0),
            "by_tier": by_tier,
            "top_reasons": {k: int(v) for k, v in sorted(by_reason.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:20]},
        },
        "results": [
            {
                "kernel": r.kernel,
                "ok": bool(r.ok),
                "anchor_tier": r.anchor_tier,
                "reasons": list(r.reasons),
                "per_frontend": r.sigs,
            }
            for r in results
        ],
    }
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()

