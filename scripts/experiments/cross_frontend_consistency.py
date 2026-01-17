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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontends.common.certificate_v2 import SemanticCertificateV2
from intent_ir.ir import IntentFunction, Op, TensorType


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


def _kernel_kind_hint(anchors: Dict[str, Any]) -> str:
    k = anchors.get("kernel_kind_hint") if isinstance(anchors, dict) else None
    return str(k) if k else "unknown"


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

    # Canonicalize shape symbols (alpha-renaming).
    # Cross-frontend kernels often use different symbol spellings (e.g., M/N vs m/n).
    # We assign canonical symbols S0,S1,... by first appearance in the *interface*
    # (external inputs + outputs), in a tensor-order that ignores original names.
    def dim_pat(d) -> Tuple[str, int | None]:
        if getattr(d, "kind", None) == "const":
            try:
                return ("c", int(d.value))
            except Exception:
                return ("c", None)
        return ("s", None)

    def tensor_key(name: str) -> Tuple[str, int, str, Tuple[Tuple[str, int | None], ...]]:
        tt = intent.tensors.get(name)
        if tt is None:
            return ("", 0, "", tuple())
        layout = tt.layout.name if hasattr(tt.layout, "name") else str(tt.layout)
        shp = tuple(dim_pat(d) for d in list(tt.shape))
        return (str(tt.dtype), int(len(tt.shape)), str(layout), shp)

    in_sorted = sorted(set(external_inputs), key=lambda n: (tensor_key(n), str(n)))
    out_sorted = sorted(set(intent.outputs or []), key=lambda n: (tensor_key(n), str(n)))
    sym_map: Dict[str, str] = {}
    next_id = 0
    for n in list(in_sorted) + list(out_sorted):
        tt = intent.tensors.get(n)
        if tt is None:
            continue
        for d in list(tt.shape):
            if getattr(d, "kind", None) != "sym":
                continue
            s = str(getattr(d, "value", "")).strip()
            if not s or s in sym_map:
                continue
            sym_map[s] = f"S{next_id}"
            next_id += 1

    def _tensor_sig_canon(t: TensorType) -> Dict[str, Any]:
        layout = t.layout.name if hasattr(t.layout, "name") else str(t.layout)
        shape: List[str] = []
        for d in list(t.shape):
            k = getattr(d, "kind", None)
            v = getattr(d, "value", None)
            if k == "sym":
                shape.append(f"sym:{sym_map.get(str(v), str(v))}")
            else:
                shape.append(f"{k}:{v}")
        return {"dtype": str(t.dtype), "shape": shape, "layout": str(layout)}

    # Deterministic renaming:
    # - external inputs: sorted by tensor signature, then name
    # - op outputs: in program order (intent.ops order)
    def in_key(n: str) -> Tuple[str, int, str]:
        t = intent.tensors.get(n)
        if t is None:
            return ("", 0, n)
        sig = _tensor_sig_canon(t)
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
        canon_name(k): _tensor_sig_canon(v)
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

    commutative = {"add", "mul", "max", "min", "and", "or", "ne"}
    ops = [
        (
            lambda _op=op: {
                "op": str(_op.op),
                "inputs": (
                    sorted([canon_name(x) for x in list(_op.inputs)])
                    if str(_op.op) in commutative
                    else [canon_name(x) for x in list(_op.inputs)]
                ),
                "output": canon_name(_op.output),
                "attrs": canon_attrs(_op),
            }
        )()
        for op in intent.ops
    ]

    axis_roles = {
        str(sym_map.get(str(k), str(k))): str(v)
        for k, v in sorted((intent.axis_roles or {}).items(), key=lambda kv: str(kv[0]))
    }
    parallel_axes = sorted([str(sym_map.get(str(x), str(x))) for x in list(intent.parallel_axes or [])], key=lambda x: str(x))
    # Canonicalize output order by tensor signature (avoid spurious mismatches).
    inputs = [canon_name(x) for x in ext_sorted]
    outputs = [canon_name(x) for x in sorted(set(intent.outputs), key=in_key)]

    return {
        "inputs": inputs,
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


def _parse_dim(s: str) -> Tuple[str, str]:
    parts = str(s).split(":", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "unknown", str(s)


def _shape_compatible(a: List[str], b: List[str]) -> bool:
    if len(a) != len(b):
        return False
    for da, db in zip(a, b):
        ka, va = _parse_dim(str(da))
        kb, vb = _parse_dim(str(db))
        if ka == "const" and kb == "const" and va != vb:
            return False
        # sym matches anything (specialization tolerant)
    return True


def _tensor_compatible(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if str(a.get("dtype")) != str(b.get("dtype")):
        return False
    if str(a.get("layout")) != str(b.get("layout")):
        return False
    sa = a.get("shape")
    sb = b.get("shape")
    if not isinstance(sa, list) or not isinstance(sb, list):
        return False
    return _shape_compatible([str(x) for x in sa], [str(x) for x in sb])


def _match_tensor_sigs(base: List[Dict[str, Any]], other: List[Dict[str, Any]], *, allow_extra_other: bool) -> bool:
    pool = list(other)
    for b in base:
        hit = None
        for i, o in enumerate(pool):
            if _tensor_compatible(b, o):
                hit = i
                break
        if hit is None:
            return False
        pool.pop(hit)
    return True if allow_extra_other else (len(pool) == 0)


def _interface_compatible(base_sig: Dict[str, Any], other_sig: Dict[str, Any]) -> bool:
    bt = base_sig.get("tensors")
    ot = other_sig.get("tensors")
    if not isinstance(bt, dict) or not isinstance(ot, dict):
        return False
    bi = base_sig.get("inputs")
    oi = other_sig.get("inputs")
    bo = base_sig.get("outputs")
    oo = other_sig.get("outputs")
    if not isinstance(bi, list) or not isinstance(oi, list) or not isinstance(bo, list) or not isinstance(oo, list):
        return False
    if len(bi) != len(oi):
        return False
    b_inputs = [bt.get(str(n)) for n in bi]
    o_inputs = [ot.get(str(n)) for n in oi]
    if any(not isinstance(x, dict) for x in b_inputs + o_inputs):
        return False
    if not _match_tensor_sigs([x for x in b_inputs if isinstance(x, dict)], [x for x in o_inputs if isinstance(x, dict)], allow_extra_other=False):
        return False
    b_outputs = [bt.get(str(n)) for n in bo]
    o_outputs = [ot.get(str(n)) for n in oo]
    if any(not isinstance(x, dict) for x in b_outputs + o_outputs):
        return False
    # Allow the other frontend to expose auxiliary outputs (base outputs must be matchable).
    return _match_tensor_sigs([x for x in b_outputs if isinstance(x, dict)], [x for x in o_outputs if isinstance(x, dict)], allow_extra_other=True)


def _ops_skeleton(sig: Dict[str, Any]) -> List[str]:
    ops = sig.get("ops")
    if not isinstance(ops, list):
        return []
    drop = {"identity", "const"}
    out: List[str] = []
    for o in ops:
        if not isinstance(o, dict):
            continue
        name = str(o.get("op"))
        if name in drop:
            continue
        out.append(name)
    return out


def _axis_roles_recall(base_sig: Dict[str, Any], other_sig: Dict[str, Any]) -> float:
    # Compare *role values* as a multiset (symbol names may differ across frontends).
    from collections import Counter

    br = base_sig.get("axis_roles")
    orr = other_sig.get("axis_roles")
    if not isinstance(br, dict) or not isinstance(orr, dict):
        return 0.0
    bvals = [str(v) for v in br.values()]
    ovals = [str(v) for v in orr.values()]
    if not bvals:
        return 1.0
    bc = Counter(bvals)
    oc = Counter(ovals)
    hit = 0
    for k, n in bc.items():
        hit += min(n, int(oc.get(k, 0)))
    return float(hit) / float(sum(bc.values()))


@dataclass(frozen=True)
class PairResult:
    kernel: str
    anchor_tier: str
    kernel_kind: str
    ok_intent: bool
    ok_expanded: bool
    ok_intent_structural: bool
    ok_expanded_structural: bool
    axis_roles_recall_intent: float
    axis_roles_recall_expanded: float
    reasons_intent: List[str]
    reasons_expanded: List[str]
    reasons_intent_structural: List[str]
    reasons_expanded_structural: List[str]
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


def _kernels_from_artifacts(frontend: str) -> List[str]:
    out_dir = _artifact_dir(frontend)
    if not out_dir.exists():
        return []
    # Only accept the primary report files: <kernel>.json (and skip auxiliary
    # <kernel>.*.json such as certificate/contract dumps).
    return sorted({p.stem for p in out_dir.glob("*.json") if "." not in p.stem})


def _mismatch_category(
    per_fe: Dict[str, Dict[str, Any]],
    *,
    ok_intent_strict: bool,
    ok_intent_structural: bool,
    ok_expanded_strict: bool,
    ok_expanded_structural: bool,
) -> str:
    # Heuristic taxonomy aligned with the paper:
    # - semantic/interface mismatch: IO skeleton differs (likely truly different kernel interface/semantics)
    # - extraction drift: structurally same, but strict signature differs (axis_roles/parallel_axes/specialization)
    # - lowering drift: intent layer consistent but expanded differs (macro lowering/decomposition differences)
    if any((v.get("contract") == "OUT_OF_SCOPE") for v in per_fe.values() if isinstance(v, dict)):
        return "facts_missing_or_out_of_scope"
    if not ok_intent_structural:
        return "intent_structural_mismatch"
    if ok_intent_structural and not ok_intent_strict:
        return "intent_minor_drift"
    if ok_intent_strict and not ok_expanded_structural:
        return "lowering_structural_mismatch"
    if ok_expanded_structural and not ok_expanded_strict:
        return "lowering_minor_drift"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", action="append", default=[], help="repeatable; default compares triton+tilelang")
    ap.add_argument(
        "--suite",
        choices=["artifact_intersection", "pipeline_coverage"],
        default="artifact_intersection",
        help="kernel universe: use cached artifacts (default) or import pipeline coverage lists (heavier)",
    )
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

    # Candidate kernels = intersection of all per-frontend sets.
    # Default: artifact-only (no heavy frontend imports). This avoids "hangs" when
    # Triton/TileLang/CUDA toolchains are not available, while still letting us
    # analyze cached reports.
    if str(args.suite) == "pipeline_coverage":
        sets = [set(_kernels_from_pipeline(fe)) for fe in frontends]
    else:
        sets = [set(_kernels_from_artifacts(fe)) for fe in frontends]
    kernels = sorted(set.intersection(*sets)) if sets else []
    if wanted:
        kernels = [k for k in kernels if k in wanted]

    results: List[PairResult] = []
    for k in kernels:
        per_fe: Dict[str, Dict[str, Any]] = {}
        tier = "D_none"
        kind = "unknown"
        ok_intent = True  # strict
        ok_expanded = True  # strict
        ok_intent_struct = True
        ok_expanded_struct = True
        axis_recall_intent = 1.0
        axis_recall_expanded = 1.0
        reasons_intent: List[str] = []
        reasons_expanded: List[str] = []
        reasons_intent_struct: List[str] = []
        reasons_expanded_struct: List[str] = []

        # Load per-frontend report + intent signature.
        for fe in frontends:
            out_dir = _artifact_dir(fe)
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / f"{k}.json"
            if args.refresh_artifacts:
                _run_pipeline(fe, k, out_dir=out_dir, cases_limit=int(args.cases_limit))
            if not report_path.exists():
                ok_intent = False
                ok_expanded = False
                ok_intent_struct = False
                ok_expanded_struct = False
                axis_recall_intent = 0.0
                axis_recall_expanded = 0.0
                reasons_intent.append(f"missing_report:{fe}")
                reasons_expanded.append(f"missing_report:{fe}")
                reasons_intent_struct.append(f"missing_report:{fe}")
                reasons_expanded_struct.append(f"missing_report:{fe}")
                continue
            rep = _read_json(report_path)
            intent_json = rep.get("intent")
            expanded_json = rep.get("intent_expanded") or intent_json
            cert_json = rep.get("certificate_v2")
            if not isinstance(intent_json, dict) or not isinstance(expanded_json, dict) or not isinstance(cert_json, dict):
                ok_intent = False
                ok_expanded = False
                ok_intent_struct = False
                ok_expanded_struct = False
                axis_recall_intent = 0.0
                axis_recall_expanded = 0.0
                reasons_intent.append(f"missing_intent_or_cert:{fe}")
                reasons_expanded.append(f"missing_intent_or_cert:{fe}")
                reasons_intent_struct.append(f"missing_intent_or_cert:{fe}")
                reasons_expanded_struct.append(f"missing_intent_or_cert:{fe}")
                continue
            try:
                intent = IntentFunction.from_json_dict(intent_json)
                intent_expanded = IntentFunction.from_json_dict(expanded_json)
            except Exception as e:
                ok_intent = False
                ok_expanded = False
                ok_intent_struct = False
                ok_expanded_struct = False
                axis_recall_intent = 0.0
                axis_recall_expanded = 0.0
                reasons_intent.append(f"intent_parse_error:{fe}:{type(e).__name__}")
                reasons_expanded.append(f"intent_parse_error:{fe}:{type(e).__name__}")
                reasons_intent_struct.append(f"intent_parse_error:{fe}:{type(e).__name__}")
                reasons_expanded_struct.append(f"intent_parse_error:{fe}:{type(e).__name__}")
                continue
            cert_v2 = _cert_v2_from_json(cert_json)
            anchors = _anchors_from_cert(cert_v2)
            tier = _anchor_tier(anchors)
            kind = _kernel_kind_hint(anchors)
            per_fe[fe] = {
                "contract": (rep.get("contract") or {}).get("level"),
                "sig_intent": _canonicalize_intent(intent),
                "sig_expanded": _canonicalize_intent(intent_expanded),
            }

        # Compare signatures (ignore schedule/meta).
        base_fe = frontends[0]
        base_intent_sig = per_fe.get(base_fe, {}).get("sig_intent")
        base_exp_sig = per_fe.get(base_fe, {}).get("sig_expanded")
        if not isinstance(base_intent_sig, dict):
            ok_intent = False
            ok_intent_struct = False
            axis_recall_intent = 0.0
            reasons_intent.append("missing_base_signature")
            reasons_intent_struct.append("missing_base_signature")
        if not isinstance(base_exp_sig, dict):
            ok_expanded = False
            ok_expanded_struct = False
            axis_recall_expanded = 0.0
            reasons_expanded.append("missing_base_signature")
            reasons_expanded_struct.append("missing_base_signature")
        for fe in frontends[1:]:
            sig_i = per_fe.get(fe, {}).get("sig_intent")
            sig_e = per_fe.get(fe, {}).get("sig_expanded")
            if isinstance(base_intent_sig, dict) and isinstance(sig_i, dict):
                if sig_i != base_intent_sig:
                    ok_intent = False
                    reasons_intent.extend([f"{fe}:{r}" for r in _diff_reasons(base_intent_sig, sig_i)])
                if not _interface_compatible(base_intent_sig, sig_i):
                    ok_intent_struct = False
                    reasons_intent_struct.append(f"{fe}:io_mismatch")
                if _ops_skeleton(base_intent_sig) != _ops_skeleton(sig_i):
                    ok_intent_struct = False
                    reasons_intent_struct.append(f"{fe}:ops_skeleton_mismatch")
                axis_recall_intent = min(axis_recall_intent, _axis_roles_recall(base_intent_sig, sig_i))
            else:
                ok_intent_struct = False
                axis_recall_intent = 0.0
            if isinstance(base_exp_sig, dict) and isinstance(sig_e, dict):
                if sig_e != base_exp_sig:
                    ok_expanded = False
                    reasons_expanded.extend([f"{fe}:{r}" for r in _diff_reasons(base_exp_sig, sig_e)])
                if not _interface_compatible(base_exp_sig, sig_e):
                    ok_expanded_struct = False
                    reasons_expanded_struct.append(f"{fe}:io_mismatch")
                if _ops_skeleton(base_exp_sig) != _ops_skeleton(sig_e):
                    ok_expanded_struct = False
                    reasons_expanded_struct.append(f"{fe}:ops_skeleton_mismatch")
                axis_recall_expanded = min(axis_recall_expanded, _axis_roles_recall(base_exp_sig, sig_e))
            else:
                ok_expanded_struct = False
                axis_recall_expanded = 0.0

        results.append(
            PairResult(
                kernel=k,
                anchor_tier=str(tier),
                kernel_kind=str(kind),
                ok_intent=bool(ok_intent),
                ok_expanded=bool(ok_expanded),
                ok_intent_structural=bool(ok_intent_struct),
                ok_expanded_structural=bool(ok_expanded_struct),
                axis_roles_recall_intent=float(axis_recall_intent),
                axis_roles_recall_expanded=float(axis_recall_expanded),
                reasons_intent=sorted(set(reasons_intent)),
                reasons_expanded=sorted(set(reasons_expanded)),
                reasons_intent_structural=sorted(set(reasons_intent_struct)),
                reasons_expanded_structural=sorted(set(reasons_expanded_struct)),
                sigs=per_fe,
            )
        )
        print(
            f"[E4:{'+'.join(frontends)}:{k}] intent={('OK' if ok_intent else 'FAIL')}/{('OK' if ok_intent_struct else 'FAIL')} expanded={('OK' if ok_expanded else 'FAIL')}/{('OK' if ok_expanded_struct else 'FAIL')}",
            flush=True,
        )

    # Summaries
    by_tier: Dict[str, Dict[str, int]] = {}
    by_kind: Dict[str, Dict[str, int]] = {}
    by_reason_intent: Dict[str, int] = {}
    by_reason_expanded: Dict[str, int] = {}
    by_reason_intent_struct: Dict[str, int] = {}
    by_reason_expanded_struct: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    axis_recalls_intent: List[float] = []
    axis_recalls_expanded: List[float] = []
    for r in results:
        cat = _mismatch_category(
            r.sigs,
            ok_intent_strict=r.ok_intent,
            ok_intent_structural=r.ok_intent_structural,
            ok_expanded_strict=r.ok_expanded,
            ok_expanded_structural=r.ok_expanded_structural,
        )
        by_tier.setdefault(r.anchor_tier, {"n": 0, "ok_intent": 0, "ok_expanded": 0})
        by_tier[r.anchor_tier]["n"] += 1
        if r.ok_intent:
            by_tier[r.anchor_tier]["ok_intent"] += 1
        if r.ok_expanded:
            by_tier[r.anchor_tier]["ok_expanded"] += 1
        by_tier[r.anchor_tier].setdefault("ok_intent_structural", 0)
        by_tier[r.anchor_tier].setdefault("ok_expanded_structural", 0)
        if r.ok_intent_structural:
            by_tier[r.anchor_tier]["ok_intent_structural"] += 1
        if r.ok_expanded_structural:
            by_tier[r.anchor_tier]["ok_expanded_structural"] += 1
        by_kind.setdefault(r.kernel_kind, {"n": 0, "ok_intent": 0, "ok_expanded": 0})
        by_kind[r.kernel_kind]["n"] += 1
        if r.ok_intent:
            by_kind[r.kernel_kind]["ok_intent"] += 1
        if r.ok_expanded:
            by_kind[r.kernel_kind]["ok_expanded"] += 1
        by_kind[r.kernel_kind].setdefault("ok_intent_structural", 0)
        by_kind[r.kernel_kind].setdefault("ok_expanded_structural", 0)
        if r.ok_intent_structural:
            by_kind[r.kernel_kind]["ok_intent_structural"] += 1
        if r.ok_expanded_structural:
            by_kind[r.kernel_kind]["ok_expanded_structural"] += 1
        for rr in r.reasons_intent:
            by_reason_intent[rr] = by_reason_intent.get(rr, 0) + 1
        for rr in r.reasons_expanded:
            by_reason_expanded[rr] = by_reason_expanded.get(rr, 0) + 1
        for rr in r.reasons_intent_structural:
            by_reason_intent_struct[rr] = by_reason_intent_struct.get(rr, 0) + 1
        for rr in r.reasons_expanded_structural:
            by_reason_expanded_struct[rr] = by_reason_expanded_struct.get(rr, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1
        axis_recalls_intent.append(float(r.axis_roles_recall_intent))
        axis_recalls_expanded.append(float(r.axis_roles_recall_expanded))

    out = {
        "experiment": "E4_cross_frontend_consistency",
        "frontends": list(frontends),
        "suite": str(args.suite),
        "kernels": list(kernels),
        "summary": {
            "n": int(len(results)),
            "intent_ok": int(sum(1 for r in results if r.ok_intent)),
            "intent_ok_rate": (float(sum(1 for r in results if r.ok_intent)) / float(len(results)) if results else 0.0),
            "intent_structural_ok": int(sum(1 for r in results if r.ok_intent_structural)),
            "intent_structural_ok_rate": (
                float(sum(1 for r in results if r.ok_intent_structural)) / float(len(results)) if results else 0.0
            ),
            "expanded_ok": int(sum(1 for r in results if r.ok_expanded)),
            "expanded_ok_rate": (float(sum(1 for r in results if r.ok_expanded)) / float(len(results)) if results else 0.0),
            "expanded_structural_ok": int(sum(1 for r in results if r.ok_expanded_structural)),
            "expanded_structural_ok_rate": (
                float(sum(1 for r in results if r.ok_expanded_structural)) / float(len(results)) if results else 0.0
            ),
            "axis_roles_recall_intent_avg": (float(sum(axis_recalls_intent)) / float(len(axis_recalls_intent)) if axis_recalls_intent else 0.0),
            "axis_roles_recall_expanded_avg": (
                float(sum(axis_recalls_expanded)) / float(len(axis_recalls_expanded)) if axis_recalls_expanded else 0.0
            ),
            "by_tier": by_tier,
            "by_kind": by_kind,
            "mismatch_categories": {k: int(v) for k, v in sorted(by_category.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))},
            "top_reasons_intent": {
                k: int(v) for k, v in sorted(by_reason_intent.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:20]
            },
            "top_reasons_expanded": {
                k: int(v)
                for k, v in sorted(by_reason_expanded.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:20]
            },
            "top_reasons_intent_structural": {
                k: int(v)
                for k, v in sorted(by_reason_intent_struct.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:20]
            },
            "top_reasons_expanded_structural": {
                k: int(v)
                for k, v in sorted(by_reason_expanded_struct.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:20]
            },
        },
        "results": [
            {
                "kernel": r.kernel,
                "anchor_tier": r.anchor_tier,
                "kernel_kind": r.kernel_kind,
                "ok": {
                    "intent_strict": bool(r.ok_intent),
                    "intent_structural": bool(r.ok_intent_structural),
                    "expanded_strict": bool(r.ok_expanded),
                    "expanded_structural": bool(r.ok_expanded_structural),
                },
                "axis_roles_recall": {"intent": float(r.axis_roles_recall_intent), "expanded": float(r.axis_roles_recall_expanded)},
                "reasons": {
                    "intent_strict": list(r.reasons_intent),
                    "intent_structural": list(r.reasons_intent_structural),
                    "expanded_strict": list(r.reasons_expanded),
                    "expanded_structural": list(r.reasons_expanded_structural),
                },
                "category": _mismatch_category(
                    r.sigs,
                    ok_intent_strict=r.ok_intent,
                    ok_intent_structural=r.ok_intent_structural,
                    ok_expanded_strict=r.ok_expanded,
                    ok_expanded_structural=r.ok_expanded_structural,
                ),
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
