"""
Paper experiment (E1 baseline): rule-only Intent recovery (no LLM).

Goal:
  Provide a conservative baseline that uses only frontend anchors/evidence
  (CertificateV2.semantic_facts) to build a minimal IntentIR skeleton.

This baseline is intentionally simple:
  - it does NOT inspect kernel source (beyond what the frontend already used)
  - it does NOT attempt full semantic reconstruction (that is what LLM is for)
  - it targets "static_validate" pass-rate as a lower bound for recoverability

Why this matters for the paper:
  Reviewer question: "Do you really need the LLM?"
  We answer by showing that anchor-only recovery is much weaker than the full
  LLM+verification pipeline (E1 + E3).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontends.common.certificate_v2 import SemanticCertificateV2
from frontends.common.static_validate import static_validate
from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get(d: Dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


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


def _kernels_from_pipeline(frontend: str, suite: str) -> List[str]:
    if suite not in {"smoke", "coverage", "all"}:
        raise ValueError(f"unknown suite: {suite}")
    if frontend == "triton":
        from pipeline.triton.core import coverage_kernel_specs, default_kernel_specs  # noqa: PLC0415

        specs_fn = default_kernel_specs if suite == "smoke" else coverage_kernel_specs
        return [s.name for s in specs_fn()]
    if frontend == "tilelang":
        from pipeline.tilelang.core import coverage_kernel_specs, default_kernel_specs  # noqa: PLC0415

        specs_fn = default_kernel_specs if suite == "smoke" else coverage_kernel_specs
        return [s.name for s in specs_fn()]
    if frontend == "cuda":
        from pipeline.cuda.core import coverage_kernel_specs, default_kernel_specs  # noqa: PLC0415

        specs_fn = default_kernel_specs if suite == "smoke" else coverage_kernel_specs
        return [s.name for s in specs_fn()]
    raise ValueError(f"unknown frontend: {frontend}")


def _iter_accesses_from_cert(cert: SemanticCertificateV2) -> List[Dict[str, Any]]:
    sf = cert.semantic_facts or {}
    ce = sf.get("canonical_evidence")
    if isinstance(ce, dict):
        acc = ce.get("accesses")
        if isinstance(acc, list):
            return [a for a in acc if isinstance(a, dict)]
    return []


def _anchors_from_cert(cert: SemanticCertificateV2) -> Dict[str, Any]:
    sf = cert.semantic_facts or {}
    anchors = sf.get("anchors")
    return dict(anchors) if isinstance(anchors, dict) else {}


def _kernel_kind(anchors: Dict[str, Any]) -> str:
    if bool(anchors.get("has_dot")) and bool(anchors.get("has_reduce")):
        return "attention"
    if bool(anchors.get("has_dot")):
        return "matmul"
    if bool(anchors.get("has_reduce")):
        return "reduce"
    if bool(anchors.get("has_copy")):
        return "copy"
    return "unknown"


def _anchor_tier(anchors: Dict[str, Any]) -> str:
    if bool(anchors.get("has_dot")):
        return "A_dot"
    if bool(anchors.get("has_reduce")):
        return "B_reduce"
    if bool(anchors.get("has_copy")):
        return "C_copy"
    return "D_none"


def _contract_level_from_cert(cert: SemanticCertificateV2) -> str:
    # Pipelines attach contract to cert.meta to keep semantic_facts stable.
    meta = cert.meta or {}
    c = meta.get("contract") if isinstance(meta, dict) else None
    if isinstance(c, dict) and isinstance(c.get("level"), str):
        return str(c.get("level"))
    # Fallback: some reports copy contract into semantic_facts.
    sf = cert.semantic_facts or {}
    c2 = sf.get("contract")
    if isinstance(c2, dict) and isinstance(c2.get("level"), str):
        return str(c2.get("level"))
    return "N/A"


def _tensor_roles_from_accesses(accesses: List[Dict[str, Any]]) -> Tuple[List[str], List[str], Dict[str, Dict[str, Any]]]:
    """
    Infer (inputs, outputs, tensor_meta) from access kinds:
      - any tensor with a store access is considered an output
      - tensors with only loads are inputs
    """
    meta: Dict[str, Dict[str, Any]] = {}
    stores: set[str] = set()
    loads: set[str] = set()
    for a in accesses:
        t = a.get("tensor")
        if not isinstance(t, str) or not t:
            continue
        kind = str(a.get("kind") or "")
        dt = str(a.get("dtype") or "f32")
        rank = int(a.get("rank") or 0)
        meta.setdefault(t, {"dtype": dt, "rank": rank, "kinds": set()})
        meta[t]["dtype"] = dt
        meta[t]["rank"] = rank
        meta[t]["kinds"].add(kind)
        if kind == "store":
            stores.add(t)
        if kind == "load":
            loads.add(t)
    outs = sorted(stores)
    ins = sorted(loads - stores)
    # Normalize kinds to JSON-friendly.
    for t in list(meta.keys()):
        ks = meta[t].get("kinds")
        meta[t]["kinds"] = sorted(str(x) for x in (ks or []))
    return ins, outs, meta


def _make_tensor_type(name: str, *, dtype: str, rank: int) -> TensorType:
    rm = TensorLayout(kind="row_major", params={})
    dt = "bool" if str(dtype) in {"i1", "bool"} else str(dtype)
    r = max(0, int(rank))
    shape = [Dim("sym", f"{name}_d{i}") for i in range(r)]
    return TensorType(dtype=dt, shape=shape, layout=rm)


def _build_rule_only_intent(*, kernel: str, cert_v2: SemanticCertificateV2) -> IntentFunction:
    anchors = _anchors_from_cert(cert_v2)
    kind = _kernel_kind(anchors)
    accesses = _iter_accesses_from_cert(cert_v2)
    ins, outs, meta = _tensor_roles_from_accesses(accesses)

    # Fallback naming when evidence is missing.
    if not ins:
        ins = ["A", "B"]
    if not outs:
        outs = ["O"]

    tensors: Dict[str, TensorType] = {}
    for t in sorted(set(ins + outs)):
        info = meta.get(t, {})
        tensors[t] = _make_tensor_type(t, dtype=str(info.get("dtype") or "f32"), rank=int(info.get("rank") or 1))

    ops: List[Op] = []
    main_out = outs[0]

    def ensure_tensor(name: str, like: str) -> None:
        if name in tensors:
            return
        tensors[name] = tensors[like]

    if kind == "matmul":
        a = ins[0]
        b = ins[1] if len(ins) > 1 else ins[0]
        ops.append(Op(op="matmul", inputs=[a, b], output=main_out, attrs={}))
    elif kind == "attention":
        q = ins[0]
        k = ins[1] if len(ins) > 1 else ins[0]
        v = ins[2] if len(ins) > 2 else ins[0]
        ensure_tensor("scores", q)
        ensure_tensor("probs", q)
        ops.append(Op(op="matmul", inputs=[q, k], output="scores", attrs={"transpose_b": True}))
        ops.append(Op(op="softmax", inputs=["scores"], output="probs", attrs={"axis": -1}))
        ops.append(Op(op="matmul", inputs=["probs", v], output=main_out, attrs={}))
    elif kind == "reduce":
        a = ins[0]
        ops.append(Op(op="reduce_sum", inputs=[a], output=main_out, attrs={"dims": [0], "keepdims": False}))
    elif kind == "copy":
        a = ins[0]
        ops.append(Op(op="identity", inputs=[a], output=main_out, attrs={}))
    else:
        # Unknown: at least produce the output from the first input to satisfy SV output checks.
        a = ins[0]
        ops.append(Op(op="identity", inputs=[a], output=main_out, attrs={}))

    # Ensure every output is produced by some op (SV requirement).
    for out in outs[1:]:
        if out not in tensors:
            tensors[out] = tensors[main_out]
        ops.append(Op(op="identity", inputs=[main_out], output=out, attrs={}))

    return IntentFunction(
        name=str(kernel),
        tensors=tensors,
        ops=ops,
        outputs=list(outs),
        schedule=ScheduleSketch(),
        axis_roles={},
    )


@dataclass(frozen=True)
class Row:
    kernel: str
    contract: str
    anchor_tier: str
    kernel_kind: str
    ok: bool
    category: str
    reasons: List[str]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "kernel": str(self.kernel),
            "contract": str(self.contract),
            "anchor_tier": str(self.anchor_tier),
            "kernel_kind": str(self.kernel_kind),
            "ok": bool(self.ok),
            "category": str(self.category),
            "reasons": list(self.reasons),
        }


def _summarize(rows: List[Row]) -> Dict[str, Any]:
    total = len(rows)
    ok = sum(1 for r in rows if r.ok)
    by_tier: Dict[str, Dict[str, int]] = {}
    fails: Dict[str, int] = {}
    contracts: Dict[str, int] = {}
    for r in rows:
        by_tier.setdefault(r.anchor_tier, {"n": 0, "ok": 0})
        by_tier[r.anchor_tier]["n"] += 1
        by_tier[r.anchor_tier]["ok"] += 1 if r.ok else 0
        fails[r.category] = fails.get(r.category, 0) + 1
        contracts[r.contract] = contracts.get(r.contract, 0) + 1
    return {
        "n": int(total),
        "ok": int(ok),
        "ok_rate": (float(ok) / float(total) if total else 0.0),
        "by_tier": by_tier,
        "failures": fails,
        "contracts": contracts,
    }


def _prepare_frontend_artifacts(*, frontend: str, kernel: str, artifacts_dir: Path) -> Dict[str, Any]:
    """
    Prepare descriptor + CertificateV2 without invoking the LLM.

    For Triton this may trigger a single JIT run to dump TTIR (via ensure_artifacts).
    """
    from pipeline import registry as pipeline_registry  # noqa: PLC0415

    from frontends.common.contract_v2 import evaluate_contract_v2  # noqa: PLC0415
    from frontends.common.obligations import evaluate_obligations  # noqa: PLC0415

    spec = _spec_from_pipeline(frontend, kernel)
    adapter = pipeline_registry.get(str(frontend))
    desc = adapter.build_descriptor(spec)
    desc.meta["artifact_dir"] = str(artifacts_dir)
    desc = adapter.ensure_artifacts(desc, spec)
    facts = adapter.extract_facts(desc)
    constraints = adapter.extract_constraints(desc, facts)
    if str(frontend) == "triton":
        # Triton adapter still returns the legacy v1 certificate; build v2 directly.
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

    return {"descriptor": desc.to_json_dict(), "certificate_v2": cert_v2.to_json_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang", "cuda", "both", "all"], default="both")
    ap.add_argument("--suite", choices=["smoke", "coverage", "all"], default="coverage")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; restrict to kernel name(s)")
    ap.add_argument("--refresh-artifacts", action="store_true", help="rebuild descriptor/cert without LLM (stage4 only)")
    ap.add_argument("--cases-limit", type=int, default=1, help="unused (compat placeholder)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "experiments" / "e1_rule_only_latest.json"))
    args = ap.parse_args()

    if str(args.frontend) == "both":
        frontends = ["triton", "tilelang"]
    elif str(args.frontend) == "all":
        frontends = ["triton", "tilelang", "cuda"]
    else:
        frontends = [str(args.frontend)]

    wanted = {str(x) for x in (args.kernel or []) if str(x).strip()}

    out_rows: List[Row] = []
    for fe in frontends:
        art_dir = _artifact_dir(fe)
        art_dir.mkdir(parents=True, exist_ok=True)
        kernels = _kernels_from_pipeline(fe, str(args.suite))
        if wanted:
            kernels = [k for k in kernels if k in wanted]
        for k in kernels:
            report_path = art_dir / f"{k}.json"
            rep: Dict[str, Any] | None = None
            if report_path.exists() and not bool(args.refresh_artifacts):
                try:
                    rep = _read_json(report_path)
                except Exception:
                    rep = None
            if rep is None:
                try:
                    rep = _prepare_frontend_artifacts(frontend=str(fe), kernel=str(k), artifacts_dir=art_dir)
                except Exception as e:
                    out_rows.append(
                        Row(
                            kernel=str(k),
                            contract="N/A",
                            anchor_tier="D_none",
                            kernel_kind="unknown",
                            ok=False,
                            category=f"artifact_error:{type(e).__name__}",
                            reasons=[str(e)],
                        )
                    )
                    print(f"[rule_only:{fe}:{k}] FAIL (artifact_error:{type(e).__name__})", flush=True)
                    continue

            cert_json = rep.get("certificate_v2")
            if not isinstance(cert_json, dict):
                out_rows.append(
                    Row(
                        kernel=str(k),
                        contract="N/A",
                        anchor_tier="D_none",
                        kernel_kind="unknown",
                        ok=False,
                        category="missing_certificate_v2",
                        reasons=[],
                    )
                )
                print(f"[rule_only:{fe}:{k}] FAIL (missing_certificate_v2)", flush=True)
                continue

            cert_v2 = SemanticCertificateV2(
                schema_version=str(cert_json.get("schema_version") or "cert_v2.0"),
                semantic_facts=(cert_json.get("semantic_facts") if isinstance(cert_json.get("semantic_facts"), dict) else {}),
                schedule_hints=(cert_json.get("schedule_hints") if isinstance(cert_json.get("schedule_hints"), dict) else {}),
                meta=(cert_json.get("meta") if isinstance(cert_json.get("meta"), dict) else {}),
            ).canonicalize()
            anchors = _anchors_from_cert(cert_v2)
            intent = _build_rule_only_intent(kernel=str(k), cert_v2=cert_v2)
            sv = static_validate(intent, cert_v2)
            tier = _anchor_tier(anchors)
            kind = _kernel_kind(anchors)
            contract = _contract_level_from_cert(cert_v2)
            category = "ok" if bool(sv.ok) else "static_validate_fail"
            out_rows.append(
                Row(
                    kernel=f"{fe}:{k}",
                    contract=str(contract),
                    anchor_tier=str(tier),
                    kernel_kind=str(kind),
                    ok=bool(sv.ok),
                    category=category,
                    reasons=list(sv.reasons),
                )
            )
            status = "OK" if sv.ok else "FAIL"
            print(f"[rule_only:{fe}:{k}] {status} ({category})", flush=True)

    out: Dict[str, Any] = {
        "experiment": "E1_rule_only_baseline",
        "frontends": list(frontends),
        "suite": str(args.suite),
        "kernels": sorted({r.kernel for r in out_rows}),
        "summary": _summarize(out_rows),
        "results": [r.to_json_dict() for r in out_rows],
    }
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
