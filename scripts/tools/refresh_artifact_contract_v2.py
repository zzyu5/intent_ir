#!/usr/bin/env python3
"""
Refresh (gitignored) `artifacts/full_pipeline_verify/*.json` with the *current*
CertificateV2 + obligations + ContractV2, without rerunning the full Triton
pipeline or recompiling kernels.

Why this exists:
  - We often iterate on certificate extraction / obligation checking logic.
  - Old reports may embed stale `contract_v2` inside `intent.meta`, which then
    affects backend decisions (e.g., CUDA specialization gating).
  - Full pipeline refresh is slow and noisy; for paper/debug we want a fast
    "recompute evidence from TTIR + descriptor" path.

What it updates per kernel:
  - `<kernel>.certificate_v2.json`
  - `<kernel>.contract.json`
  - `<kernel>.json` report fields: `certificate_v2`, `obligations`, `contract`
  - `report["intent"]["meta"]["contract_v2"]` (and keeps other meta intact)
  - `report["static_validation"]` (Stage-A static validation against cert_v2)

It expects the artifacts directory already contains:
  - `<kernel>.ttir`
  - `<kernel>.descriptor.json` (or embedded `report["descriptor"]`)
  - `<kernel>.json` report
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.interfaces import KernelArtifactBundle, KernelDescriptor  # noqa: E402

from frontends.common.contract_v2 import evaluate_contract_v2  # noqa: E402
from frontends.common.obligations import evaluate_obligations  # noqa: E402
from frontends.common.static_validate import static_validate  # noqa: E402
from frontends.triton.certificate import build_certificate_v2  # noqa: E402
from frontends.triton.facts import extract_constraints, extract_facts  # noqa: E402

from intent_ir.ir import IntentFunction  # noqa: E402


DEFAULT_DIR = ROOT / "artifacts" / "full_pipeline_verify"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_descriptor(report: Dict[str, Any], *, artifact_dir: Path, kernel: str) -> KernelDescriptor:
    dp = report.get("descriptor_path")
    if isinstance(dp, str) and dp:
        p = Path(dp)
        if not p.is_absolute():
            p = artifact_dir / p
        if p.is_file():
            d = _read_json(p)
            return _descriptor_from_json(d)
    # Fallback to canonical path under artifact_dir
    p2 = artifact_dir / f"{kernel}.descriptor.json"
    if p2.is_file():
        return _descriptor_from_json(_read_json(p2))
    # Fallback to embedded descriptor
    d2 = report.get("descriptor")
    if isinstance(d2, dict):
        return _descriptor_from_json(d2)
    raise FileNotFoundError(f"descriptor not found for {kernel}")


def _descriptor_from_json(d: Dict[str, Any]) -> KernelDescriptor:
    art = KernelArtifactBundle(**(d.get("artifacts") or {}))
    return KernelDescriptor(
        schema_version=str(d.get("schema_version") or ""),
        name=str(d.get("name") or ""),
        frontend=str(d.get("frontend") or "triton"),
        source_kind=str(d.get("source_kind") or "source"),
        source_text=str(d.get("source_text") or ""),
        launch=d.get("launch") if isinstance(d.get("launch"), dict) else {},
        io_spec=d.get("io_spec") if isinstance(d.get("io_spec"), dict) else {},
        artifacts=art,
        frontend_facts=d.get("frontend_facts") if isinstance(d.get("frontend_facts"), dict) else {},
        frontend_constraints=d.get("frontend_constraints") if isinstance(d.get("frontend_constraints"), dict) else {},
        meta=d.get("meta") if isinstance(d.get("meta"), dict) else {},
    )


def _load_ttir_text(*, artifact_dir: Path, kernel: str) -> str:
    p = artifact_dir / f"{kernel}.ttir"
    if not p.is_file():
        raise FileNotFoundError(f"TTIR not found: {p}")
    return p.read_text(encoding="utf-8")


def _load_intent(report: Dict[str, Any], kernel: str) -> IntentFunction:
    it = report.get("intent")
    if not isinstance(it, dict):
        raise KeyError(f"report missing intent for {kernel}")
    return IntentFunction.from_json_dict(it)


def _refresh_one(artifact_dir: Path, kernel: str) -> Dict[str, Any]:
    report_path = artifact_dir / f"{kernel}.json"
    report = _read_json(report_path)

    desc = _load_descriptor(report, artifact_dir=artifact_dir, kernel=kernel)
    ttir_text = _load_ttir_text(artifact_dir=artifact_dir, kernel=kernel)
    facts = extract_facts(ttir_text)
    constraints = extract_constraints(ttir_text, facts=facts)

    cert_v2 = build_certificate_v2(ttir_text, desc=desc, facts=facts)
    obligations = evaluate_obligations(desc, cert_v2)
    # Keep obligations in semantic_facts (schema-versioned).
    cert_v2.semantic_facts["obligations"] = [o.to_json_dict() for o in obligations]

    contract = evaluate_contract_v2(desc, cert_v2, obligations, constraints=constraints)
    # Store contract summary in cert_v2.meta (NOT semantic_facts).
    cert_v2.meta = dict(getattr(cert_v2, "meta", {}) or {})
    cert_v2.meta["contract"] = {"level": str(contract.level), "reasons": list(contract.reasons), "assumptions": list(contract.assumptions)}

    # Update report fields.
    report["certificate_v2"] = cert_v2.to_json_dict()
    report["obligations"] = [o.to_json_dict() for o in obligations]
    report["contract"] = {
        "level": str(contract.level),
        "reasons": list(contract.reasons),
        "assumptions": list(contract.assumptions),
        "signals": dict(contract.signals),
    }

    # Keep backend-facing meta up to date.
    try:
        intent = _load_intent(report, kernel)
        meta = dict(getattr(intent, "meta", {}) or {})
        meta["contract_v2"] = {"level": str(contract.level), "reasons": list(contract.reasons), "assumptions": list(contract.assumptions)}
        # Preserve canonical_shapes if present in descriptor launch.
        if isinstance(desc.launch, dict) and isinstance(desc.launch.get("canonical_shapes"), dict):
            meta.setdefault("canonical_shapes", dict(desc.launch.get("canonical_shapes") or {}))
        # Refresh schedule hints + access witnesses used by GPU backends.
        try:
            sh = getattr(cert_v2, "schedule_hints", {}) or {}
            if isinstance(sh, dict):
                meta["schedule_hints_v2"] = {
                    "tile_hints": list(sh.get("tile_hints") or []) if isinstance(sh.get("tile_hints"), list) else [],
                    "symbol_ranges": dict(sh.get("symbol_ranges") or {}) if isinstance(sh.get("symbol_ranges"), dict) else {},
                }
        except Exception:
            pass
        try:
            from frontends.common.access_witness import build_stride_summary  # noqa: PLC0415

            shape_bindings: dict[str, int] = {}
            cs = meta.get("canonical_shapes")
            if isinstance(cs, dict):
                for kk, vv in cs.items():
                    if isinstance(kk, str) and isinstance(vv, int):
                        shape_bindings[str(kk)] = int(vv)
                    elif isinstance(kk, str) and isinstance(vv, float) and float(vv).is_integer():
                        shape_bindings[str(kk)] = int(vv)
            ss = build_stride_summary(cert_v2.to_json_dict(), shape_bindings=shape_bindings)
            j = ss.to_json_dict()
            tp = j.get("tensor_penalty") if isinstance(j.get("tensor_penalty"), dict) else {}
            top = sorted(((str(k), float(v)) for k, v in tp.items()), key=lambda kv: kv[1], reverse=True)[:8]
            axis_contig_len: dict[str, int] = {}
            for a in (j.get("accesses") or []):
                if not isinstance(a, dict):
                    continue
                axis_bind = a.get("axis_bindings") if isinstance(a.get("axis_bindings"), dict) else {}
                for r in (a.get("range_strides") or []):
                    if not isinstance(r, dict):
                        continue
                    try:
                        stride_elems = r.get("stride_elems")
                        range_len = r.get("range_len")
                        if stride_elems is None or int(stride_elems) != 1:
                            continue
                        if range_len is None:
                            continue
                        sym = str(r.get("range_sym") or "")
                        axis = axis_bind.get(sym)
                        if not isinstance(axis, str) or not axis:
                            continue
                        axis_contig_len[axis] = max(int(axis_contig_len.get(axis, 0)), int(range_len))
                    except Exception:
                        continue
            meta["access_witness"] = {
                "dominant_axis": j.get("dominant_axis"),
                "dominant_range": j.get("dominant_range"),
                "dominant_range_len": j.get("dominant_range_len"),
                "has_contiguous_range": bool(j.get("has_contiguous_range")),
                "tensor_penalty_top": top,
                "notes": list(j.get("notes") or []) if isinstance(j.get("notes"), list) else [],
            }
            if axis_contig_len:
                meta["access_witness"]["axis_contig_len"] = dict(axis_contig_len)
        except Exception:
            pass
        intent.meta = meta
        report["intent"] = intent.to_json_dict()
        # Also refresh Stage-A static validation report (purely for paper/debug).
        try:
            sv = static_validate(intent, cert_v2)
            report["static_validation"] = {
                "ok": bool(sv.ok),
                "reasons": list(sv.reasons),
                "obligations": [asdict(o) for o in (sv.obligations or [])],
            }
        except Exception as e:
            report["static_validation_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        report["intent_refresh_error"] = f"{type(e).__name__}: {e}"

    # Write sidecar JSONs (matches pipeline naming).
    _write_json(artifact_dir / f"{kernel}.certificate_v2.json", report["certificate_v2"])
    _write_json(artifact_dir / f"{kernel}.contract.json", report["contract"])
    _write_json(report_path, report)
    return {"kernel": kernel, "contract_level": str(contract.level), "obligations": {o.id: o.status for o in obligations}}


def _discover_kernels(artifact_dir: Path) -> List[str]:
    out: List[str] = []
    for p in sorted(artifact_dir.glob("*.json")):
        name = p.name
        if name.endswith(".descriptor.json") or name.endswith(".certificate_v2.json") or name.endswith(".contract.json"):
            continue
        k = name[:-5]
        # Require TTIR alongside.
        if (artifact_dir / f"{k}.ttir").is_file():
            out.append(k)
    return out


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR, help="Artifacts directory (default: artifacts/full_pipeline_verify)")
    ap.add_argument("--kernel", action="append", default=[], help="Kernel name (repeatable). Default: refresh all kernels in dir.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    artifact_dir = Path(args.artifact_dir)
    kernels = [str(x) for x in (args.kernel or []) if str(x)]
    if not kernels:
        kernels = _discover_kernels(artifact_dir)
    if not kernels:
        raise SystemExit(f"No kernels found under {artifact_dir}")

    rows = []
    for k in kernels:
        try:
            rows.append(_refresh_one(artifact_dir, k))
            print(f"[refresh] {k}: contract={rows[-1]['contract_level']}")
        except Exception as e:
            print(f"[refresh] {k}: FAIL {type(e).__name__}: {e}")

    # Small summary for sanity.
    by_level: Dict[str, int] = {}
    for r in rows:
        by_level[str(r.get("contract_level") or "")] = by_level.get(str(r.get("contract_level") or ""), 0) + 1
    print(f"[summary] kernels={len(rows)} by_contract_level={by_level}")


if __name__ == "__main__":
    main()
