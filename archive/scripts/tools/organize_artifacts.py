#!/usr/bin/env python3
"""
Organize local (gitignored) `artifacts/` into a clearer per-experiment layout.

This script is intentionally lightweight and safe to rerun. It:
  - buckets `artifacts/experiments/*.json` into `E1/..E5/` and `summaries/`
  - buckets common root-level smoke logs into `artifacts/remote/` / `artifacts/llm_raw/`
  - writes `artifacts/experiments/_LATEST_INDEX.json` pointing at the newest results

Notes:
  - `artifacts/` is gitignored; this is purely for local sanity.
  - We avoid moving canonical "latest" paths that other scripts may reference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
EXP = ART / "experiments"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _latest(dir_path: Path, patterns: List[str]) -> Optional[Path]:
    cand: List[Path] = []
    for pat in patterns:
        cand.extend(list(dir_path.rglob(pat)))
    cand = [p for p in cand if p.is_file()]
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]


def _classify_experiment_file(name: str) -> Optional[str]:
    # Paper summary snapshots.
    if name.startswith("paper_summary_"):
        return "summaries"

    # Experiment families (keep original filenames; just bucket them).
    if name.startswith("e1_"):
        return "E1"
    if name.startswith("e2_"):
        return "E2"
    if name.startswith("e3_") or name.startswith("e1e3_") or name.startswith("llm_regression_suite"):
        return "E1E3"
    if name.startswith("e4_"):
        return "E4"
    if name.startswith("e5_") or name.startswith("experiment_a_") or name.startswith("portability_vs_perf"):
        return "E5"

    return None


def _move_experiment_jsons() -> int:
    if not EXP.exists():
        return 0
    moved = 0
    for p in list(EXP.iterdir()):
        if not p.is_file():
            continue
        bucket = _classify_experiment_file(p.name)
        if bucket is None:
            continue
        # Keep shared profiles at the root for now (many experiment JSONs embed this path).
        if p.name.startswith("rvv_profile_"):
            continue
        dst = EXP / bucket / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        p.rename(dst)
        moved += 1
    return moved


def _move_root_smoke_logs() -> int:
    if not ART.exists():
        return 0
    moved = 0
    remote_dir = ART / "remote"
    llm_raw_dir = ART / "llm_raw"
    remote_dir.mkdir(parents=True, exist_ok=True)
    llm_raw_dir.mkdir(parents=True, exist_ok=True)
    for p in list(ART.iterdir()):
        if not p.is_file():
            continue
        # Keep canonical suites in-place (scripts default paths).
        if p.name in {"rvv_remote_suite_latest.json", "rvv_remote_suite_smoke_latest.json"}:
            continue
        if (
            (p.name.startswith("rvv_remote_all_smoke_case0") and p.suffix == ".json")
            or (p.name.startswith("rvv_remote_") and "_smoke_case" in p.name and p.suffix == ".json")
        ):
            p.rename(remote_dir / p.name)
            moved += 1
            continue
        if p.name.startswith("llm_raw_") and p.suffix in {".txt", ".log"}:
            p.rename(llm_raw_dir / p.name)
            moved += 1
            continue
    return moved


def _summarize_e5_1(path: Path) -> Dict[str, Any]:
    d = _read_json(path)
    rows = []
    for it in d.get("kernels") or []:
        if not isinstance(it, dict):
            continue
        name = it.get("ours", {}).get("kernel") or it.get("baseline_name") or it.get("kernel")
        sp = it.get("speedup_ours_over_baseline")
        if isinstance(name, str) and isinstance(sp, (int, float)) and float(sp) > 0:
            rows.append((name, float(sp)))
    rows.sort(key=lambda x: x[1])
    geo = math.exp(sum(math.log(s) for _, s in rows) / len(rows)) if rows else None
    return {
        "path": str(path),
        "kernels": len(rows),
        "geomean_speedup_ours_over_baseline": geo,
        "min": rows[0] if rows else None,
        "max": rows[-1] if rows else None,
        "slower_count": sum(1 for _, s in rows if s < 1.0),
    }


def _summarize_e5_2(path: Path) -> Dict[str, Any]:
    d = _read_json(path)
    rows = []
    for it in d.get("kernels") or []:
        if not isinstance(it, dict) or it.get("status") != "ok":
            continue
        name = it.get("kernel")
        sp = it.get("speedup_retune_vs_freeze")
        if isinstance(name, str) and isinstance(sp, (int, float)) and float(sp) > 0:
            rows.append((name, float(sp)))
    rows.sort(key=lambda x: x[1])
    geo = math.exp(sum(math.log(s) for _, s in rows) / len(rows)) if rows else None
    return {
        "path": str(path),
        "kernels_ok": len(rows),
        "geomean_speedup_retune_vs_freeze": geo,
        "min": rows[0] if rows else None,
        "max": rows[-1] if rows else None,
        "regressions": sum(1 for _, s in rows if s < 1.0),
        "top5": rows[-5:] if len(rows) >= 5 else rows,
        "bottom5": rows[:5] if len(rows) >= 5 else rows,
    }


def _summarize_e4(path: Path) -> Dict[str, Any]:
    d = _read_json(path)
    s = d.get("summary") or {}
    if not isinstance(s, dict):
        s = {}
    return {
        "path": str(path),
        "n": s.get("n"),
        "intent_structural_ok_rate": s.get("intent_structural_ok_rate"),
        "expanded_structural_ok_rate": s.get("expanded_structural_ok_rate"),
        "axis_roles_recall_intent_avg": s.get("axis_roles_recall_intent_avg"),
        "mismatch_categories": s.get("mismatch_categories"),
    }


def _summarize_e3(path: Path) -> Dict[str, Any]:
    d = _read_json(path)
    out: Dict[str, Any] = {"path": str(path), "delta": d.get("delta")}
    v = d.get("variants")
    if not isinstance(v, dict):
        return out
    for variant in ["one_shot", "feedback"]:
        vv = v.get(variant) or {}
        s = (vv.get("summary") or {}).get("by_frontend") or {}
        if not isinstance(s, dict):
            continue
        out[variant] = {}
        for fe, ss in s.items():
            if not isinstance(ss, dict):
                continue
            out[variant][fe] = {
                "n": ss.get("n"),
                "ok": ss.get("ok"),
                "ok_rate": ss.get("ok_rate"),
                "llm_cost": ss.get("llm_cost"),
                "failures": ss.get("failures"),
            }
    return out


def _summarize_e2_masked6(path: Path) -> Dict[str, Any]:
    d = _read_json(path)
    agg = ((d.get("summary") or {}).get("aggregate")) or {}
    return {"path": str(path), "frontend": d.get("frontend"), "suite": d.get("suite"), "aggregate": agg}


def _summarize_e1_rule_only(path: Path) -> Dict[str, Any]:
    d = _read_json(path)
    return {"path": str(path), "summary": d.get("summary")}


def _summarize_e1_semantic_acc() -> Dict[str, Any]:
    labels_path = ROOT / "data" / "human_labels.json"
    rule_only_path = _latest(EXP, ["e1_rule_only*.json"])
    if not labels_path.exists() or rule_only_path is None:
        return {}

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict):
        return {}

    d = _read_json(rule_only_path)
    results = d.get("results") or []
    if not isinstance(results, list):
        return {}

    by_fe: Dict[str, Dict[str, str]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        k = r.get("kernel")
        if not isinstance(k, str):
            continue
        fe, name = (k.split(":", 1) + [""])[:2] if ":" in k else ("unknown", k)
        if name not in labels or name == "__meta__":
            continue
        by_fe.setdefault(fe, {}).setdefault(name, str(r.get("kernel_kind") or ""))

    out: Dict[str, Any] = {"labels": str(labels_path), "rule_only_path": str(rule_only_path), "by_frontend": {}}
    for fe, mp in by_fe.items():
        n = 0
        ok = 0
        for name, pred in mp.items():
            gt = (labels.get(name) or {}).get("semantic_class")
            if not isinstance(gt, str):
                continue
            n += 1
            ok += int(pred == gt)
        out["by_frontend"][fe] = {"n": n, "ok": ok, "acc": (ok / n) if n else None}
    return out


def write_latest_index() -> Path:
    EXP.mkdir(parents=True, exist_ok=True)
    idx: Dict[str, Any] = {"root": str(ROOT), "experiments_dir": str(EXP)}

    idx["E1"] = {
        "rule_only": (_summarize_e1_rule_only(p) if (p := _latest(EXP, ["e1_rule_only*.json"])) else None),
        "rule_only_semantic_acc": _summarize_e1_semantic_acc() or None,
    }
    # LLM semantic-class accuracy lives in the E1E3 regression outputs (per-frontend).
    e1e3_paths = {fe: _latest(EXP, [f"e1e3_llm_regression_{fe}_coverage.json"]) for fe in ["triton", "tilelang", "cuda"]}
    e1e3_summary: Dict[str, Any] = {}
    for fe, p in e1e3_paths.items():
        if p is None:
            continue
        d = _read_json(p)
        ev = (((d.get("label_eval") or {}).get("feedback") or {}).get("by_frontend") or {}).get(fe) or {}
        sc = ev.get("semantic_class") if isinstance(ev, dict) else None
        e1e3_summary[fe] = {"path": str(p), "semantic_class": sc}
    idx["E1"]["llm_semantic_acc"] = e1e3_summary or None

    idx["E2"] = {
        "masked6": {
            fe: (_summarize_e2_masked6(p) if (p := _latest(EXP, [f"e2_trust_ablation_{fe}_masked6.json"])) else None)
            for fe in ["triton", "tilelang", "cuda"]
        }
    }

    idx["E1E3"] = (
        _summarize_e3(p)
        if (p := _latest(EXP, ["e3_llm_regression_coverage_oneshot_vs_feedback*.json", "llm_regression_suite*.json", "e1e3_llm_regression*.json"]))
        else None
    )
    idx["E4"] = _summarize_e4(p) if (p := _latest(EXP, ["e4_cross_frontend_triton_tilelang_intersection.json"])) else None
    idx["E5"] = {
        "E5_1": _summarize_e5_1(p) if (p := _latest(EXP, ["e5_1_external*_v2.json", "e5_1_external*.json"])) else None,
        "E5_2": _summarize_e5_2(p) if (p := _latest(EXP, ["e5_2_triton_coverage_freeze_vs_retune*_v2.json"])) else None,
    }

    out_path = EXP / "_LATEST_INDEX.json"
    _write_json(out_path, idx)
    return out_path


def main() -> None:
    moved_root = _move_root_smoke_logs()
    moved_exp = _move_experiment_jsons()
    idx_path = write_latest_index()
    print(f"Moved {moved_exp} experiment JSON(s) into buckets under {EXP}")
    print(f"Moved {moved_root} root artifact log(s) into {ART / 'remote'} / {ART / 'llm_raw'}")
    print(f"Wrote: {idx_path}")


if __name__ == "__main__":
    main()
