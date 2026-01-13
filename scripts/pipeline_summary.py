"""
Unified pipeline summary (P3): coverage curve + failure category distribution.

This script consumes the JSON reports emitted by:
- `scripts/full_pipeline_verify.py --frontend triton`
- `scripts/full_pipeline_verify.py --frontend tilelang`
- optional: `scripts/rvv_remote_suite.py --out <json>`
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRITON_DIR = ROOT / "artifacts" / "full_pipeline_verify"
DEFAULT_TILELANG_DIR = ROOT / "artifacts" / "tilelang_full_pipeline"
DEFAULT_CUDA_DIR = ROOT / "artifacts" / "cuda_full_pipeline"
DEFAULT_REMOTE_JSON = ROOT / "artifacts" / "rvv_remote_suite_latest.json"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_reports(dir_path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if not dir_path.exists():
        return []
    for p in sorted(dir_path.glob("*.json")):
        # Skip auxiliary JSONs (pipeline emits per-kernel descriptor/certificate/contract sidecars).
        if (
            p.name.endswith(".certificate_v2.json")
            or p.name.endswith(".contract.json")
            or p.name.endswith(".descriptor.json")
            or p.name.endswith(".certificate.json")
        ):
            continue
        try:
            d = _read_json(p)
        except Exception:
            continue
        # Heuristic: accept only full pipeline reports (have contract+diff) or explicit error reports.
        is_error = isinstance(d.get("error"), dict) or d.get("ok") is False
        is_report = isinstance(d.get("contract"), dict) and isinstance(d.get("diff"), dict)
        if not (is_error or is_report):
            continue
        kernel = str(d.get("kernel") or p.stem)
        yield kernel, d


def _get(d: Dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _anchors_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    cert = report.get("certificate_v2")
    if isinstance(cert, dict):
        anchors = _get(cert, "semantic_facts", "anchors", default={})
        if isinstance(anchors, dict):
            return dict(anchors)
    return {}


def _anchor_tier(anchors: Dict[str, Any]) -> str:
    if bool(anchors.get("has_dot")):
        return "A_dot"
    if bool(anchors.get("has_reduce")):
        return "B_reduce"
    if bool(anchors.get("has_copy")):
        return "C_copy"
    return "D_none"


def _anchor_score(anchors: Dict[str, Any]) -> int:
    # Deterministic coarse score for sorting / curves.
    score = 0
    if bool(anchors.get("has_dot")):
        score += 3
    if bool(anchors.get("has_reduce")):
        score += 2
    if bool(anchors.get("has_copy")):
        score += 1
    return int(score)


def _failure_category(report: Dict[str, Any], *, remote_ok: Optional[bool]) -> str:
    if isinstance(report.get("error"), dict) or report.get("ok") is False:
        et = _get(report, "error", "type", default="Exception")
        return f"pipeline_error:{et}"
    contract_level = str(_get(report, "contract", "level", default="")).upper()
    if contract_level == "OUT_OF_SCOPE":
        reasons = _get(report, "contract", "reasons", default=[])
        if isinstance(reasons, list) and reasons:
            return f"out_of_scope:{str(reasons[0])}"
        return "out_of_scope"
    sv_ok = _get(report, "static_validation", "ok", default=None)
    if sv_ok is False:
        reasons = _get(report, "static_validation", "reasons", default=[])
        if isinstance(reasons, list) and reasons:
            return f"static_validate_fail:{str(reasons[0])}"
        return "static_validate_fail"
    diff_ok = bool(_get(report, "diff", "ok", default=False))
    if not diff_ok:
        worst = _get(report, "diff", "worst", "summary", default=None)
        return f"diff_fail:{worst}" if isinstance(worst, str) and worst else "diff_fail"
    if remote_ok is False:
        return "remote_fail"
    return "ok"


@dataclass(frozen=True)
class Entry:
    frontend: str
    kernel: str
    contract: str
    diff_ok: bool
    remote_ok: Optional[bool]
    anchors: Dict[str, Any]
    anchor_tier: str
    anchor_score: int
    failure: str
    contract_reasons: List[str]


def _load_remote_map(path: Path) -> Dict[Tuple[str, str], bool]:
    if not path.exists():
        return {}
    try:
        d = _read_json(path)
    except Exception:
        return {}
    out: Dict[Tuple[str, str], bool] = {}
    for r in d.get("results") or []:
        if not isinstance(r, dict):
            continue
        fe = r.get("frontend")
        k = r.get("kernel")
        if not isinstance(fe, str) or not isinstance(k, str):
            continue
        ok = r.get("ok")
        if isinstance(ok, bool):
            out[(fe, k)] = bool(ok)
    return out


def _print_table(title: str, rows: List[List[str]]) -> None:
    if not rows:
        print(f"{title}: (no data)")
        return
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    print(title)
    for r in rows:
        print("  " + "  ".join(r[i].ljust(widths[i]) for i in range(len(r))))


def _summarize(entries: List[Entry]) -> Dict[str, Any]:
    tiers = ["A_dot", "B_reduce", "C_copy", "D_none"]
    tier_counts: Dict[str, Dict[str, int]] = {t: {"n": 0, "ok": 0, "remote_ok": 0, "remote_total": 0} for t in tiers}
    contract_counts: Dict[str, int] = {}
    failures: Dict[str, int] = {}
    failures_by_tier: Dict[str, Dict[str, int]] = {t: {} for t in tiers}
    contracts_by_tier: Dict[str, Dict[str, int]] = {t: {} for t in tiers}
    reason_counts: Dict[str, int] = {}

    for e in entries:
        tier_counts.setdefault(e.anchor_tier, {"n": 0, "ok": 0, "remote_ok": 0, "remote_total": 0})
        tier_counts[e.anchor_tier]["n"] += 1
        if e.failure == "ok":
            tier_counts[e.anchor_tier]["ok"] += 1
        if e.remote_ok is not None:
            tier_counts[e.anchor_tier]["remote_total"] += 1
            if e.remote_ok:
                tier_counts[e.anchor_tier]["remote_ok"] += 1

        contract_counts[e.contract] = contract_counts.get(e.contract, 0) + 1
        failures[e.failure] = failures.get(e.failure, 0) + 1
        failures_by_tier.setdefault(e.anchor_tier, {})
        failures_by_tier[e.anchor_tier][e.failure] = failures_by_tier[e.anchor_tier].get(e.failure, 0) + 1
        contracts_by_tier.setdefault(e.anchor_tier, {})
        contracts_by_tier[e.anchor_tier][e.contract] = contracts_by_tier[e.anchor_tier].get(e.contract, 0) + 1
        for r in e.contract_reasons:
            reason_counts[str(r)] = reason_counts.get(str(r), 0) + 1

    # Coverage curve: sort by anchor score (strong→weak), then compute cumulative ok rate.
    curve: List[Dict[str, Any]] = []
    sorted_entries = sorted(entries, key=lambda e: (-int(e.anchor_score), str(e.kernel)))
    ok_prefix = 0
    for i, e in enumerate(sorted_entries, start=1):
        ok_prefix += 1 if e.failure == "ok" else 0
        curve.append({"k": int(i), "ok": int(ok_prefix), "ok_rate": float(ok_prefix / i)})

    return {
        "total": len(entries),
        "tiers": tier_counts,
        "contracts": contract_counts,
        "failures": failures,
        "failures_by_tier": failures_by_tier,
        "contracts_by_tier": contracts_by_tier,
        "contract_reasons": reason_counts,
        "coverage_curve": curve,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang", "cuda", "both", "all"], default="both")
    ap.add_argument("--triton-dir", default=str(DEFAULT_TRITON_DIR))
    ap.add_argument("--tilelang-dir", default=str(DEFAULT_TILELANG_DIR))
    ap.add_argument("--cuda-dir", default=str(DEFAULT_CUDA_DIR))
    ap.add_argument("--remote", default=str(DEFAULT_REMOTE_JSON))
    ap.add_argument("--no-remote", action="store_true")
    ap.add_argument("--json", default=None, help="write summary JSON to this path (otherwise print text)")
    args = ap.parse_args()

    remote_map = {} if args.no_remote else _load_remote_map(Path(args.remote))
    if args.frontend == "both":
        frontends = ["triton", "tilelang"]
    elif args.frontend == "all":
        frontends = ["triton", "tilelang", "cuda"]
    else:
        frontends = [str(args.frontend)]
    entries_by_fe: Dict[str, List[Entry]] = {fe: [] for fe in frontends}

    def ingest(frontend: str, reports_dir: Path) -> None:
        for kernel, report in _iter_reports(reports_dir):
            contract_level = str(_get(report, "contract", "level", default="N/A"))
            diff_ok = bool(_get(report, "diff", "ok", default=False))
            anchors = _anchors_from_report(report)
            remote_ok = remote_map.get((frontend, kernel))
            failure = _failure_category(report, remote_ok=remote_ok)
            reasons = _get(report, "contract", "reasons", default=[])
            contract_reasons = [str(x) for x in reasons] if isinstance(reasons, list) else []
            entries_by_fe[frontend].append(
                Entry(
                    frontend=frontend,
                    kernel=kernel,
                    contract=contract_level,
                    diff_ok=diff_ok,
                    remote_ok=remote_ok,
                    anchors=anchors,
                    anchor_tier=_anchor_tier(anchors),
                    anchor_score=_anchor_score(anchors),
                    failure=failure,
                    contract_reasons=contract_reasons,
                )
            )

    for fe in frontends:
        if fe == "triton":
            ingest("triton", Path(args.triton_dir))
        elif fe == "tilelang":
            ingest("tilelang", Path(args.tilelang_dir))
        elif fe == "cuda":
            ingest("cuda", Path(args.cuda_dir))
        else:
            raise ValueError(f"unknown frontend: {fe}")

    summary: Dict[str, Any] = {"frontends": {}}
    for fe, xs in entries_by_fe.items():
        summary["frontends"][fe] = _summarize(xs)

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(args.json)
        return

    # Human-readable output
    for fe, fe_sum in summary["frontends"].items():
        print(f"\n=== {fe} ===")
        rows = [["tier", "n", "ok(diff+sv)", "remote_ok/total"]]
        for tier in ["A_dot", "B_reduce", "C_copy", "D_none"]:
            tc = (fe_sum.get("tiers") or {}).get(tier) or {}
            n = int(tc.get("n", 0))
            ok = int(tc.get("ok", 0))
            rok = int(tc.get("remote_ok", 0))
            rtot = int(tc.get("remote_total", 0))
            rows.append([tier, str(n), str(ok), (f"{rok}/{rtot}" if rtot else "-")])
        _print_table("Anchor-tier coverage", rows)

        crows = [["contract", "count"]]
        for k, v in sorted((fe_sum.get("contracts") or {}).items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            crows.append([str(k), str(int(v))])
        _print_table("Contract levels", crows)

        frows = [["failure", "count"]]
        for k, v in sorted((fe_sum.get("failures") or {}).items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            frows.append([str(k), str(int(v))])
        _print_table("Failure categories", frows)

        # Per-tier failure distribution (compact): useful for coverage/anchor plots.
        fb = fe_sum.get("failures_by_tier") or {}
        fbt_rows = [["tier", "top_failure", "count"]]
        for tier in ["A_dot", "B_reduce", "C_copy", "D_none"]:
            d = fb.get(tier) if isinstance(fb, dict) else None
            if not isinstance(d, dict) or not d:
                fbt_rows.append([tier, "-", "0"])
                continue
            top_k, top_v = sorted(d.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0]
            fbt_rows.append([tier, str(top_k), str(int(top_v))])
        _print_table("Top failure by tier", fbt_rows)

        rrows = [["contract_reason(top)", "count"]]
        reasons = list((fe_sum.get("contract_reasons") or {}).items())
        reasons.sort(key=lambda kv: (-int(kv[1]), str(kv[0])))
        for k, v in reasons[:10]:
            rrows.append([str(k), str(int(v))])
        _print_table("Top contract reasons", rrows)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
