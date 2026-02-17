"""
Paper-ready Markdown report generator (E1–E6).

This script is intentionally lightweight:
  - It does **not** rerun expensive pipelines.
  - It reads `artifacts/experiments/paper/paper_index.json` (and referenced raw
    JSON artifacts) and emits a single Markdown report that is convenient for
    writing the paper's Experiments section (tables + pointers + appendices).

Artifacts are gitignored; this is intended for local paper iteration.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_rate(x: Any, ndigits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:.{ndigits}f}"


def _fmt_int(x: Any) -> str:
    try:
        return str(int(x))
    except Exception:
        return str(x)


def _md_h(title: str, level: int = 2) -> str:
    return f"\n{'#' * level} {title}\n"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ""
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        r = list(r) + [""] * (len(headers) - len(r))
        lines.append("| " + " | ".join(r[: len(headers)]) + " |")
    return "\n".join(lines) + "\n"


def _md_code_block(text: str, lang: str = "") -> str:
    fence = "```"
    return f"{fence}{lang}\n{text}\n{fence}\n"


def _md_details(summary: str, body_md: str) -> str:
    return f"<details>\n<summary>{summary}</summary>\n\n{body_md}\n\n</details>\n"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _path_str(p: Any) -> str:
    if isinstance(p, Path):
        return str(p)
    return str(p)


def _rel_path_str(p: Any) -> str:
    """
    Prefer repo-relative paths for portability and readability.
    """
    if isinstance(p, Path):
        try:
            return str(p.relative_to(ROOT))
        except Exception:
            return str(p)
    if isinstance(p, str):
        if p.startswith(ROOT_STR + "/"):
            return p[len(ROOT_STR) + 1 :]
    return str(p)


def _rel_obj(x: Any) -> Any:
    """
    Recursively rewrite repo-absolute paths under ROOT into repo-relative strings.
    """
    if isinstance(x, dict):
        return {k: _rel_obj(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rel_obj(v) for v in x]
    if isinstance(x, Path):
        return _rel_path_str(x)
    if isinstance(x, str):
        return _rel_path_str(x)
    return x


def _render_index_section(index: dict[str, Any]) -> str:
    figures: dict[str, str] = dict(index.get("figures") or {})
    rows: list[list[str]] = []
    for k in sorted(figures.keys()):
        rows.append([k, f"`{_rel_path_str(figures[k])}`"])
    out = _md_h("Canonical Paper JSON Index", 2)
    out += (
        "This report is generated from the *paper-ready* JSONs under "
        "`artifacts/experiments/paper/`. Use these JSONs as the single source of truth for plotting.\n\n"
    )
    out += f"- `git_head`: `{index.get('git_head')}`\n"
    out += f"- `paper_index`: `{_rel_path_str(ROOT / 'artifacts/experiments/paper/paper_index.json')}`\n\n"
    out += _md_table(["Figure", "Paper JSON"], rows)
    return out


def _render_e1_rule_only(e1_paper: dict[str, Any]) -> str:
    s = dict(e1_paper.get("summary") or {})
    out = _md_h("E1 — Rule-Only Baseline (No LLM)", 2)
    out += (
        "**Goal.** Quantify how far a *pure rule/anchor* baseline can go (no LLM), both for pipeline viability "
        "and as a lower-bound comparison for LLM-based extraction.\n\n"
    )
    out += f"- Paper JSON: `{_rel_path_str(ROOT / 'artifacts/experiments/paper/e1_rule_only.paper.json')}`\n"
    out += f"- Raw source: `{_rel_path_str(e1_paper.get('source'))}`\n\n"

    out += _md_h("E1 Summary", 3)
    out += _md_table(
        ["n", "ok", "ok_rate", "FULL", "PARTIAL", "OUT_OF_SCOPE"],
        [
            [
                _fmt_int(s.get("n")),
                _fmt_int(s.get("ok")),
                _fmt_rate(s.get("ok_rate")),
                _fmt_int((s.get("contracts") or {}).get("FULL")),
                _fmt_int((s.get("contracts") or {}).get("PARTIAL")),
                _fmt_int((s.get("contracts") or {}).get("OUT_OF_SCOPE")),
            ]
        ],
    )

    by_tier = s.get("by_tier") or {}
    if isinstance(by_tier, dict) and by_tier:
        out += _md_h("E1 Breakdown by Anchor Tier", 3)
        rows: list[list[str]] = []
        for tier in ["A_dot", "B_reduce", "C_copy"]:
            if tier in by_tier:
                rows.append([tier, _fmt_int(by_tier[tier].get("n")), _fmt_int(by_tier[tier].get("ok"))])
        out += _md_table(["tier", "n", "ok"], rows)

    failures = s.get("failures") or {}
    if isinstance(failures, dict) and failures:
        out += _md_h("E1 Failure Types", 3)
        rows = [[k, _fmt_int(v)] for k, v in sorted(failures.items(), key=lambda kv: (-int(kv[1]), kv[0]))]
        out += _md_table(["failure_type", "count"], rows)

    label_eval = e1_paper.get("label_eval") or {}
    acc_by_fe = (label_eval.get("semantic_class_acc_by_frontend") or {}) if isinstance(label_eval, dict) else {}
    if isinstance(acc_by_fe, dict) and acc_by_fe:
        out += _md_h("E1 Semantic-Class Label Accuracy (Rule-Only)", 3)
        rows = []
        for fe in ["triton", "tilelang", "cuda"]:
            if fe in acc_by_fe:
                rows.append(
                    [
                        fe,
                        _fmt_int(acc_by_fe[fe].get("n")),
                        _fmt_int(acc_by_fe[fe].get("ok")),
                        _fmt_rate(acc_by_fe[fe].get("acc")),
                    ]
                )
        out += _md_table(["frontend", "n", "ok", "acc"], rows)
        out += _md_details(
            "E1 confusion matrices (full JSON)",
            _md_code_block(json.dumps(acc_by_fe, indent=2, ensure_ascii=False), "json"),
        )

    return out


def _render_e1e3_llm_regression(e1e3_paper: dict[str, Any]) -> str:
    out = _md_h("E3 — LLM Regression + Repair (with E1 baseline context)", 2)
    out += (
        "**Goal.** Measure LLM-based semantic recovery (E3) and the benefit of the repair/feedback loop "
        "under realistic constraints (cold run, limited repair rounds). This section is typically presented "
        "together with E1 as the baseline.\n\n"
    )
    out += f"- Paper JSON: `{_rel_path_str(ROOT / 'artifacts/experiments/paper/e1e3_llm_regression.paper.json')}`\n"
    out += f"- Sources: `{json.dumps(_rel_obj(e1e3_paper.get('source')), ensure_ascii=False)}`\n\n"

    # Key deltas
    delta = (e1e3_paper.get("delta") or {}).get("by_frontend") or {}
    out += _md_h("E3 One-shot vs Feedback (OK rate)", 3)
    rows: list[list[str]] = []
    for fe in ["triton", "tilelang", "cuda"]:
        if fe not in delta:
            continue
        rows.append(
            [
                fe,
                _fmt_int(delta[fe]["one_shot"]["n"]),
                _fmt_rate(delta[fe]["one_shot"]["ok_rate"]),
                _fmt_rate(delta[fe]["feedback"]["ok_rate"]),
                _fmt_rate(delta[fe]["ok_rate_gain"]),
            ]
        )
    out += _md_table(["frontend", "n", "one_shot_ok_rate", "feedback_ok_rate", "gain"], rows)

    # LLM cost
    summary = e1e3_paper.get("summary") or {}
    for phase in ["one_shot", "feedback"]:
        if phase not in summary:
            continue
        by_fe = (summary[phase].get("by_frontend") or {}) if isinstance(summary[phase], dict) else {}
        out += _md_h(f"E3 LLM Cost ({phase})", 3)
        rows = []
        for fe in ["triton", "tilelang", "cuda"]:
            if fe not in by_fe:
                continue
            c = (by_fe[fe].get("llm_cost") or {}) if isinstance(by_fe[fe], dict) else {}
            rows.append(
                [
                    fe,
                    _fmt_int(by_fe[fe].get("n")),
                    _fmt_int(c.get("api_calls_total")),
                    _fmt_rate(c.get("api_calls_avg")),
                    _fmt_int(c.get("cache_hits_total")),
                    _fmt_int(c.get("cache_misses_total")),
                    _fmt_rate(c.get("rounds_used_avg")),
                ]
            )
        out += _md_table(
            ["frontend", "n", "api_calls_total", "api_calls_avg", "cache_hits", "cache_misses", "rounds_used_avg"],
            rows,
        )

    # Label accuracy (one_shot/feedback)
    le = e1e3_paper.get("label_eval") or {}
    for phase in ["one_shot", "feedback"]:
        by_fe = ((le.get(phase) or {}).get("by_frontend") or {}) if isinstance(le, dict) else {}
        if not by_fe:
            continue
        out += _md_h(f"E3 Semantic-Class Label Accuracy ({phase})", 3)
        rows = []
        for fe in ["triton", "tilelang", "cuda"]:
            if fe not in by_fe:
                continue
            sc = (by_fe[fe].get("semantic_class") or {}) if isinstance(by_fe[fe], dict) else {}
            rows.append([fe, _fmt_int(sc.get("n")), _fmt_int(sc.get("ok")), _fmt_rate(sc.get("acc"))])
        out += _md_table(["frontend", "n", "ok", "acc"], rows)

    # Failure list (one_shot only) from base source
    src = e1e3_paper.get("source") or {}
    base_path = Path(src.get("base") or "")
    if base_path.is_file():
        base = _load_json(base_path)
        one_shot_results = ((base.get("variants") or {}).get("one_shot") or {}).get("results") or []
        fails = [r for r in one_shot_results if not bool(r.get("ok"))]
        out += _md_h("E3 One-shot Failures (kernel-level)", 3)
        out += f"- Source: `{_rel_path_str(base_path)}`\n\n"
        out += f"- Total failures: `{len(fails)}`\n\n"
        rows = []
        for r in sorted(fails, key=lambda x: (str(x.get("frontend")), str(x.get("kernel")))):
            rows.append(
                [
                    str(r.get("frontend")),
                    str(r.get("kernel")),
                    str(r.get("category") or ""),
                    "; ".join([str(x) for x in (r.get("reasons") or [])][:3]),
                ]
            )
        out += _md_table(["frontend", "kernel", "category", "top_reasons"], rows)
        out += _md_details(
            "E3 one-shot failures (full JSON list)",
            _md_code_block(json.dumps(fails, indent=2, ensure_ascii=False), "json"),
        )

    return out


def _render_e2_trust_ablation(e2_paper: dict[str, Any]) -> str:
    out = _md_h("E2 — Trust Ablation (Mutation-Kill vs Diff-Only)", 2)
    out += (
        "**Goal.** Demonstrate why a *contract/certificate-aware* verifier is needed: "
        "dynamic diff alone catches some errors, but static obligations/certificates kill many more mutants.\n\n"
    )
    out += f"- Paper JSON: `{_rel_path_str(ROOT / 'artifacts/experiments/paper/e2_trust_ablation.paper.json')}`\n\n"

    by_fe = e2_paper.get("by_frontend") or {}
    rows = []
    for fe in ["triton", "tilelang", "cuda"]:
        if fe not in by_fe:
            continue
        agg = ((by_fe[fe].get("summary") or {}).get("aggregate") or {}) if isinstance(by_fe[fe], dict) else {}
        rows.append(
            [
                fe,
                _fmt_rate(agg.get("diff_only", {}).get("kill_rate")),
                _fmt_rate(agg.get("generic", {}).get("kill_rate")),
                _fmt_rate(agg.get("full", {}).get("kill_rate")),
            ]
        )
    out += _md_h("E2 Aggregate Kill Rate", 3)
    out += _md_table(["frontend", "diff_only", "generic", "full"], rows)

    # Add stage breakdown and per-kernel "full" kill rate for each frontend from raw sources.
    out += _md_h("E2 Per-kernel Full-mode Kill Rate (from raw JSON)", 3)
    for fe in ["triton", "tilelang", "cuda"]:
        if fe not in by_fe:
            continue
        src = Path((by_fe[fe].get("source") or "").strip())
        if not src.is_file():
            continue
        raw = _load_json(src)
        results = list(raw.get("results") or [])
        rows = []
        for r in results:
            mode = (r.get("modes") or {}).get("full") or {}
            rows.append([str(r.get("kernel")), _fmt_rate(mode.get("kill_rate")), _fmt_int(mode.get("total"))])
        rows.sort(key=lambda x: float(x[1]))
        out += f"\n**{fe}** (`{_rel_path_str(src)}`)\n\n"
        out += _md_table(["kernel", "full_kill_rate", "mutants_total"], rows)

        # Stage breakdown (full)
        stage_counts: dict[str, int] = {}
        for r in results:
            mode = (r.get("modes") or {}).get("full") or {}
            kbs = mode.get("killed_by_stage") or {}
            for k, v in kbs.items():
                stage_counts[k] = int(stage_counts.get(k, 0)) + int(v)
        if stage_counts:
            out += _md_details(
                f"E2 {fe} killed_by_stage (full) — aggregated",
                _md_code_block(json.dumps(stage_counts, indent=2, ensure_ascii=False), "json"),
            )

    return out


def _render_e4_cross_frontend(e4_paper: dict[str, Any]) -> str:
    out = _md_h("E4 — Cross-Frontend Consistency (Triton vs TileLang)", 2)
    out += (
        "**Goal.** Show that IntentIR is a *frontend-agnostic semantic carrier*: "
        "given different frontends producing the same kernel intent, the recovered IntentIR should be structurally consistent.\n\n"
    )
    out += f"- Paper JSON: `{_rel_path_str(ROOT / 'artifacts/experiments/paper/e4_cross_frontend_consistency.paper.json')}`\n"
    out += f"- Raw source: `{_rel_path_str(e4_paper.get('source'))}`\n\n"

    s = e4_paper.get("summary") or {}
    out += _md_h("E4 Summary", 3)
    out += _md_table(
        ["n", "intent_structural_ok_rate", "expanded_structural_ok_rate", "intent_strict_ok_rate", "expanded_strict_ok_rate"],
        [
            [
                _fmt_int(s.get("n")),
                _fmt_rate(s.get("intent_structural_ok_rate")),
                _fmt_rate(s.get("expanded_structural_ok_rate")),
                _fmt_rate(s.get("intent_ok_rate")),
                _fmt_rate(s.get("expanded_ok_rate")),
            ]
        ],
    )

    by_tier = s.get("by_tier") or {}
    if isinstance(by_tier, dict) and by_tier:
        out += _md_h("E4 Breakdown by Anchor Tier", 3)
        rows = []
        for tier in ["A_dot", "B_reduce", "C_copy"]:
            if tier in by_tier:
                rows.append(
                    [
                        tier,
                        _fmt_int(by_tier[tier].get("n")),
                        _fmt_int(by_tier[tier].get("ok_intent_structural")),
                        _fmt_int(by_tier[tier].get("ok_intent")),
                    ]
                )
        out += _md_table(["tier", "n", "structural_ok_intent", "strict_ok_intent"], rows)

    src = Path(str(e4_paper.get("source") or ""))
    if src.is_file():
        raw = _load_json(src)
        results = list(raw.get("results") or [])
        strict_ok = [r for r in results if bool(((r.get("ok") or {}).get("intent_strict")))]
        strict_bad = [r for r in results if not bool(((r.get("ok") or {}).get("intent_strict")))]
        out += _md_h("E4 Kernel-level Strict Mismatches (IntentIR)", 3)
        out += f"- strict_ok: `{len(strict_ok)}` / `{len(results)}`\n"
        out += f"- strict_bad: `{len(strict_bad)}` / `{len(results)}`\n\n"
        rows = []
        for r in strict_bad:
            rs = (r.get("reasons") or {}).get("intent_strict") or []
            rows.append([str(r.get("kernel")), str(r.get("anchor_tier")), str(r.get("category") or ""), "; ".join(rs[:3])])
        rows.sort(key=lambda x: (x[1], x[0]))
        out += _md_table(["kernel", "tier", "category", "top_reasons"], rows)

    return out


def _render_e5_1_external_baseline(e5_1_paper: dict[str, Any]) -> str:
    out = _md_h("E5.1 — External Baseline (AI-Benchmark 8 kernels)", 2)
    out += (
        "**Goal.** Compare our end-to-end toolchain against an external baseline suite "
        "(Triton CPU → LLVM IR → RISC-V compilation pipeline), on real hardware.\n\n"
    )
    out += f"- Paper JSON: `{_rel_path_str(ROOT / 'artifacts/experiments/paper/e5_1_external_baseline.paper.json')}`\n"
    out += f"- Raw source: `{_rel_path_str(e5_1_paper.get('source'))}`\n\n"

    s = e5_1_paper.get("summary") or {}
    out += _md_h("E5.1 Summary", 3)
    out += _md_table(
        ["n", "geom_speedup", "slower_count", "min(kernel,speedup)", "max(kernel,speedup)"],
        [
            [
                _fmt_int(s.get("n")),
                _fmt_rate(s.get("geom_speedup_ours_over_baseline")),
                _fmt_int(s.get("slower_count")),
                str(s.get("min")),
                str(s.get("max")),
            ]
        ],
    )

    per = list(e5_1_paper.get("per_kernel") or [])
    if per:
        out += _md_h("E5.1 Per-kernel Speedup (ours / baseline)", 3)
        rows = [[str(r.get("name")), _fmt_rate(r.get("speedup_ours_over_baseline"))] for r in per]
        out += _md_table(["kernel", "speedup_ours_over_baseline"], rows)

    return out


def _render_e5_2_portability_vs_perf(e5_2_paper: dict[str, Any]) -> str:
    out = _md_h("E5.2 — Portability vs Performance (Freeze vs Retune)", 2)
    out += (
        "**Goal.** Evaluate the *tuning interface*: how much performance we lose if we freeze schedules "
        "(portable defaults) vs retune per target (guided by witness + cost model).\n\n"
    )
    out += f"- Paper JSON: `{_rel_path_str(ROOT / 'artifacts/experiments/paper/e5_2_portability_vs_perf.paper.json')}`\n"
    out += f"- Raw source: `{_rel_path_str(e5_2_paper.get('source'))}`\n\n"

    bench = e5_2_paper.get("bench") or {}
    out += _md_h("E5.2 Bench Configuration", 3)
    out += _md_code_block(json.dumps(bench, indent=2, ensure_ascii=False), "json")

    s = e5_2_paper.get("summary") or {}
    out += _md_h("E5.2 Summary", 3)
    out += _md_table(
        ["n", "geom_speedup_unpaired", "geom_speedup_paired", "regressions_unpaired_n", "regressions_paired_n"],
        [
            [
                _fmt_int(s.get("n")),
                _fmt_rate(s.get("geom_speedup_unpaired")),
                _fmt_rate(s.get("geom_speedup_paired")),
                _fmt_int(s.get("regressions_unpaired_n")),
                _fmt_int(s.get("regressions_paired_n")),
            ]
        ],
    )

    per = list(e5_2_paper.get("per_kernel") or [])
    if per:
        out += _md_h("E5.2 Per-kernel Speedup (retune / freeze)", 3)
        rows = [
            [
                str(r.get("kernel")),
                str(r.get("anchor_tier")),
                _fmt_rate(r.get("speedup_unpaired")),
                _fmt_rate(r.get("speedup_paired")),
            ]
            for r in per
        ]
        out += _md_table(["kernel", "tier", "speedup_unpaired", "speedup_paired"], rows)

    return out


def _render_e6_2_ir_usability(e6_paper: dict[str, Any]) -> str:
    out = _md_h("E6.2 — IR Usability & Contract Calibration (IntentIR vs Linalg)", 2)
    out += (
        "**Goal.** Fairly compare IntentIR against a Linalg baseline *as LLM output carriers*: "
        "both must output **IR + contract** (FULL/PARTIAL/OOS + assumptions). We test whether the "
        "representation makes it easier to stay honest and consistent under missing evidence.\n\n"
    )
    out += f"- Paper JSON: `{_rel_path_str(ROOT / 'artifacts/experiments/paper/e6_ir_usability.paper.json')}`\n"
    out += f"- Raw source: `{_rel_path_str(e6_paper.get('source'))}`\n\n"

    out += _md_h("E6.2 Setup", 3)
    out += _md_table(
        ["suite", "frontends", "reps", "ablations", "model", "cache", "repair_rounds"],
        [
            [
                str(e6_paper.get("suite")),
                ",".join(e6_paper.get("frontends") or []),
                ",".join(e6_paper.get("reps") or []),
                ",".join(e6_paper.get("ablations") or []),
                str(e6_paper.get("model")),
                str(e6_paper.get("cache")),
                _fmt_int(e6_paper.get("repair_rounds")),
            ]
        ],
    )

    summ = e6_paper.get("summary") or {}
    src_sum = (summ.get("source_summary") or {}).get("by_rep") or {}
    out += _md_h("E6.2 High-level Metrics (by rep)", 3)
    rows: list[list[str]] = []
    for rep in ["intentir", "linalg"]:
        if rep not in src_sum:
            continue
        m = src_sum[rep]
        rows.append(
            [
                rep,
                _fmt_int(m.get("n")),
                _fmt_rate(m.get("ok_rate")),
                _fmt_rate(m.get("binding_ok_rate")),
                _fmt_rate(m.get("overclaim_rate")),
                _fmt_rate(m.get("underclaim_rate")),
                _fmt_rate(m.get("full_false_accept_rate")),
            ]
        )
    out += _md_table(
        ["rep", "n", "ok_rate", "binding_ok_rate", "overclaim_rate", "underclaim_rate", "full_false_accept_rate"],
        rows,
    )

    paper_sum = summ.get("paper_summary") or {}
    by_rep_ablation = paper_sum.get("by_rep_ablation") or {}
    if by_rep_ablation:
        out += _md_h("E6.2 Breakdown: by rep × evidence ablation", 3)
        rows = []
        for rep in ["intentir", "linalg"]:
            if rep not in by_rep_ablation:
                continue
            for abl in ["full", "no_mask", "no_anchors"]:
                if abl not in by_rep_ablation[rep]:
                    continue
                m = by_rep_ablation[rep][abl]
                rows.append(
                    [
                        rep,
                        abl,
                        _fmt_int(m.get("n")),
                        _fmt_rate(m.get("ok_rate")),
                        _fmt_rate(m.get("binding_ok_rate")),
                        _fmt_rate(m.get("full_false_accept_rate")),
                    ]
                )
        out += _md_table(["rep", "ablation", "n", "ok_rate", "binding_ok_rate", "full_false_accept_rate"], rows)

    # List linalg failures (kernel+ablation) from raw run, to make plotting/debugging easier.
    src = Path(str(e6_paper.get("source") or ""))
    if src.is_file():
        raw = _load_json(src)
        rows = []
        for r in raw.get("results") or []:
            if str(r.get("rep")) != "linalg":
                continue
            if bool(r.get("ok")):
                continue
            rows.append(
                [
                    str(r.get("kernel")),
                    str(r.get("ablation")),
                    str(r.get("oracle_level")),
                    str(r.get("contract_level")),
                    str(r.get("failure_type") or ""),
                    "; ".join([str(x) for x in (r.get("reasons") or [])][:3]),
                ]
            )
        if rows:
            out += _md_h("E6.2 Linalg failures (kernel-level, for debugging/figures)", 3)
            out += f"- Total failures: `{len(rows)}`\n\n"
            rows.sort(key=lambda x: (x[0], x[1], x[2]))
            out += _md_table(
                ["kernel", "ablation", "oracle", "contract", "failure_type", "top_reasons"],
                rows,
            )

    out += _md_details("E6.2 paper_summary (full JSON)", _md_code_block(json.dumps(paper_sum, indent=2, ensure_ascii=False), "json"))
    return out


def _render_appendix_json_dumps(paper_paths: dict[str, Path]) -> str:
    out = _md_h("Appendix: Full Paper JSON Dumps (verbatim)", 2)
    out += (
        "These are the verbatim `*.paper.json` files used for plotting. "
        "They are embedded here to avoid any accidental omission while writing the paper.\n\n"
    )
    for key in sorted(paper_paths.keys()):
        p = paper_paths[key]
        if not p.is_file():
            continue
        out += _md_details(f"{key}: `{_rel_path_str(p)}`", _md_code_block(_read_text(p), "json"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paper-index",
        type=str,
        default=str(ROOT / "artifacts/experiments/paper/paper_index.json"),
        help="Path to artifacts/experiments/paper/paper_index.json",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "doc/paper/EXPERIMENTS_RESULTS_DATA.md"),
        help="Output Markdown path",
    )
    ap.add_argument("--no-embed-json", action="store_true", help="Do not embed full paper JSON dumps")
    args = ap.parse_args()

    idx_path = Path(args.paper_index)
    if not idx_path.is_file():
        raise SystemExit(f"paper_index not found: {idx_path}")

    index = _load_json(idx_path)
    figures = dict(index.get("figures") or {})
    paper_paths = {k: Path(v) for k, v in figures.items()}

    # Load per-figure paper jsons.
    e1_p = _load_json(paper_paths["E1"])
    e1e3_p = _load_json(paper_paths["E1E3"])
    e2_p = _load_json(paper_paths["E2"])
    e4_p = _load_json(paper_paths["E4"])
    e5_1_p = _load_json(paper_paths["E5_1"])
    e5_2_p = _load_json(paper_paths["E5_2"])
    e6_p = _load_json(paper_paths["E6"])

    out_md = ""
    out_md += "# IntentIR Experiments (E1–E6): Data + Analysis\n"
    out_md += "\n"
    out_md += f"Generated: `{datetime.now().isoformat(timespec='seconds')}`\n\n"
    out_md += f"Paper index: `{_rel_path_str(idx_path)}`\n\n"

    out_md += _render_index_section(index)
    out_md += _render_e1_rule_only(e1_p)
    out_md += _render_e1e3_llm_regression(e1e3_p)
    out_md += _render_e2_trust_ablation(e2_p)
    out_md += _render_e4_cross_frontend(e4_p)
    out_md += _render_e5_1_external_baseline(e5_1_p)
    out_md += _render_e5_2_portability_vs_perf(e5_2_p)
    out_md += _render_e6_2_ir_usability(e6_p)
    if not args.no_embed_json:
        out_md += _md_details(
            f"paper_index.json (verbatim): `{_rel_path_str(idx_path)}`",
            _md_code_block(_read_text(idx_path), "json"),
        )
        out_md += _render_appendix_json_dumps({k: v for k, v in paper_paths.items() if k != "E2"})
        # E2 paper JSON is small; include too.
        out_md += _md_details("E2: `e2_trust_ablation.paper.json` (verbatim)", _md_code_block(_read_text(paper_paths["E2"]), "json"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_md, encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
