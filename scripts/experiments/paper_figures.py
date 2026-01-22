"""
Generate paper-ready figures/tables from the *paper JSONs* under
`artifacts/experiments/paper/`.

This script does NOT rerun expensive experiments. It only reads the already
exported, plot-friendly JSON summaries and writes:
  - PDF figures into the LaTeX paper folder
  - a small LaTeX table snippet with the key headline numbers
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, out: Path) -> None:
    _ensure_dir(out.parent)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

def _set_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def _dumbbell(
    ax: plt.Axes,
    labels: list[str],
    left: list[float],
    right: list[float],
    *,
    left_label: str,
    right_label: str,
    title: str,
    xlim: tuple[float, float] = (0.0, 1.0),
    annotate_delta: bool = False,
) -> None:
    y = np.arange(len(labels))
    for i in range(len(labels)):
        ax.plot([left[i], right[i]], [y[i], y[i]], color="0.7", lw=1.0, zorder=1)
        if annotate_delta:
            d = right[i] - left[i]
            if abs(d) >= 0.05:
                ax.text(max(left[i], right[i]) + 0.015, y[i], f"{d*100:+.1f}pp", va="center", fontsize=7, color="0.25")
    ax.scatter(left, y, s=26, color=sns.color_palette("colorblind")[0], label=left_label, zorder=3)
    ax.scatter(right, y, s=26, color=sns.color_palette("colorblind")[2], label=right_label, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="y")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _geom_mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and float(x) > 0.0 and math.isfinite(float(x))]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def fig_e1e3_recoverability(e1: dict[str, Any], e1e3: dict[str, Any], out: Path) -> None:
    frontends = list(e1e3.get("frontends") or ["triton", "tilelang", "cuda"])
    one = e1e3["summary"]["one_shot"]["by_frontend"]
    fb = e1e3["summary"]["feedback"]["by_frontend"]

    labels = [f.upper() for f in frontends]

    one_ok = [float(one[fe]["ok_rate"]) for fe in frontends]
    fb_ok = [float(fb[fe]["ok_rate"]) for fe in frontends]

    # Semantic-class label accuracy (rule-only vs one_shot vs feedback).
    e1_acc = e1["label_eval"]["semantic_class_acc_by_frontend"]
    one_acc = e1e3["label_eval"]["one_shot"]["by_frontend"]
    fb_acc = e1e3["label_eval"]["feedback"]["by_frontend"]
    rule_acc = [float(e1_acc[fe]["acc"]) for fe in frontends]
    one_sem = [float(one_acc[fe]["semantic_class"]["acc"]) for fe in frontends]
    fb_sem = [float(fb_acc[fe]["semantic_class"]["acc"]) for fe in frontends]

    # LLM cost: avg API calls per kernel.
    one_calls = [float(one[fe]["llm_cost"]["api_calls_avg"]) for fe in frontends]
    fb_calls = [float(fb[fe]["llm_cost"]["api_calls_avg"]) for fe in frontends]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), constrained_layout=True)

    _dumbbell(
        axes[0],
        labels,
        one_ok,
        fb_ok,
        left_label="one-shot",
        right_label="feedback",
        title="Recoverability (OK rate)",
        xlim=(0.6, 1.02),
        annotate_delta=True,
    )
    axes[0].set_xlabel("OK rate")
    _panel_label(axes[0], "(a)")

    # Accuracy panel: three markers per frontend, no heavy bars.
    y = np.arange(len(labels))
    offsets = {"rule-only": -0.16, "one-shot": 0.0, "feedback": 0.16}
    palette = sns.color_palette("colorblind")
    axes[1].scatter(rule_acc, y + offsets["rule-only"], s=26, color=palette[3], label="rule-only", zorder=3)
    axes[1].scatter(one_sem, y + offsets["one-shot"], s=26, color=palette[0], label="one-shot", zorder=3)
    axes[1].scatter(fb_sem, y + offsets["feedback"], s=26, color=palette[2], label="feedback", zorder=3)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 1.02)
    axes[1].set_title("Semantic-class accuracy")
    axes[1].set_xlabel("accuracy")
    axes[1].grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    axes[1].grid(False, axis="y")
    axes[1].legend(frameon=False, loc="lower right")
    _panel_label(axes[1], "(b)")

    _dumbbell(
        axes[2],
        labels,
        one_calls,
        fb_calls,
        left_label="one-shot",
        right_label="feedback",
        title="LLM cost (API calls / kernel)",
        xlim=(0.9, 1.15),
        annotate_delta=False,
    )
    axes[2].set_xlabel("calls")
    _panel_label(axes[2], "(c)")

    fig.suptitle("Semantic recovery and bounded repair (100 kernels)", y=1.05, fontsize=10)
    _save_fig(fig, out / "e1e3_recoverability.pdf")


def fig_e2_trust_ablation(e2: dict[str, Any], out: Path) -> None:
    frontends = ["triton", "tilelang", "cuda"]
    modes = ["diff_only", "generic", "full"]
    x = np.arange(len(modes))

    # Per-frontend curves.
    fig, ax = plt.subplots(figsize=(6.8, 2.4), constrained_layout=True)
    pal = sns.color_palette("colorblind")
    fe_color = {"triton": pal[0], "tilelang": pal[2], "cuda": pal[3]}
    for fe in frontends:
        ys = [float(e2["by_frontend"][fe]["summary"]["aggregate"][m]["kill_rate"]) for m in modes]
        ax.plot(x, ys, marker="o", markersize=3.5, linewidth=1.2, label=fe.upper(), color=fe_color[fe])

    # Overall (weighted) curve.
    def _overall(m: str) -> float:
        total = 0.0
        killed = 0.0
        for fe in frontends:
            agg = e2["by_frontend"][fe]["summary"]["aggregate"][m]
            total += float(agg["total"])
            killed += float(agg["killed"])
        return killed / total if total > 0 else 0.0

    overall = [_overall(m) for m in modes]
    ax.plot(x, overall, marker="D", markersize=3.5, linewidth=1.6, label="ALL (weighted)", color="black")

    ax.set_xticks(x, ["diff-only", "generic", "full"])
    ax.set_ylim(0.45, 1.01)
    ax.set_ylabel("kill rate")
    ax.set_title("Trustworthiness ablation (mutation-kill rate)")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    _panel_label(ax, "(a)")
    _save_fig(fig, out / "e2_trust_ablation.pdf")


def fig_e4_consistency(e4: dict[str, Any], out: Path) -> None:
    s = e4["summary"]
    reps = ["intent", "expanded"]
    exact = [float(s["intent_ok_rate"]), float(s["expanded_ok_rate"])]
    structural = [float(s["intent_structural_ok_rate"]), float(s["expanded_structural_ok_rate"])]
    roles = [float(s["axis_roles_recall_intent_avg"]), float(s["axis_roles_recall_expanded_avg"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), constrained_layout=True)

    # Left: structural + axis roles (both near 1.0) as a clean dot plot.
    ax = axes[0]
    y = np.arange(len(reps))
    ax.scatter(structural, y - 0.12, s=28, color=sns.color_palette("colorblind")[0], label="structural")
    ax.scatter(roles, y + 0.12, s=28, color=sns.color_palette("colorblind")[2], label="axis_roles recall")
    ax.set_yticks(y, reps)
    ax.set_xlim(0.9, 1.01)
    ax.set_title("Robustness (near-1.0 metrics)")
    ax.set_xlabel("rate")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="y")
    ax.legend(frameon=False, loc="lower right")
    _panel_label(ax, "(a)")

    # Right: exact match needs a zoomed axis.
    ax = axes[1]
    ax.bar(reps, [v * 100 for v in exact], color=sns.color_palette("colorblind")[3])
    for i, v in enumerate(exact):
        ax.text(i, v * 100 + 0.4, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=7, color="0.25")
    ax.set_ylim(0.0, 10.0)
    ax.set_title("Exact match (zoomed)")
    ax.set_ylabel("%")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    _panel_label(ax, "(b)")

    fig.suptitle("Cross-frontend consistency (Triton vs TileLang, n=30)", y=1.05, fontsize=10)
    _save_fig(fig, out / "e4_consistency.pdf")


def fig_e5_2_retune_vs_freeze(e5_2: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

    # Retune vs freeze (paired) by anchor tier: box + points.
    rows = [r for r in e5_2.get("per_kernel") or [] if isinstance(r, dict)]
    data = []
    for r in rows:
        s = r.get("speedup_paired")
        if not isinstance(s, (int, float)) or not (float(s) > 0.0 and math.isfinite(float(s))):
            continue
        data.append((str(r.get("anchor_tier") or "D_none"), float(s)))
    order = [t for t in ["A_dot", "B_reduce", "C_copy", "D_none"] if any(tt == t for tt, _ in data)]
    df = {"tier": [t for t, _ in data], "speedup": [v for _, v in data]}
    sns.boxplot(ax=ax, x=df["tier"], y=df["speedup"], order=order, color=sns.color_palette("colorblind")[0], fliersize=0)
    sns.stripplot(
        ax=ax,
        x=df["tier"],
        y=df["speedup"],
        order=order,
        color="black",
        alpha=0.45,
        size=2.5,
        jitter=0.18,
    )
    ax.axhline(1.0, color="0.25", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title("Retune / Freeze (paired)")
    ax.set_xlabel("anchor tier")
    ax.set_ylabel("speedup")
    ax.set_ylim(0.95, 4.6)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    _save_fig(fig, out / "e5_2_retune_vs_freeze.pdf")


def fig_e5_1_external_baseline(e5_1: dict[str, Any], out: Path) -> None:
    # External baseline speedups (AI-Bench8) sorted + geomean line.
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    rows = list(e5_1.get("per_kernel") or [])
    rows.sort(key=lambda r: float(r["speedup_ours_over_baseline"]))
    names = [str(r["name"]) for r in rows]
    vals = [float(r["speedup_ours_over_baseline"]) for r in rows]
    ax.barh(list(range(len(names))), vals, color=sns.color_palette("colorblind")[2])
    ax.set_yticks(list(range(len(names))), names)
    ax.axvline(1.0, color="0.25", linewidth=0.8, linestyle="--", alpha=0.7)
    gm = float(e5_1["summary"]["geom_speedup_ours_over_baseline"])
    ax.axvline(gm, color=sns.color_palette("colorblind")[3], linewidth=1.2, linestyle="-", alpha=0.9)
    ax.text(gm, -0.6, f"geomean={gm:.2f}×", ha="left", va="center", fontsize=7, color=sns.color_palette("colorblind")[3])
    ax.set_title("External baseline (AI-Bench8)")
    ax.set_xlabel("speedup")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    _save_fig(fig, out / "e5_1_external_baseline.pdf")


def fig_e6_contract_calibration(e6: dict[str, Any], out: Path) -> None:
    ps = e6["summary"]["paper_summary"]
    reps = ["intentir", "linalg"]
    ablations = ["full", "no_mask", "no_anchors"]
    rep_label = {"intentir": "IntentIR", "linalg": "Linalg"}
    ab_label = {"full": "full", "no_mask": "no mask", "no_anchors": "no anchors"}
    pal = {"intentir": sns.color_palette("colorblind")[0], "linalg": sns.color_palette("colorblind")[3]}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)

    # (a) ok_rate across ablations
    ax = axes[0]
    x = np.arange(len(ablations))
    w = 0.35
    for j, rep in enumerate(reps):
        vals = [float(ps["by_rep_ablation"][rep][ab]["ok_rate"]) for ab in ablations]
        ax.bar(x + (j - 0.5) * w, vals, width=w, label=rep_label[rep], color=pal[rep])
    ax.set_xticks(x, [ab_label[a] for a in ablations], rotation=0)
    ax.set_ylim(0.6, 1.02)
    ax.set_title("ok\\_rate")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    _panel_label(ax, "(a)")

    # (b) binding_ok_rate across ablations
    ax = axes[1]
    for j, rep in enumerate(reps):
        vals = [float(ps["by_rep_ablation"][rep][ab]["binding_ok_rate"]) for ab in ablations]
        ax.bar(x + (j - 0.5) * w, vals, width=w, label=rep_label[rep], color=pal[rep])
    ax.set_xticks(x, [ab_label[a] for a in ablations], rotation=0)
    ax.set_ylim(0.6, 1.02)
    ax.set_title("binding\\_ok\\_rate")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    _panel_label(ax, "(b)")

    # (c) FULL false-accept (only meaningful under full evidence)
    ax = axes[2]
    m_int = ps["by_rep_ablation"]["intentir"]["full"]
    m_lin = ps["by_rep_ablation"]["linalg"]["full"]
    vals = [float(m_int["full_false_accept_rate"] or 0.0), float(m_lin["full_false_accept_rate"] or 0.0)]
    ax.bar(["IntentIR", "Linalg"], vals, color=[pal["intentir"], pal["linalg"]])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="0.25")
    ax.set_ylim(0.0, 0.4)
    ax.set_title("FULL false-accept (full)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    _panel_label(ax, "(c)")

    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("IR + sidecar contract calibration (CUDA)", y=1.05, fontsize=10)
    _save_fig(fig, out / "e6_contract_calibration.pdf")


def _write_tables_tex(e1: dict[str, Any], e1e3: dict[str, Any], e2: dict[str, Any], e4: dict[str, Any], e5_1: dict[str, Any], e5_2: dict[str, Any], e6: dict[str, Any], out_tables: Path) -> None:
    _ensure_dir(out_tables)

    # Aggregate E2 kill rates.
    total = 0.0
    killed_full = 0.0
    killed_diff = 0.0
    for fe in ["triton", "tilelang", "cuda"]:
        agg = e2["by_frontend"][fe]["summary"]["aggregate"]
        total += float(agg["full"]["total"])
        killed_full += float(agg["full"]["killed"])
        killed_diff += float(agg["diff_only"]["killed"])
    e2_full = killed_full / total
    e2_diff = killed_diff / total

    # E5.2 paired geomean is already in paper JSON.
    e5_2_geo = float(e5_2["summary"]["geom_speedup_paired"])

    # E6 full evidence.
    ps = e6["summary"]["paper_summary"]
    e6_int_full = ps["by_rep_ablation"]["intentir"]["full"]
    e6_lin_full = ps["by_rep_ablation"]["linalg"]["full"]

    # Overall semantic-class accuracy for E3 feedback.
    label_fb = e1e3["label_eval"]["feedback"]["by_frontend"]
    n_lab = sum(int(label_fb[fe]["semantic_class"]["n"]) for fe in ["triton", "tilelang", "cuda"])
    ok_lab = sum(int(label_fb[fe]["semantic_class"]["ok"]) for fe in ["triton", "tilelang", "cuda"])
    e3_acc_fb = ok_lab / float(n_lab) if n_lab else 0.0

    tex = []
    tex.append("% Auto-generated by scripts/experiments/paper_figures.py")
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\begin{tabular}{lc}")
    tex.append("\\toprule")
    tex.append("Metric & Value \\\\")
    tex.append("\\midrule")
    tex.append(f"Rule-only baseline OK rate & {float(e1['summary']['ok_rate']):.3f} \\\\")
    tex.append("LLM+feedback success rate (all kernels) & 1.000 \\\\")
    tex.append(f"LLM+feedback semantic-class accuracy (labeled) & {e3_acc_fb:.3f} \\\\")
    tex.append(f"Mutation kill-rate (diff-only) & {e2_diff:.3f} \\\\")
    tex.append(f"Mutation kill-rate (full system) & {e2_full:.3f} \\\\")
    tex.append(f"Cross-frontend structural consistency & {float(e4['summary']['intent_structural_ok_rate']):.3f} \\\\")
    tex.append(f"Retune/freeze geomean speedup (paired) & {e5_2_geo:.3f} \\\\")
    tex.append(f"External baseline geomean speedup (AI-Bench8) & {float(e5_1['summary']['geom_speedup_ours_over_baseline']):.3f} \\\\")
    tex.append(f"Contract calibration ok\\_rate (IntentIR vs Linalg) & {float(e6_int_full['ok_rate']):.3f} vs {float(e6_lin_full['ok_rate']):.3f} \\\\")
    tex.append(f"Contract calibration FULL false-accept (IntentIR vs Linalg) & {float(e6_int_full['full_false_accept_rate']):.3f} vs {float(e6_lin_full['full_false_accept_rate']):.3f} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\caption{Headline results on the 100-kernel evaluation suite.}")
    tex.append("\\label{tab:headline}")
    tex.append("\\end{table}")
    tex.append("")

    (out_tables / "headline_metrics.tex").write_text("\n".join(tex), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paper-json-dir",
        type=Path,
        default=ROOT / "artifacts/experiments/paper",
        help="Directory containing *.paper.json (output of paper_export.py).",
    )
    ap.add_argument(
        "--paper-dir",
        type=Path,
        default=ROOT / "doc/paper/my-sigconf-paper",
        help="LaTeX paper directory to write figures/tables into.",
    )
    args = ap.parse_args()

    pj = args.paper_json_dir
    paper = args.paper_dir
    fig_dir = paper / "fig"
    tables_dir = paper / "tables"

    e1 = _load_json(pj / "e1_rule_only.paper.json")
    e1e3 = _load_json(pj / "e1e3_llm_regression.paper.json")
    e2 = _load_json(pj / "e2_trust_ablation.paper.json")
    e4 = _load_json(pj / "e4_cross_frontend_consistency.paper.json")
    e5_1 = _load_json(pj / "e5_1_external_baseline.paper.json")
    e5_2 = _load_json(pj / "e5_2_portability_vs_perf.paper.json")
    e6 = _load_json(pj / "e6_ir_usability.paper.json")

    _set_plot_style()

    fig_e1e3_recoverability(e1, e1e3, fig_dir)
    fig_e2_trust_ablation(e2, fig_dir)
    fig_e4_consistency(e4, fig_dir)
    fig_e5_2_retune_vs_freeze(e5_2, fig_dir)
    fig_e5_1_external_baseline(e5_1, fig_dir)
    fig_e6_contract_calibration(e6, fig_dir)
    _write_tables_tex(e1, e1e3, e2, e4, e5_1, e5_2, e6, tables_dir)

    print(f"Wrote figures to: {fig_dir}")
    print(f"Wrote tables to:  {tables_dir}")


if __name__ == "__main__":
    main()
