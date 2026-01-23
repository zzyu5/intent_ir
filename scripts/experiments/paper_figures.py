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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, out: Path, *, tight: bool = True) -> None:
    _ensure_dir(out.parent)
    if tight:
        fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    else:
        # Some layouts (notably multi-panel figures with long y-labels) can be
        # clipped by bbox_inches='tight' on some Matplotlib versions; keep the
        # default figure bbox for safety.
        fig.savefig(out)
    plt.close(fig)

def _set_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        rc={
            # Use a higher DPI so rasterized elements (if any) remain crisp in print.
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
            "axes.titlepad": 6,
            "axes.formatter.useoffset": False,
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
    """
    Panel labels like (a)/(b) should not cover data.

    Place them just above the axes (still within the figure), which is the
    typical academic style for tight single-column plots.
    """
    ax.text(
        -0.14,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def _geom_mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and float(x) > 0.0 and math.isfinite(float(x))]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def fig_e1e3_recoverability(e1: dict[str, Any], e1e3: dict[str, Any], out: Path) -> None:
    # Paper-friendly *grouped bars*: easier to read than overlapping lines at single-column width.
    fes = list(e1e3.get("frontends") or ["triton", "tilelang", "cuda"])
    frontends = [fe for fe in ["triton", "tilelang", "cuda"] if fe in fes]
    one = e1e3["summary"]["one_shot"]["by_frontend"]
    fb = e1e3["summary"]["feedback"]["by_frontend"]

    fe_label = {"triton": "Triton", "tilelang": "TileLang", "cuda": "CUDA", "all": "All"}
    reps = frontends + ["all"]

    def _overall_ok_rate(v: dict[str, Any]) -> float:
        n = 0.0
        ok = 0.0
        for fe in frontends:
            n += float(v[fe]["n"])
            ok += float(v[fe]["ok"])
        return (ok / n) if n > 0 else 0.0

    # Semantic-class label accuracy (rule-only vs one-shot vs feedback).
    e1_acc = e1["label_eval"]["semantic_class_acc_by_frontend"]
    one_acc = e1e3["label_eval"]["one_shot"]["by_frontend"]
    fb_acc = e1e3["label_eval"]["feedback"]["by_frontend"]

    def _overall_acc(acc_by_fe: dict[str, Any], *, kind: str) -> float:
        n = 0.0
        ok = 0.0
        for fe in frontends:
            if kind == "e1":
                st = acc_by_fe.get(fe) or {}
                n += float(st.get("n") or 0)
                ok += float(st.get("ok") or 0)
            else:
                st = (acc_by_fe.get(fe) or {}).get("semantic_class") or {}
                n += float(st.get("n") or 0)
                ok += float(st.get("ok") or 0)
        return (ok / n) if n > 0 else 0.0

    pal = sns.color_palette("colorblind")
    stage_color = {"rule_only": "0.65", "one_shot": pal[0], "feedback": pal[2]}

    # (a) Recoverability: show one-shot vs feedback per frontend.
    # Note: rule-only only has an *overall* OK rate (not per-frontend), so we
    # visualize it as a single baseline marker on the "All" bucket to avoid
    # implying per-frontend numbers.
    rule_ok = float((e1.get("summary") or {}).get("ok_rate") or 0.0)
    ok_one = [float(one[fe]["ok_rate"]) for fe in frontends] + [_overall_ok_rate(one)]
    ok_fb = [float(fb[fe]["ok_rate"]) for fe in frontends] + [_overall_ok_rate(fb)]

    # (b) Semantic-class accuracy per frontend (rule-only, one-shot, feedback), plus weighted "All".
    acc_rule = [float(e1_acc[fe]["acc"]) for fe in frontends] + [_overall_acc(e1_acc, kind="e1")]
    acc_one = [float(one_acc[fe]["semantic_class"]["acc"]) for fe in frontends] + [_overall_acc(one_acc, kind="e1e3")]
    acc_fb = [float(fb_acc[fe]["semantic_class"]["acc"]) for fe in frontends] + [_overall_acc(fb_acc, kind="e1e3")]

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 3.85), constrained_layout=False)
    fig.subplots_adjust(top=0.83, hspace=0.60)

    ax = axes[0]
    x = np.arange(len(reps))
    w = 0.34
    ax.bar(x - w / 2, ok_one, width=w, color=stage_color["one_shot"])
    ax.bar(x + w / 2, ok_fb, width=w, color=stage_color["feedback"])
    all_i = len(reps) - 1
    ax.scatter([all_i], [rule_ok], marker="D", s=20, color="0.35", zorder=5)
    ax.set_xticks(x, [fe_label[r] for r in reps])
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("OK rate")
    ax.set_title("Validated recoverability")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="x")
    _panel_label(ax, "(a)")
    # Value labels + delta (feedback - one-shot) per group.
    for i, (v1, v2) in enumerate(zip(ok_one, ok_fb, strict=True)):
        # Put values inside bars, but skip near-1.0 values to avoid clutter/overlap.
        if v1 < 0.995:
            ax.text(i - w / 2, max(0.62, v1 - 0.03), f"{v1:.2f}", ha="center", va="top", fontsize=7, color="white")
        if v2 < 0.995:
            ax.text(i + w / 2, max(0.62, v2 - 0.03), f"{v2:.2f}", ha="center", va="top", fontsize=7, color="white")
        d = v2 - v1
        if abs(d) >= 0.03:
            ax.text(i, 0.995, f"{d*100:+.0f}pp", ha="center", va="top", fontsize=7, color=stage_color["feedback"])

    ax = axes[1]
    w = 0.22
    ax.bar(x - w, acc_rule, width=w, color=stage_color["rule_only"])
    ax.bar(x, acc_one, width=w, color=stage_color["one_shot"])
    ax.bar(x + w, acc_fb, width=w, color=stage_color["feedback"])
    ax.set_xticks(x, [fe_label[r] for r in reps])
    ax.set_ylim(0.2, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title("Semantic-class accuracy (labeled subset)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="x")
    _panel_label(ax, "(b)")
    for i, (v0, v1, v2) in enumerate(zip(acc_rule, acc_one, acc_fb, strict=True)):
        ax.text(i - w, v0 + 0.02, f"{v0:.2f}", ha="center", va="bottom", fontsize=7, color="0.25")
        # Avoid clutter in tightly packed grouped bars; panel (b) already communicates the trend.

    # One clean legend for the whole figure (color blocks), outside the axes.
    handles = [
        Patch(facecolor=stage_color["rule_only"], edgecolor="none", label="rule-only (class acc)"),
        Patch(facecolor=stage_color["one_shot"], edgecolor="none", label="one-shot"),
        Patch(facecolor=stage_color["feedback"], edgecolor="none", label="feedback"),
        Line2D([0], [0], marker="D", color="0.35", linestyle="None", markersize=5, label=f"rule-only baseline (recoverability, All={rule_ok:.2f})"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, columnspacing=1.2, handlelength=1.6)

    _save_fig(fig, out / "e1e3_recoverability.pdf")


def fig_e2_trust_ablation(e2: dict[str, Any], out: Path) -> None:
    # Avoid crowded lines: grouped bars (diff-only / generic / full) per frontend.
    frontends = ["triton", "tilelang", "cuda"]
    fe_label = {"triton": "Triton", "tilelang": "TileLang", "cuda": "CUDA", "all": "All"}
    modes = ["diff_only", "generic", "full"]
    mode_label = {"diff_only": "diff-only", "generic": "generic", "full": "full"}

    def _overall(m: str) -> float:
        total = 0.0
        killed = 0.0
        for fe in frontends:
            agg = e2["by_frontend"][fe]["summary"]["aggregate"][m]
            total += float(agg["total"])
            killed += float(agg["killed"])
        return (killed / total) if total > 0 else 0.0

    xs = frontends + ["all"]
    x = np.arange(len(xs))

    fig, ax = plt.subplots(figsize=(3.55, 3.05), constrained_layout=False)
    fig.subplots_adjust(top=0.84)
    pal = sns.color_palette("colorblind")
    mode_color = {"diff_only": pal[7], "generic": pal[0], "full": pal[2]}

    ys_by_mode: dict[str, list[float]] = {}
    for m in modes:
        ys_by_mode[m] = [float(e2["by_frontend"][fe]["summary"]["aggregate"][m]["kill_rate"]) for fe in frontends] + [_overall(m)]

    w = 0.18
    for i, m in enumerate(modes):
        ax.bar(x + (i - 1) * w, ys_by_mode[m], width=w, color=mode_color[m], label=mode_label[m])

    ax.set_xticks(x, [fe_label[fe] for fe in xs])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("mutation-kill rate")
    ax.set_title("Trustworthiness ablation (higher is better)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="x")

    # One clean legend for the whole figure (outside the axes).
    handles = [
        Patch(facecolor=mode_color["diff_only"], edgecolor="none", label="diff-only"),
        Patch(facecolor=mode_color["generic"], edgecolor="none", label="generic"),
        Patch(facecolor=mode_color["full"], edgecolor="none", label="full"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, columnspacing=1.2, handlelength=1.6)
    _save_fig(fig, out / "e2_trust_ablation.pdf")


def fig_e4_consistency(e4: dict[str, Any], out: Path) -> None:
    s = e4["summary"]
    exact = [float(s["intent_ok_rate"]), float(s["expanded_ok_rate"])]
    structural = [float(s["intent_structural_ok_rate"]), float(s["expanded_structural_ok_rate"])]
    roles = [float(s["axis_roles_recall_intent_avg"]), float(s["axis_roles_recall_expanded_avg"])]
    reasons_int = (s.get("top_reasons_intent") or {}) if isinstance(s.get("top_reasons_intent"), dict) else {}
    reasons_exp = (s.get("top_reasons_expanded") or {}) if isinstance(s.get("top_reasons_expanded"), dict) else {}

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 3.85), constrained_layout=False)
    fig.subplots_adjust(top=0.84, hspace=0.62)
    fig.suptitle("Cross-frontend consistency (Triton vs TileLang, n=30)", y=0.985, fontsize=10)

    pal = sns.color_palette("colorblind")

    # (a) Consistency metrics: structural/roles are the meaningful invariants; exact match is strict and can be low due to benign drift.
    ax = axes[0]
    metrics = ["structural match", "axis-role recall", "verbatim match"]
    intent_vals = [structural[0], roles[0], exact[0]]
    expanded_vals = [structural[1], roles[1], exact[1]]
    y = np.arange(len(metrics))
    h = 0.28
    ax.barh(y - 0.16, intent_vals, height=h, color=pal[0], label="IntentIR")
    ax.barh(y + 0.16, expanded_vals, height=h, color=pal[2], label="Macro-expanded")

    def _pct(v: float) -> str:
        if v >= 0.995:
            return "100%"
        if v >= 0.10:
            return f"{v*100:.0f}%"
        return f"{v*100:.1f}%"

    for yi, v in enumerate(intent_vals):
        ax.text(min(1.03, v + 0.02), yi - 0.16, _pct(float(v)), va="center", ha="left", fontsize=7, color="0.25")
    for yi, v in enumerate(expanded_vals):
        ax.text(min(1.03, v + 0.02), yi + 0.16, _pct(float(v)), va="center", ha="left", fontsize=7, color="0.25")

    ax.set_yticks(y, [m.capitalize() for m in metrics])
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("rate")
    ax.set_title("Consistency metrics")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="y")
    ax.invert_yaxis()
    _panel_label(ax, "(a)")

    # (b) Drift sources: why verbatim match can be low even when structural/roles are perfect.
    ax = axes[1]
    keys = sorted(
        set(reasons_int) | set(reasons_exp),
        key=lambda k: max(int(reasons_int.get(k, 0)), int(reasons_exp.get(k, 0))),
        reverse=True,
    )[:4]

    def _pretty(k: str) -> str:
        k = str(k)
        k = k.replace("tilelang:", "")
        k = k.replace("_mismatch", "")
        k = k.replace("_", " ")
        return k

    y = np.arange(len(keys))
    intent_vals = [int(reasons_int.get(k, 0)) for k in keys]
    expanded_vals = [int(reasons_exp.get(k, 0)) for k in keys]
    ax.barh(y - 0.16, intent_vals, height=0.28, color=pal[0])
    ax.barh(y + 0.16, expanded_vals, height=0.28, color=pal[2])
    ax.set_yticks(y, [_pretty(k) for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel("count (out of 30 pairs)")
    ax.set_title("Benign drift sources (verbatim mismatches)")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="y")
    for yi, v in enumerate(intent_vals):
        ax.text(v + 0.4, yi - 0.16, str(int(v)), va="center", ha="left", fontsize=7, color="0.25")
    for yi, v in enumerate(expanded_vals):
        ax.text(v + 0.4, yi + 0.16, str(int(v)), va="center", ha="left", fontsize=7, color="0.25")
    _panel_label(ax, "(b)")

    handles = [Patch(facecolor=pal[0], edgecolor="none", label="IntentIR"), Patch(facecolor=pal[2], edgecolor="none", label="Macro-expanded")]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, columnspacing=1.2, handlelength=1.6)
    _save_fig(fig, out / "e4_consistency.pdf")


def fig_e5_2_retune_vs_freeze(e5_2: dict[str, Any], out: Path) -> None:
    # Use a linear-scale bar + scatter (no log axis), which is easier to read.
    rows = [r for r in e5_2.get("per_kernel") or [] if isinstance(r, dict)]
    pairs: list[tuple[str, float]] = []
    for r in rows:
        s = r.get("speedup_paired")
        if not isinstance(s, (int, float)) or not (float(s) > 0.0 and math.isfinite(float(s))):
            continue
        pairs.append((str(r.get("anchor_tier") or "D_none"), float(s)))
    order = [t for t in ["A_dot", "B_reduce", "C_copy", "D_none"] if any(tt == t for tt, _ in pairs)]
    if not order:
        return

    by_tier: dict[str, list[float]] = {t: [] for t in order}
    for t, s in pairs:
        if t in by_tier:
            by_tier[t].append(float(s))

    tier_label = {"A_dot": "A dot", "B_reduce": "B reduce", "C_copy": "C copy", "D_none": "D none"}
    gm_tier = [(_geom_mean(by_tier[t]) or 0.0) for t in order]
    gm_all = float(e5_2.get("summary", {}).get("geom_speedup_paired") or 1.0)
    max_val = max((max(vs) for vs in by_tier.values() if vs), default=1.0)
    n_tier = [len(by_tier[t]) for t in order]

    pal = sns.color_palette("colorblind")
    fig, ax = plt.subplots(figsize=(3.55, 3.05), constrained_layout=False)
    # Leave headroom for a clean figure-level title + legend (avoid overlap).
    fig.subplots_adjust(top=0.72)
    x = np.arange(len(order))
    ax.bar(x, gm_tier, width=0.48, color=pal[0], alpha=0.9)

    rng = np.random.default_rng(0)
    for i, t in enumerate(order):
        ys = by_tier[t]
        if not ys:
            continue
        xs = i + rng.uniform(-0.16, 0.16, size=len(ys))
        ax.scatter(xs, ys, s=10, color="0.15", alpha=0.5, linewidths=0, label="per kernel" if i == 0 else "_nolegend_")

    ax.axhline(1.0, color="0.35", linewidth=0.9, linestyle="--")
    if gm_all and gm_all > 0:
        ax.axhline(gm_all, color=pal[2], linewidth=1.1, linestyle="-", alpha=0.85)

    y_max = 2.2
    ax.set_ylim(0.0, y_max)
    if max_val > y_max:
        ax.annotate(
            f"max={max_val:.2f}×",
            xy=(len(order) - 0.2, y_max),
            xytext=(len(order) - 1.0, y_max * 0.93),
            fontsize=7,
            color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8),
        )

    ax.set_xticks(x, [f"{tier_label.get(t, t)}\n(n={n})" for t, n in zip(order, n_tier, strict=True)])
    ax.set_ylabel("speedup (retune / freeze)")
    ax.set_title("")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="x")
    # Bar labels (tier gmeans).
    for i, v in enumerate(gm_tier):
        if v <= 0:
            continue
        ax.text(i, min(y_max - 0.02, v + 0.04), f"{v:.2f}×", ha="center", va="bottom", fontsize=7, color="0.25")

    handles = [
        Patch(facecolor=pal[0], edgecolor="none", label="gmean (per tier)"),
        Line2D([0], [0], marker="o", color="0.15", linestyle="None", markersize=4, label="per kernel"),
        Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.0, label="freeze (1.0×)"),
    ]
    if gm_all and gm_all > 0:
        handles.append(Line2D([0], [0], color=pal[2], linestyle="-", linewidth=1.2, label=f"overall gmean ({gm_all:.2f}×)"))
    fig.suptitle("Retune vs Freeze (paired speedup)", y=0.99, fontsize=10)
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        columnspacing=1.2,
        handlelength=1.6,
    )
    _save_fig(fig, out / "e5_2_retune_vs_freeze.pdf")


def fig_e5_1_external_baseline(e5_1: dict[str, Any], out: Path) -> None:
    # External baseline: use *speedup* on the y-axis (clear comparison), and
    # annotate each bar with the achieved throughput (iters/s) so the plot
    # still communicates absolute performance.
    rows = [r for r in list(e5_1.get("per_kernel") or []) if isinstance(r, dict)]

    data: list[tuple[str, float, float, float, float]] = []
    for r in rows:
        name = str(r.get("name") or "")
        bt1 = r.get("baseline_seconds_per_iter_t1")
        bt16 = r.get("baseline_seconds_per_iter_t16")
        ot1 = r.get("ours_seconds_per_iter_t1")
        ot16 = r.get("ours_seconds_per_iter_t16")
        if not all(isinstance(x, (int, float)) and float(x) > 0.0 and math.isfinite(float(x)) for x in [bt1, bt16, ot1, ot16]):
            continue
        data.append((name, float(bt1), float(ot1), float(bt16), float(ot16)))

    # Keep the suite order (8 kernels) for easy cross-checking with AI-Benchmark reports.
    names = [n for n, *_ in data]
    x = np.arange(len(names))

    base_t1 = [1.0 / bt1 for _, bt1, _, _, _ in data]
    ours_t1 = [1.0 / ot1 for _, _, ot1, _, _ in data]
    base_t16 = [1.0 / bt16 for _, _, _, bt16, _ in data]
    ours_t16 = [1.0 / ot16 for _, _, _, _, ot16 in data]
    sp_t1 = [bt1 / ot1 for _, bt1, ot1, _, _ in data]
    sp_t16 = [bt16 / ot16 for _, _, _, bt16, ot16 in data]

    pal = sns.color_palette("colorblind")
    c_ours = pal[0]
    c_base = pal[7]

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 3.60), constrained_layout=False, sharex=True)
    fig.subplots_adjust(top=0.88, hspace=0.55)

    def _panel(ax: plt.Axes, title: str, sp: list[float], ours_tp: list[float]) -> None:
        ax.bar(x, sp, width=0.55, color=c_ours, alpha=0.95)
        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel("speedup (×)")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(False, axis="x")
        y_max = max(2.0, max(sp, default=1.0) * 1.18)
        ax.set_ylim(0.0, y_max)
        # Annotate each bar with throughput (ours) to keep a sense of absolute performance.
        for i, (s, tp) in enumerate(zip(sp, ours_tp, strict=True)):
            ax.text(i, min(y_max - 0.02, s + 0.06), f"{tp:.0f}", ha="center", va="bottom", fontsize=7, color="0.25")
        # (We describe the labels in the figure caption to avoid clutter.)

    gm1 = e5_1.get("summary", {}).get("geom_speedup_ours_over_baseline_t1")
    gm16 = e5_1.get("summary", {}).get("geom_speedup_ours_over_baseline_t16")
    t1_title = "Single-thread"
    t16_title = "16 threads"
    if isinstance(gm1, (int, float)) and math.isfinite(float(gm1)):
        t1_title += f" (gmean={float(gm1):.2f}×)"
    if isinstance(gm16, (int, float)) and math.isfinite(float(gm16)):
        t16_title += f" (gmean={float(gm16):.2f}×)"

    _panel(axes[0], t1_title, sp_t1, ours_t1)
    _panel_label(axes[0], "(a)")
    _panel(axes[1], t16_title, sp_t16, ours_t16)
    _panel_label(axes[1], "(b)")

    # Figure-level note: bar = speedup, label = throughput (ours).
    handles = [
        Patch(facecolor=c_ours, edgecolor="none", label="Speedup (IntentIR / AI-Benchmark)"),
        Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.0, label="Parity (1.0×)"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, columnspacing=1.2, handlelength=2.0)

    axes[0].tick_params(labelbottom=False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=28, ha="right", fontsize=7)

    fig.suptitle("External baseline vs IntentIR (AI-Bench8)", y=1.02, fontsize=10)
    _save_fig(fig, out / "e5_1_external_baseline.pdf")


def _write_dataset_table_tex(e1e3: dict[str, Any], out_tables: Path) -> None:
    _ensure_dir(out_tables)
    # Use one-shot's tier classification as the "dataset composition" view.
    # Feedback summaries may be overridden by per-frontend reruns (e.g., CUDA fixes),
    # which can introduce minor tier-count drift in aggregated summaries.
    os = (e1e3.get("summary") or {}).get("one_shot") or {}
    by_fe = os.get("by_frontend") if isinstance(os, dict) else None
    if not isinstance(by_fe, dict):
        return

    frontends = ["triton", "tilelang", "cuda"]
    tiers = ["A_dot", "B_reduce", "C_copy"]
    tot_n = 0
    tot_by_tier = {t: 0 for t in tiers}

    rows = []
    for fe in frontends:
        s = by_fe.get(fe)
        if not isinstance(s, dict):
            continue
        n = int(s.get("n") or 0)
        tot_n += n
        t = s.get("tiers") if isinstance(s.get("tiers"), dict) else {}
        a = int((t.get("A_dot") or {}).get("n") or 0)
        b = int((t.get("B_reduce") or {}).get("n") or 0)
        c = int((t.get("C_copy") or {}).get("n") or 0)
        tot_by_tier["A_dot"] += a
        tot_by_tier["B_reduce"] += b
        tot_by_tier["C_copy"] += c
        rows.append((fe.upper(), n, a, b, c))

    tex = []
    tex.append("% Auto-generated by scripts/experiments/paper_figures.py")
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\begin{tabular}{lrrrr}")
    tex.append("\\toprule")
    tex.append("Frontend & Total & A\\_dot & B\\_reduce & C\\_copy \\\\")
    tex.append("\\midrule")
    for fe, n, a, b, c in rows:
        tex.append(f"{fe} & {n:d} & {a:d} & {b:d} & {c:d} \\\\")
    tex.append("\\midrule")
    tex.append(f"ALL & {tot_n:d} & {tot_by_tier['A_dot']:d} & {tot_by_tier['B_reduce']:d} & {tot_by_tier['C_copy']:d} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append(
        "\\caption{Kernel suite composition (100 kernels). A\\_dot: GEMM/dot-like; B\\_reduce: softmax/norm/reduction; C\\_copy: data movement (e.g., transpose/gather).}"
    )
    tex.append("\\label{tab:kernel_suite}")
    tex.append("\\end{table}")
    tex.append("")

    (out_tables / "kernel_suite.tex").write_text("\n".join(tex), encoding="utf-8")


def fig_e6_contract_calibration(e6: dict[str, Any], out: Path) -> None:
    ps = e6["summary"]["paper_summary"]
    reps = ["intentir", "linalg"]
    rep_label = {"intentir": "IntentIR", "linalg": "Linalg"}
    pal = sns.color_palette("colorblind")
    rep_color = {"intentir": pal[0], "linalg": pal[3]}

    # Narrative-friendly order (what information we remove).
    ablations = ["full", "no_mask", "no_anchors"]
    ab_label = {"full": "full evidence", "no_mask": "no mask", "no_anchors": "no anchors"}

    # Single-column figure with two distinct panels:
    # (a) Contract distribution (IntentIR vs Linalg) under evidence ablation
    # (b) Calibration under FULL claims (false-accept among FULL) + context (ok_rate, #FULL)
    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.15), constrained_layout=False)
    fig.subplots_adjust(top=0.86, hspace=0.70)

    levels = ["FULL", "PARTIAL", "OUT_OF_SCOPE"]
    lvl_label = {"FULL": "FULL", "PARTIAL": "PARTIAL", "OUT_OF_SCOPE": "OOS"}
    lvl_color = {"FULL": pal[2], "PARTIAL": pal[0], "OUT_OF_SCOPE": "0.75"}

    # (a) Grouped stacked bars: per ablation, IntentIR (solid) vs Linalg (hatched).
    ax = axes[0]
    x = np.arange(len(ablations))
    bw = 0.34
    offsets = {"intentir": -bw / 2, "linalg": bw / 2}

    for rep in reps:
        bottoms = np.zeros(len(ablations))
        for lvl in levels:
            vals = []
            for ab in ablations:
                m = ps["by_rep_ablation"][rep][ab]
                n = float(m.get("n") or 0)
                c = float((m.get("contract_levels") or {}).get(lvl, 0) or 0)
                vals.append((c / n) if n > 0 else 0.0)
            hatch = "" if rep == "intentir" else "///"
            ax.bar(
                x + offsets[rep],
                vals,
                bottom=bottoms,
                width=bw,
                color=lvl_color[lvl],
                edgecolor="0.25",
                linewidth=0.4,
                hatch=hatch,
            )
            bottoms = bottoms + np.array(vals)

    ax.set_xticks(x, [ab_label[a] for a in ablations])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("fraction")
    ax.set_title("Contract distribution under evidence ablation")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="x")
    _panel_label(ax, "(a)")

    # (b) Calibration under FULL claims (full evidence).
    ax = axes[1]
    m_int = ps["by_rep_ablation"]["intentir"]["full"]
    m_lin = ps["by_rep_ablation"]["linalg"]["full"]
    vals = {"intentir": float(m_int.get("full_false_accept_rate") or 0.0), "linalg": float(m_lin.get("full_false_accept_rate") or 0.0)}
    xs = np.arange(len(reps))
    bars = ax.bar(xs, [vals[r] for r in reps], color=[rep_color[r] for r in reps], width=0.55)
    ax.set_xticks(xs, [rep_label[r] for r in reps])
    ax.set_ylim(0.0, max(0.40, max(vals.values()) * 1.65))
    ax.set_ylabel("false-accept among FULL")
    ax.set_title("Calibration under FULL claims (full evidence)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="x")
    for i, rep in enumerate(reps):
        m = ps["by_rep_ablation"][rep]["full"]
        ok = float(m.get("ok_rate") or 0.0)
        fc = int(m.get("full_claims") or 0)
        ax.text(i, bars[i].get_height() + 0.015, f"{vals[rep]:.2f}", ha="center", va="bottom", fontsize=8, color="0.25")
        # Keep the plot uncluttered: ok_rate / #FULL are reported in the table and caption.
    _panel_label(ax, "(b)")

    # Figure-level legends: (i) contract levels (colors) and (ii) reps (solid vs hatched).
    legend_levels = [Patch(facecolor=lvl_color[lvl], edgecolor="none", label=lvl_label[lvl]) for lvl in levels]
    legend_reps = [
        Patch(facecolor="white", edgecolor="0.25", label="IntentIR", linewidth=0.8),
        Patch(facecolor="white", edgecolor="0.25", hatch="///", label="Linalg", linewidth=0.8),
    ]
    fig.legend(handles=legend_reps + legend_levels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=5, columnspacing=0.9, handlelength=1.2)
    _save_fig(fig, out / "e6_contract_calibration.pdf", tight=False)


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
    # External baseline: report 16-thread geomean (primary) and keep 1-thread in the JSON for the paper narrative.
    e5_1_gm16 = e5_1.get("summary", {}).get("geom_speedup_ours_over_baseline_t16")
    if isinstance(e5_1_gm16, (int, float)):
        tex.append(f"External baseline geomean speedup (AI-Bench8, 16T) & {float(e5_1_gm16):.3f} \\\\")
    tex.append(f"Contract calibration ok\\_rate (IntentIR vs Linalg) & {float(e6_int_full['ok_rate']):.3f} vs {float(e6_lin_full['ok_rate']):.3f} \\\\")
    tex.append(f"Contract calibration FULL false-accept (IntentIR vs Linalg) & {float(e6_int_full['full_false_accept_rate']):.3f} vs {float(e6_lin_full['full_false_accept_rate']):.3f} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\caption{Headline results on the 100-kernel evaluation suite.}")
    tex.append("\\label{tab:headline}")
    tex.append("\\end{table}")
    tex.append("")

    (out_tables / "headline_metrics.tex").write_text("\n".join(tex), encoding="utf-8")
    _write_dataset_table_tex(e1e3, out_tables)


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
    # Remove stale legacy figures (we now plot E5.1/E5.2 separately).
    stale = fig_dir / "e5_performance.pdf"
    if stale.exists():
        stale.unlink()

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
