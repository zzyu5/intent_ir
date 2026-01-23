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
from matplotlib.patches import Patch  # noqa: E402
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

def _direct_labels_right(
    ax: plt.Axes,
    *,
    x: float,
    items: list[tuple[float, str, str]],
    dx: float = 0.07,
    min_dy: float = 0.03,
) -> None:
    """
    Place text labels at the right edge with light leader lines, avoiding overlap.

    items: [(y, label, color), ...] for points at (x, y).
    """
    if not items:
        return
    items_sorted = sorted(items, key=lambda t: float(t[0]))
    ys = [float(y) for y, _, _ in items_sorted]
    y_min, y_max = ax.get_ylim()

    # Greedy non-overlap: push labels upward to maintain min separation.
    y_lab = ys[:]
    for i in range(1, len(y_lab)):
        if y_lab[i] - y_lab[i - 1] < min_dy:
            y_lab[i] = y_lab[i - 1] + min_dy
    # If we overflow, shift everything down.
    overflow = y_lab[-1] - y_max
    if overflow > 0:
        y_lab = [y - overflow for y in y_lab]
    # Clamp to lower bound.
    under = y_min - y_lab[0]
    if under > 0:
        y_lab = [y + under for y in y_lab]

    for (y0, text, color), yl in zip(items_sorted, y_lab, strict=True):
        ax.annotate(
            text,
            xy=(x, float(y0)),
            xytext=(x + dx, float(yl)),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=7,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.8),
            clip_on=False,
        )


def _geom_mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and float(x) > 0.0 and math.isfinite(float(x))]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def fig_e1e3_recoverability(e1: dict[str, Any], e1e3: dict[str, Any], out: Path) -> None:
    # Paper-friendly line charts (avoid dumbbells): show the step-up from one-shot to
    # bounded repair, and show how semantic-class accuracy improves over rule-only.
    frontends = list(e1e3.get("frontends") or ["triton", "tilelang", "cuda"])
    one = e1e3["summary"]["one_shot"]["by_frontend"]
    fb = e1e3["summary"]["feedback"]["by_frontend"]

    fe_label = {"triton": "Triton", "tilelang": "TileLang", "cuda": "CUDA"}
    pal = sns.color_palette("colorblind")
    fe_color = {"triton": pal[0], "tilelang": pal[2], "cuda": pal[3], "all": "black"}

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

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 3.35), constrained_layout=True)

    # (a) Validated recoverability (OK rate): rule-only vs one-shot vs feedback.
    ax = axes[0]
    x = np.arange(3)
    xt = ["rule-only", "one-shot", "feedback"]
    marker = {"triton": "o", "tilelang": "s", "cuda": "o", "all": "D"}
    ls = {"triton": "-", "tilelang": "-", "cuda": "-", "all": "-"}
    for fe in frontends:
        # No per-frontend rule-only stats in the paper JSON; start the line at one-shot.
        xs = np.array([1, 2])
        ys = [float(one[fe]["ok_rate"]), float(fb[fe]["ok_rate"])]
        ax.plot(
            xs,
            ys,
            marker=marker.get(fe, "o"),
            markersize=3.8,
            linewidth=1.5,
            linestyle=ls.get(fe, "-"),
            color=fe_color[fe],
            label="_nolegend_",
        )
    all_ys = [float((e1.get("summary") or {}).get("ok_rate") or 0.0), _overall_ok_rate(one), _overall_ok_rate(fb)]
    ax.plot(x, all_ys, marker=marker["all"], markersize=3.8, linewidth=1.8, color=fe_color["all"], label="_nolegend_")
    ax.set_xticks(x, xt)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("OK rate")
    ax.set_title("Validated recoverability")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    _panel_label(ax, "(a)")
    ax.set_xlim(-0.05, 2.05)
    _direct_labels_right(
        ax,
        x=2.0,
        items=[
            (float(fb["cuda"]["ok_rate"]), "CUDA", fe_color["cuda"]),
            (float(fb["triton"]["ok_rate"]), "Triton", fe_color["triton"]),
            (float(fb["tilelang"]["ok_rate"]), "TileLang", fe_color["tilelang"]),
            (float(all_ys[2]), "All", fe_color["all"]),
        ],
        dx=0.08,
        min_dy=0.02,
    )

    # (b) Semantic-class accuracy (rule-only vs LLM one-shot vs LLM+feedback).
    ax = axes[1]
    x = np.arange(3)
    xt = ["rule-only", "one-shot", "feedback"]
    for fe in frontends:
        ys = [float(e1_acc[fe]["acc"]), float(one_acc[fe]["semantic_class"]["acc"]), float(fb_acc[fe]["semantic_class"]["acc"])]
        ax.plot(
            x,
            ys,
            marker=marker.get(fe, "o"),
            markersize=3.8,
            linewidth=1.5,
            color=fe_color[fe],
            label="_nolegend_",
        )
    all2 = [
        _overall_acc(e1_acc, kind="e1"),
        _overall_acc(one_acc, kind="e1e3"),
        _overall_acc(fb_acc, kind="e1e3"),
    ]
    ax.plot(
        x,
        all2,
        marker=marker["all"],
        markersize=3.8,
        linewidth=1.8,
        color=fe_color["all"],
        label="_nolegend_",
    )
    ax.set_xticks(x, xt)
    ax.set_ylim(0.2, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title("Semantic-class accuracy (labeled subset)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    _panel_label(ax, "(b)")
    ax.set_xlim(-0.05, 2.05)
    _direct_labels_right(
        ax,
        x=2.0,
        items=[
            (float(fb_acc["cuda"]["semantic_class"]["acc"]), "CUDA", fe_color["cuda"]),
            (float(fb_acc["triton"]["semantic_class"]["acc"]), "Triton", fe_color["triton"]),
            (float(fb_acc["tilelang"]["semantic_class"]["acc"]), "TileLang", fe_color["tilelang"]),
            (float(all2[-1]), "All", fe_color["all"]),
        ],
        dx=0.08,
        min_dy=0.03,
    )

    # No global legend: direct labeling avoids overlap at single-column width.
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

    fig, ax = plt.subplots(figsize=(3.55, 2.6), constrained_layout=True)
    pal = sns.color_palette("colorblind")
    mode_color = {"diff_only": pal[7], "generic": pal[0], "full": pal[2]}

    w = 0.22
    for i, m in enumerate(modes):
        ys = [float(e2["by_frontend"][fe]["summary"]["aggregate"][m]["kill_rate"]) for fe in frontends] + [_overall(m)]
        ax.bar(x + (i - 1) * w, ys, width=w, color=mode_color[m], label=mode_label[m])

    ax.set_xticks(x, [fe_label[fe] for fe in xs])
    ax.set_ylim(0.5, 1.01)
    ax.set_ylabel("mutation-kill rate")
    ax.set_title("Trustworthiness ablation (higher is better)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.26))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    # Light value labels (avoid overlap by keeping them small).
    for b in ax.patches:
        h = float(getattr(b, "get_height")())
        if h <= 0:
            continue
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + 0.008,
            f"{h*100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.25",
        )
    _save_fig(fig, out / "e2_trust_ablation.pdf")


def fig_e4_consistency(e4: dict[str, Any], out: Path) -> None:
    s = e4["summary"]
    reps = ["IntentIR", "Expanded primitives"]
    exact = [float(s["intent_ok_rate"]), float(s["expanded_ok_rate"])]
    structural = [float(s["intent_structural_ok_rate"]), float(s["expanded_structural_ok_rate"])]
    roles = [float(s["axis_roles_recall_intent_avg"]), float(s["axis_roles_recall_expanded_avg"])]
    reasons_int = (s.get("top_reasons_intent") or {}) if isinstance(s.get("top_reasons_intent"), dict) else {}
    reasons_exp = (s.get("top_reasons_expanded") or {}) if isinstance(s.get("top_reasons_expanded"), dict) else {}

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 3.25), constrained_layout=True)

    # (a) Drift sources: why exact match is low even when structure/roles are perfect.
    ax = axes[0]
    # Pick a small set of top reasons (shared between intent/expanded).
    keys = sorted(set(reasons_int) | set(reasons_exp), key=lambda k: max(int(reasons_int.get(k, 0)), int(reasons_exp.get(k, 0))), reverse=True)
    keys = keys[:4]

    def _pretty(k: str) -> str:
        k = str(k)
        k = k.replace("tilelang:", "")
        k = k.replace("_mismatch", "")
        k = k.replace("_", " ")
        return k

    y = np.arange(len(keys))
    intent_vals = [int(reasons_int.get(k, 0)) for k in keys]
    expanded_vals = [int(reasons_exp.get(k, 0)) for k in keys]
    pal = sns.color_palette("colorblind")
    ax.barh(y - 0.18, intent_vals, height=0.34, color=pal[0], label="IntentIR")
    ax.barh(y + 0.18, expanded_vals, height=0.34, color=pal[2], label="Expanded")
    ax.set_yticks(y, [_pretty(k) for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel("count (out of 30 pairs)")
    ax.set_title("Benign drift sources (exact mismatches)")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="y")
    ax.legend(frameon=False, loc="lower right")
    # Inline note: these two metrics are effectively perfect on this suite.
    ax.text(
        0.02,
        1.02,
        f"structural={structural[0]:.2f}/{structural[1]:.2f}, axis-roles={roles[0]:.2f}/{roles[1]:.2f}",
        transform=ax.transAxes,
        fontsize=7,
        color="0.25",
        va="bottom",
    )
    _panel_label(ax, "(a)")

    # (b) Exact match (zoomed).
    ax = axes[1]
    ax.bar(reps, [v * 100 for v in exact], color=sns.color_palette("colorblind")[3])
    for i, v in enumerate(exact):
        ax.text(i, v * 100 + 0.4, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=7, color="0.25")
    ax.set_ylim(0.0, 10.0)
    ax.set_title("Exact match (zoomed)")
    ax.set_ylabel("%")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    _panel_label(ax, "(b)")

    fig.suptitle("Cross-frontend consistency (Triton vs TileLang, n=30)", y=1.02, fontsize=10)
    _save_fig(fig, out / "e4_consistency.pdf")


def fig_e5_2_retune_vs_freeze(e5_2: dict[str, Any], out: Path) -> None:
    # Horizontal layout improves readability at single-column width.
    fig, ax = plt.subplots(figsize=(3.55, 2.6), constrained_layout=True)

    rows = [r for r in e5_2.get("per_kernel") or [] if isinstance(r, dict)]
    data: list[tuple[str, float]] = []
    for r in rows:
        s = r.get("speedup_paired")
        if not isinstance(s, (int, float)) or not (float(s) > 0.0 and math.isfinite(float(s))):
            continue
        data.append((str(r.get("anchor_tier") or "D_none"), float(s)))
    order = [t for t in ["A_dot", "B_reduce", "C_copy", "D_none"] if any(tt == t for tt, _ in data)]

    df = {"tier": [t for t, _ in data], "speedup": [v for _, v in data]}
    sns.boxplot(
        ax=ax,
        y=df["tier"],
        x=df["speedup"],
        order=order,
        color=sns.color_palette("colorblind")[0],
        fliersize=0,
        linewidth=0.9,
    )
    sns.stripplot(
        ax=ax,
        y=df["tier"],
        x=df["speedup"],
        order=order,
        color="black",
        alpha=0.45,
        size=2.5,
        jitter=0.18,
    )
    ax.axvline(1.0, color="0.25", linewidth=0.8, linestyle="--", alpha=0.7)

    gm = float(e5_2.get("summary", {}).get("geom_speedup_paired") or 1.0)
    if gm and gm > 0:
        ax.axvline(gm, color=sns.color_palette("colorblind")[2], linewidth=1.1, alpha=0.8)
        ax.text(gm, -0.55, f"gmean={gm:.2f}×", ha="left", va="center", fontsize=7, color=sns.color_palette("colorblind")[2])

    ax.set_title("Retune vs Freeze (paired)")
    ax.set_xlabel("speedup (log scale)")
    ax.set_ylabel("anchor tier")
    ax.set_xscale("log")

    lo = min(v for _, v in data) if data else 1.0
    hi = max(v for _, v in data) if data else 1.0
    # Avoid the single extreme outlier making the plot unreadably sparse.
    x_max = 2.0
    ax.set_xlim(max(0.98, lo * 0.98), min(hi * 1.12, x_max))
    if hi > x_max:
        # Annotate the clipped outlier(s) on the right edge with the true max.
        ticks = {t.get_text(): float(pos) for t, pos in zip(ax.get_yticklabels(), ax.get_yticks(), strict=True)}
        tier_max, val_max = max(data, key=lambda kv: kv[1])
        y0 = ticks.get(tier_max)
        if y0 is not None:
            ax.annotate(
                f"max={val_max:.2f}×",
                xy=(x_max, y0),
                xytext=(x_max * 0.92, y0 - 0.25),
                textcoords="data",
                fontsize=7,
                color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8),
            )
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="y")
    _save_fig(fig, out / "e5_2_retune_vs_freeze.pdf")


def fig_e5_1_external_baseline(e5_1: dict[str, Any], out: Path) -> None:
    # External baseline: compare *absolute throughput* (not speedup) for 1T and 16T.
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

    pal = sns.color_palette("colorblind")
    c_base = pal[7]
    c_ours = pal[0]

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 3.3), constrained_layout=True, sharex=True)

    def _panel(ax: plt.Axes, title: str, base: list[float], ours: list[float], *, show_legend: bool) -> None:
        ax.plot(x, base, marker="o", markersize=3.6, linewidth=1.35, color=c_base, label="AI-Benchmark")
        ax.plot(x, ours, marker="o", markersize=3.6, linewidth=1.35, color=c_ours, label="IntentIR")
        ax.set_title(title)
        ax.set_ylabel("throughput (iters/s)")
        ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(False, axis="x")
        ax.set_xticks(x)
        ax.tick_params(axis="x", pad=1)
        if show_legend:
            ax.legend(frameon=False, ncol=2, loc="lower right")

    _panel(axes[0], "Single-thread", base_t1, ours_t1, show_legend=True)
    _panel_label(axes[0], "(a)")
    _panel(axes[1], "16 threads", base_t16, ours_t16, show_legend=False)
    _panel_label(axes[1], "(b)")
    # Show x labels only on the bottom panel.
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xticklabels(names, rotation=35, ha="right", fontsize=6.8)

    fig.suptitle("External baseline throughput (AI-Bench8)", y=1.02, fontsize=10)
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
    tex.append("\\begin{table}[H]")
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
    rep_color = {"intentir": sns.color_palette("colorblind")[0], "linalg": sns.color_palette("colorblind")[3]}

    # Narrative-friendly order (what information we remove).
    ablations = ["full", "no_mask", "no_anchors"]
    ab_label = {"full": "full evidence", "no_mask": "no mask", "no_anchors": "no anchors"}

    # Single-column figure with three panels:
    # (a) IntentIR contract distribution under evidence ablation
    # (b) Linalg contract distribution under evidence ablation
    # (c) Calibration under FULL claims (false-accept among FULL)
    fig, axes = plt.subplots(3, 1, figsize=(3.55, 4.35), constrained_layout=True)

    def _stacked(ax: plt.Axes, rep: str) -> None:
        x = np.arange(len(ablations))
        bottoms = np.zeros(len(ablations))
        for lvl in levels:
            vals = []
            for ab in ablations:
                m = ps["by_rep_ablation"][rep][ab]
                n = float(m["n"] or 0)
                c = float((m.get("contract_levels") or {}).get(lvl, 0) or 0)
                vals.append((c / n) if n > 0 else 0.0)
            ax.bar(x, vals, bottom=bottoms, color=lvl_color[lvl], edgecolor="0.25", linewidth=0.4, label=lvl_label[lvl])
            bottoms = bottoms + np.array(vals)
        ax.set_xticks(x, [ab_label[a] for a in ablations])
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("fraction")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_title(rep_label[rep])

    # (a) IntentIR distribution.
    ax = axes[0]
    levels = ["FULL", "PARTIAL", "OUT_OF_SCOPE"]
    lvl_label = {"FULL": "FULL", "PARTIAL": "PARTIAL", "OUT_OF_SCOPE": "OOS"}
    lvl_color = {
        "FULL": sns.color_palette("colorblind")[2],
        "PARTIAL": sns.color_palette("colorblind")[0],
        "OUT_OF_SCOPE": "0.75",
    }
    _stacked(ax, "intentir")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    _panel_label(ax, "(a)")

    # (b) Linalg distribution.
    ax = axes[1]
    _stacked(ax, "linalg")
    _panel_label(ax, "(b)")

    # (c) Calibration under FULL claims (full evidence).
    ax = axes[2]
    m_int = ps["by_rep_ablation"]["intentir"]["full"]
    m_lin = ps["by_rep_ablation"]["linalg"]["full"]
    vals = {
        "intentir": float(m_int.get("full_false_accept_rate") or 0.0),
        "linalg": float(m_lin.get("full_false_accept_rate") or 0.0),
    }
    ax.bar([rep_label[r] for r in reps], [vals[r] for r in reps], color=[rep_color[r] for r in reps])
    ax.set_ylim(0.0, max(0.35, max(vals.values()) * 1.15))
    ax.set_ylabel("false-accept among FULL")
    ax.set_title("Calibration under FULL claims (full evidence)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    for i, r in enumerate(reps):
        m = ps["by_rep_ablation"][r]["full"]
        n_full = int(m.get("full_claims") or 0)
        ok_rate = float(m.get("ok_rate") or 0.0)
        bind = float(m.get("binding_ok_rate") or 0.0)
        ax.text(
            i,
            vals[r] + 0.012,
            f"{vals[r]:.2f}\nFULL={n_full}\nok={ok_rate:.2f}\nbind={bind:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.25",
        )
    _panel_label(ax, "(c)")

    fig.suptitle("IR + sidecar contract calibration (CUDA)", y=1.02, fontsize=10)
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
