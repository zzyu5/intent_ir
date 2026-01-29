"""
Generate publication-quality figures with strict alignment and consistency.
Style: Professional Academic (Times New Roman, Morandi Colors).
Layout: Top-aligned legends, Horizontal bars for long labels, Explicit separation.

[Modified Version]: Adjusted margins (top) to prevent legend overlap.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# 1. Professional Style & Palette
# =============================================================================

def _set_pro_style() -> None:
    """
    Sets strict formatting rules for consistent, publication-ready figures.
    """
    plt.rcParams.update({
        # Fonts: Times New Roman is the gold standard
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 10,     # Bold, distinct
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        # Spines & Grid
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,   # Grid BEHIND data
        "grid.color": "#EAEAEA",  # Very subtle grid
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,

        # Hatching (Refined)
        "hatch.linewidth": 0.5,
        "hatch.color": "white",

        # Layout
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "legend.frameon": False,  # Clean look
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.2,
    })

# High-Contrast Morandi Palette
PALETTE = {
    "blue": "#6B8E9B",       # Muted Blue
    "red": "#B6635E",        # Brick Red
    "green": "#8F9E8B",      # Sage Green
    "grey": "#A8A29E",       # Warm Grey
    "light_grey": "#D3D3D3", # Background context
    "dark": "#4A4A4A",       # Text / Scatter points
}

# =============================================================================
# 2. Helper Utilities
# =============================================================================

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _latest_file(dir_path: Path, pattern: str) -> Path | None:
    cand = [p for p in dir_path.glob(pattern) if p.is_file()]
    if not cand:
        return None
    cand.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return cand[-1]

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _save_fig(fig: plt.Figure, out: Path) -> None:
    _ensure_dir(out.parent)
    fig.savefig(out)
    plt.close(fig)

def _panel_label(ax: plt.Axes, label: str) -> None:
    """Standardized panel label (a)/(b) placement."""
    # 稍微调整了 Y 位置，配合 margin 调整，防止贴得太近
    ax.text(-0.15, 1.02, label, transform=ax.transAxes, fontsize=11, 
            fontweight="bold", va="bottom", ha="right", color="black")

def _draw_x_separator(ax: plt.Axes, x_pos: float, y_limit: float) -> None:
    """Draws a vertical dotted line to separate 'All' from specific frontends."""
    ax.axvline(x=x_pos, color=PALETTE["grey"], linestyle=":", linewidth=1.0, alpha=0.6)

def _geom_mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and float(x) > 0.0]
    if not vals: return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))

def _common_legend(fig, handles, labels, ncol=3, y_pos=0.99):
    """
    Forced unified legend at the top of the figure.
    y_pos defaults to 0.99 to sit in the newly created whitespace.
    """
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y_pos), 
               ncol=ncol, frameon=False, fontsize=9)


# =============================================================================
# 3. Plotting Functions (Modified Margins)
# =============================================================================

def fig_e1e3_recoverability(e1: dict[str, Any], e1e3: dict[str, Any], out: Path) -> None:
    # Data Processing
    fes = ["triton", "tilelang", "cuda"]
    
    def _get_data(source_dict, key_chain=None):
        vals = []
        n_sum, ok_sum = 0, 0
        for fe in fes:
            d = source_dict.get(fe, {})
            if key_chain:
                for k in key_chain: d = d.get(k, {})
            n, ok = float(d.get("n", 0)), float(d.get("ok", 0))
            vals.append(ok/n if n>0 else 0)
            n_sum += n; ok_sum += ok
        avg = ok_sum/n_sum if n_sum>0 else 0
        return vals, avg

    # Fetch Data
    ok_one, avg_one = _get_data(e1e3["summary"]["one_shot"]["by_frontend"])
    ok_fb, avg_fb = _get_data(e1e3["summary"]["feedback"]["by_frontend"])
    acc_rule, avg_rule = _get_data(e1["label_eval"]["semantic_class_acc_by_frontend"])
    acc_one, avg_one_acc = _get_data(e1e3["label_eval"]["one_shot"]["by_frontend"], ["semantic_class"])
    acc_fb, avg_fb_acc = _get_data(e1e3["label_eval"]["feedback"]["by_frontend"], ["semantic_class"])

    # Layout Setup
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.5))
    
    # [FIX] Adjusted top from 0.88 to 0.82 to prevent overlap
    plt.subplots_adjust(top=0.82, bottom=0.10, hspace=0.5)

    # X-Axis Construction
    x = np.arange(len(fes))
    x_all = len(fes) + 0.6 
    x_locs = list(x) + [x_all]
    x_labels = [f.capitalize() for f in fes] + ["All"]

    # --- Panel A: Pass Rate ---
    ax = axes[0]
    w = 0.35
    
    ax.bar([i - w/2 for i in x_locs], ok_one + [avg_one], width=w, color=PALETTE["blue"], label="One-shot")
    ax.bar([i + w/2 for i in x_locs], ok_fb + [avg_fb], width=w, color=PALETTE["red"], label="Feedback")
    
    _draw_x_separator(ax, len(fes), 1.0)
    
    for i, (base, impr) in enumerate(zip(ok_one+[avg_one], ok_fb+[avg_fb])):
        if (impr - base) > 0.02:
            ax.text(x_locs[i], impr + 0.02, f"+{(impr-base)*100:.0f}%", 
                    ha='center', fontsize=7, color=PALETTE["red"], fontweight='bold')

    ax.set_ylim(0.6, 1.1)
    ax.set_ylabel("Pass Rate")
    ax.set_xticks(x_locs)
    ax.set_xticklabels(x_labels)
    ax.set_title("Validated Recoverability")
    _panel_label(ax, "(a)")

    # --- Panel B: Accuracy ---
    ax = axes[1]
    w = 0.25
    
    ax.bar([i - w for i in x_locs], acc_rule + [avg_rule], width=w, color=PALETTE["grey"], label="Rule-only")
    ax.bar([i for i in x_locs], acc_one + [avg_one_acc], width=w, color=PALETTE["blue"], label="One-shot")
    ax.bar([i + w for i in x_locs], acc_fb + [avg_fb_acc], width=w, color=PALETTE["red"], label="Feedback")
    
    _draw_x_separator(ax, len(fes), 1.0)

    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Semantic Accuracy")
    ax.set_xticks(x_locs)
    ax.set_xticklabels(x_labels)
    ax.set_title("Semantic-Class Accuracy")
    _panel_label(ax, "(b)")

    # Unified Legend
    handles = [
        Patch(facecolor=PALETTE["grey"], label="Rule-only"),
        Patch(facecolor=PALETTE["blue"], label="One-shot"),
        Patch(facecolor=PALETTE["red"], label="Feedback"),
    ]
    _common_legend(fig, handles, [h.get_label() for h in handles], ncol=3)
    
    _save_fig(fig, out / "e1e3_recoverability.pdf")


def fig_e5_1_external_baseline(e5_1: dict[str, Any], out: Path) -> None:
    rows = [r for r in list(e5_1.get("per_kernel") or []) if isinstance(r, dict)]
    data = []
    for r in rows:
        vals = [r.get(k) for k in ["baseline_seconds_per_iter_t1", "ours_seconds_per_iter_t1", 
                                   "baseline_seconds_per_iter_t16", "ours_seconds_per_iter_t16"]]
        if all(isinstance(v, (int, float)) and v > 0 for v in vals):
            data.append({
                "name": str(r.get("name")),
                "base_t1": 1.0/float(vals[0]), "ours_t1": 1.0/float(vals[1]),
                "base_t16": 1.0/float(vals[2]), "ours_t16": 1.0/float(vals[3])
            })
    
    data.sort(key=lambda x: x["ours_t16"]/x["base_t16"])
    names = [d["name"] for d in data]
    y = np.arange(len(names))

    fig, axes = plt.subplots(2, 1, figsize=(3.8, 6.0), sharex=False)
    
    # [FIX] Adjusted top from 0.90 to 0.85
    plt.subplots_adjust(top=0.85, bottom=0.08, hspace=0.4, left=0.25)

    def _plot_h(ax, k_base, k_ours, title):
        base = [d[k_base] for d in data]
        ours = [d[k_ours] for d in data]
        speedups = [o/b for o, b in zip(ours, base)]
        
        h = 0.35
        ax.barh(y + h/2, base, height=h, color=PALETTE["grey"], label="Baseline")
        ax.barh(y - h/2, ours, height=h, color=PALETTE["red"], label="IntentIR (Ours)")
        
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(title, pad=5)
        ax.grid(axis='x', which='major', alpha=0.5)

        xmax = max(base + ours) if (base + ours) else 1.0
        ax.set_xlim(0.0, xmax * 1.25)
        
        for i, (v, sp) in enumerate(zip(ours, speedups)):
            ax.text(v + xmax * 0.03, i - h/2, f"{sp:.1f}x", va='center', fontsize=7,
                    color=PALETTE["red"], fontweight='bold')

    gm1 = e5_1.get("summary", {}).get("geom_speedup_ours_over_baseline_t1")
    title1 = f"Single-Thread (Geomean: {float(gm1):.2f}x)" if gm1 else "Single-Thread"
    _plot_h(axes[0], "base_t1", "ours_t1", title1)
    _panel_label(axes[0], "(a)")

    gm16 = e5_1.get("summary", {}).get("geom_speedup_ours_over_baseline_t16")
    title16 = f"16-Thread (Geomean: {float(gm16):.2f}x)" if gm16 else "16-Thread"
    _plot_h(axes[1], "base_t16", "ours_t16", title16)
    _panel_label(axes[1], "(b)")
    axes[1].set_xlabel("Throughput (iter/s)")

    handles = [Patch(facecolor=PALETTE["grey"], label="Triton-CPU LLVM baseline"),
               Patch(facecolor=PALETTE["red"], label="IntentIR (Ours)")]
    _common_legend(fig, handles, [h.get_label() for h in handles], ncol=2)

    _save_fig(fig, out / "e5_1_external_baseline.pdf")


def fig_e5_2_retune_vs_freeze(e5_2: dict[str, Any], out: Path) -> None:
    rows = [r for r in e5_2.get("per_kernel") or [] if isinstance(r, dict)]
    pairs = []
    for r in rows:
        if isinstance(r.get("speedup_paired"), (int, float)):
            pairs.append((str(r.get("anchor_tier", "D_none")), float(r["speedup_paired"])))
    
    order = ["A_dot", "B_reduce", "C_copy"]
    labels = {"A_dot": "GEMM", "B_reduce": "Reduce", "C_copy": "Copy"}
    
    by_tier = {t: [] for t in order}
    for t, s in pairs: 
        if t in by_tier: by_tier[t].append(s)
    
    gm_tier = [_geom_mean(by_tier[t]) or 0.0 for t in order]
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    
    # [FIX] Adjusted top from 0.85 to 0.78 (Short figure needs more relative legend space)
    plt.subplots_adjust(top=0.78, bottom=0.15)
    
    x = np.arange(len(order))
    
    ax.bar(x, gm_tier, width=0.5, color=PALETTE["red"], alpha=0.9, label="Tier Geomean")
    
    rng = np.random.default_rng(42)
    for i, t in enumerate(order):
        ys = by_tier[t]
        if ys:
            xs = i + rng.uniform(-0.1, 0.1, size=len(ys))
            ax.scatter(xs, ys, s=12, color=PALETTE["dark"], alpha=0.5, 
                       linewidths=0, zorder=3, label="Kernel" if i==0 else None)

    ax.axhline(1.0, color=PALETTE["grey"], linestyle="--", linewidth=1.5, label="Freeze (1.0x)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{labels[t]}\n(n={len(by_tier[t])})" for t in order])
    ax.set_ylabel("Speedup (Retune / Freeze)")
    ax.set_title("Portability: Retuning Benefit")
    ax.set_ylim(0, 2.4)
    
    handles = [
        Patch(facecolor=PALETTE["red"], label="Tier Geomean"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE["dark"], alpha=0.5, label='Kernel'),
        plt.Line2D([0], [0], color=PALETTE["grey"], linestyle='--', label='Freeze (1.0x)')
    ]
    _common_legend(fig, handles, [h.get_label() for h in handles], ncol=3, y_pos=0.99)
    
    _save_fig(fig, out / "e5_2_retune_vs_freeze.pdf")


def fig_e6_contract_calibration(e6: dict[str, Any], out: Path) -> None:
    ps = e6["summary"]["paper_summary"]
    ablations = ["full", "no_mask", "no_anchors"]
    x_labels = ["Full", "No Mask", "No Anchors"]
    levels = ["FULL", "PARTIAL", "OUT_OF_SCOPE"]
    c_map = {"FULL": PALETTE["green"], "PARTIAL": PALETTE["blue"], "OUT_OF_SCOPE": "#F0F0F0"}

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.5))
    
    # [FIX] Adjusted top from 0.92 to 0.84 (Significant reduction for 2-row legend safety)
    plt.subplots_adjust(top=0.84, bottom=0.08, hspace=0.45)

    # --- Panel A ---
    ax = axes[0]
    x = np.arange(len(ablations))
    w = 0.35
    
    for i, rep in enumerate(["intentir", "linalg"]):
        offset = -w/2 if rep == "intentir" else w/2
        bottoms = np.zeros(len(ablations))
        for lvl in levels:
            vals = []
            for ab in ablations:
                m = ps["by_rep_ablation"][rep][ab]
                n = float(m.get("n", 0))
                c = float(m.get("contract_levels", {}).get(lvl, 0))
                vals.append(c/n if n > 0 else 0)
            
            hatch = "/////" if rep == "linalg" else None
            edge = "white"
            ax.bar(x + offset, vals, width=w, bottom=bottoms, 
                   color=c_map[lvl], edgecolor=edge, linewidth=0.5, hatch=hatch)
            bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Fraction of Kernels")
    ax.set_title("Contract Distribution")
    _panel_label(ax, "(a)")

    # --- Panel B ---
    ax = axes[1]
    reps = ["intentir", "linalg"]
    vals = [float(ps["by_rep_ablation"][r]["full"]["full_false_accept_rate"]) for r in reps]
    
    bars = ax.bar(np.arange(len(reps)), vals, width=0.5, 
                  color=[PALETTE["red"], PALETTE["grey"]])
    bars[1].set_hatch("/////")
    bars[1].set_edgecolor("white")
    
    ax.set_xticks(np.arange(len(reps)))
    ax.set_xticklabels(["IntentIR", "Linalg"])
    ax.set_ylabel("False Accept Rate")
    ax.set_title("Calibration Error (Lower is Better)")
    ax.set_ylim(0, max(vals)*1.4)
    
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}", ha='center', fontsize=9)
    _panel_label(ax, "(b)")

    leg_handles = [
        Patch(facecolor=c_map["FULL"], label="Full"),
        Patch(facecolor=c_map["PARTIAL"], label="Partial"),
        Patch(facecolor=c_map["OUT_OF_SCOPE"], label="OOS"),
        Patch(facecolor="white", edgecolor="gray", label="IntentIR"),
        Patch(facecolor="white", edgecolor="gray", hatch="/////", label="Linalg"),
    ]
    _common_legend(fig, leg_handles, [h.get_label() for h in leg_handles], ncol=3, y_pos=0.99)

    _save_fig(fig, out / "e6_contract_calibration.pdf")


def fig_e2_trust_ablation(e2: dict[str, Any], out: Path) -> None:
    frontends = ["triton", "tilelang", "cuda"]
    modes = ["diff_only", "generic", "full"]
    colors = ["#C4D4E0", "#8FA9BF", PALETTE["red"]]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    
    # [FIX] Adjusted top from 0.85 to 0.78 (Very short figure needs room)
    plt.subplots_adjust(top=0.78)

    x = np.arange(len(frontends) + 1)
    w = 0.22

    for i, m in enumerate(modes):
        vals = []
        for fe in frontends:
            d = e2["by_frontend"][fe]["summary"]["aggregate"][m]
            vals.append(d["killed"] / d["total"] if d["total"] > 0 else 0)
        tot, kill = 0, 0
        for fe in frontends:
            d = e2["by_frontend"][fe]["summary"]["aggregate"][m]
            tot += d["total"]; kill += d["killed"]
        vals.append(kill / tot if tot > 0 else 0)
        
        ax.bar(x + (i-1)*w, vals, width=w, label=m.replace("_", "-").title(), 
               color=colors[i], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in frontends] + ["All"])
    ax.set_ylabel("Mutation Kill Rate")
    ax.set_title("Trustworthiness Ablation")
    ax.set_ylim(0, 1.05)
    
    _common_legend(fig, *ax.get_legend_handles_labels(), ncol=3)
    _save_fig(fig, out / "e2_trust_ablation.pdf")


def fig_e4_consistency(e4: dict[str, Any], out: Path) -> None:
    s = e4["summary"]
    # We report a "consistency vs strictness" view:
    # - canonical: exact equality under a portable normalization (drops schedule hints + size specialization)
    # - structural: order-insensitive graph hash + interface compatibility
    # - roles: axis-role recall on labeled subset
    labels_a = ["Canonical match", "Structural match", "Axis-role recall"]
    intent = [s["intent_ok_rate"], s["intent_structural_ok_rate"], s["axis_roles_recall_intent_avg"]]
    expanded = [s["expanded_ok_rate"], s["expanded_structural_ok_rate"], s["axis_roles_recall_expanded_avg"]]
    
    r_int = s.get("top_reasons_intent", {})
    r_exp = s.get("top_reasons_expanded", {})
    keys = sorted(set(r_int)|set(r_exp), key=lambda k: max(r_int.get(k,0), r_exp.get(k,0)), reverse=True)[:4]
    
    fig, axes = plt.subplots(2, 1, figsize=(3.8, 5.0))
    
    # [FIX] Adjusted top from 0.90 to 0.85
    plt.subplots_adjust(top=0.85, bottom=0.10, hspace=0.5, left=0.3)

    def _do_barh(ax, y_labels, v1, v2, title, panel):
        y = np.arange(len(y_labels))
        h = 0.35
        ax.barh(y - h/2, v1, height=h, color=PALETTE["blue"], label="Macro-level")
        ax.barh(y + h/2, v2, height=h, color="#D99F94", label="Expanded primitives")
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()
        ax.set_title(title)
        _panel_label(ax, panel)

    _do_barh(axes[0], labels_a, intent, expanded, "Consistency at Macro vs Primitive Levels", "(a)")
    axes[0].set_xlim(0, 1.15)
    for i, v in enumerate(intent): axes[0].text(v+0.02, i-0.17, f"{v:.1%}", fontsize=8, va='center')
    for i, v in enumerate(expanded): axes[0].text(v+0.02, i+0.17, f"{v:.1%}", fontsize=8, va='center')

    clean_keys = [k.replace("tilelang:", "").replace("_", " ").capitalize() for k in keys]
    _do_barh(axes[1], clean_keys, [r_int.get(k,0) for k in keys], [r_exp.get(k,0) for k in keys], "Benign Drift Sources", "(b)")
    axes[1].set_xlabel("Count (out of 30)")

    handles = [Patch(facecolor=PALETTE["blue"], label="Macro-level"),
               Patch(facecolor="#D99F94", label="Expanded primitives")]
    _common_legend(fig, handles, [h.get_label() for h in handles], ncol=2)

    _save_fig(fig, out / "e4_consistency.pdf")


def fig_e5_cuda_triton_vs_intentir(
    *,
    quick: dict[str, Any],
    ablation: dict[str, Any] | None,
    out_dir: Path,
    out_name: str,
) -> None:
    """
    E5 CUDA (GPU): Triton vs IntentIR-CUDA (quick), plus ablations if available.

    `quick` is expected to be the default perf run (`--bench-mode graph`, `CUDA_SPECIALIZE_DIMS=1`).
    `ablation` is expected to come from `--ablation --ablation-modes evidence_on,dispatch_off,contract_off`.
    """

    order = [
        "ai_bench_matmul",
        "ai_bench_dropout",
        "ai_bench_softmax",
        "ai_bench_layernorm",
        "ai_bench_correlation",
        "ai_bench_resize",
        "ai_bench_rope",
        "ai_bench_warp",
    ]
    pretty = {
        "ai_bench_matmul": "MatMul",
        "ai_bench_dropout": "Dropout",
        "ai_bench_softmax": "Softmax",
        "ai_bench_layernorm": "LayerNorm",
        "ai_bench_correlation": "Correlation",
        "ai_bench_resize": "Resize",
        "ai_bench_rope": "RoPE",
        "ai_bench_warp": "Warp",
    }

    def _by_kernel(obj: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        if not isinstance(obj, dict):
            return {}
        rows = [r for r in list(obj.get("results") or []) if isinstance(r, dict)]
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            k = r.get("kernel")
            if isinstance(k, str) and k:
                out[k] = r
        return out

    def _get_speedup(r: dict[str, Any] | None, key: str) -> float | None:
        if not isinstance(r, dict):
            return None
        v = r.get(key)
        if isinstance(v, (int, float)):
            vf = float(v)
            return vf if vf > 0 else None
        return None

    by_q = _by_kernel(quick)
    by_a = _by_kernel(ablation)

    kernels = [k for k in order if k in by_q]
    labels = [pretty.get(k, k) for k in kernels]
    x = np.arange(len(kernels))

    # Build series (speedup vs Triton, so Triton itself is 1.0×).
    sp_quick = [_get_speedup(by_q.get(k), "speedup_ours_over_triton") for k in kernels]
    sp_dispatch_off = [_get_speedup(by_a.get(k), "speedup_ours_dispatch_off_over_triton") for k in kernels]
    sp_contract_off = [_get_speedup(by_a.get(k), "speedup_ours_contract_off_over_triton") for k in kernels]

    have_dispatch_off = any(v is not None for v in sp_dispatch_off)
    have_contract_off = any(v is not None for v in sp_contract_off)

    series: list[tuple[str, list[float | None], str]] = [("IntentIR-CUDA (quick)", sp_quick, PALETTE["red"])]
    if have_dispatch_off:
        series.append(("Host dispatch off", sp_dispatch_off, PALETTE["blue"]))
    if have_contract_off:
        series.append(("Contract off", sp_contract_off, PALETTE["light_grey"]))

    n_series = len(series)
    width = 0.22 if n_series <= 3 else 0.18
    n_series = len(series)
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * width

    all_vals: list[float] = []
    series_vals: list[np.ndarray] = []
    for _, vals_raw, _ in series:
        vals = np.array([float(v) if isinstance(v, (int, float)) else np.nan for v in vals_raw], dtype=np.float64)
        series_vals.append(vals)
        if np.any(np.isfinite(vals)):
            all_vals.extend([float(x) for x in vals[np.isfinite(vals)].tolist()])

    vmax = max(all_vals) if all_vals else 1.0
    vmin = min(all_vals) if all_vals else 1.0

    gpu = None
    try:
        meta = quick.get("meta") if isinstance(quick.get("meta"), dict) else {}
        gpu = meta.get("gpu")
    except Exception:
        gpu = None

    gm = None
    try:
        s = quick.get("summary") if isinstance(quick.get("summary"), dict) else {}
        gm = s.get("geom_speedup_ours_over_triton")
    except Exception:
        gm = None

    title = str(gpu) if isinstance(gpu, str) and gpu else "CUDA GPU"
    if isinstance(gm, (int, float)):
        title = f"{title} (Geomean: {float(gm):.2f}×)"

    # Figure: vertical bars + broken y-axis (not log) to keep 1× details readable.
    y_break = 1.30
    y_lo = max(0.0, min(0.85, vmin * 0.95))
    use_broken_axis = vmax > (y_break * 1.20)

    if use_broken_axis:
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(6.8, 3.4),
            gridspec_kw={"height_ratios": [1, 2]},
        )
        plt.subplots_adjust(top=0.80, bottom=0.27, left=0.10, right=0.99, hspace=0.05)
        ax_top.set_title(title)

        for i, (name, _, color) in enumerate(series):
            vals = series_vals[i]
            pos = x + offsets[i]
            ax_bot.bar(pos, vals, width=width * 0.90, color=color, edgecolor="white", linewidth=0.5, label=name)
            ax_top.bar(pos, vals, width=width * 0.90, color=color, edgecolor="white", linewidth=0.5)

        for ax in (ax_top, ax_bot):
            ax.axhline(1.0, color=PALETTE["dark"], linestyle="--", linewidth=1.0, alpha=0.8)
            ax.grid(axis="y", which="major", alpha=0.5)

        ax_bot.set_ylim(y_lo, y_break)
        ax_top.set_ylim(y_break, max(y_break + 0.05, vmax * 1.10))

        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(labeltop=False)
        ax_bot.xaxis.tick_bottom()

        # Diagonal marks to indicate the axis break.
        d = 0.008
        kwargs = dict(transform=ax_top.transAxes, color=PALETTE["dark"], clip_on=False, linewidth=0.9)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs = dict(transform=ax_bot.transAxes, color=PALETTE["dark"], clip_on=False, linewidth=0.9)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        # Labels / ticks.
        ax_bot.set_ylabel("Speedup over Triton (×)")
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(labels, rotation=20, ha="right")

        # Annotate very large outliers on the top axis (keeps the plot readable).
        for i, (_, _, _) in enumerate(series):
            vals = series_vals[i]
            for j in range(len(kernels)):
                v = vals[j]
                if not np.isfinite(v) or v <= y_break:
                    continue
                ax_top.text(
                    float(x[j] + offsets[i]),
                    float(v) + 0.02 * float(vmax),
                    f"{float(v):.1f}×",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=PALETTE["dark"],
                )

        _common_legend(fig, *ax_bot.get_legend_handles_labels(), ncol=min(3, n_series), y_pos=0.99)
        _save_fig(fig, out_dir / f"{out_name}.pdf")
    else:
        fig, ax = plt.subplots(figsize=(6.8, 2.6))
        plt.subplots_adjust(top=0.80, bottom=0.27, left=0.10, right=0.99)
        ax.set_title(title)
        for i, (name, _, color) in enumerate(series):
            vals = series_vals[i]
            ax.bar(x + offsets[i], vals, width=width * 0.90, color=color, edgecolor="white", linewidth=0.5, label=name)
        ax.axhline(1.0, color=PALETTE["dark"], linestyle="--", linewidth=1.0, alpha=0.8)
        ax.grid(axis="y", which="major", alpha=0.5)
        ax.set_ylabel("Speedup over Triton (×)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(y_lo, max(1.2, vmax * 1.10))
        _common_legend(fig, *ax.get_legend_handles_labels(), ncol=min(3, n_series), y_pos=0.99)
        _save_fig(fig, out_dir / f"{out_name}.pdf")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-json-dir", type=Path, default=ROOT / "artifacts/experiments/paper")
    ap.add_argument("--paper-dir", type=Path, default=ROOT / "doc/paper/my-sigconf-paper")
    ap.add_argument("--e5-cuda-4080s-quick", type=Path, default=None)
    ap.add_argument("--e5-cuda-4080s-ablation", type=Path, default=None)
    ap.add_argument("--e5-cuda-h100-quick", type=Path, default=None)
    ap.add_argument("--e5-cuda-h100-ablation", type=Path, default=None)
    args = ap.parse_args()

    pj = args.paper_json_dir
    fig_dir = args.paper_dir / "fig"

    # Load
    try:
        e1 = _load_json(pj / "e1_rule_only.paper.json")
        e1e3 = _load_json(pj / "e1e3_llm_regression.paper.json")
        e2 = _load_json(pj / "e2_trust_ablation.paper.json")
        e4 = _load_json(pj / "e4_cross_frontend_consistency.paper.json")
        e5_1 = _load_json(pj / "e5_1_external_baseline.paper.json")
        e5_2 = _load_json(pj / "e5_2_portability_vs_perf.paper.json")
        e6 = _load_json(pj / "e6_ir_usability.paper.json")
    except FileNotFoundError as e:
        print(f"Data not found: {e}"); return

    _set_pro_style()

    print("Generating figures with STRICT alignment...")
    fig_e1e3_recoverability(e1, e1e3, fig_dir)
    fig_e2_trust_ablation(e2, fig_dir)
    fig_e4_consistency(e4, fig_dir)
    fig_e5_1_external_baseline(e5_1, fig_dir)
    fig_e5_2_retune_vs_freeze(e5_2, fig_dir)
    fig_e6_contract_calibration(e6, fig_dir)

    # Optional: E5 CUDA GPU figures (H100 + 4080S).
    cuda_dir = ROOT / "artifacts" / "experiments" / "E5"
    p_4080_q = args.e5_cuda_4080s_quick or _latest_file(cuda_dir, "e5_cuda_4080s*quick*.json")
    p_4080_a = args.e5_cuda_4080s_ablation or _latest_file(cuda_dir, "e5_cuda_4080s*ablation*.json")
    p_h100_q = args.e5_cuda_h100_quick or _latest_file(cuda_dir, "e5_cuda_h100*quick*.json")
    p_h100_a = args.e5_cuda_h100_ablation or _latest_file(cuda_dir, "e5_cuda_h100*ablation*.json")

    if p_h100_q and p_h100_q.is_file():
        h100_q = _load_json(p_h100_q)
        h100_a = _load_json(p_h100_a) if (p_h100_a and p_h100_a.is_file()) else None
        fig_e5_cuda_triton_vs_intentir(quick=h100_q, ablation=h100_a, out_dir=fig_dir, out_name="e5_cuda_gpu_h100")

    if p_4080_q and p_4080_q.is_file():
        g4080_q = _load_json(p_4080_q)
        g4080_a = _load_json(p_4080_a) if (p_4080_a and p_4080_a.is_file()) else None
        fig_e5_cuda_triton_vs_intentir(quick=g4080_q, ablation=g4080_a, out_dir=fig_dir, out_name="e5_cuda_gpu_4080s")
    
    print(f"Done. Figures written to: {fig_dir}")

if __name__ == "__main__":
    main()
