"""Figure 2. Functional and structural alignment in Full, all ten policies."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis_pipeline.paper_figures_v2.plot_style import (
    COLORS, LABELS, POLICIES, add_legend, draw_curves, figure_title, forest, panel_label,
)

PARAMS = (("wx", r"$w_x$"), ("wy", r"$w_y$"), ("b", r"$b$"))


def draw(bundle):
    fig = plt.figure(figsize=(7.45, 8.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.32, left=0.09, right=0.98, top=0.90, bottom=0.14)
    figure_title(fig, "Figure 2. Functional and structural alignment in Full")
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    inner = gs[1, 0].subgridspec(1, 2, wspace=0.45)
    ax_cf = fig.add_subplot(inner[0, 0])
    ax_cs = fig.add_subplot(inner[0, 1])
    dgs = gs[1, 1].subgridspec(1, 3, wspace=0.12)
    daxes = [fig.add_subplot(dgs[0, i]) for i in range(3)]

    draw_curves(ax_a, bundle, "full", "func")
    ax_a.set_ylabel("Functional MAE")
    ax_a.set_title("Functional alignment", loc="left", fontsize=8)
    panel_label(ax_a, "A")

    draw_curves(ax_b, bundle, "full", "struct")
    ax_b.set_ylabel(r"Structural distance $\|\hat\theta-\theta^\star\|_2$")
    ax_b.set_title("Structural alignment", loc="left", fontsize=8)
    panel_label(ax_b, "B")

    rows_f, rows_s = [], []
    for pol in POLICIES:
        est, lo, hi = bundle["levels"][("full", pol, "auc_func")]
        rows_f.append({"label": LABELS[pol], "estimate": est, "lo": lo, "hi": hi, "color": COLORS[pol]})
        est, lo, hi = bundle["levels"][("full", pol, "auc_struct")]
        rows_s.append({"label": LABELS[pol], "estimate": est, "lo": lo, "hi": hi, "color": COLORS[pol]})
    forest(ax_cf, rows_f)
    forest(ax_cs, rows_s)
    ax_cs.set_yticklabels([])
    ax_cf.set_xlabel("Functional AUC\n(lower is better)")
    ax_cs.set_xlabel("Structural AUC\n(lower is better)")
    ax_cf.set_title("Integrated AUC", loc="left", fontsize=8)
    panel_label(ax_cf, "C", dx=-0.45)

    y = np.arange(len(POLICIES))[::-1]
    for ax, (key, name), letter in zip(daxes, PARAMS, ("D", "", "")):
        xs, los, his = [], [], []
        for pol in POLICIES:
            est, lo, hi = bundle["levels"][("full", pol, f"abs_{key}")]
            xs.append(est)
            los.append(lo)
            his.append(hi)
        xs, los, his = map(np.asarray, (xs, los, his))
        ax.errorbar(xs, y, xerr=[xs - los, his - xs], fmt="none", ecolor="#8A8A8A", lw=0.7, capsize=0)
        ax.scatter(xs, y, c=[COLORS[p] for p in POLICIES], s=16, zorder=3, linewidths=0)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[p] for p in POLICIES] if key == "wx" else [])
        ax.set_xlabel(f"Absolute error\n{name}")
        ax.set_xlim(left=0)
        if letter:
            panel_label(ax, letter, dx=-0.62)
    daxes[0].set_title("Final absolute error", loc="left", fontsize=8)
    add_legend(fig, y=0.075, ncol=5)
    return fig
