"""Figure 4. Noisy observations. All ten policies."""
from __future__ import annotations

import matplotlib.pyplot as plt

from analysis_pipeline.paper_figures_v2.plot_style import (
    COLORS, LABELS, POLICIES, add_legend, draw_curves, figure_title, forest, panel_label,
)


def draw(bundle):
    fig = plt.figure(figsize=(7.45, 8.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.15], hspace=0.38, wspace=0.36, left=0.10, right=0.98, top=0.90, bottom=0.13)
    figure_title(fig, "Figure 4. Alignment under unmodelled label flips")
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_curves(ax_a, bundle, "noisy", "func")
    ax_a.set_ylabel("Functional MAE")
    ax_a.set_title("Functional MAE, Noisy", loc="left", fontsize=8)
    panel_label(ax_a, "A")

    draw_curves(ax_b, bundle, "noisy", "struct")
    ax_b.set_ylabel(r"Structural distance $\|\hat\theta-\theta^\star\|_2$")
    ax_b.set_title("Structural distance, Noisy", loc="left", fontsize=8)
    panel_label(ax_b, "B")

    rows = []
    for pol in POLICIES:
        if pol == "dne":
            continue
        est, lo, hi = bundle["diffs"][("noisy", "func_auc", pol)]
        rows.append({"label": LABELS[pol], "estimate": est, "lo": lo, "hi": hi, "color": COLORS[pol]})
    forest(ax_c, rows)
    ax_c.set_title("Functional AUC differences", loc="left", fontsize=8)
    ax_c.set_xlabel("Functional AUC, DNE − comparator")
    panel_label(ax_c, "C")

    rows = []
    for pol in POLICIES:
        est, lo, hi = bundle["changes"][(pol, "func_final")]
        rows.append({"label": LABELS[pol], "estimate": est, "lo": lo, "hi": hi, "color": COLORS[pol]})
    forest(ax_d, rows)
    ax_d.set_title("Change from Full to Noisy", loc="left", fontsize=8)
    ax_d.set_xlabel("Final functional MAE, Noisy − Full")
    panel_label(ax_d, "D")
    add_legend(fig, y=0.065, ncol=5)
    return fig
