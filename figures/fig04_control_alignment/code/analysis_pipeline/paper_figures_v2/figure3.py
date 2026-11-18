"""Figure 3. Entropy and functional alignment. All policies are drawn; the contrast is DNE versus EIG."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis_pipeline.paper_figures_v2.entropy_matched import analyse, plot_on_axis
from analysis_pipeline.paper_figures_v2.plot_style import (
    COLORS, ENV_LABEL, POLICIES, add_legend, draw_curves, figure_title, forest, panel_label,
)

CONDITIONS = ("full", "full_trap", "noisy")


def _rows(bundle, endpoint):
    rows = []
    for env in CONDITIONS:
        est, lo, hi = bundle["diffs"][(env, endpoint, "eig")]
        rows.append({"label": ENV_LABEL[env], "estimate": est, "lo": lo, "hi": hi, "color": COLORS["dne"]})
    return rows


def _bin_means(bundle, n_bins=24, min_count=8):
    """Previous Figure 3B: equal-width bins of functional MAE, retained for that archived analysis."""
    pooled = np.concatenate([bundle["raw"]["full"][pol]["func"].ravel() for pol in POLICIES])
    edges = np.linspace(float(pooled.min()), float(pooled.max()), n_bins + 1)
    out = {}
    for pol in POLICIES:
        func = bundle["raw"]["full"][pol]["func"].ravel()
        entropy = bundle["raw"]["full"][pol]["entropy"].ravel()
        xs, ys = [], []
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (func >= edges[i]) & (func <= edges[i + 1])
            else:
                mask = (func >= edges[i]) & (func < edges[i + 1])
            if int(mask.sum()) < min_count:
                continue
            xs.append(float(func[mask].mean()))
            ys.append(float(entropy[mask].mean()))
        out[pol] = (np.asarray(xs), np.asarray(ys))
    return out


def draw(bundle):
    fig = plt.figure(figsize=(7.45, 7.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], hspace=0.36, wspace=0.34, left=0.10, right=0.98, top=0.90, bottom=0.14)
    figure_title(fig, "Figure 3. Posterior entropy and target alignment")
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_curves(ax_a, bundle, "full", "entropy", ymin=None)
    ax_a.set_ylabel("Posterior entropy (nats)")
    ax_a.set_title("Posterior entropy, Full", loc="left", fontsize=8)
    panel_label(ax_a, "A")

    plot_on_axis(ax_b, analyse(bundle)["summary"], POLICIES)
    panel_label(ax_b, "B")

    forest(ax_c, _rows(bundle, "entropy_auc"))
    ax_c.set_title("Entropy difference", loc="left", fontsize=8)
    ax_c.set_xlabel("Entropy AUC, DNE − EIG")
    ax_c.set_xlim(-18, None)
    panel_label(ax_c, "C")
    forest(ax_d, _rows(bundle, "func_auc"))
    ax_d.set_title("Functional difference", loc="left", fontsize=8)
    ax_d.set_xlabel("Functional AUC, DNE − EIG")
    ax_d.set_xlim(None, 1.0)
    panel_label(ax_d, "D")
    add_legend(fig, y=0.07, ncol=5)
    return fig
