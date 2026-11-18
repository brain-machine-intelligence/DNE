"""Shared appearance for the four main manuscript figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

COLORS = {
    "dne": "#087F78",
    "eig": "#3776B6",
    "boed_h3": "#8863B5",
    "tiny_rnn_target": "#D19A38",
    "thompson": "#C65B52",
    "random": "#858B91",
    "bald": "#5C8FA3",
    "boed_h2": "#B089A8",
    "uncertainty": "#7E9A72",
    "tiny_rnn_info": "#C4A36A",
}
LABELS = {
    "dne": "DNE",
    "eig": "EIG",
    "boed_h3": "BOED-H3",
    "tiny_rnn_target": "RNN-target",
    "thompson": "Thompson",
    "random": "Random",
    "bald": "BALD",
    "boed_h2": "BOED-H2",
    "uncertainty": "Uncertainty",
    "tiny_rnn_info": "RNN-info",
}
# Solid lines are the pre-specified comparison set. Dashed lines are the remaining policies.
LS = {
    "dne": "-",
    "eig": "-",
    "boed_h3": "-",
    "tiny_rnn_target": "-",
    "thompson": "-",
    "random": "-",
    "bald": (0, (4, 1.4)),
    "boed_h2": (0, (1.2, 1.1)),
    "uncertainty": (0, (5, 1.2, 1, 1.2)),
    "tiny_rnn_info": (0, (2.2, 1.0)),
}
LW = {k: (1.55 if k == "dne" else 1.15 if k in ("eig", "boed_h3", "tiny_rnn_target", "thompson", "random") else 1.05) for k in COLORS}
POLICIES = ["dne", "eig", "boed_h3", "tiny_rnn_target", "thompson", "random", "bald", "boed_h2", "uncertainty", "tiny_rnn_info"]
PRIMARY = ["dne", "eig", "boed_h3", "tiny_rnn_target", "thompson", "random"]
SECONDARY = ["bald", "boed_h2", "uncertainty", "tiny_rnn_info"]
ENV_LABEL = {
    "full": "Full",
    "full_trap": "Full with spatial logging",
    "noisy": "Noisy",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "text.color": "#222222",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "legend.frameon": False,
    })


def figure_title(fig, text: str) -> None:
    fig.text(0.02, 0.995, text, fontsize=11, fontweight="bold", ha="left", va="top", color="#111111")


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.02) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left", color="#111111")


def finish_trial_axis(ax) -> None:
    ax.set_xlim(0, 199)
    ax.set_xlabel("Trial")


def draw_order(policies):
    return [p for p in policies if p != "dne"] + ["dne"]


def draw_curves(ax, bundle, env, metric, policies=None, ymin=0):
    policies = list(policies or POLICIES)
    x = np.arange(200)
    for pol in draw_order(policies):
        mean, lo, hi = bundle["series"][(env, pol, metric)]
        ax.fill_between(x, lo, hi, color=COLORS[pol], alpha=0.12, linewidth=0, zorder=1)
        ax.plot(x, mean, color=COLORS[pol], lw=LW[pol], ls=LS[pol], zorder=3 if pol != "dne" else 4)
    finish_trial_axis(ax)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)


def legend_handles(policies=None):
    policies = policies or POLICIES
    return [
        Line2D([0], [0], color=COLORS[p], lw=LW[p], ls=LS[p], label=LABELS[p])
        for p in policies
    ]


def add_legend(fig, policies=None, y=0.01, ncol=5):
    fig.legend(
        handles=legend_handles(policies),
        loc="upper center", bbox_to_anchor=(0.5, y),
        ncol=ncol, frameon=False, handlelength=2.4, columnspacing=1.0,
    )


def forest(ax, rows):
    """rows: list of dicts with label, estimate, lo, hi, color."""
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color="#222222", lw=0.7, zorder=0)
    for i, row in enumerate(rows):
        yy = y[i]
        ax.plot([row["lo"], row["hi"]], [yy, yy], color=row["color"], lw=1.25, solid_capstyle="butt", zorder=2)
        ax.scatter([row["estimate"]], [yy], s=22, color=row["color"], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
