#!/usr/bin/env python3
"""Clean perturbation tables and the Post-attention LSTM intervention map.

The map uses raw mean ± SEM from the same seed-level losses as the tests.
Normalized scores are not plotted.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

V3 = Path(__file__).resolve().parents[2]
SRC = V3 / "figure2_dual_metric_perturbation" / "data"
STATS = Path(__file__).resolve().parents[1] / "data" / "Perturbation_raw_statistics.xlsx"
OUT_DIR = Path(__file__).resolve().parents[1]
FIG = OUT_DIR / "figures"
DATA = OUT_DIR / "data"

FOCAL = "lstm_attn"
POINTS = [
    ("shuffled", "attention", "1. shuffled", "#B03A2E"),
    ("uniform", "attention", "2. uniform", "#B03A2E"),
    ("no past action", "channel", "3. no past action", "#7FB3D5"),
    ("no past reward", "channel", "4. no past reward", "#7FB3D5"),
    ("temporal-order shuffled", "order", "5. temporal-order shuffled", "#1A5276"),
    ("stimulus only (A+R removed)", "channel", "6. stimulus only", "#7FB3D5"),
]
MODEL_ORDER = [
    "Transformer",
    "Pre-attention BiLSTM",
    "Pre-attention LSTM",
    "BiLSTM",
    "LSTM",
    "Post-attention BiLSTM",
    "Post-attention LSTM",
]
COND_ORDER = [p[0] for p in POINTS]


def summarise(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    sd = float(np.std(v, ddof=1)) if n > 1 else np.nan
    return {
        "mean": float(np.mean(v)) if n else np.nan,
        "sd": sd,
        "sem": sd / np.sqrt(n) if n > 1 else np.nan,
        "n": n,
    }


def supported(g: pd.DataFrame) -> pd.DataFrame:
    if "valid_intervention" not in g.columns:
        return g
    ok = g["valid_intervention"].astype(str).str.lower().isin(["true", "1", "yes"])
    return g.loc[ok]


def focal_seeds() -> dict[str, dict[str, np.ndarray]]:
    attn = pd.read_csv(SRC / "Attn_raw.csv")
    chan = pd.read_csv(SRC / "Channel_raw.csv")
    order = pd.read_csv(SRC / "Order_raw.csv")
    out = {}
    sub = attn[attn.architecture.eq(FOCAL)]
    for cond in ("uniform", "shuffled"):
        g = supported(sub[sub.attention_condition.eq(cond)])
        out[cond] = {
            "prediction": g["Prediction Loss"].to_numpy(float),
            "profile": g["Agent Accuracy Loss"].to_numpy(float),
            "seed": g["seed"].to_numpy(int),
        }
    key_to_label = {
        "action": "no past action",
        "reward": "no past reward",
        "stimulus": "stimulus only (A+R removed)",
    }
    for key, label in key_to_label.items():
        g = chan[(chan.architecture.eq(FOCAL)) & (chan.channel.eq(key))]
        out[label] = {
            "prediction": g["Prediction Loss"].to_numpy(float),
            "profile": g["Agent Accuracy Loss"].to_numpy(float),
            "seed": g["seed"].to_numpy(int),
        }
    g = order[order.architecture.eq(FOCAL)]
    out["temporal-order shuffled"] = {
        "prediction": g["Prediction Loss"].to_numpy(float),
        "profile": g["Agent Accuracy Loss"].to_numpy(float),
        "seed": g["seed"].to_numpy(int),
    }
    return out


def map_tables(seeds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    seed_rows = []
    for i, (cond, family, label, _color) in enumerate(POINTS, start=1):
        pred = summarise(seeds[cond]["prediction"])
        prof = summarise(seeds[cond]["profile"])
        summary_rows.append(
            {
                "point": i,
                "label": label,
                "family": family,
                "condition": cond,
                "architecture": "Post-attention LSTM",
                "prediction_mean": pred["mean"],
                "prediction_sd": pred["sd"],
                "prediction_sem": pred["sem"],
                "prediction_n": pred["n"],
                "profile_mean": prof["mean"],
                "profile_sd": prof["sd"],
                "profile_sem": prof["sem"],
                "profile_n": prof["n"],
            }
        )
        for seed, a, p in zip(seeds[cond]["seed"], seeds[cond]["prediction"], seeds[cond]["profile"]):
            seed_rows.append(
                {
                    "point": i,
                    "label": label,
                    "family": family,
                    "condition": cond,
                    "seed": int(seed),
                    "prediction_loss": float(a),
                    "profile_loss": float(p),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(seed_rows)


def ordered(df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    df = df.copy()
    df["_m"] = pd.Categorical(df[model_col], MODEL_ORDER, ordered=True)
    df["_c"] = pd.Categorical(df["condition"], COND_ORDER, ordered=True)
    df["_metric"] = pd.Categorical(df["metric"], ["prediction", "profile"], ordered=True)
    return df.sort_values(["_c", "_metric", "_m"]).drop(columns=["_m", "_c", "_metric"])


def clean_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    vs0 = pd.read_excel(STATS, sheet_name="vs_zero")
    vsf = pd.read_excel(STATS, sheet_name="vs_PostAttnLSTM")
    zero = ordered(vs0, "architecture_display")[
        [
            "family",
            "condition",
            "metric",
            "architecture_display",
            "mean",
            "sem",
            "n",
            "t_stat",
            "df",
            "p_one_sided_mean_gt_0",
            "p_one_sided_Holm",
            "p_two_sided",
            "p_two_sided_Holm",
            "holm_family_size",
            "significant_one_sided_Holm_0.05",
        ]
    ].rename(columns={"architecture_display": "model"})
    pair = ordered(vsf, "comparison_display")[
        [
            "family",
            "condition",
            "metric",
            "comparison_display",
            "other_mean",
            "other_sem",
            "other_n",
            "focal_mean",
            "focal_sem",
            "focal_n",
            "mean_difference_other_minus_focal",
            "welch_t_stat",
            "welch_df",
            "p_one_sided_other_gt_focal",
            "p_one_sided_Holm",
            "p_two_sided",
            "p_two_sided_Holm",
            "holm_family_size",
            "significant_one_sided_Holm_0.05",
        ]
    ].rename(
        columns={
            "comparison_display": "model",
            "mean_difference_other_minus_focal": "difference_model_minus_post_attn_lstm",
        }
    )
    return zero, pair


def write_sheet(wb: Workbook, name: str, frame: pd.DataFrame, first: bool = False):
    ws = wb.active if first else wb.create_sheet(name[:31])
    ws.title = name[:31]
    for r in dataframe_to_rows(frame, index=False, header=True):
        ws.append(list(r))
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9E2F3")
        cell.alignment = Alignment(wrap_text=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 32


def plot_map(summary: pd.DataFrame):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    offsets = {
        1: (8, 8),
        2: (8, -14),
        3: (8, -12),
        4: (8, 8),
        5: (8, 8),
        6: (8, 8),
    }
    for rec in summary.itertuples(index=False):
        color = POINTS[rec.point - 1][3]
        ax.errorbar(
            rec.profile_mean,
            rec.prediction_mean,
            xerr=rec.profile_sem,
            yerr=rec.prediction_sem,
            fmt="o",
            color=color,
            ms=8,
            capsize=3,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            str(rec.point),
            (rec.profile_mean, rec.prediction_mean),
            textcoords="offset points",
            xytext=offsets[rec.point],
            fontsize=9,
            color=color,
            fontweight="medium",
        )
    ax.axvline(0, color="#888888", ls="--", lw=0.8, zorder=1)
    ax.axhline(0, color="#888888", ls="--", lw=0.8, zorder=1)
    ax.set_xlabel("Profile loss")
    ax.set_ylabel("Prediction loss")
    ax.set_title("D. Asymmetric intervention map\nPost-attention LSTM", loc="left", fontsize=12)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="white", markersize=8, label=lab)
        for _cond, _fam, lab, c in POINTS
    ]
    ax.legend(handles=handles, frameon=True, fontsize=8, loc="center right", bbox_to_anchor=(1.02, 0.55))
    ax.text(
        0.0,
        -0.16,
        "Points are seed means. Bars are SEM. N = 20. Raw loss, not the normalized score.",
        transform=ax.transAxes,
        fontsize=8,
        style="italic",
        color="#555555",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    stem = FIG / "PostAttentionLSTM_asymmetric_intervention_map"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem


def main():
    seeds = focal_seeds()
    summary, seed_long = map_tables(seeds)
    zero, pair = clean_stats()
    methods = pd.DataFrame(
        [
            ("map_model", "Post-attention LSTM only"),
            ("map_y", "Prediction loss = native accuracy - perturbed accuracy"),
            ("map_x", "Profile loss = native within-task agent decoding - perturbed decoding"),
            ("map_point", "Mean across 20 seeds"),
            ("map_errorbar", "SEM, sd / sqrt(N), ddof=1"),
            ("map_source", str(SRC)),
            ("stats_unit", "Seed-level raw loss. Normalized values are not tested or plotted."),
            ("stats_vs_zero", "One-sample t-test, alternative mean > 0. Holm within condition x metric."),
            ("stats_vs_focal", "Unpaired Welch t-test. One-sided alternative: the named model has a larger raw loss than Post-attention LSTM."),
            ("stats_difference", "model mean - Post-attention LSTM mean. Negative means Post-attention LSTM lost more."),
            ("excluded", "Transformer shuffled attention is undefined and is omitted."),
        ],
        columns=["item", "description"],
    )
    DATA.mkdir(parents=True, exist_ok=True)
    xlsx = DATA / "PostAttentionLSTM_intervention_map_and_statistics.xlsx"
    wb = Workbook()
    write_sheet(wb, "Methods", methods, first=True)
    write_sheet(wb, "Map_summary", summary)
    write_sheet(wb, "Map_seeds", seed_long)
    write_sheet(wb, "Stats_vs_zero", zero)
    write_sheet(wb, "Stats_vs_PostAttnLSTM", pair)
    wb.save(xlsx)
    stem = plot_map(summary)
    print(summary.round(4).to_string(index=False))
    print(stem)
    print(xlsx)


if __name__ == "__main__":
    main()
