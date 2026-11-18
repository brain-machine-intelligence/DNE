#!/usr/bin/env python3
"""Seed-level tests of raw perturbation losses, plus the normalization note.

Tests use raw Prediction Loss and Agent Accuracy Loss. Normalized scores
are not tested. Welch tests are unpaired across architectures. Holm is
applied inside each condition x metric family.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy import stats

V3 = Path(__file__).resolve().parents[2]
SRC = V3 / "figure2_dual_metric_perturbation" / "data"
OUT = Path(__file__).resolve().parents[1] / "data" / "Perturbation_raw_statistics.xlsx"
FOCAL = "lstm_attn"
FOCAL_LABEL = "Post-attention LSTM"

ATTN_MODELS = [
    ("transformer", "Transformer"),
    ("pre_bilstm", "Pre-attention BiLSTM"),
    ("pre_lstm", "Pre-attention LSTM"),
    ("post_bilstm", "Post-attention BiLSTM"),
    ("lstm_attn", "Post-attention LSTM"),
]
RECURRENT6 = [
    ("bilstm", "BiLSTM"),
    ("lstm", "LSTM"),
    ("pre_bilstm", "Pre-attention BiLSTM"),
    ("pre_lstm", "Pre-attention LSTM"),
    ("post_bilstm", "Post-attention BiLSTM"),
    ("lstm_attn", "Post-attention LSTM"),
]
METRICS = [
    ("Prediction Loss", "prediction"),
    ("Agent Accuracy Loss", "profile"),
]


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        rank = n - i
        val = min(1.0, p[idx] * rank)
        running = max(running, val)
        adj[idx] = running
    adj_sorted = adj[order].copy()
    for i in range(1, n):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])
    out = np.empty(n, dtype=float)
    out[order] = adj_sorted
    return out


def supported(g: pd.DataFrame) -> pd.DataFrame:
    if "valid_intervention" not in g.columns:
        return g
    ok = g["valid_intervention"].astype(str).str.lower().isin(["true", "1", "yes"])
    return g.loc[ok]


def summarise(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    sd = float(np.std(v, ddof=1)) if n > 1 else np.nan
    sem = sd / np.sqrt(n) if n > 1 else np.nan
    return {"mean": float(np.mean(v)) if n else np.nan, "sd": sd, "sem": sem, "n": n, "values": v}


def collect() -> pd.DataFrame:
    attn = pd.read_csv(SRC / "Attn_raw.csv")
    chan = pd.read_csv(SRC / "Channel_raw.csv")
    order = pd.read_csv(SRC / "Order_raw.csv")
    rows = []
    for code, label in ATTN_MODELS:
        sub = attn[attn.architecture.eq(code)]
        for cond in ("uniform", "shuffled"):
            g = supported(sub[sub.attention_condition.eq(cond)])
            if g.empty:
                continue
            for col, metric in METRICS:
                rows.append(( "attention", cond, code, label, metric, g[col].to_numpy(float)))
    channel_keys = [
        ("action", "no past action"),
        ("reward", "no past reward"),
        ("stimulus", "stimulus only (A+R removed)"),
    ]
    for key, cond in channel_keys:
        for code, label in RECURRENT6:
            g = chan[(chan.architecture.eq(code)) & (chan.channel.eq(key))]
            for col, metric in METRICS:
                rows.append(("channel", cond, code, label, metric, g[col].to_numpy(float)))
    for code, label in RECURRENT6:
        g = order[order.architecture.eq(code)]
        for col, metric in METRICS:
            rows.append(("order", "temporal-order shuffled", code, label, metric, g[col].to_numpy(float)))
    return pd.DataFrame(rows, columns=["family", "condition", "architecture", "architecture_display", "metric", "values"])


def test_vs_zero(blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, cond, metric), g in blocks.groupby(["family", "condition", "metric"], sort=False):
        local = []
        for rec in g.itertuples(index=False):
            s = summarise(rec.values)
            one = stats.ttest_1samp(s["values"], 0.0, alternative="greater")
            two = stats.ttest_1samp(s["values"], 0.0, alternative="two-sided")
            local.append(
                {
                    "family": family,
                    "condition": cond,
                    "metric": metric,
                    "architecture": rec.architecture,
                    "architecture_display": rec.architecture_display,
                    "mean": s["mean"],
                    "sd": s["sd"],
                    "sem": s["sem"],
                    "n": s["n"],
                    "cohens_d_vs_0": s["mean"] / s["sd"] if s["sd"] and s["sd"] > 0 else np.nan,
                    "t_stat": float(one.statistic),
                    "df": float(one.df),
                    "p_one_sided_mean_gt_0": float(one.pvalue),
                    "p_two_sided": float(two.pvalue),
                }
            )
        part = pd.DataFrame(local)
        part["p_one_sided_Holm"] = holm(part["p_one_sided_mean_gt_0"].to_numpy())
        part["p_two_sided_Holm"] = holm(part["p_two_sided"].to_numpy())
        part["holm_family_size"] = len(part)
        part["significant_one_sided_Holm_0.05"] = part["p_one_sided_Holm"] < 0.05
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def test_vs_focal(blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, cond, metric), g in blocks.groupby(["family", "condition", "metric"], sort=False):
        focal_row = g[g.architecture.eq(FOCAL)]
        if focal_row.empty:
            continue
        focal = summarise(focal_row.iloc[0]["values"])
        local = []
        for rec in g[g.architecture.ne(FOCAL)].itertuples(index=False):
            other = summarise(rec.values)
            one = stats.ttest_ind(other["values"], focal["values"], equal_var=False, alternative="greater")
            two = stats.ttest_ind(other["values"], focal["values"], equal_var=False, alternative="two-sided")
            s1, s2 = other["sd"], focal["sd"]
            sp = np.sqrt((s1**2 + s2**2) / 2.0)
            diff = other["mean"] - focal["mean"]
            local.append(
                {
                    "family": family,
                    "condition": cond,
                    "metric": metric,
                    "comparison_architecture": rec.architecture,
                    "comparison_display": rec.architecture_display,
                    "focal_architecture": FOCAL,
                    "focal_display": FOCAL_LABEL,
                    "other_mean": other["mean"],
                    "other_sem": other["sem"],
                    "other_n": other["n"],
                    "focal_mean": focal["mean"],
                    "focal_sem": focal["sem"],
                    "focal_n": focal["n"],
                    "mean_difference_other_minus_focal": diff,
                    "cohens_d_other_minus_focal": diff / sp if sp > 1e-12 else np.nan,
                    "welch_t_stat": float(one.statistic),
                    "welch_df": float(one.df),
                    "p_one_sided_other_gt_focal": float(one.pvalue),
                    "p_two_sided": float(two.pvalue),
                }
            )
        part = pd.DataFrame(local)
        part["p_one_sided_Holm"] = holm(part["p_one_sided_other_gt_focal"].to_numpy())
        part["p_two_sided_Holm"] = holm(part["p_two_sided"].to_numpy())
        part["holm_family_size"] = len(part)
        part["significant_one_sided_Holm_0.05"] = part["p_one_sided_Holm"] < 0.05
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def write_sheet(wb: Workbook, name: str, frame: pd.DataFrame, first: bool = False):
    ws = wb.active if first else wb.create_sheet(name[:31])
    ws.title = name[:31]
    for r in dataframe_to_rows(frame, index=False, header=True):
        ws.append(list(r))
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9E2F3")
        cell.alignment = Alignment(wrap_text=True)
    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 30


def main():
    blocks = collect()
    vs0 = test_vs_zero(blocks)
    vs_focal = test_vs_focal(blocks)
    methods = pd.DataFrame(
        [
            ("tested_values", "Raw seed-level losses. Normalized scores are not tested."),
            ("prediction_loss", "native accuracy - perturbed accuracy. Positive means prediction got worse."),
            ("profile_loss", "native within-task agent decoding - perturbed decoding. Column name in the raw file: Agent Accuracy Loss."),
            ("unit", "One trained seed. N=20 per architecture when the intervention is defined."),
            ("excluded", "Transformer shuffled attention is undefined (valid_intervention false, raw loss stored as 0). It is not a zero effect and is omitted."),
            ("vs_zero", "One-sample t-test. Primary alternative: mean loss > 0."),
            ("vs_post_attention_lstm", "Unpaired Welch t-test. Primary alternative: the other architecture has a larger raw loss than Post-attention LSTM."),
            ("why_unpaired", "Seeds index independent training runs. The same seed number is not a paired experimental unit across architectures."),
            ("holm", "Holm adjustment inside each family x condition x metric. Attention uniform has 5 models; attention shuffled has 4; channel and order have 6."),
            ("alpha", "0.05 on the Holm-adjusted one-sided p."),
            ("normalization_not_in_test", "Normalization is a display rescaling of architecture means. It does not change these tests."),
            ("prediction_norm", "A' = -0.1 - 0.9 * (A - A_min) / (A_max - A_min), so the smallest mean maps to -0.1 and the largest to -1."),
            ("profile_norm", "P' = 0.1 + 0.9 * (P - P_min) / (P_max - P_min), so the smallest mean maps to 0.1 and the largest to 1."),
            ("norm_span", "Both scales have width 0.9 and neither passes through 0."),
            ("norm_pool", "Min and max are the architecture means inside that one condition. Uniform is not pooled with shuffled, and channel conditions are not pooled with each other."),
            ("norm_missing", "Means with no supported seeds are left blank and are excluded from that condition's min and max."),
            ("source", str(SRC)),
        ],
        columns=["item", "description"],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    write_sheet(wb, "Methods", methods, first=True)
    write_sheet(wb, "vs_zero", vs0)
    write_sheet(wb, "vs_PostAttnLSTM", vs_focal)
    wb.save(OUT)
    print(OUT)
    show = vs_focal[
        [
            "family",
            "condition",
            "metric",
            "comparison_display",
            "mean_difference_other_minus_focal",
            "p_one_sided_other_gt_focal",
            "p_one_sided_Holm",
            "significant_one_sided_Holm_0.05",
        ]
    ]
    print(show.to_string(index=False))
    print("--- vs zero not significant ---")
    quiet = vs0.loc[~vs0["significant_one_sided_Holm_0.05"], ["family", "condition", "metric", "architecture_display", "mean", "p_one_sided_Holm"]]
    print(quiet.to_string(index=False) if len(quiet) else "all mean>0 tests significant after Holm")


if __name__ == "__main__":
    main()
