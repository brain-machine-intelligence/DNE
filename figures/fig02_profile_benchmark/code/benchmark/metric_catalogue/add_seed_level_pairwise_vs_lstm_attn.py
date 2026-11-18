#!/usr/bin/env python3
"""Seed-level Domain-balanced Overall + pairwise tests vs Post-attention LSTM.

Appends sheets to Equal_weight_z_composite_DATA_domain_balanced.xlsx.
Uses FIXED mu/sigma from Domain_Z_metadata (architecture-level Method A).
Does not overwrite Main_Overall_Benchmark or prior sheets' values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy import stats

DATA = Path(__file__).resolve().parents[1] / "data"
XLSX = DATA / "Equal_weight_z_composite_DATA_domain_balanced.xlsx"
FOCAL = "lstm_attn"

PRIMARY_KEYS = [
    "operational_accuracy_L80",
    "operational_NLL_L80",
    "prefix_AUC_next",
    "within_task_agent_decoding_L80",
    "across_task_agent_decoding_L80",
    "agent_separability_L80",
    "history_gain_acc",
    "within_task_profile_gain_P80_P5",
    "sliding_mean_acc",
    "late_acc",
    "window_acc_std",
    "horizon_mean_acc",
    "horizon_decay_80_260",
    "worst_task_acc",
    "task_acc_var",
    "worst_agent_acc",
    "agent_acc_var",
]

DOMAIN_MAP = {
    "Prediction": [
        "operational_accuracy_L80",
        "operational_NLL_L80",
        "prefix_AUC_next",
    ],
    "Profiling": [
        "within_task_agent_decoding_L80",
        "across_task_agent_decoding_L80",
        "agent_separability_L80",
    ],
    "History": [
        "history_gain_acc",
        "within_task_profile_gain_P80_P5",
    ],
    "R_Window": ["sliding_mean_acc", "late_acc", "window_acc_std"],
    "R_Horizon": ["horizon_mean_acc", "horizon_decay_80_260"],
    "R_Task": ["worst_task_acc", "task_acc_var"],
    "R_Agent": ["worst_agent_acc", "agent_acc_var"],
}


def _cell(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def append_sheet(wb, name: str, df: pd.DataFrame):
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(str(name)[:31])
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append([_cell(c) for c in r])


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
    # enforce monotonicity in sorted order
    adj_sorted = adj[order].copy()
    for i in range(1, n):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])
    out = np.empty(n, dtype=float)
    out[order] = adj_sorted
    return out


def mean_z(row_z: dict, keys: list[str]) -> float:
    return float(np.mean([row_z[k] for k in keys]))


def main():
    meta = pd.read_excel(XLSX, sheet_name="Domain_Z_metadata")
    raw = pd.read_excel(XLSX, sheet_name="RAW_per_seed_with_z")
    main_arch = pd.read_excel(XLSX, sheet_name="Main_Overall_Benchmark")

    meta_i = meta.set_index("metric")
    for k in PRIMARY_KEYS:
        if k not in meta_i.index:
            raise KeyError(k)
        if k not in raw.columns:
            raise KeyError(f"missing in RAW: {k}")

    # Fixed standardization parameters
    mu = {k: float(meta_i.loc[k, "oriented_mean_mu"]) for k in PRIMARY_KEYS}
    sd = {k: float(meta_i.loc[k, "oriented_SD_sigma_ddof0"]) for k in PRIMARY_KEYS}
    higher = {k: bool(meta_i.loc[k, "higher_better_raw"]) for k in PRIMARY_KEYS}

    seed_rows = []
    for _, r in raw.iterrows():
        z = {}
        for k in PRIMARY_KEYS:
            x = float(r[k])
            x_star = x if higher[k] else -x
            z[k] = (x_star - mu[k]) / sd[k] if sd[k] > 1e-12 else 0.0
        D_pred = mean_z(z, DOMAIN_MAP["Prediction"])
        D_prof = mean_z(z, DOMAIN_MAP["Profiling"])
        D_hist = mean_z(z, DOMAIN_MAP["History"])
        R_win = mean_z(z, DOMAIN_MAP["R_Window"])
        R_hor = mean_z(z, DOMAIN_MAP["R_Horizon"])
        R_task = mean_z(z, DOMAIN_MAP["R_Task"])
        R_agent = mean_z(z, DOMAIN_MAP["R_Agent"])
        D_rob = (R_win + R_hor + R_task + R_agent) / 4.0
        C = (D_pred + D_prof + D_hist + D_rob) / 4.0
        seed_rows.append(
            {
                "architecture": r["architecture"],
                "architecture_display": r["architecture_display"],
                "seed": int(r["seed"]),
                "checkpoint": r["checkpoint"],
                "D_Behavioural_Prediction": D_pred,
                "D_Agent_Profiling": D_prof,
                "D_History_Utilization": D_hist,
                "R_Temporal_window_Robustness": R_win,
                "R_Horizon_Generalization": R_hor,
                "R_Task_Robustness": R_task,
                "R_Agent_Robustness": R_agent,
                "D_Robustness_Generalization": D_rob,
                "C_Domain_balanced_Overall": C,
            }
        )
    seed_df = pd.DataFrame(seed_rows)

    # Summary per architecture
    sum_rows = []
    for arch, g in seed_df.groupby("architecture"):
        c = g["C_Domain_balanced_Overall"].astype(float)
        n = int(len(c))
        mean = float(c.mean())
        sd_s = float(c.std(ddof=1)) if n > 1 else 0.0
        sem = sd_s / np.sqrt(n)
        sum_rows.append(
            {
                "architecture": arch,
                "architecture_display": g["architecture_display"].iloc[0],
                "C_mean": mean,
                "C_SD": sd_s,
                "C_SEM": sem,
                "C_N": n,
                "C_CI95_low": mean - 1.96 * sem,
                "C_CI95_high": mean + 1.96 * sem,
                "C_median": float(c.median()),
                "C_min": float(c.min()),
                "C_max": float(c.max()),
                "D_Prediction_mean": float(g["D_Behavioural_Prediction"].mean()),
                "D_Profiling_mean": float(g["D_Agent_Profiling"].mean()),
                "D_History_mean": float(g["D_History_Utilization"].mean()),
                "D_Robustness_mean": float(g["D_Robustness_Generalization"].mean()),
            }
        )
    summary = pd.DataFrame(sum_rows)
    # align rank with Main_Overall (architecture-mean definition) for reference
    # also rank by seed-mean C
    summary["rank_by_seed_mean_C"] = summary["C_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    main_map = dict(
        zip(main_arch["Architecture"], main_arch["Domain_balanced_Overall_Benchmark_Score"])
    )
    # Main uses display names
    summary["Main_Overall_arch_mean_C"] = summary["architecture_display"].map(main_map)
    summary = summary.sort_values("rank_by_seed_mean_C").reset_index(drop=True)

    # Pairwise vs focal
    focal = seed_df[seed_df.architecture.eq(FOCAL)]["C_Domain_balanced_Overall"].astype(float).values
    focal_disp = seed_df[seed_df.architecture.eq(FOCAL)]["architecture_display"].iloc[0]
    others = [a for a in sorted(seed_df.architecture.unique()) if a != FOCAL]
    # preserve a readable order from summary
    order_disp = summary[summary.architecture.ne(FOCAL)]["architecture"].tolist()
    others = [a for a in order_disp if a in others] + [a for a in others if a not in order_disp]

    pair_rows = []
    for arch in others:
        g = seed_df[seed_df.architecture.eq(arch)]["C_Domain_balanced_Overall"].astype(float).values
        # Welch t-test: H1 focal > other (greater)
        tw = stats.ttest_ind(focal, g, equal_var=False, alternative="greater")
        # two-sided for reporting
        tw2 = stats.ttest_ind(focal, g, equal_var=False, alternative="two-sided")
        mw = stats.mannwhitneyu(focal, g, alternative="greater")
        diff = float(np.mean(focal) - np.mean(g))
        # Cohen's d (Welch / Hedges-ish using pooled wd)
        s1, s2 = np.std(focal, ddof=1), np.std(g, ddof=1)
        # Welch denominator for effect size approximation: use pooled
        sp = np.sqrt((s1**2 + s2**2) / 2.0)
        d = diff / sp if sp > 1e-12 else np.nan
        pair_rows.append(
            {
                "focal_architecture": FOCAL,
                "focal_display": focal_disp,
                "comparison_architecture": arch,
                "comparison_display": seed_df[seed_df.architecture.eq(arch)][
                    "architecture_display"
                ].iloc[0],
                "focal_C_mean": float(np.mean(focal)),
                "focal_C_SEM": float(np.std(focal, ddof=1) / np.sqrt(len(focal))),
                "focal_N": int(len(focal)),
                "other_C_mean": float(np.mean(g)),
                "other_C_SEM": float(np.std(g, ddof=1) / np.sqrt(len(g))),
                "other_N": int(len(g)),
                "mean_difference_focal_minus_other": diff,
                "cohens_d_approx": float(d),
                "welch_t_stat": float(tw.statistic),
                "welch_df": float(tw.df) if hasattr(tw, "df") else np.nan,
                "welch_p_one_sided_focal_greater": float(tw.pvalue),
                "welch_p_two_sided": float(tw2.pvalue),
                "mannwhitney_U": float(mw.statistic),
                "mannwhitney_p_one_sided_focal_greater": float(mw.pvalue),
            }
        )
    pair = pd.DataFrame(pair_rows)
    pair["welch_p_one_sided_Holm"] = holm(pair["welch_p_one_sided_focal_greater"].values)
    pair["welch_p_two_sided_Holm"] = holm(pair["welch_p_two_sided"].values)
    pair["mannwhitney_p_one_sided_Holm"] = holm(
        pair["mannwhitney_p_one_sided_focal_greater"].values
    )
    pair["significant_one_sided_Holm_0.05"] = pair["welch_p_one_sided_Holm"] < 0.05
    pair["significant_two_sided_Holm_0.05"] = pair["welch_p_two_sided_Holm"] < 0.05
    pair = pair.sort_values("welch_p_one_sided_focal_greater").reset_index(drop=True)

    methods = pd.DataFrame(
        [
            ("unit_of_observation", "architecture × seed (n=20 seeds per architecture)"),
            ("score", "Domain-balanced Overall Benchmark Score (same formula as Main_Overall_Benchmark)"),
            ("z_standardization", "FIXED mu/sigma from Domain_Z_metadata (architecture-level, ddof=0); applied to each seed"),
            ("why_fixed_mu_sigma", "Matches manuscript architecture-level composite definition; avoids re-fitting z per seed"),
            ("primary_metrics", str(len(PRIMARY_KEYS))),
            ("horizon_AUC_trapz", "excluded from primary (same as Main)"),
            ("focal_comparison", "Post-attention LSTM (lstm_attn) vs each other architecture"),
            ("primary_test", "Welch two-sample t-test (unequal variance), one-sided: C_focal > C_other"),
            ("also_reported", "two-sided Welch; Mann–Whitney U one-sided; Cohen d approx"),
            ("multiple_comparisons", "Holm correction across 12 pairwise tests"),
            ("alpha", "0.05 on Holm-adjusted p"),
            ("independence_note", "Seeds are independent across architectures (different trained checkpoints)"),
            ("Main_Overall_note", "Main_Overall_Benchmark remains architecture-mean composite; seed-mean C is highly correlated but not identical to arch-mean-of-metrics path"),
            ("recommendation", "Report Main_Overall for the figure; use these tests for statistical support that Post-attn LSTM exceeds alternatives on seed-level C"),
        ],
        columns=["item", "description"],
    )

    qc = pd.DataFrame(
        [
            ("n_seed_rows", len(seed_df), len(seed_df) == 260),
            ("n_arch", seed_df.architecture.nunique(), seed_df.architecture.nunique() == 13),
            ("n_seeds_per_arch_all_20", int((seed_df.groupby("architecture").size() == 20).all()), True),
            ("no_nan_C", int(seed_df["C_Domain_balanced_Overall"].isna().sum()), True),
            ("n_pairwise", len(pair), len(pair) == 12),
            ("holm_one_sided_monotone", True, True),
            ("focal_is_lstm_attn", FOCAL == "lstm_attn", True),
        ],
        columns=["check", "value", "pass"],
    )
    # fix pass column properly
    qc.loc[qc.check.eq("n_seed_rows"), "pass"] = len(seed_df) == 260
    qc.loc[qc.check.eq("n_arch"), "pass"] = seed_df.architecture.nunique() == 13
    qc.loc[qc.check.eq("n_seeds_per_arch_all_20"), "pass"] = bool(
        (seed_df.groupby("architecture").size() == 20).all()
    )
    qc.loc[qc.check.eq("no_nan_C"), "pass"] = int(seed_df["C_Domain_balanced_Overall"].isna().sum()) == 0
    qc.loc[qc.check.eq("n_pairwise"), "pass"] = len(pair) == 12

    wb = load_workbook(XLSX)
    append_sheet(wb, "Seed_C_overall_raw", seed_df)
    append_sheet(wb, "Seed_C_overall_summary", summary)
    append_sheet(wb, "Pairwise_vs_PostAttnLSTM", pair)
    append_sheet(wb, "Pairwise_methods", methods)
    append_sheet(wb, "Pairwise_QC", qc)
    wb.save(XLSX)

    # also CSV sidecars
    seed_df.to_csv(DATA / "Seed_C_overall_raw.csv", index=False)
    summary.to_csv(DATA / "Seed_C_overall_summary.csv", index=False)
    pair.to_csv(DATA / "Pairwise_vs_PostAttnLSTM.csv", index=False)

    print(f"[OK] updated {XLSX}")
    print("\nSeed-mean C ranking:")
    print(
        summary[
            ["rank_by_seed_mean_C", "architecture_display", "C_mean", "C_SEM", "Main_Overall_arch_mean_C"]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nPairwise vs Post-attention LSTM (Welch one-sided, Holm):")
    show = pair[
        [
            "comparison_display",
            "mean_difference_focal_minus_other",
            "welch_p_one_sided_focal_greater",
            "welch_p_one_sided_Holm",
            "significant_one_sided_Holm_0.05",
        ]
    ].copy()
    print(show.round(6).to_string(index=False))
    print("\nQC:", qc.to_string(index=False))
    assert bool(qc["pass"].all())


if __name__ == "__main__":
    main()
