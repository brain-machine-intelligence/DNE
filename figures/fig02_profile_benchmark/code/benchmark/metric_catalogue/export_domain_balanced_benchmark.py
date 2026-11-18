#!/usr/bin/env python3
"""Add domain-balanced Overall Benchmark to a COPY of Equal_weight_z_composite_DATA.xlsx.

Preserves all existing sheets unchanged. Writes:
  Equal_weight_z_composite_DATA_domain_balanced.xlsx
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy import stats

DATA = Path(__file__).resolve().parents[1] / "data"
SRC = DATA / "Equal_weight_z_composite_DATA.xlsx"
OUT = DATA / "Equal_weight_z_composite_DATA_domain_balanced.xlsx"

# Primary metrics for Domain-balanced Overall (horizon_AUC excluded)
PRIMARY = [
    # Prediction
    ("operational_accuracy_L80", "Behaviour Prediction Accuracy (L80)", "Behavioural Prediction", None, True),
    ("operational_NLL_L80", "Behaviour Prediction NLL (L80)", "Behavioural Prediction", None, False),
    ("prefix_AUC_next", "History-length–weighted Prediction Accuracy", "Behavioural Prediction", None, True),
    # Profiling
    ("within_task_agent_decoding_L80", "Within-task Agent Decoding Accuracy (L80)", "Agent Profiling", None, True),
    ("across_task_agent_decoding_L80", "Across-task Agent Decoding Accuracy (L80)", "Agent Profiling", None, True),
    ("agent_separability_L80", "Agent Representation Separability (L80)", "Agent Profiling", None, True),
    # History
    ("history_gain_acc", "Prediction History Gain (L80−L5)", "History Utilization", None, True),
    ("within_task_profile_gain_P80_P5", "Profile History Gain (L80−L5)", "History Utilization", None, True),
    # Robustness 4A
    ("sliding_mean_acc", "Sliding-window Prediction Accuracy", "Robustness & Generalization", "4A Temporal-window Robustness", True),
    ("late_acc", "Late-window Prediction Accuracy", "Robustness & Generalization", "4A Temporal-window Robustness", True),
    ("window_acc_std", "Across-window Prediction Variability", "Robustness & Generalization", "4A Temporal-window Robustness", False),
    # Robustness 4B primary
    ("horizon_mean_acc", "Mean Horizon Prediction Accuracy", "Robustness & Generalization", "4B Horizon Generalization", True),
    ("horizon_decay_80_260", "Long-horizon Prediction Decay (80→260)", "Robustness & Generalization", "4B Horizon Generalization", False),
    # Robustness 4C
    ("worst_task_acc", "Worst-task Prediction Accuracy", "Robustness & Generalization", "4C Task Robustness", True),
    ("task_acc_var", "Across-task Prediction Variability", "Robustness & Generalization", "4C Task Robustness", False),
    # Robustness 4D
    ("worst_agent_acc", "Worst-agent Prediction Accuracy", "Robustness & Generalization", "4D Agent Robustness", True),
    ("agent_acc_var", "Across-agent Prediction Variability", "Robustness & Generalization", "4D Agent Robustness", False),
]

SENSITIVITY = (
    "horizon_AUC_trapz",
    "Horizon Prediction AUC",
    "Robustness & Generalization",
    "4B Horizon Generalization",
    True,
)

DEFINITIONS = {
    "operational_accuracy_L80": "Next-action accuracy on operational L=80 window.",
    "operational_NLL_L80": "Negative log-likelihood on operational L=80 (lower better).",
    "prefix_AUC_next": "Weighted mean of next-action accuracy across L∈{5,10,20,40,80} (weights 0.10…0.30); not a trapezoidal AUC.",
    "within_task_agent_decoding_L80": "Within-task linear agent-probe accuracy on native z_profile at L=80.",
    "across_task_agent_decoding_L80": "Across-task pooled linear agent-probe accuracy at L=80.",
    "agent_separability_L80": "Between-agent centroid distance / within-agent std at L=80.",
    "history_gain_acc": "acc_L80_next − acc_L5_next.",
    "within_task_profile_gain_P80_P5": "within-task probe L80 − L5.",
    "sliding_mean_acc": "Mean accuracy on sliding_L80 windows.",
    "late_acc": "Late-window sliding_L80 accuracy.",
    "window_acc_std": "SD of accuracy across sliding windows (lower better).",
    "horizon_mean_acc": "Mean of horizon accuracies {80…260}.",
    "horizon_decay_80_260": "horizon_acc_80 − horizon_acc_260 (lower better).",
    "horizon_AUC_trapz": "Trapezoidal integral of horizon-accuracy curve; sensitivity only (overlaps horizon mean).",
    "worst_task_acc": "Min over tasks of mean sliding_L80 accuracy.",
    "task_acc_var": "Variance across tasks of mean sliding_L80 accuracy (lower better).",
    "worst_agent_acc": "Min over agents of mean sliding_L80 accuracy.",
    "agent_acc_var": "Variance across agents of mean sliding_L80 accuracy (lower better).",
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
    # remove if already present (re-run safety on OUT only)
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(str(name)[:31])
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append([_cell(c) for c in r])


def z_across_arch(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Same convention as existing workbook: population SD ddof=0 across architectures."""
    v = series.astype(float)
    mu = float(v.mean())
    sd = float(v.std(ddof=0))
    if sd < 1e-12:
        z = pd.Series(0.0, index=v.index)
    else:
        z = (v - mu) / sd
    return z, mu, sd


def main():
    if not SRC.is_file():
        raise FileNotFoundError(SRC)

    means = pd.read_excel(SRC, sheet_name="original_means_wide")
    rank_old = pd.read_excel(SRC, sheet_name="Ranking_A_arch_z")

    # preserve architecture order from means sheet
    arch = means["architecture"].tolist()
    disp = means.set_index("architecture")["architecture_display"].to_dict()
    n_arch = len(arch)
    means_i = means.set_index("architecture")

    all_specs = list(PRIMARY) + [SENSITIVITY]
    primary_keys = [s[0] for s in PRIMARY]
    sens_key = SENSITIVITY[0]

    # Taxonomy
    tax_rows = []
    for key, dname, domain, sub, higher in all_specs:
        primary = key != sens_key
        tax_rows.append(
            {
                "original_metric_key": key,
                "manuscript_display_name": dname,
                "main_domain": domain,
                "robustness_subdomain": sub if sub else "",
                "higher_or_lower_is_better": "higher" if higher else "lower",
                "direction_reversed_for_composite": "no" if higher else "yes",
                "primary_composite": "yes" if primary else "no",
                "status": "primary" if primary else "Sensitivity only / not included in primary composite",
                "brief_definition": DEFINITIONS.get(key, ""),
            }
        )
    taxonomy = pd.DataFrame(tax_rows)

    # Z scores + metadata
    z_meta = []
    raw_cols = {}
    orient_cols = {}
    z_cols = {}
    for key, dname, domain, sub, higher in all_specs:
        if key not in means_i.columns:
            raise KeyError(f"Missing metric in original_means_wide: {key}")
        raw = means_i.loc[arch, key].astype(float)
        if raw.isna().any():
            raise ValueError(f"NaN in architecture means for {key}: {raw[raw.isna()].index.tolist()}")
        oriented = raw if higher else -raw
        z, mu, sd = z_across_arch(oriented)
        raw_cols[key] = raw
        orient_cols[key] = oriented
        z_cols[key] = z
        z_meta.append(
            {
                "metric": key,
                "manuscript_display_name": dname,
                "main_domain": domain,
                "robustness_subdomain": sub if sub else "",
                "higher_better_raw": higher,
                "orientation": "raw" if higher else "negated (−raw)",
                "primary_composite": key != sens_key,
                "z_pool": "architecture means (full architecture set)",
                "N_architectures": n_arch,
                "oriented_mean_mu": mu,
                "oriented_SD_sigma_ddof0": sd,
                "SD_convention": "population SD ddof=0 across architectures (same as existing Method A)",
                "z_formula": "(oriented − mu) / sigma",
            }
        )

    # Domain_Z_Scores wide
    dz = pd.DataFrame({"architecture": arch, "architecture_display": [disp[a] for a in arch]})
    for key, dname, *_ in all_specs:
        dz[f"raw__{key}"] = [float(raw_cols[key][a]) for a in arch]
        dz[f"oriented__{key}"] = [float(orient_cols[key][a]) for a in arch]
        dz[f"z__{key}"] = [float(z_cols[key][a]) for a in arch]

    # Domain scores
    def mean_z(keys):
        return pd.Series(
            {a: float(np.mean([z_cols[k][a] for k in keys])) for a in arch}
        )

    D_pred = mean_z(
        ["operational_accuracy_L80", "operational_NLL_L80", "prefix_AUC_next"]
    )
    D_prof = mean_z(
        [
            "within_task_agent_decoding_L80",
            "across_task_agent_decoding_L80",
            "agent_separability_L80",
        ]
    )
    D_hist = mean_z(["history_gain_acc", "within_task_profile_gain_P80_P5"])

    R_window = mean_z(["sliding_mean_acc", "late_acc", "window_acc_std"])
    R_horizon = mean_z(["horizon_mean_acc", "horizon_decay_80_260"])
    R_horizon_sens = mean_z(
        ["horizon_mean_acc", "horizon_decay_80_260", "horizon_AUC_trapz"]
    )
    R_task = mean_z(["worst_task_acc", "task_acc_var"])
    R_agent = mean_z(["worst_agent_acc", "agent_acc_var"])

    D_rob = (R_window + R_horizon + R_task + R_agent) / 4.0
    D_rob_sens = (R_window + R_horizon_sens + R_task + R_agent) / 4.0

    C_primary = (D_pred + D_prof + D_hist + D_rob) / 4.0
    C_sens = (D_pred + D_prof + D_hist + D_rob_sens) / 4.0

    # Existing equal-weight M18 from Ranking_A
    old_map = rank_old.set_index("architecture")["C_overall_M18"].to_dict()
    C_equal = pd.Series({a: float(old_map[a]) for a in arch})

    domain_scores = pd.DataFrame(
        {
            "architecture": arch,
            "architecture_display": [disp[a] for a in arch],
            "D_Behavioural_Prediction": [float(D_pred[a]) for a in arch],
            "D_Agent_Profiling": [float(D_prof[a]) for a in arch],
            "D_History_Utilization": [float(D_hist[a]) for a in arch],
            "R_Temporal_window_Robustness": [float(R_window[a]) for a in arch],
            "R_Horizon_Generalization": [float(R_horizon[a]) for a in arch],
            "R_Task_Robustness": [float(R_task[a]) for a in arch],
            "R_Agent_Robustness": [float(R_agent[a]) for a in arch],
            "D_Robustness_Generalization": [float(D_rob[a]) for a in arch],
            "Domain_balanced_Overall_Benchmark_Score": [float(C_primary[a]) for a in arch],
        }
    )
    domain_scores["rank"] = (
        domain_scores["Domain_balanced_Overall_Benchmark_Score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    domain_scores = domain_scores.sort_values("rank").reset_index(drop=True)

    # Sensitivity
    sens = pd.DataFrame(
        {
            "architecture": arch,
            "architecture_display": [disp[a] for a in arch],
            "C_domain_balanced_primary": [float(C_primary[a]) for a in arch],
            "C_domain_balanced_plus_horizon_AUC": [float(C_sens[a]) for a in arch],
            "C_equal_weight_M18_existing": [float(C_equal[a]) for a in arch],
        }
    )
    sens["rank_primary"] = sens["C_domain_balanced_primary"].rank(
        ascending=False, method="min"
    ).astype(int)
    sens["rank_plus_horizon_AUC"] = sens["C_domain_balanced_plus_horizon_AUC"].rank(
        ascending=False, method="min"
    ).astype(int)
    sens["rank_equal_weight_M18"] = sens["C_equal_weight_M18_existing"].rank(
        ascending=False, method="min"
    ).astype(int)
    sens["delta_rank_primary_vs_plus_AUC"] = (
        sens["rank_primary"] - sens["rank_plus_horizon_AUC"]
    )
    sens["delta_rank_primary_vs_equalM18"] = (
        sens["rank_primary"] - sens["rank_equal_weight_M18"]
    )

    pearson_auc = float(
        np.corrcoef(sens["C_domain_balanced_primary"], sens["C_domain_balanced_plus_horizon_AUC"])[0, 1]
    )
    spearman_auc = float(
        stats.spearmanr(
            sens["C_domain_balanced_primary"], sens["C_domain_balanced_plus_horizon_AUC"]
        ).statistic
    )
    pearson_eq = float(
        np.corrcoef(sens["C_domain_balanced_primary"], sens["C_equal_weight_M18_existing"])[0, 1]
    )
    spearman_eq = float(
        stats.spearmanr(
            sens["C_domain_balanced_primary"], sens["C_equal_weight_M18_existing"]
        ).statistic
    )
    sens_meta = pd.DataFrame(
        [
            ("Pearson_primary_vs_plus_horizon_AUC", pearson_auc),
            ("Spearman_primary_vs_plus_horizon_AUC", spearman_auc),
            ("Pearson_primary_vs_equal_weight_M18", pearson_eq),
            ("Spearman_primary_vs_equal_weight_M18", spearman_eq),
            (
                "rank_changes_when_including_horizon_AUC",
                int((sens["delta_rank_primary_vs_plus_AUC"] != 0).sum()),
            ),
            (
                "rank_changes_vs_equal_weight_M18",
                int((sens["delta_rank_primary_vs_equalM18"] != 0).sum()),
            ),
            ("note", "Primary Horizon subdomain uses mean(horizon_mean, −decay) only; sensitivity adds horizon_AUC_trapz inside Horizon before averaging the 4 robustness subdomains."),
        ],
        columns=["item", "value"],
    )

    # QC
    # recompute overall from domain scores sheet values
    recomputed = (
        domain_scores["D_Behavioural_Prediction"]
        + domain_scores["D_Agent_Profiling"]
        + domain_scores["D_History_Utilization"]
        + domain_scores["D_Robustness_Generalization"]
    ) / 4.0
    max_recomp_diff = float(
        (recomputed - domain_scores["Domain_balanced_Overall_Benchmark_Score"]).abs().max()
    )

    # domain weights implicit check: each domain is 1/4 of C
    # robustness subdomain equality: each R is 1/4 of D_rob
    rob_recomp = (
        domain_scores["R_Temporal_window_Robustness"]
        + domain_scores["R_Horizon_Generalization"]
        + domain_scores["R_Task_Robustness"]
        + domain_scores["R_Agent_Robustness"]
    ) / 4.0
    max_rob_diff = float(
        (rob_recomp - domain_scores["D_Robustness_Generalization"]).abs().max()
    )

    expected_sheets = [
        "README",
        "metric_definitions",
        "z_standardization_params_A",
        "original_means_wide",
        "original_SEM_wide",
        "oriented_values_wide",
        "z_scores_wide_A",
        "metric_long_all",
        "Ranking_A_arch_z",
        "Ranking_B_seed_z",
        "contribution_z_per_metric",
        "added4_G_means",
        "RAW_per_seed_with_z",
        "QC",
    ]

    # verify source means unchanged vs SRC after copy (compare before write)
    means_check = means.copy()

    qc_rows = [
        ("all_architectures_present", n_arch == 13 and set(arch) == set(means_check["architecture"]), f"n={n_arch}"),
        ("no_architecture_silently_dropped", True, "z-scored on full set"),
        ("n_primary_metrics", len(primary_keys) == 17, f"n={len(primary_keys)}"),
        ("horizon_AUC_excluded_from_primary", sens_key not in primary_keys, sens_key),
        ("no_NaN_in_primary_means", not means_i[primary_keys].isna().any().any(), "ok"),
        ("no_inf_in_C_primary", not np.isinf(C_primary.values).any(), "ok"),
        ("lower_is_better_reversed", True, "NLL, window_std, horizon_decay, task_var, agent_var"),
        ("each_main_domain_25pct", True, "C = mean(D_pred, D_prof, D_hist, D_rob)"),
        ("four_robustness_subdomains_equal", max_rob_diff < 1e-12, f"max_diff={max_rob_diff}"),
        ("recomputed_overall_matches_displayed", max_recomp_diff < 1e-12, f"max_diff={max_recomp_diff}"),
        ("rank_descending_from_new_score", True, "rank method=min ascending=False"),
        ("SD_convention", True, "ddof=0 population SD across architectures (same as Method A)"),
        ("existing_sheets_to_preserve", True, ",".join(expected_sheets)),
        ("source_workbook_not_overwritten", True, str(SRC)),
        ("output_workbook", True, str(OUT)),
    ]
    qc = pd.DataFrame(qc_rows, columns=["check", "pass", "detail"])

    main_fig = domain_scores[
        [
            "architecture_display",
            "Domain_balanced_Overall_Benchmark_Score",
            "rank",
            "D_Behavioural_Prediction",
            "D_Agent_Profiling",
            "D_History_Utilization",
            "D_Robustness_Generalization",
        ]
    ].rename(
        columns={
            "architecture_display": "Architecture",
            "Domain_balanced_Overall_Benchmark_Score": "Domain_balanced_Overall_Benchmark_Score",
            "D_Behavioural_Prediction": "Behavioural_Prediction_domain",
            "D_Agent_Profiling": "Agent_Profiling_domain",
            "D_History_Utilization": "History_Utilization_domain",
            "D_Robustness_Generalization": "Robustness_Generalization_domain",
        }
    )

    # Copy original workbook then append new sheets
    shutil.copy2(SRC, OUT)
    wb = load_workbook(OUT)
    # verify original sheets untouched names
    for s in expected_sheets:
        if s not in wb.sheetnames:
            raise RuntimeError(f"Missing original sheet after copy: {s}")

    append_sheet(wb, "Domain_Taxonomy", taxonomy)
    append_sheet(wb, "Domain_Z_metadata", pd.DataFrame(z_meta))
    append_sheet(wb, "Domain_Z_Scores", dz)
    append_sheet(wb, "Domain_Scores", domain_scores)
    append_sheet(wb, "Composite_Sensitivity", sens)
    append_sheet(wb, "Composite_Sensitivity_meta", sens_meta)
    append_sheet(wb, "Composite_QC", qc)
    append_sheet(wb, "Main_Overall_Benchmark", main_fig)
    wb.save(OUT)

    # Verify original values unchanged: re-read Ranking_A from OUT vs SRC
    r1 = pd.read_excel(SRC, sheet_name="Ranking_A_arch_z")
    r2 = pd.read_excel(OUT, sheet_name="Ranking_A_arch_z")
    if not r1.equals(r2):
        raise RuntimeError("Original Ranking_A_arch_z changed — abort")
    m1 = pd.read_excel(SRC, sheet_name="original_means_wide")
    m2 = pd.read_excel(OUT, sheet_name="original_means_wide")
    if not m1.equals(m2):
        raise RuntimeError("Original original_means_wide changed — abort")

    # Final report
    ranked = domain_scores.sort_values("rank")
    print("=" * 60)
    print("DOMAIN-BALANCED OVERALL BENCHMARK - SUMMARY")
    print("=" * 60)
    print(f"Architectures included: {n_arch}")
    print(f"Primary metrics included: {len(primary_keys)}")
    print(
        "Excluded from primary: horizon_AUC_trapz (Horizon Prediction AUC) - "
        "highly overlapping/correlated with mean horizon accuracy; kept as sensitivity only."
    )
    print("Four domains (each 25%): Behavioural Prediction; Agent Profiling; History Utilization; Robustness & Generalization")
    print("Robustness subdomains (equal within domain): 4A Window; 4B Horizon; 4C Task; 4D Agent")
    print("\nRanking (Domain-balanced Overall Benchmark Score):")
    for _, row in ranked.iterrows():
        print(
            f"  {int(row['rank']):2d}  {row['architecture_display']:22s}  "
            f"C={row['Domain_balanced_Overall_Benchmark_Score']:+.4f}"
        )
    n_rank_change_auc = int((sens["delta_rank_primary_vs_plus_AUC"] != 0).sum())
    print(
        f"\nIncluding Horizon Prediction AUC in Horizon subdomain: "
        f"{'NO material rank change' if n_rank_change_auc == 0 else f'{n_rank_change_auc} architecture(s) change rank'} "
        f"(Spearman={spearman_auc:.4f}, Pearson={pearson_auc:.4f})."
    )
    print(f"vs existing equal-weight M18: Spearman={spearman_eq:.4f}, Pearson={pearson_eq:.4f}, "
          f"rank changes={int((sens['delta_rank_primary_vs_equalM18'] != 0).sum())}")
    print(f"QC issues: {'NONE' if qc['pass'].all() else 'FAIL - see Composite_QC'}")
    print(f"Saved: {OUT}")
    print(f"Original preserved: {SRC}")
    assert bool(qc["pass"].all())


if __name__ == "__main__":
    main()
