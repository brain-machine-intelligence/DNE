#!/usr/bin/env python3
"""Equal-weight z composite with catalogue 14 + 4 G-metrics (late, window_std, task/agent var).

M = 18. Writes Equal_weight_z_composite_DATA.xlsx (overwrites prior 14-metric workbook
with expanded content; same path).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
QC = ROOT / "qc"
SLIDING = Path(__file__).resolve().parents[4]
PER_SEED = SLIDING / "analysis_summary_inputpad_v2" / "per_seed_summary.csv"
RESULTS = SLIDING / "results_inputpad_v2"
OUT_XLSX = DATA / "Equal_weight_z_composite_DATA.xlsx"
CAT_RAW = DATA / "RAW_per_seed.csv"

ORDER = [
    "lstm_attn",
    "post_bilstm",
    "lstm",
    "bilstm",
    "pre_bilstm",
    "pre_lstm",
    "gru",
    "cnn",
    "transformer",
    "single_layer",
    "mlp",
    "tinyrnn_paper",
    "mlp_pool",
]

METRICS = [
    # --- original catalogue 14 ---
    {
        "metric": "within_task_agent_decoding_L80",
        "side": "Profile",
        "higher_better": True,
        "short_name": "Within-task agent decoding (L80)",
        "meaning": "Linear probe recovers agent identity from native z_profile within each task.",
        "how_calculated": "Within-task linear agent-probe accuracy on native z_profile at L=80.",
        "source": "catalogue / within_task_agent_probe_L80",
        "in_original_14": True,
        "in_PG": "P",
    },
    {
        "metric": "within_task_profile_gain_P80_P5",
        "side": "Profile",
        "higher_better": True,
        "short_name": "Profile gain (P80−P5)",
        "meaning": "Agent-decoding improvement from short to long history.",
        "how_calculated": "within_task_agent_probe_L80 − within_task_agent_probe_L5.",
        "source": "catalogue derived",
        "in_original_14": True,
        "in_PG": "P",
    },
    {
        "metric": "across_task_agent_decoding_L80",
        "side": "Profile",
        "higher_better": True,
        "short_name": "Across-task agent decoding (L80)",
        "meaning": "Agent identity recoverable from z_profile pooled across tasks.",
        "how_calculated": "Across-task linear agent-probe accuracy at L=80.",
        "source": "catalogue / agent_probe_L80",
        "in_original_14": True,
        "in_PG": "no (catalogue add)",
    },
    {
        "metric": "agent_separability_L80",
        "side": "Profile",
        "higher_better": True,
        "short_name": "Agent separability (L80)",
        "meaning": "Geometric separation of agent centroids in z_profile space.",
        "how_calculated": "Between-agent centroid distance / within-agent std at L=80.",
        "source": "catalogue / agent_separability_L80",
        "in_original_14": True,
        "in_PG": "no (catalogue add)",
    },
    {
        "metric": "operational_accuracy_L80",
        "side": "Prediction",
        "higher_better": True,
        "short_name": "Operational accuracy (L80)",
        "meaning": "Next-action prediction accuracy on operational L=80 window.",
        "how_calculated": "operational_L80_acc_mean.",
        "source": "catalogue",
        "in_original_14": True,
        "in_PG": "P",
    },
    {
        "metric": "operational_NLL_L80",
        "side": "Prediction",
        "higher_better": False,
        "short_name": "Operational NLL (L80)",
        "meaning": "Negative log-likelihood of next actions (lower better).",
        "how_calculated": "operational_L80_nll_mean; z uses −NLL.",
        "source": "catalogue",
        "in_original_14": True,
        "in_PG": "P (−z)",
    },
    {
        "metric": "prefix_AUC_next",
        "side": "Prediction",
        "higher_better": True,
        "short_name": "Prefix AUC (next-action)",
        "meaning": "Weighted next-action accuracy across prefix lengths.",
        "how_calculated": "0.10·L5+0.15·L10+0.20·L20+0.25·L40+0.30·L80.",
        "source": "catalogue",
        "in_original_14": True,
        "in_PG": "P",
    },
    {
        "metric": "history_gain_acc",
        "side": "Prediction",
        "higher_better": True,
        "short_name": "History gain (prediction)",
        "meaning": "Prediction improvement from L5 to L80.",
        "how_calculated": "acc_L80_next − acc_L5_next.",
        "source": "catalogue",
        "in_original_14": True,
        "in_PG": "P",
    },
    {
        "metric": "sliding_mean_acc",
        "side": "Robustness",
        "higher_better": True,
        "short_name": "Sliding mean accuracy",
        "meaning": "Mean prediction accuracy on sliding_L80 windows.",
        "how_calculated": "sliding_L80_acc_mean.",
        "source": "catalogue",
        "in_original_14": True,
        "in_PG": "G",
    },
    {
        "metric": "horizon_mean_acc",
        "side": "Robustness",
        "higher_better": True,
        "short_name": "Horizon mean accuracy",
        "meaning": "Mean accuracy across longer horizons from L80.",
        "how_calculated": "Mean of horizon_acc_{80,100,140,180,220,260}.",
        "source": "catalogue",
        "in_original_14": True,
        "in_PG": "G (legacy horizon_auc was this mean)",
    },
    {
        "metric": "horizon_AUC_trapz",
        "side": "Robustness",
        "higher_better": True,
        "short_name": "Horizon AUC (trapz)",
        "meaning": "Trapezoidal integral of horizon-accuracy curve.",
        "how_calculated": "Stored horizon_auc_acc trapz (correlated with horizon_mean).",
        "source": "catalogue",
        "in_original_14": True,
        "in_PG": "partial (catalogue trapz; legacy G used mean)",
    },
    {
        "metric": "horizon_decay_80_260",
        "side": "Robustness",
        "higher_better": False,
        "short_name": "Horizon decay (80→260)",
        "meaning": "Accuracy drop from near to far horizon (lower better).",
        "how_calculated": "horizon_acc_80 − horizon_acc_260; z uses −decay.",
        "source": "catalogue",
        "in_original_14": True,
        "in_PG": "G (−z)",
    },
    {
        "metric": "worst_task_acc",
        "side": "Robustness",
        "higher_better": True,
        "short_name": "Worst-task accuracy",
        "meaning": "Accuracy on the hardest task.",
        "how_calculated": "Min over tasks of mean sliding_L80 acc (episode CSV).",
        "source": "catalogue",
        "in_original_14": True,
        "in_PG": "G",
    },
    {
        "metric": "worst_agent_acc",
        "side": "Robustness",
        "higher_better": True,
        "short_name": "Worst-agent accuracy",
        "meaning": "Accuracy on the hardest agent.",
        "how_calculated": "Min over agents of mean sliding_L80 acc (episode CSV).",
        "source": "catalogue",
        "in_original_14": True,
        "in_PG": "G",
    },
    # --- added 4 from G ---
    {
        "metric": "late_acc",
        "side": "Robustness",
        "higher_better": True,
        "short_name": "Late-window accuracy",
        "meaning": "Prediction accuracy on late portion of sliding_L80 evaluation (G metric).",
        "how_calculated": "sliding_L80_late_acc from per_seed_summary.",
        "source": "analysis_summary_inputpad_v2 / sliding_L80_late_acc",
        "in_original_14": False,
        "in_PG": "G (added now)",
    },
    {
        "metric": "window_acc_std",
        "side": "Robustness",
        "higher_better": False,
        "short_name": "Window accuracy SD",
        "meaning": "Variability of accuracy across sliding windows (lower = more stable).",
        "how_calculated": "sliding_L80_acc_std; z uses −std.",
        "source": "analysis_summary_inputpad_v2 / sliding_L80_acc_std",
        "in_original_14": False,
        "in_PG": "G (−z) (added now)",
    },
    {
        "metric": "task_acc_var",
        "side": "Robustness",
        "higher_better": False,
        "short_name": "Task accuracy variance",
        "meaning": "Variance of mean accuracy across tasks (lower = more even across tasks).",
        "how_calculated": "Var over tasks of mean sliding_L80 acc (episode CSV, pandas default ddof=1); z uses −var.",
        "source": "results_inputpad_v2/*_sliding_episode.csv",
        "in_original_14": False,
        "in_PG": "G (−z) (added now)",
    },
    {
        "metric": "agent_acc_var",
        "side": "Robustness",
        "higher_better": False,
        "short_name": "Agent accuracy variance",
        "meaning": "Variance of mean accuracy across agents (lower = more even across agents).",
        "how_calculated": "Var over agents of mean sliding_L80 acc (episode CSV, ddof=1); z uses −var.",
        "source": "results_inputpad_v2/*_sliding_episode.csv",
        "in_original_14": False,
        "in_PG": "G (−z) (added now)",
    },
]


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


def write_wb(path: Path, sheets: dict[str, pd.DataFrame]):
    wb = Workbook()
    wb.remove(wb.active)
    for name, df in sheets.items():
        ws = wb.create_sheet(str(name)[:31])
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append([_cell(c) for c in r])
    wb.save(path)


def episode_vars(model: str, seed: int) -> dict:
    path = RESULTS / f"{model}_seed{seed}_sliding_episode.csv"
    if not path.is_file():
        return {"task_acc_var": np.nan, "agent_acc_var": np.nan}
    g = pd.read_csv(path)
    g = g[g["eval_tag"] == "sliding_L80"]
    if g.empty:
        return {"task_acc_var": np.nan, "agent_acc_var": np.nan}
    by_task = g.groupby("task", observed=True)["acc"].mean()
    by_agent = g.groupby("agent", observed=True)["acc"].mean()
    return {
        "task_acc_var": float(by_task.var()) if len(by_task) > 1 else np.nan,
        "agent_acc_var": float(by_agent.var()) if len(by_agent) > 1 else np.nan,
    }


def build_raw18() -> pd.DataFrame:
    cat = pd.read_csv(CAT_RAW)
    per = pd.read_csv(PER_SEED)
    # index late/std from per_seed
    per_key = per.set_index(["model", "seed"])
    rows = []
    for _, r in cat.iterrows():
        model, seed = str(r["architecture"]), int(r["seed"])
        pr = per_key.loc[(model, seed)]
        ev = episode_vars(model, seed)
        row = r.to_dict()
        row["late_acc"] = float(pr["sliding_L80_late_acc"])
        row["window_acc_std"] = float(pr["sliding_L80_acc_std"])
        row["task_acc_var"] = ev["task_acc_var"]
        row["agent_acc_var"] = ev["agent_acc_var"]
        rows.append(row)
    raw = pd.DataFrame(rows)
    metric_names = [m["metric"] for m in METRICS]
    keep = ["architecture", "architecture_display", "seed", "checkpoint"] + metric_names
    return raw[keep]


def main():
    raw = build_raw18()
    archs = [a for a in ORDER if a in set(raw.architecture)]
    disp = (
        raw.drop_duplicates("architecture")
        .set_index("architecture")["architecture_display"]
        .to_dict()
    )
    metric_names = [m["metric"] for m in METRICS]
    assert len(metric_names) == 18

    # QC completeness
    n_miss = {m: int(raw[m].isna().sum()) for m in metric_names}

    means = raw.groupby("architecture")[metric_names].mean().loc[archs]
    sems = raw.groupby("architecture")[metric_names].sem().loc[archs]
    ns = raw.groupby("architecture")[metric_names].count().loc[archs]

    oriented = means.copy()
    z_arch = pd.DataFrame(index=archs, columns=metric_names, dtype=float)
    z_meta = []
    for mdef in METRICS:
        m = mdef["metric"]
        v = means[m].astype(float)
        v_o = v if mdef["higher_better"] else -v
        oriented[m] = v_o
        mu, sd = float(v_o.mean()), float(v_o.std(ddof=0))
        z_arch[m] = (v_o - mu) / sd if sd > 1e-12 else 0.0
        z_meta.append(
            {
                "metric": m,
                "side": mdef["side"],
                "higher_better_raw": mdef["higher_better"],
                "orientation": "raw" if mdef["higher_better"] else "negated (−raw)",
                "in_original_14": mdef["in_original_14"],
                "in_PG": mdef["in_PG"],
                "N_architectures": len(archs),
                "oriented_mean_across_arch": mu,
                "oriented_SD_across_arch_ddof0": sd,
                "n_missing_seed_rows": n_miss[m],
            }
        )

    side_cols = {
        s: [m["metric"] for m in METRICS if m["side"] == s]
        for s in ("Profile", "Prediction", "Robustness")
    }
    C_overall = z_arch.mean(axis=1)
    C_profile = z_arch[side_cols["Profile"]].mean(axis=1)
    C_pred = z_arch[side_cols["Prediction"]].mean(axis=1)
    C_rob = z_arch[side_cols["Robustness"]].mean(axis=1)
    C_14 = z_arch[[m["metric"] for m in METRICS if m["in_original_14"]]].mean(axis=1)
    C_4 = z_arch[[m["metric"] for m in METRICS if not m["in_original_14"]]].mean(axis=1)

    # Method B seed-level
    z_seed = pd.DataFrame(index=raw.index)
    for mdef in METRICS:
        m = mdef["metric"]
        v = raw[m].astype(float)
        v_o = v if mdef["higher_better"] else -v
        mu, sd = float(v_o.mean()), float(v_o.std(ddof=0))
        z_seed[m] = (v_o - mu) / sd if sd > 1e-12 else 0.0
    raw_b = raw.copy()
    raw_b["C_overall_seed"] = z_seed.mean(axis=1)
    for side, cols in side_cols.items():
        raw_b[f"C_{side.lower()}_seed"] = z_seed[cols].mean(axis=1)

    B_sum = (
        raw_b.groupby("architecture")
        .agg(
            C_overall_mean=("C_overall_seed", "mean"),
            C_overall_SEM=("C_overall_seed", "sem"),
            C_overall_SD=("C_overall_seed", "std"),
            C_overall_N=("C_overall_seed", "count"),
            C_profile_mean=("C_profile_seed", "mean"),
            C_profile_SEM=("C_profile_seed", "sem"),
            C_prediction_mean=("C_prediction_seed", "mean"),
            C_prediction_SEM=("C_prediction_seed", "sem"),
            C_robustness_mean=("C_robustness_seed", "mean"),
            C_robustness_SEM=("C_robustness_seed", "sem"),
        )
        .loc[archs]
    )

    readme = pd.DataFrame(
        [
            ("M_metrics", "18 = previous catalogue 14 + 4 G metrics (late_acc, window_acc_std, task_acc_var, agent_acc_var)"),
            ("formula", "C_overall = (1/M) sum_m z_m ; z across architectures on oriented means (Method A)"),
            ("added_metrics", "late_acc, window_acc_std, task_acc_var, agent_acc_var"),
            ("data_source_catalogue", str(CAT_RAW)),
            ("data_source_per_seed", str(PER_SEED)),
            ("data_source_episodes", str(RESULTS)),
            ("implementation", "inputpad_v2"),
            ("orientation", "If higher_better=False: use −value before z-scoring"),
            ("primary_ranking", "Ranking_A_arch_z (M=18)"),
            ("comparison_14", "C_overall_14 column = equal-weight over previous 14 only"),
        ],
        columns=["item", "description"],
    )

    metric_definitions = pd.DataFrame(METRICS)

    orig_mean = means.reset_index()
    orig_mean.insert(1, "architecture_display", orig_mean["architecture"].map(disp))
    orig_sem = sems.reset_index()
    orig_sem.insert(1, "architecture_display", orig_sem["architecture"].map(disp))
    oriented_wide = oriented.reset_index()
    oriented_wide.insert(1, "architecture_display", oriented_wide["architecture"].map(disp))
    z_wide = z_arch.reset_index().rename(columns={"index": "architecture"})
    z_wide.insert(1, "architecture_display", z_wide["architecture"].map(disp))

    metric_long = []
    for a in archs:
        for mdef in METRICS:
            m = mdef["metric"]
            metric_long.append(
                {
                    "architecture": a,
                    "architecture_display": disp[a],
                    "side": mdef["side"],
                    "metric": m,
                    "short_name": mdef["short_name"],
                    "higher_better": mdef["higher_better"],
                    "in_original_14": mdef["in_original_14"],
                    "in_PG": mdef["in_PG"],
                    "raw_mean": float(means.loc[a, m]),
                    "raw_SEM": float(sems.loc[a, m]),
                    "N_seeds": int(ns.loc[a, m]),
                    "oriented_value": float(oriented.loc[a, m]),
                    "z_across_architectures": float(z_arch.loc[a, m]),
                }
            )
    metric_long = pd.DataFrame(metric_long)

    ranking_a = pd.DataFrame(
        {
            "architecture": archs,
            "architecture_display": [disp[a] for a in archs],
            "C_overall_M18": [float(C_overall[a]) for a in archs],
            "C_overall_M14_prev": [float(C_14[a]) for a in archs],
            "C_added4_G_only": [float(C_4[a]) for a in archs],
            "C_profile": [float(C_profile[a]) for a in archs],
            "C_prediction": [float(C_pred[a]) for a in archs],
            "C_robustness_M10": [float(C_rob[a]) for a in archs],
            "M_overall": 18,
            "M_profile": len(side_cols["Profile"]),
            "M_prediction": len(side_cols["Prediction"]),
            "M_robustness": len(side_cols["Robustness"]),
        }
    )
    ranking_a["rank_overall_M18"] = ranking_a["C_overall_M18"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking_a["rank_overall_M14"] = ranking_a["C_overall_M14_prev"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking_a["rank_profile"] = ranking_a["C_profile"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking_a["rank_prediction"] = ranking_a["C_prediction"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking_a["rank_robustness"] = ranking_a["C_robustness_M10"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking_a = ranking_a.sort_values("rank_overall_M18")

    ranking_b = B_sum.reset_index()
    ranking_b.insert(1, "architecture_display", ranking_b["architecture"].map(disp))
    ranking_b["rank_overall"] = ranking_b["C_overall_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking_b = ranking_b.sort_values("rank_overall")

    seed_export = raw.copy()
    for m in metric_names:
        seed_export[f"z_{m}"] = z_seed[m].values
    seed_export["C_overall"] = raw_b["C_overall_seed"].values
    seed_export["C_profile"] = raw_b["C_profile_seed"].values
    seed_export["C_prediction"] = raw_b["C_prediction_seed"].values
    seed_export["C_robustness"] = raw_b["C_robustness_seed"].values

    contrib = z_wide.copy()
    contrib["C_overall_M18"] = contrib["architecture"].map(C_overall)
    contrib = contrib.sort_values("C_overall_M18", ascending=False)

    added4_means = means[["late_acc", "window_acc_std", "task_acc_var", "agent_acc_var"]].reset_index()
    added4_means.insert(1, "architecture_display", added4_means["architecture"].map(disp))

    qc = pd.DataFrame(
        [{"metric": m, "n_missing": n_miss[m], "pass": n_miss[m] == 0} for m in metric_names]
        + [
            {
                "metric": "M_equals_18",
                "n_missing": 0,
                "pass": len(metric_names) == 18,
            },
            {
                "metric": "added4_present",
                "n_missing": 0,
                "pass": all(
                    x in metric_names
                    for x in ("late_acc", "window_acc_std", "task_acc_var", "agent_acc_var")
                ),
            },
        ]
    )

    sheets = {
        "README": readme,
        "metric_definitions": metric_definitions,
        "z_standardization_params_A": pd.DataFrame(z_meta),
        "original_means_wide": orig_mean,
        "original_SEM_wide": orig_sem,
        "oriented_values_wide": oriented_wide,
        "z_scores_wide_A": z_wide,
        "metric_long_all": metric_long,
        "Ranking_A_arch_z": ranking_a,
        "Ranking_B_seed_z": ranking_b,
        "contribution_z_per_metric": contrib,
        "added4_G_means": added4_means,
        "RAW_per_seed_with_z": seed_export,
        "QC": qc,
    }
    write_wb(OUT_XLSX, sheets)
    raw.to_csv(DATA / "RAW_per_seed_M18.csv", index=False)
    ranking_a.to_csv(DATA / "equal_weight_z_Ranking_A_M18.csv", index=False)

    print(f"[OK] M={len(metric_names)} -> {OUT_XLSX}")
    print("missing:", n_miss)
    print(ranking_a[["rank_overall_M18", "rank_overall_M14", "architecture_display", "C_overall_M18", "C_overall_M14_prev"]].to_string(index=False))
    assert qc["pass"].all()


if __name__ == "__main__":
    main()
