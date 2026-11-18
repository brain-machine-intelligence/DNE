#!/usr/bin/env python3
"""Reproduce Fig. 6 panel statistics from shipped subject-level intermediates.

Panels
------
a  Alignment distance: paired S0 vs mean(S1–4), Welch Δ(DNE−Static), mixed ANOVA
b  Trade-off correlations (S2 and mean14 windows)
c  M* cue-mass centroids
d/e Forest of Δ relative proximity (DNE; bootstrap CI + Holm)

Inputs (relative to this figure folder)
---------------------------------------
  data/input/D_mstar_subject_session.csv
  data/input/subject_session_split.csv
  data/input/panel_c_mstar_xy_raw.csv
  data/input/subject_gain_{mean14,S2}_{incre,os}.csv

Full model re-fit / eval (Oneshot + M*) is not run here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests

FIG = Path(__file__).resolve().parents[1]
DATA_IN = FIG / "data" / "input"
OUT_A = FIG / "results" / "panel_a_alignment_distance"
OUT_B = FIG / "results" / "panel_b_tradeoff"
OUT_C = FIG / "results" / "panel_c_mstar_profiles"
OUT_D = FIG / "results" / "panel_d_proximity_control"
OUT_E = FIG / "results" / "panel_e_proximity_os"

N_BOOT = 10_000
RNG = np.random.default_rng(20260605)
POST = [1, 2, 3, 4]

ALT_ORDER = [
    "simple_bayesian",
    "reward_association",
    "uniform",
    "recency",
    "primacy",
    "novelty",
    "context_sensitive_os",
    "frequency",
]
ALT_LABEL = {
    "simple_bayesian": "Simple Bayesian",
    "reward_association": "Reward association",
    "uniform": "Uniform",
    "recency": "Recency",
    "primacy": "Primacy",
    "novelty": "Novelty",
    "context_sensitive_os": "Context-sensitive OS",
    "frequency": "Frequency",
}


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def bootstrap_mean_ci(vals: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    boots = RNG.choice(vals, size=(N_BOOT, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_split() -> pd.DataFrame:
    path = DATA_IN / "subject_session_split.csv"
    if path.exists():
        return pd.read_csv(path)
    dist = pd.read_csv(DATA_IN / "D_mstar_subject_session.csv")
    rows = []
    for (group, sid, os_label), g in dist.groupby(["group", "subject_id", "os_type_label"]):
        wide = g.set_index("session")["D_mstar"]
        if 0 not in wide.index or not all(s in wide.index for s in POST):
            continue
        rows.append(
            {
                "group": group,
                "subject_id": sid,
                "os_type_label": os_label,
                "l1_session0": float(wide[0]),
                "mean_l1_sessions_1_4": float(wide[POST].mean()),
                "l1_session2": float(wide[2]) if 2 in wide.index else np.nan,
            }
        )
    return pd.DataFrame(rows)


def panel_a(split: pd.DataFrame) -> None:
    OUT_A.mkdir(parents=True, exist_ok=True)
    paired_rows = []
    for os_label in ["Incre", "OS"]:
        for group in ["without", "with"]:
            g = split[(split.os_type_label == os_label) & (split.group == group)]
            delta = g["mean_l1_sessions_1_4"] - g["l1_session0"]
            t, p = stats.ttest_rel(g["mean_l1_sessions_1_4"], g["l1_session0"])
            paired_rows.append(
                {
                    "panel": "a",
                    "os_type_label": os_label,
                    "group": group,
                    "contrast": "mean(S1-4) vs S0",
                    "diff": float(delta.mean()),
                    "t": float(t),
                    "p": float(p),
                    "sig": stars(p),
                    "n": int(len(g)),
                }
            )
    st_a = pd.DataFrame(paired_rows)
    st_a.to_csv(OUT_A / "panel_a_stats.csv", index=False)

    # Holm within the 4 paired tests
    _, holm, _, _ = multipletests(st_a["p"].values, method="holm")
    follow = st_a.drop(columns=["panel", "sig"]).copy()
    follow["p_Holm"] = holm
    follow["sig"] = [stars(p) for p in follow["p"]]
    follow["sig_Holm"] = [stars(p) for p in holm]
    follow.to_csv(OUT_A / "panel_a_paired_followup.csv", index=False)

    # Welch: Δ_with − Δ_without per context
    welch_rows = []
    p_raw = []
    for os_label in ["Incre", "OS"]:
        w = split[(split.os_type_label == os_label) & (split.group == "with")]
        wo = split[(split.os_type_label == os_label) & (split.group == "without")]
        dw = (w["mean_l1_sessions_1_4"] - w["l1_session0"]).values
        dwo = (wo["mean_l1_sessions_1_4"] - wo["l1_session0"]).values
        t, p = stats.ttest_ind(dw, dwo, equal_var=False)
        p_raw.append(p)
        welch_rows.append(
            {
                "os_type_label": os_label,
                "contrast": "delta(with) vs delta(without)",
                "mean_delta_with": float(np.mean(dw)),
                "mean_delta_without": float(np.mean(dwo)),
                "diff_with_minus_without": float(np.mean(dw) - np.mean(dwo)),
                "t": float(t),
                "p": float(p),
                "n_with": int(len(dw)),
                "n_without": int(len(dwo)),
            }
        )
    _, holm_w, _, _ = multipletests(p_raw, method="holm")
    for r, ph in zip(welch_rows, holm_w):
        r["p_Holm"] = float(ph)
        r["sig"] = stars(r["p"])
        r["sig_Holm"] = stars(ph)
    pd.DataFrame(welch_rows).to_csv(OUT_A / "panel_a_welch_delta.csv", index=False)

    # Mixed ANOVA: D ~ group × time (S0 vs mean S1–4)
    aov_rows = []
    for os_label in ["Incre", "OS"]:
        sub = split[split.os_type_label == os_label]
        long = pd.concat(
            [
                sub.assign(time="S0", value=sub["l1_session0"]),
                sub.assign(time="mean_S1_4", value=sub["mean_l1_sessions_1_4"]),
            ],
            ignore_index=True,
        )
        aov = pg.mixed_anova(
            data=long, dv="value", within="time", subject="subject_id", between="group"
        )
        for _, row in aov.iterrows():
            aov_rows.append(
                {
                    "os_type_label": os_label,
                    "method": "pingouin_mixed_anova",
                    "effect": row["Source"],
                    "ddof1": row["DF1"],
                    "ddof2": row["DF2"],
                    "F": row["F"],
                    "p": row["p-unc"],
                    "np2": row["np2"],
                    "sig": stars(float(row["p-unc"])),
                }
            )
    pd.DataFrame(aov_rows).to_csv(OUT_A / "panel_a_mixed_anova.csv", index=False)

    # Keep subject distances alongside stats
    dist = pd.read_csv(DATA_IN / "D_mstar_subject_session.csv")
    dist.to_csv(OUT_A / "D_mstar_subject_session.csv", index=False)
    print("panel a →", OUT_A)


def alignment_gains(dist: pd.DataFrame, window: str) -> pd.DataFrame:
    rows = []
    for (group, sid, os_label), g in dist.groupby(["group", "subject_id", "os_type_label"]):
        wide = g.set_index("session")["D_mstar"]
        if 0 not in wide.index:
            continue
        d0 = float(wide[0])
        if window == "S2":
            if 2 not in wide.index:
                continue
            d_post = float(wide[2])
        else:
            if not all(s in wide.index for s in POST):
                continue
            d_post = float(wide[POST].mean())
        rows.append(
            {
                "group": group,
                "subject_id": sid,
                "os_type_label": os_label,
                "delta_D_mstar": d_post - d0,
                "alignment_gain": d0 - d_post,
            }
        )
    return pd.DataFrame(rows)


def panel_b(dist: pd.DataFrame) -> None:
    OUT_B.mkdir(parents=True, exist_ok=True)
    for window, tag in [("S2", "S2"), ("mean14", "mean14")]:
        gains = alignment_gains(dist, window)
        piv = gains.pivot_table(
            index=["group", "subject_id"], columns="os_type_label", values="alignment_gain"
        ).reset_index()
        piv = piv.rename(columns={"OS": "G_OS", "Incre": "G_Incre"})
        corr_rows = []
        for label, sub in [
            ("all", piv),
            ("with-DNE", piv[piv.group == "with"]),
            ("without-DNE", piv[piv.group == "without"]),
        ]:
            s = sub.dropna(subset=["G_OS", "G_Incre"])
            r, p = stats.pearsonr(s["G_OS"], s["G_Incre"])
            rho, p_s = stats.spearmanr(s["G_OS"], s["G_Incre"])
            corr_rows.append(
                {
                    "subset": label,
                    "n": int(len(s)),
                    "pearson_r": float(r),
                    "pearson_p": float(p),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_s),
                }
            )
        piv.to_csv(OUT_B / f"tradeoff_wide_{tag}.csv", index=False)
        pd.DataFrame(corr_rows).to_csv(OUT_B / f"tradeoff_corr_{tag}.csv", index=False)
    print("panel b →", OUT_B)


def panel_c() -> None:
    OUT_C.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_IN / "panel_c_mstar_xy_raw.csv")
    raw.to_csv(OUT_C / "panel_c_mstar_xy_raw.csv", index=False)
    cents = (
        raw.groupby("os_type_label", as_index=False)
        .agg(non_novel=("non_novel", "mean"), novel=("novel", "mean"), n=("eval_id", "count"))
    )
    cents.to_csv(OUT_C / "panel_c_mstar_xy_centroids.csv", index=False)
    print("panel c →", OUT_C)


def forest_from_gains(gain: pd.DataFrame, gain_col: str, window: str) -> pd.DataFrame:
    sub = gain[gain.group == "with"]
    tmp, pvals = [], []
    for alt in ALT_ORDER:
        vals = sub[sub.alternative_model == alt][gain_col].dropna().values
        mean, lo, hi = bootstrap_mean_ci(vals)
        t, p = stats.ttest_1samp(vals, 0.0)
        tmp.append(
            {
                "alternative_model": alt,
                "alternative_label": ALT_LABEL[alt],
                "mean": mean,
                "ci_lo_2p5": lo,
                "ci_hi_97p5": hi,
                "ci_lo_err": mean - lo,
                "ci_hi_err": hi - mean,
                "t": float(t),
                "p": float(p),
                "n": int(len(vals)),
                "window": window,
            }
        )
        pvals.append(p)
    _, holm, _, _ = multipletests(pvals, method="holm")
    for r, ph in zip(tmp, holm):
        r["p_Holm"] = float(ph)
        r["sig"] = stars(ph)
    return pd.DataFrame(tmp)


def panels_de() -> None:
    OUT_D.mkdir(parents=True, exist_ok=True)
    OUT_E.mkdir(parents=True, exist_ok=True)
    for window, tag, gcol in [
        ("mean14", "mean14", "gain_s0_mean14"),
        ("S2", "S2", "gain_s0_s2"),
    ]:
        g_inc = pd.read_csv(DATA_IN / f"subject_gain_{tag}_incre.csv")
        g_os = pd.read_csv(DATA_IN / f"subject_gain_{tag}_os.csv")
        g_inc.to_csv(OUT_D / f"subject_gain_{tag}_incre.csv", index=False)
        g_os.to_csv(OUT_E / f"subject_gain_{tag}_os.csv", index=False)
        forest_from_gains(g_inc, gcol, window).to_csv(OUT_D / f"forest_incre_{tag}.csv", index=False)
        forest_from_gains(g_os, gcol, window).to_csv(OUT_E / f"forest_os_{tag}.csv", index=False)
    print("panels d/e →", OUT_D, OUT_E)


def main() -> None:
    for p in (DATA_IN / "D_mstar_subject_session.csv", DATA_IN / "panel_c_mstar_xy_raw.csv"):
        if not p.exists():
            raise FileNotFoundError(p)
    split = load_split()
    dist = pd.read_csv(DATA_IN / "D_mstar_subject_session.csv")
    panel_a(split)
    panel_b(dist)
    panel_c()
    panels_de()
    print("Done. Stats written under results/panel_*/")


if __name__ == "__main__":
    main()
