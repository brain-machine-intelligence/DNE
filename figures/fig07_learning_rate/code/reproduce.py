#!/usr/bin/env python3
"""Reproduce Fig. 7 panel statistics from shipped intermediates.

Panels
------
a  Session-wise param trajectories: means/SEM, session Welch (DNE vs Static),
   2-way OLS ANOVA (group × session), S0 vs S2 paired
b  Single-parameter reset counterfactual \(D^{M*}\) (with-DNE): paired contrasts + Holm
c  Diagnostic ΔLR vs ΔL1 (later−now): per-session and mean(S1–4) Pearson (with-DNE)

Inputs
------
  data/input/param_session_indep.csv
  data/input/param_reset_subject_long.csv
  data/input/diag_incre_lr_l1_trajectory_subject_session.csv
  data/input/diag_lr_l1_trajectory_subject_session.csv

Full session-wise re-fit / Oneshot CF forward passes are not run here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

FIG = Path(__file__).resolve().parents[1]
DATA_IN = FIG / "data" / "input"
OUT_A = FIG / "results" / "panel_a_param_trajectories"
OUT_B = FIG / "results" / "panel_b_param_reset"
OUT_C = FIG / "results" / "panel_c_lr_vs_alignment"

PARAM_COLS = ["primacy", "recency", "learning rate", "inv_temp"]
PARAM_LABEL = {
    "primacy": "Primacy",
    "recency": "Recency",
    "learning rate": "Learning rate",
    "inv_temp": "Inverse temperature τ",
}


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def panel_a(params: pd.DataFrame) -> None:
    OUT_A.mkdir(parents=True, exist_ok=True)

    # tidy means
    tidy_rows = []
    for col in PARAM_COLS:
        for sess in range(5):
            row = {"parameter": col, "parameter_label": PARAM_LABEL[col], "session": sess}
            for group, tag in [("with", "with_DNE"), ("without", "without_DNE")]:
                vals = params[(params.type == group) & (params.session == sess)][col]
                row[f"{tag}_mean"] = float(vals.mean())
                row[f"{tag}_sem"] = float(vals.sem())
                row[f"{tag}_n"] = int(len(vals))
            tidy_rows.append(row)
    tidy = pd.DataFrame(tidy_rows)
    tidy.to_csv(OUT_A / "param_trajectories_tidy.csv", index=False)

    # session Welch between groups
    sess_rows = []
    for col in PARAM_COLS:
        for sess in range(5):
            w = params[(params.type == "with") & (params.session == sess)][col]
            wo = params[(params.type == "without") & (params.session == sess)][col]
            t, p = stats.ttest_ind(w, wo, equal_var=False)
            sess_rows.append(
                {
                    "parameter": col,
                    "session": sess,
                    "with_mean": float(w.mean()),
                    "without_mean": float(wo.mean()),
                    "diff": float(w.mean() - wo.mean()),
                    "t": float(t),
                    "p": float(p),
                    "sig": stars(p),
                    "n_with": int(len(w)),
                    "n_without": int(len(wo)),
                }
            )
    sess = pd.DataFrame(sess_rows)
    pieces = []
    for col, g in sess.groupby("parameter"):
        g = g.copy()
        _, holm, _, _ = multipletests(g["p"].values, method="holm")
        g["p_Holm"] = holm
        g["sig_Holm"] = [stars(p) for p in holm]
        pieces.append(g)
    sess = pd.concat(pieces, ignore_index=True)
    sess.to_csv(OUT_A / "param_trajectory_session_between.csv", index=False)

    # 2-way ANOVA via OLS
    anova_rows = []
    for col in PARAM_COLS:
        sub = params[["type", "session", "sbj_id", col]].rename(columns={"type": "group", col: "y"})
        model = smf.ols("y ~ C(group) * C(session)", data=sub).fit()
        aov = anova_lm(model, typ=2)
        for effect in aov.index:
            p = float(aov.loc[effect, "PR(>F)"])
            anova_rows.append(
                {
                    "parameter": col,
                    "effect": str(effect),
                    "F": float(aov.loc[effect, "F"]),
                    "p": p,
                    "sig": stars(p),
                }
            )
    pd.DataFrame(anova_rows).to_csv(OUT_A / "param_trajectory_anova.csv", index=False)

    # S0 vs S2 paired + between @ S2
    s0s2 = []
    for col in PARAM_COLS:
        for group in ["with", "without"]:
            wide = (
                params[params.type == group]
                .pivot_table(index="sbj_id", columns="session", values=col, aggfunc="first")
                .dropna(subset=[0, 2])
            )
            t, p = stats.ttest_rel(wide[2], wide[0])
            s0s2.append(
                {
                    "parameter": col,
                    "group": group,
                    "contrast": "S2 vs S0",
                    "diff": float((wide[2] - wide[0]).mean()),
                    "t": float(t),
                    "p": float(p),
                    "sig": stars(p),
                    "n": int(len(wide)),
                }
            )
        w = params[(params.type == "with") & (params.session == 2)][col]
        wo = params[(params.type == "without") & (params.session == 2)][col]
        t, p = stats.ttest_ind(w, wo, equal_var=False)
        s0s2.append(
            {
                "parameter": col,
                "group": "between",
                "contrast": "with vs without @ S2",
                "diff": float(w.mean() - wo.mean()),
                "t": float(t),
                "p": float(p),
                "sig": stars(p),
                "n": int(len(w) + len(wo)),
            }
        )
    pd.DataFrame(s0s2).to_csv(OUT_A / "param_S0_vs_S2_tests.csv", index=False)
    print("panel a →", OUT_A)


def panel_b(long: pd.DataFrame) -> None:
    OUT_B.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT_B / "param_reset_subject_long.csv", index=False)
    sub = long[long.group == "with"].copy()
    wide = (
        sub.pivot_table(index=["subject_id", "os_type_label"], columns="scenario", values="D_mstar_mean")
        .reset_index()
    )
    contrasts = [
        ("S0 vs S2", "actual_s0", "actual_s2"),
        ("S0 vs LR→S0", "actual_s0", "lr_reset_s0"),
        ("S0 vs Primacy→S0", "actual_s0", "primacy_reset_s0"),
        ("S0 vs Recency→S0", "actual_s0", "recency_reset_s0"),
        ("S0 vs τ→S0", "actual_s0", "tau_reset_s0"),
        ("S2 vs LR→S0", "actual_s2", "lr_reset_s0"),
        ("S2 vs Primacy→S0", "actual_s2", "primacy_reset_s0"),
        ("S2 vs Recency→S0", "actual_s2", "recency_reset_s0"),
        ("S2 vs τ→S0", "actual_s2", "tau_reset_s0"),
    ]
    out = []
    for panel in ("Incre", "OS"):
        w = wide[wide.os_type_label == panel]
        tmp = []
        for name, a, b in contrasts:
            x, y = w[a].values, w[b].values
            diff = x - y
            if np.allclose(diff, 0):
                t, p = 0.0, 1.0
            else:
                t, p = stats.ttest_rel(x, y)
            tmp.append(
                {
                    "panel": "Control rounds" if panel == "Incre" else "OS rounds",
                    "os_type_label": panel,
                    "contrast": name,
                    "mean_A": float(np.mean(x)),
                    "mean_B": float(np.mean(y)),
                    "diff_A_minus_B": float(np.mean(diff)),
                    "sem_diff": float(np.std(diff, ddof=1) / np.sqrt(len(diff))),
                    "n": int(len(diff)),
                    "t": float(t),
                    "p": float(p),
                }
            )
        _, holm, _, _ = multipletests([r["p"] for r in tmp], method="holm")
        for r, ph in zip(tmp, holm):
            r["p_Holm"] = float(ph)
            r["sig"] = stars(ph)
            out.append(r)
    pd.DataFrame(out).to_csv(OUT_B / "param_reset_pairwise.csv", index=False)
    print("panel b →", OUT_B)


def _corr_table(df: pd.DataFrame, os_label: str, subset: str) -> pd.DataFrame:
    rows = []
    for sess in [1, 2, 3, 4]:
        for cond in ["with-DNE", "without-DNE"]:
            s = df[(df.session == sess) & (df.condition == cond)]
            r, p = stats.pearsonr(s["delta_lr"], s["delta_l1_from_s0"])
            rows.append(
                {
                    "os_type_label": os_label,
                    "subset": subset,
                    "condition": cond,
                    "session": sess,
                    "n": int(len(s)),
                    "pearson_r_delta_lr_vs_delta_l1": float(r),
                    "p_delta_lr_vs_delta_l1": float(p),
                }
            )
    # mean14 (subject mean of S1–4 deltas)
    post = (
        df[df.session.isin([1, 2, 3, 4])]
        .groupby(["group", "subject_id", "condition"], as_index=False)[["delta_lr", "delta_l1_from_s0"]]
        .mean()
    )
    for cond in ["with-DNE", "without-DNE"]:
        s = post[post.condition == cond]
        r, p = stats.pearsonr(s["delta_lr"], s["delta_l1_from_s0"])
        rows.append(
            {
                "os_type_label": os_label,
                "subset": subset,
                "condition": cond,
                "session": "mean14",
                "n": int(len(s)),
                "pearson_r_delta_lr_vs_delta_l1": float(r),
                "p_delta_lr_vs_delta_l1": float(p),
            }
        )
    return pd.DataFrame(rows)


def panel_c() -> None:
    OUT_C.mkdir(parents=True, exist_ok=True)
    control = pd.read_csv(DATA_IN / "diag_incre_lr_l1_trajectory_subject_session.csv")
    os_df = pd.read_csv(DATA_IN / "diag_lr_l1_trajectory_subject_session.csv")
    corr_c = _corr_table(control, "Incre", "incre_diagnostic_top50")
    corr_o = _corr_table(os_df, "OS", "os_diagnostic_top50")
    corr_c.to_csv(OUT_C / "corr_control_by_session.csv", index=False)
    corr_o.to_csv(OUT_C / "corr_os_by_session.csv", index=False)

    # figure-reported window: mean14, with-DNE
    fig_rows = []
    for lab, df, subset in [
        ("Control rounds", control, "incre_diagnostic_top50"),
        ("OS rounds", os_df, "os_diagnostic_top50"),
    ]:
        post = (
            df[(df.session.isin([1, 2, 3, 4])) & (df.condition == "with-DNE")]
            .groupby(["subject_id"], as_index=False)[["delta_lr", "delta_l1_from_s0"]]
            .mean()
        )
        r, p = stats.pearsonr(post["delta_lr"], post["delta_l1_from_s0"])
        fig_rows.append(
            {
                "panel": lab,
                "window": "mean14",
                "condition": "with-DNE",
                "n": int(len(post)),
                "pearson_r": float(r),
                "p": float(p),
                "sig": stars(p),
                "subset": subset,
            }
        )
        post.assign(panel=lab).to_csv(
            OUT_C / f"subject_mean14_{'control' if 'Control' in lab else 'os'}_with_dne.csv",
            index=False,
        )
    pd.DataFrame(fig_rows).to_csv(OUT_C / "panel_c_figure_correlations.csv", index=False)

    # also keep S2 summary (often used in supplements)
    s2_rows = []
    for variant, df in [("incre", control), ("os", os_df)]:
        for cond in ["with-DNE", "without-DNE"]:
            s = df[(df.session == 2) & (df.condition == cond)]
            r, p = stats.pearsonr(s["delta_lr"], s["delta_l1_from_s0"])
            s2_rows.append(
                {
                    "variant": variant,
                    "condition": cond,
                    "session": 2,
                    "n": int(len(s)),
                    "r_delta_lr_vs_delta_l1": float(r),
                    "p": float(p),
                }
            )
    pd.DataFrame(s2_rows).to_csv(OUT_C / "session2_correlation_summary.csv", index=False)
    print("panel c →", OUT_C)


def main() -> None:
    params = pd.read_csv(DATA_IN / "param_session_indep.csv")
    long = pd.read_csv(DATA_IN / "param_reset_subject_long.csv")
    panel_a(params)
    panel_b(long)
    panel_c()
    print("Done. Stats written under results/panel_*/")


if __name__ == "__main__":
    main()
