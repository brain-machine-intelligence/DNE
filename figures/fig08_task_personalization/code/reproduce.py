#!/usr/bin/env python3
"""Reproduce Fig. 8 panel statistics from shipped intermediates.

Panels
------
b  Sequence-condition decoding AUC (S0 vs S1–S4 windows; frozen OOF + perm)
c  Cross-participant 30D L2 distance (Control / OS × Static / DNE)
d  TPS axis construction (weights + example projections; tables only)
e  Mean TPS (CV) by round type × group (Welch)
f  Preceding behavioural separation → subsequent TPS (MixedLM + perm table)
g  TPS → ΔLR / Δ alignment (session FE OLS, participant-clustered SE)

Schematic panel (a) has no numeric recompute.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import t as tdist

FIG = Path(__file__).resolve().parents[1]
DATA_IN = FIG / "data" / "input"
OUT = {
    "a": FIG / "results" / "panel_a_schematic",
    "b": FIG / "results" / "panel_b_sequence_decoding",
    "c": FIG / "results" / "panel_c_task_distance",
    "d": FIG / "results" / "panel_d_tps_axis",
    "e": FIG / "results" / "panel_e_tps_means",
    "f": FIG / "results" / "panel_f_behav_to_tps",
    "g": FIG / "results" / "panel_g_tps_coupling",
}


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def panel_a() -> None:
    OUT["a"].mkdir(parents=True, exist_ok=True)
    note = pd.DataFrame(
        [
            {
                "panel": "a",
                "content": "30D cue–outcome pair-type counts (5 trials × 6 pair types)",
                "note": "schematic only (no numeric table)",
            }
        ]
    )
    note.to_csv(OUT["a"] / "README_note.csv", index=False)
    print("panel a →", OUT["a"])


def panel_b() -> None:
    OUT["b"].mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(DATA_IN / "panel_C_contrast_decoding_windows.csv")
    redraw = pd.read_csv(DATA_IN / "panel_b_decoding_session_bars.csv")
    windows.to_csv(OUT["b"] / "contrast_decoding_windows.csv", index=False)
    # figure bars: session_1 (=S0) vs sessions_2_5 (=S1–S4)
    bars = redraw[redraw["label"].isin(["session_1", "sessions_2_5"])].copy()
    bars["paper_label"] = bars["label"].map(
        {"session_1": "S0", "sessions_2_5": "S1–S4"}
    )
    bars["sig"] = bars["p_perm"].map(stars)
    bars.to_csv(OUT["b"] / "panel_b_bars.csv", index=False)
    print("panel b →", OUT["b"])


def panel_c() -> None:
    OUT["c"].mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(DATA_IN / "panel_A_exposure_summary_by_trial_type.csv")
    # Control = Incre, OS = OS (full_30D_*)
    rows = []
    for trial, paper in [("Incre", "Control"), ("OS", "OS")]:
        tag = f"full_30D_{trial}"
        sub = summ[summ.feature_set == tag]
        w = sub[sub.group == "with"].iloc[0]
        wo = sub[sub.group == "without"].iloc[0]
        rows.append(
            {
                "round": paper,
                "condition": "DNE",
                "group": "with",
                "mean_cross_session": float(w.mean_cross_session),
                "sem_cross_session": float(w.sem_cross_session),
                "sd_cross_session": float(w.sd_cross_session),
                "n_sessions": int(w.n_sessions),
                "obs_cross_diff": float(w.obs_cross_diff),
                "perm_p_cross_diff": float(w.perm_p_cross_diff),
                "sig": stars(float(w.perm_p_cross_diff)),
            }
        )
        rows.append(
            {
                "round": paper,
                "condition": "Static",
                "group": "without",
                "mean_cross_session": float(wo.mean_cross_session),
                "sem_cross_session": float(wo.sem_cross_session),
                "sd_cross_session": float(wo.sd_cross_session),
                "n_sessions": int(wo.n_sessions),
                "obs_cross_diff": float(wo.obs_cross_diff),
                "perm_p_cross_diff": float(wo.perm_p_cross_diff),
                "sig": stars(float(wo.perm_p_cross_diff)),
            }
        )
    pd.DataFrame(rows).to_csv(OUT["c"] / "mean_pairwise_distance.csv", index=False)
    # round to 3 dp companion
    out3 = pd.DataFrame(rows).copy()
    for c in ["mean_cross_session", "sem_cross_session", "sd_cross_session", "obs_cross_diff"]:
        out3[c] = out3[c].round(3)
    out3.to_csv(OUT["c"] / "mean_pairwise_distance_3dp.csv", index=False)
    print("panel c →", OUT["c"])


def panel_d() -> None:
    OUT["d"].mkdir(parents=True, exist_ok=True)
    for name in [
        "panel_A_tps_axis_group_summary_cv.csv",
        "panel_B_examples_cv.csv",
        "panel_C_personalization_weights_cv.csv",
    ]:
        pd.read_csv(DATA_IN / name).to_csv(OUT["d"] / name, index=False)
    print("panel d →", OUT["d"])


def panel_e() -> None:
    OUT["e"].mkdir(parents=True, exist_ok=True)
    subj = pd.read_csv(DATA_IN / "panel_D_dne_score_s25_subject.csv")
    summ = pd.read_csv(DATA_IN / "panel_D_dne_score_s25_summary.csv")
    summ.to_csv(OUT["e"] / "tps_s25_summary.csv", index=False)
    rows = []
    for subset, paper in [("Incre", "Control rounds"), ("OS", "OS rounds")]:
        w = subj[(subj.trial_subset == subset) & (subj.group == "with")]["dne_score_cv"]
        wo = subj[(subj.trial_subset == subset) & (subj.group == "without")]["dne_score_cv"]
        t, p = stats.ttest_ind(w, wo, equal_var=False)
        rows.append(
            {
                "panel": paper,
                "trial_subset": subset,
                "mean_DNE": float(w.mean()),
                "sem_DNE": float(w.sem()),
                "n_DNE": int(len(w)),
                "mean_Static": float(wo.mean()),
                "sem_Static": float(wo.sem()),
                "n_Static": int(len(wo)),
                "diff": float(w.mean() - wo.mean()),
                "t_welch": float(t),
                "p_welch": float(p),
                "sig": stars(p),
            }
        )
    pd.DataFrame(rows).to_csv(OUT["e"] / "tps_means_welch.csv", index=False)
    subj.to_csv(OUT["e"] / "tps_s25_subject.csv", index=False)
    print("panel e →", OUT["e"])


def panel_f() -> None:
    OUT["f"].mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(DATA_IN / "panel_A_behaviour_panel_full.csv")
    d = panel.dropna(subset=["dne_score_cv", "behav_dev_z", "prev_dne_score_cv"]).copy()
    d = d[d.session >= 4]
    d["condition_with"] = (d.group == "with").astype(int)
    formula = "dne_score_cv ~ prev_dne_score_cv + behav_dev_z * condition_with + C(session)"
    m = smf.mixedlm(formula, d, groups=d["subid"]).fit(reml=False, method="powell")
    term = "behav_dev_z:condition_with"
    beta = float(m.params[term])
    p_lmm = float(m.pvalues[term])
    # Pearson on scatter export (figure annotation)
    sc = pd.read_csv(DATA_IN / "panel_A_scatter_data.csv")
    r, r_p = stats.pearsonr(sc["behav_dev"], sc["dne_score_cv"])
    stats_df = pd.DataFrame(
        [
            {
                "beta_int": beta,
                "p_lmm": p_lmm,
                "p_perm": 0.0009990009990009,  # frozen 1000-perm from evidence pipeline
                "pearson_r_scatter": float(r),
                "pearson_p_scatter": float(r_p),
                "n_obs": int(len(d)),
                "formula": formula,
                "note": "p_perm from behaviour_contingent_permutation (shipped)",
            }
        ]
    )
    stats_df.to_csv(OUT["f"] / "behav_to_tps_stats.csv", index=False)
    # short form matching prior Desktop CSV
    stats_df[["beta_int", "p_lmm", "p_perm"]].to_csv(
        OUT["f"] / "behav_to_tps_stats_short.csv", index=False
    )
    sc.to_csv(OUT["f"] / "panel_A_scatter_data.csv", index=False)
    print("panel f →", OUT["f"], f"β={beta:.3f}, P_LMM={p_lmm:.3g}")


def _fit_clustered(sub: pd.DataFrame) -> dict:
    d = sub.copy()
    d["condition_with"] = (d.group == "with").astype(int)
    ols = smf.ols("y ~ x * condition_with + C(session_t)", d).fit()
    cl = ols.get_robustcov_results(
        cov_type="cluster", groups=d["subid"].to_numpy(), use_t=True
    )
    names = list(ols.model.exog_names)
    ix = names.index("x")
    ii = names.index("x:condition_with")
    cov = np.asarray(cl.cov_params())
    b_s = float(cl.params[ix])
    b_i = float(cl.params[ii])
    b_d = b_s + b_i
    se_s = float(np.sqrt(cov[ix, ix]))
    se_i = float(np.sqrt(cov[ii, ii]))
    se_d = float(np.sqrt(cov[ix, ix] + cov[ii, ii] + 2 * cov[ix, ii]))
    df = int(d["subid"].nunique() - 1)

    def p_t(b, se):
        return float(2 * tdist.sf(abs(b / se), df))

    out = {
        "interaction": {"beta": b_i, "p_clustered": p_t(b_i, se_i)},
        "without": {"beta": b_s, "p_clustered": p_t(b_s, se_s)},
        "with": {"beta": b_d, "p_clustered": p_t(b_d, se_d)},
    }
    for group, key in (("without", "without"), ("with", "with")):
        g = d[d.group == group]
        lr = stats.linregress(g["x"], g["y"])
        out[key].update({"r": float(lr.rvalue), "r_p": float(lr.pvalue), "n": int(len(g))})
    out["interaction"].update({"r": np.nan, "r_p": np.nan, "n": int(len(d))})
    return out


def panel_g() -> None:
    OUT["g"].mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(DATA_IN / "panel_g_scatter_data.csv")
    pros = pd.read_csv(DATA_IN / "prospective_enriched.csv")
    panel["subid"] = panel["subid"].astype(str)
    pros["subid"] = pros["subid"].astype(str)
    panel = panel.merge(
        pros[["group", "subid", "session_t", "G_Incre"]],
        on=["group", "subid", "session_t"],
        how="left",
    )
    defs = [
        ("Incre", "delta_LR", False, "Incre delta_LR"),
        ("OS", "delta_LR", False, "OS delta_LR"),
        ("Incre", "G_Incre", True, "Incre G_Incre"),
        ("OS", "G_OS", True, "OS G_OS"),
    ]
    rows = []
    for subset, ycol, negate, label in defs:
        sub = (
            panel[panel.trial_subset == subset][["group", "subid", "session_t", "dne_score_cv", ycol]]
            .dropna()
            .copy()
        )
        sub["x"] = sub["dne_score_cv"]
        sub["y"] = -sub[ycol] if negate else sub[ycol]
        fits = _fit_clustered(sub)
        for scope, key in (("Static", "without"), ("DNE", "with"), ("interaction", "interaction")):
            rec = fits[key]
            rows.append(
                {
                    "panel": label,
                    "scope": scope,
                    "beta": rec["beta"],
                    "p_clustered": rec["p_clustered"],
                    "r": rec.get("r", np.nan),
                    "r_p": rec.get("r_p", np.nan),
                }
            )
    st = pd.DataFrame(rows)
    st.to_csv(OUT["g"] / "tps_to_delta_params_stats.csv", index=False)
    st3 = st.copy()
    st3["beta"] = st3["beta"].round(3)
    st3["p_clustered"] = st3["p_clustered"].round(3)
    st3["r"] = st3["r"].round(3)
    st3["r_p"] = st3["r_p"].round(3)
    st3.to_csv(OUT["g"] / "tps_to_delta_params_stats_3dp.csv", index=False)
    print("panel g →", OUT["g"])


def main() -> None:
    panel_a()
    panel_b()
    panel_c()
    panel_d()
    panel_e()
    panel_f()
    panel_g()
    print("Done. Stats written under results/panel_*/")


if __name__ == "__main__":
    main()
