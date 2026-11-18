#!/usr/bin/env python3
"""Reproduce main Fig. 5 / Supp. Fig. 12 statistics (matched cohort).

Analyses
--------
Fig. 5c  : one-way RM-ANOVA (cue) + Greenhouse–Geisser + Šidák pairwise,
           separately for each Condition × RoundType cell.
Fig. 5d left : two-way mixed ANOVA
           os_index_norm ~ Condition (between) × RoundType (within)
           + uncorrected between-condition t-tests within each round type.
Fig. 5d right / Supp. Fig. 12 :
           os_index_norm ~ Condition × Session (within OS or control)
           + GG correction + session-wise Welch t-tests (uncorrected and Šidák m=5).

OS index (normalized)
---------------------
  (R_novel - mean(R_8x, R_16x)) / (R_novel + R_8x + R_16x)

Inputs (relative to repository root)
------------------------------------
  data/matched_ids/static_ids.txt
  data/matched_ids/dne_ids.txt
  data/behavioural/wo_dne/all_subjects_view.csv
  data/behavioural/w_dne/all_subjects_view.csv

Outputs are written to results/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

FIG = Path(__file__).resolve().parents[1]
DATA = FIG / "data"
OUT = FIG / "results"
OUT.mkdir(parents=True, exist_ok=True)

SESSIONS = ["S0", "S1", "S2", "S3", "S4"]


def sidak_adjust(pvals, m: int):
    p = np.asarray(pvals, float)
    return 1.0 - (1.0 - np.clip(p, 0, 1)) ** m


def load_ids(path: Path):
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_raw():
    static_ids = set(load_ids(DATA / "matched_ids" / "static_ids.txt"))
    dne_ids = set(load_ids(DATA / "matched_ids" / "dne_ids.txt"))
    paths = {
        "static": DATA / "behavioural" / "wo_dne" / "all_subjects_view.csv",
        "dne": DATA / "behavioural" / "w_dne" / "all_subjects_view.csv",
    }
    frames = []
    for cond, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Place aggregated behavioural CSVs under data/behavioural/."
            )
        df = pd.read_csv(path)
        df["subid"] = df["subid"].astype(str)
        keep = static_ids if cond == "static" else dne_ids
        df = df[df["subid"].isin(keep)].copy()
        df["Condition"] = cond
        df["RoundType"] = df["type"].map({0: "OS", 1: "control"})
        df["Session"] = "S" + (df["session"].astype(int) - 1).astype(str)
        # Prefer CSV columns; recompute if needed
        num = df["rating_unique_cr"] - 0.5 * (
            df["rating_stim8_cr"] + df["rating_stim16_cr"]
        )
        denom = (
            df["rating_unique_cr"] + df["rating_stim8_cr"] + df["rating_stim16_cr"]
        )
        recomputed = num / denom.replace(0, np.nan)
        if "os_index_norm" in df.columns:
            df["os_index_norm"] = pd.to_numeric(df["os_index_norm"], errors="coerce")
            miss = df["os_index_norm"].isna()
            df.loc[miss, "os_index_norm"] = recomputed[miss]
        else:
            df["os_index_norm"] = recomputed
        if "os_index" not in df.columns:
            df["os_index"] = num
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    print(
        "Loaded matched cohort:",
        raw.groupby("Condition")["subid"].nunique().to_dict(),
        "rows",
        len(raw),
    )
    return raw


def fig5c(raw: pd.DataFrame):
    """Cue-type RM-ANOVA + Šidák (session-averaged ratings)."""
    subj = (
        raw.groupby(["subid", "Condition", "RoundType"], as_index=False)
        .agg(
            cr_16x=("rating_stim16_cr", "mean"),
            cr_8x=("rating_stim8_cr", "mean"),
            novel=("rating_unique_cr", "mean"),
        )
    )
    aov_rows, pw_rows, mean_rows = [], [], []
    for rt in ["control", "OS"]:
        for cond in ["static", "dne"]:
            g = subj[(subj.RoundType == rt) & (subj.Condition == cond)].copy()
            mean_rows.append(
                dict(
                    RoundType=rt,
                    Condition=cond,
                    n=len(g),
                    mean_16x=g.cr_16x.mean(),
                    sem_16x=g.cr_16x.sem(),
                    mean_8x=g.cr_8x.mean(),
                    sem_8x=g.cr_8x.sem(),
                    mean_novel=g.novel.mean(),
                    sem_novel=g.novel.sem(),
                )
            )
            long = g.melt(
                id_vars=["subid"],
                value_vars=["cr_16x", "cr_8x", "novel"],
                var_name="cue",
                value_name="CR",
            )
            long["cue"] = long["cue"].map(
                {"cr_16x": "16x", "cr_8x": "8x", "novel": "novel"}
            )
            aov = pg.rm_anova(
                data=long, dv="CR", within="cue", subject="subid", detailed=True
            )
            cue = aov[aov.Source == "cue"].iloc[0]
            err = aov[aov.Source == "Error"].iloc[0]
            aov_rows.append(
                dict(
                    RoundType=rt,
                    Condition=cond,
                    F=float(cue.F),
                    df1=int(cue.DF),
                    df2=int(err.DF),
                    p_unc=float(cue["p-unc"]),
                    p_GG=float(cue["p-GG-corr"]),
                    eps=float(cue.eps),
                    np2=float(cue.ng2),
                )
            )
            pw = pg.pairwise_tests(
                data=long,
                dv="CR",
                within="cue",
                subject="subid",
                padjust="sidak",
            )
            for _, r in pw.iterrows():
                pw_rows.append(
                    dict(
                        RoundType=rt,
                        Condition=cond,
                        A=r["A"],
                        B=r["B"],
                        t=float(r["T"]),
                        dof=float(r["dof"]),
                        p_unc=float(r["p-unc"]),
                        p_sidak=float(r["p-corr"]),
                    )
                )
    aov_df = pd.DataFrame(aov_rows)
    pw_df = pd.DataFrame(pw_rows)
    mean_df = pd.DataFrame(mean_rows)
    aov_df.to_csv(OUT / "fig5c_rm_anova.csv", index=False)
    pw_df.to_csv(OUT / "fig5c_sidak_pairwise.csv", index=False)
    mean_df.to_csv(OUT / "fig5c_cue_means.csv", index=False)
    print("\n=== Fig. 5c RM-ANOVA ===")
    print(aov_df.to_string(index=False))
    print("\n=== Fig. 5c Šidák pairwise ===")
    print(pw_df.to_string(index=False))
    return aov_df, pw_df


def fig5d_left(raw: pd.DataFrame):
    """RoundType × Condition on subject-mean os_index_norm (all sessions)."""
    subj = (
        raw.groupby(["subid", "Condition", "RoundType"], as_index=False)[
            "os_index_norm"
        ]
        .mean()
    )
    aov = pg.mixed_anova(
        data=subj,
        dv="os_index_norm",
        within="RoundType",
        between="Condition",
        subject="subid",
    )
    rows = []
    for rt in ["control", "OS"]:
        a = subj[(subj.RoundType == rt) & (subj.Condition == "static")][
            "os_index_norm"
        ]
        b = subj[(subj.RoundType == rt) & (subj.Condition == "dne")]["os_index_norm"]
        t, p = stats.ttest_ind(a, b, equal_var=True)
        rows.append(
            dict(
                RoundType=rt,
                mean_static=float(a.mean()),
                mean_dne=float(b.mean()),
                t=float(t),
                dof=len(a) + len(b) - 2,
                p_uncorrected=float(p),
            )
        )
    simple = pd.DataFrame(rows)
    aov.to_csv(OUT / "fig5d_left_mixed_anova.csv", index=False)
    simple.to_csv(OUT / "fig5d_left_simple_effects.csv", index=False)
    print("\n=== Fig. 5d left mixed ANOVA ===")
    print(aov.to_string(index=False))
    print("\n=== Fig. 5d left between-condition (uncorrected) ===")
    print(simple.to_string(index=False))
    return aov, simple


def session_wise(raw: pd.DataFrame, round_type: str, tag: str):
    """Condition × Session mixed ANOVA + Welch (and Šidák m=5)."""
    subj = (
        raw[raw.RoundType == round_type]
        .groupby(["subid", "Condition", "Session"], as_index=False)["os_index_norm"]
        .mean()
    )
    subj["Session"] = pd.Categorical(subj["Session"], categories=SESSIONS)
    aov = pg.mixed_anova(
        data=subj,
        dv="os_index_norm",
        within="Session",
        between="Condition",
        subject="subid",
        correction="auto",
    )
    # ensure GG p for Session & Interaction
    eps = float(aov.loc[aov.Source == "Session", "eps"].iloc[0])
    for src in ["Session", "Interaction"]:
        row = aov.Source == src
        F = float(aov.loc[row, "F"].iloc[0])
        df1 = float(aov.loc[row, "DF1"].iloc[0])
        df2 = float(aov.loc[row, "DF2"].iloc[0])
        aov.loc[row, "p-GG-corr"] = stats.f.sf(F, df1 * eps, df2 * eps)
        aov.loc[row, "df1_GG"] = df1 * eps
        aov.loc[row, "df2_GG"] = df2 * eps
        aov.loc[row, "eps"] = eps

    sess_rows = []
    for sess in SESSIONS:
        chunk = subj[subj.Session == sess]
        a = chunk.loc[chunk.Condition == "static", "os_index_norm"].astype(float)
        b = chunk.loc[chunk.Condition == "dne", "os_index_norm"].astype(float)
        t_w, p_w = stats.ttest_ind(a, b, equal_var=False)
        t_eq, p_eq = stats.ttest_ind(a, b, equal_var=True)
        va, vb = a.var(ddof=1), b.var(ddof=1)
        na, nb = len(a), len(b)
        dof_w = (va / na + vb / nb) ** 2 / (
            (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        )
        sess_rows.append(
            dict(
                Session=sess,
                mean_static=float(a.mean()),
                mean_dne=float(b.mean()),
                diff_dne_minus_static=float(b.mean() - a.mean()),
                t_welch=float(t_w),
                dof_welch=float(dof_w),
                p_welch=float(p_w),
                t_equal_var=float(t_eq),
                p_equal_var=float(p_eq),
                n_static=na,
                n_dne=nb,
            )
        )
    post = pd.DataFrame(sess_rows)
    post["p_welch_sidak_m5"] = sidak_adjust(post["p_welch"], 5)
    post["p_equal_sidak_m5"] = sidak_adjust(post["p_equal_var"], 5)

    aov.to_csv(OUT / f"{tag}_mixed_anova.csv", index=False)
    post.to_csv(OUT / f"{tag}_session_welch.csv", index=False)
    print(f"\n=== {tag} mixed ANOVA ===")
    print(aov.to_string(index=False))
    print(f"\n=== {tag} session-wise Welch ===")
    print(post.to_string(index=False))
    return aov, post


def main():
    raw = load_raw()
    fig5c(raw)
    fig5d_left(raw)
    session_wise(raw, "OS", "fig5d_right_OS")
    session_wise(raw, "control", "supp_fig12_control")
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
