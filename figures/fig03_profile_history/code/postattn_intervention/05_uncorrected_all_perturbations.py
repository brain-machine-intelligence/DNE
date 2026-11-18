#!/usr/bin/env python3
"""Uncorrected tests for every perturbation condition.

Two questions, both on raw seed-level losses, with no multiple-comparison adjustment:
1. Does each model's mean loss differ from 0?
2. Do models differ from each other?
"""
from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy import stats

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pert_tests", HERE / "03_perturbation_raw_tests.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

OUT = HERE.parent / "data" / "Perturbation_uncorrected_tests.xlsx"


def write_sheet(wb, name, frame, first=False):
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
    ws.row_dimensions[1].height = 30


def vs_zero(blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in blocks.itertuples(index=False):
        s = mod.summarise(rec.values)
        two = stats.ttest_1samp(s["values"], 0.0, alternative="two-sided")
        rows.append(
            {
                "family": rec.family,
                "condition": rec.condition,
                "metric": rec.metric,
                "model": rec.architecture_display,
                "mean": s["mean"],
                "sem": s["sem"],
                "n": s["n"],
                "t_stat": float(two.statistic),
                "df": float(s["n"] - 1),
                "p_two_sided": float(two.pvalue),
                "different_from_0_p05": bool(two.pvalue < 0.05),
            }
        )
    return pd.DataFrame(rows)


def between_models(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows = []
    pair_rows = []
    for (family, cond, metric), g in blocks.groupby(["family", "condition", "metric"], sort=False):
        samples = [mod.summarise(rec.values)["values"] for rec in g.itertuples(index=False)]
        labels = [rec.architecture_display for rec in g.itertuples(index=False)]
        # Welch ANOVA (unequal-variance one-way), not Alexander-Govern
        long = pd.DataFrame(
            {
                "loss": np.concatenate(samples),
                "model": np.concatenate(
                    [np.repeat(lab, len(s)) for lab, s in zip(labels, samples)]
                ),
            }
        )
        wa = pg.welch_anova(data=long, dv="loss", between="model")
        omnibus_rows.append(
            {
                "family": family,
                "condition": cond,
                "metric": metric,
                "n_models": len(samples),
                "welch_anova_stat": float(wa["F"].iloc[0]),
                "welch_anova_df1": float(wa["ddof1"].iloc[0]),
                "welch_anova_df2": float(wa["ddof2"].iloc[0]),
                "welch_anova_p": float(wa["p-unc"].iloc[0]),
                "models_differ_p05": bool(wa["p-unc"].iloc[0] < 0.05),
            }
        )
        indexed = list(g.itertuples(index=False))
        for a, b in combinations(indexed, 2):
            sa, sb = mod.summarise(a.values), mod.summarise(b.values)
            tw = stats.ttest_ind(sa["values"], sb["values"], equal_var=False, alternative="two-sided")
            # Welch–Satterthwaite df (scipy<1.11 may omit .df on the result)
            v1 = float(np.var(sa["values"], ddof=1)) / sa["n"]
            v2 = float(np.var(sb["values"], ddof=1)) / sb["n"]
            welch_df = (v1 + v2) ** 2 / (
                v1**2 / (sa["n"] - 1) + v2**2 / (sb["n"] - 1)
            )
            pair_rows.append(
                {
                    "family": family,
                    "condition": cond,
                    "metric": metric,
                    "model_a": a.architecture_display,
                    "model_b": b.architecture_display,
                    "mean_a": sa["mean"],
                    "mean_b": sb["mean"],
                    "sem_a": sa["sem"],
                    "sem_b": sb["sem"],
                    "n_a": sa["n"],
                    "n_b": sb["n"],
                    "difference_a_minus_b": sa["mean"] - sb["mean"],
                    "welch_t": float(tw.statistic),
                    "welch_df": float(welch_df),
                    "p_two_sided": float(tw.pvalue),
                    "different_p05": bool(tw.pvalue < 0.05),
                }
            )
    return pd.DataFrame(omnibus_rows), pd.DataFrame(pair_rows)


def main():
    blocks = mod.collect()
    zero = vs_zero(blocks)
    omnibus, pairs = between_models(blocks)
    methods = pd.DataFrame(
        [
            ("values", "Raw seed-level Prediction Loss and Agent Accuracy Loss. Normalized scores are not tested."),
            ("vs_zero", "One-sample t-test, two-sided: mean loss differs from 0. No multiplicity correction."),
            ("between_models_omnibus", "Welch ANOVA (unequal-variance one-way ANOVA; pingouin.welch_anova), within each condition and metric."),
            ("between_models_pairs", "Unpaired Welch t-test, two-sided, for every model pair inside that condition and metric. No multiplicity correction."),
            ("excluded", "Transformer shuffled attention is undefined and is omitted."),
            ("alpha", "0.05 on the uncorrected two-sided p."),
            ("n", "20 seeds per model when the intervention is defined."),
        ],
        columns=["item", "description"],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    write_sheet(wb, "Methods", methods, first=True)
    write_sheet(wb, "vs_zero", zero)
    write_sheet(wb, "models_omnibus", omnibus)
    write_sheet(wb, "models_pairwise", pairs)
    wb.save(OUT)
    print(OUT)
    print("--- not different from 0 ---")
    print(zero.loc[~zero.different_from_0_p05, ["family", "condition", "metric", "model", "mean", "sem", "p_two_sided"]].to_string(index=False))
    print("--- omnibus ---")
    print(omnibus.round(6).to_string(index=False))
    print("pairs", len(pairs), "different", int(pairs.different_p05.sum()))
    quiet = pairs.loc[~pairs.different_p05, ["family", "condition", "metric", "model_a", "model_b", "difference_a_minus_b", "p_two_sided"]]
    print("--- pairs not different ---")
    print(quiet.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
