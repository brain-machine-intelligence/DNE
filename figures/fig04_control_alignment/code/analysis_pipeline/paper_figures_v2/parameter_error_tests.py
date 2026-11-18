"""Signed-error tests against zero, and the stored absolute-error contrasts.

Signed error is the trial-199 posterior mean minus θ*. The null for each
policy is no signed bias. The test flips the signs of the 20 run-level
errors (seed 20261107, 20,000 flips, add-one P). Intervals are percentile
bootstrap intervals from seed 20261106. Holm adjustment is confined to the
ten policies within one parameter and one environment.

Absolute-error rows are copied from the existing workbook contrasts. A
negative difference means a smaller absolute error for DNE. Those families
each contain nine DNE-minus-comparator tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_pipeline.paper_figures_v2.data_bundle import OUT
from analysis_pipeline.paper_figures_v2.plot_style import POLICIES

DRAWS = 20000
BOOT_SEED = 20261106
PERM_SEED = 20261107
CONTRAST_BOOT_SEED = 20261108
CONTRAST_PERM_SEED = 20261109
RAW = OUT / "figure_statistics_tables" / "S6b_Parameter_Error__raw_runs.csv"
ABSOLUTE = OUT / "figure_statistics_tables" / "S6b_Parameter_Error__comparisons.csv"
KEYS = ("wx", "wy", "b")


def _holm(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * float(pvals[idx])))
        adjusted[idx] = running
    return adjusted


def _star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _mean_ci(values: np.ndarray, rng: np.random.Generator):
    idx = rng.integers(0, len(values), size=(DRAWS, len(values)))
    lo, hi = np.quantile(values[idx].mean(axis=1), [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def _signflip_p(values: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(DRAWS, len(values)))
    diffs = np.abs((signs * values).mean(axis=1))
    return float((np.sum(diffs >= observed - 1e-12) + 1) / (DRAWS + 1))


def signed_tests() -> pd.DataFrame:
    raw = pd.read_csv(RAW)
    rng_boot = np.random.default_rng(BOOT_SEED)
    rng_perm = np.random.default_rng(PERM_SEED)
    rows = []
    for env in ("full", "noisy"):
        for key in KEYS:
            block = []
            for pol in POLICIES:
                values = raw.loc[(raw.environment == env) & (raw.policy == pol), f"signed_error_{key}"].to_numpy(float)
                mean, lo, hi = _mean_ci(values, rng_boot)
                block.append({
                    "analysis": "signed_versus_zero",
                    "environment": env,
                    "parameter": key,
                    "policy": pol,
                    "n": int(len(values)),
                    "mean": mean,
                    "boot_ci_lo": lo,
                    "boot_ci_hi": hi,
                    "ci_excludes_zero": bool(hi < 0 or lo > 0),
                    "p_signflip": _signflip_p(values, rng_perm),
                })
            p_holm = _holm(np.array([row["p_signflip"] for row in block]))
            for row, p in zip(block, p_holm):
                row["p_holm"] = float(p)
                row["holm_family_size"] = 10
                row["holm_star"] = _star(float(p))
                row["family"] = f"signed_{key}_{env}"
                rows.append(row)
    return pd.DataFrame(rows)


def _boot_diff(a, b, rng):
    ia = rng.integers(0, len(a), size=(DRAWS, len(a)))
    ib = rng.integers(0, len(b), size=(DRAWS, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def _perm_p(a, b, rng) -> float:
    observed = abs(float(a.mean() - b.mean()))
    pooled = np.concatenate([a, b])
    keys = rng.random((DRAWS, pooled.size))
    chosen = np.argpartition(keys, len(a), axis=1)[:, : len(a)]
    chosen_sum = pooled[chosen].sum(axis=1)
    other_mean = (pooled.sum() - chosen_sum) / len(b)
    diffs = np.abs(chosen_sum / len(a) - other_mean)
    return float((np.sum(diffs >= observed - 1e-12) + 1) / (DRAWS + 1))


def signed_contrasts() -> pd.DataFrame:
    """DNE minus comparator. A negative value is a more negative signed error, not a smaller absolute error."""
    raw = pd.read_csv(RAW)
    rng_boot = np.random.default_rng(CONTRAST_BOOT_SEED)
    rng_perm = np.random.default_rng(CONTRAST_PERM_SEED)
    frames = []
    for env in ("full", "noisy"):
        for key in KEYS:
            dne = raw.loc[(raw.environment == env) & (raw.policy == "dne"), f"signed_error_{key}"].to_numpy(float)
            rows = []
            for pol in POLICIES:
                if pol == "dne":
                    continue
                other = raw.loc[(raw.environment == env) & (raw.policy == pol), f"signed_error_{key}"].to_numpy(float)
                diff, lo, hi = _boot_diff(dne, other, rng_boot)
                rows.append({
                    "analysis": "signed_dne_minus_comparator",
                    "family": f"signed_{key}_{env}",
                    "environment": env,
                    "parameter": key,
                    "comparator": pol,
                    "n_dne": 20,
                    "n_comparator": 20,
                    "mean_dne": float(dne.mean()),
                    "mean_comparator": float(other.mean()),
                    "diff_dne_minus_comparator": diff,
                    "boot_ci_lo": lo,
                    "boot_ci_hi": hi,
                    "ci_excludes_zero": bool(hi < 0 or lo > 0),
                    "p_permutation": _perm_p(dne, other, rng_perm),
                })
            frame = pd.DataFrame(rows)
            frame["p_holm"] = _holm(frame["p_permutation"].to_numpy())
            frame["holm_family_size"] = 9
            frame["holm_star"] = frame["p_holm"].map(_star)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def absolute_tests() -> pd.DataFrame:
    frame = pd.read_csv(ABSOLUTE)
    frame.insert(0, "analysis", "absolute_dne_minus_comparator")
    frame["holm_star"] = frame["p_holm"].map(_star)
    frame["parameter"] = frame["endpoint"].str.replace("absolute_error_", "", regex=False)
    return frame


def main() -> None:
    signed = signed_tests()
    contrasts = signed_contrasts()
    absolute = absolute_tests()
    signed.to_csv(OUT / "signed_parameter_tests.csv", index=False)
    contrasts.to_csv(OUT / "signed_parameter_contrasts.csv", index=False)
    absolute.to_csv(OUT / "absolute_parameter_tests.csv", index=False)
    show = contrasts[["environment", "parameter", "comparator", "diff_dne_minus_comparator", "boot_ci_lo", "boot_ci_hi", "p_holm", "holm_star"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
