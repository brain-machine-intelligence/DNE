"""Functional error at matched posterior entropy.

Trials are grouped by posterior entropy, not by functional MAE. Bin edges are
a fixed half-nat grid shared by every policy. Each independent run that enters
a bin contributes one mean, and those run means are averaged with equal weight.
The bootstrap resamples runs, not trials.

This is a comparison among runs that visited the same entropy bin. It is not
a causal effect of entropy, and it is not the earlier MAE-binned Figure 3B.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_pipeline.paper_figures_v2.data_bundle import OUT
from analysis_pipeline.paper_figures_v2.plot_style import (
    COLORS, LABELS, LS, LW, POLICIES, add_legend, apply_style, draw_order, panel_label,
)

EDGES = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
MIN_RUNS = 8
DRAWS = 20000
BOOT_SEED = 20261102
NOISY_BOOT_SEED = 20261110
N_RUNS = 20
FOCUS = ("dne", "eig", "boed_h3")
ARCHIVE = "Figure_3B_MAE_binned_previous"


def _check_edges() -> None:
    assert np.all(np.diff(EDGES) > 0)
    sample = np.array([2.5, 2.999, 3.0, 8.999, 9.0])
    ids = _bin_index(sample)
    assert list(ids) == [0, 0, 1, len(EDGES) - 2, len(EDGES) - 2]


def _bin_index(entropy: np.ndarray) -> np.ndarray:
    idx = np.full(entropy.shape, -1, dtype=int)
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        if i == len(EDGES) - 2:
            mask = (entropy >= lo) & (entropy <= hi)
        else:
            mask = (entropy >= lo) & (entropy < hi)
        idx[mask] = i
    return idx


def per_run_table(bundle, environment: str = "full") -> pd.DataFrame:
    rows = []
    n_bins = len(EDGES) - 1
    for pol in POLICIES:
        entropy = bundle["raw"][environment][pol]["entropy"]
        func = bundle["raw"][environment][pol]["func"]
        for run in range(N_RUNS):
            assigned = _bin_index(entropy[run])
            for bin_id in range(n_bins):
                mask = assigned == bin_id
                n_trials = int(mask.sum())
                rows.append({
                    "analysis": "entropy_binned",
                    "environment": environment,
                    "policy": pol,
                    "run_id": run,
                    "bin_id": bin_id + 1,
                    "bin_lo": float(EDGES[bin_id]),
                    "bin_hi": float(EDGES[bin_id + 1]),
                    "bin_centre": float(0.5 * (EDGES[bin_id] + EDGES[bin_id + 1])),
                    "n_trials": n_trials,
                    "entered_bin": int(n_trials > 0),
                    "mean_functional_mae": float(func[run, mask].mean()) if n_trials else np.nan,
                })
    return pd.DataFrame(rows)


def _boot_mean(values: np.ndarray, rng: np.random.Generator):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    if n < 2:
        return mean, mean, mean
    idx = rng.integers(0, n, size=(DRAWS, n))
    lo, hi = np.quantile(values[idx].mean(axis=1), [0.025, 0.975])
    return mean, float(lo), float(hi)


def _boot_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = float(a.mean() - b.mean())
    if len(a) < 1 or len(b) < 1:
        return diff, np.nan, np.nan
    ia = rng.integers(0, len(a), size=(DRAWS, len(a)))
    ib = rng.integers(0, len(b), size=(DRAWS, len(b)))
    lo, hi = np.quantile(a[ia].mean(axis=1) - b[ib].mean(axis=1), [0.025, 0.975])
    return diff, float(lo), float(hi)


def summarise(per_run: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for pol in POLICIES:
        part = per_run.loc[per_run.policy == pol]
        for bin_id, group in part.groupby("bin_id"):
            entered = group.loc[group.entered_bin == 1, "mean_functional_mae"].to_numpy(float)
            mean, lo, hi = _boot_mean(entered, rng)
            sd = float(entered.std(ddof=1)) if len(entered) > 1 else np.nan
            rows.append({
                "analysis": "entropy_binned",
                "policy": pol,
                "bin_id": int(bin_id),
                "bin_lo": float(group.bin_lo.iloc[0]),
                "bin_hi": float(group.bin_hi.iloc[0]),
                "bin_centre": float(group.bin_centre.iloc[0]),
                "n_runs_entered": int(len(entered)),
                "n_trials": int(group.n_trials.sum()),
                "equal_run_mean_functional_mae": mean,
                "sd": sd,
                "sem": float(sd / np.sqrt(len(entered))) if len(entered) > 1 else np.nan,
                "boot_ci_lo": lo,
                "boot_ci_hi": hi,
                "sufficient_runs": int(len(entered) >= MIN_RUNS),
            })
    return pd.DataFrame(rows)


def comparisons(per_run: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    dne = per_run.loc[per_run.policy == "dne"]
    for pol in POLICIES:
        if pol == "dne":
            continue
        other = per_run.loc[per_run.policy == pol]
        for bin_id in sorted(dne.bin_id.unique()):
            dne_bin = dne.loc[(dne.bin_id == bin_id) & (dne.entered_bin == 1), "mean_functional_mae"].to_numpy(float)
            other_bin = other.loc[(other.bin_id == bin_id) & (other.entered_bin == 1), "mean_functional_mae"].to_numpy(float)
            meta = dne.loc[dne.bin_id == bin_id].iloc[0]
            common = len(dne_bin) > 0 and len(other_bin) > 0
            if not common:
                diff = lo = hi = np.nan
            else:
                diff, lo, hi = _boot_diff(dne_bin, other_bin, rng)
            rows.append({
                "analysis": "entropy_binned",
                "bin_id": int(bin_id),
                "bin_lo": float(meta.bin_lo),
                "bin_hi": float(meta.bin_hi),
                "bin_centre": float(meta.bin_centre),
                "comparator": pol,
                "n_runs_dne": int(len(dne_bin)),
                "n_runs_comparator": int(len(other_bin)),
                "n_trials_dne": int(dne.loc[dne.bin_id == bin_id, "n_trials"].sum()),
                "n_trials_comparator": int(other.loc[other.bin_id == bin_id, "n_trials"].sum()),
                "mean_dne": float(dne_bin.mean()) if len(dne_bin) else np.nan,
                "mean_comparator": float(other_bin.mean()) if len(other_bin) else np.nan,
                "diff_dne_minus_comparator": diff,
                "boot_ci_lo": lo,
                "boot_ci_hi": hi,
                "common_support": int(common),
                "sufficient_support": int(len(dne_bin) >= MIN_RUNS and len(other_bin) >= MIN_RUNS),
                "ci_excludes_zero": int(common and np.isfinite(lo) and (hi < 0 or lo > 0)),
            })
    return pd.DataFrame(rows)


def support_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dne = summary.loc[summary.policy == "dne"].set_index("bin_id")
    for pol in POLICIES:
        part = summary.loc[summary.policy == pol]
        for row in part.itertuples(index=False):
            dne_runs = int(dne.loc[row.bin_id, "n_runs_entered"])
            rows.append({
                "analysis": "entropy_binned",
                "policy": pol,
                "bin_id": int(row.bin_id),
                "bin_lo": row.bin_lo,
                "bin_hi": row.bin_hi,
                "bin_centre": row.bin_centre,
                "n_runs": int(row.n_runs_entered),
                "n_trials": int(row.n_trials),
                "entered_bin": int(row.n_runs_entered > 0),
                "sufficient_runs": int(row.n_runs_entered >= MIN_RUNS),
                "dne_runs_in_bin": dne_runs,
                "common_with_dne": int(row.n_runs_entered > 0 and dne_runs > 0),
                "observed_only_by_this_policy": int(row.n_runs_entered > 0 and dne_runs == 0 and pol != "dne"),
            })
    return pd.DataFrame(rows)


def analyse_environment(bundle, environment: str = "full", seed: int = BOOT_SEED) -> dict:
    _check_edges()
    per_run = per_run_table(bundle, environment)
    rng = np.random.default_rng(seed)
    summary = summarise(per_run, rng)
    contrast = comparisons(per_run, rng)
    support = support_table(summary)
    return {"per_run": per_run, "summary": summary, "comparisons": contrast, "support": support}


def analyse(bundle) -> dict:
    return analyse_environment(bundle, "full", BOOT_SEED)


def _series(summary: pd.DataFrame, policy: str):
    part = summary.loc[(summary.policy == policy) & (summary.n_runs_entered >= 1)].sort_values("bin_id")
    centres, means, los, his = [], [], [], []
    previous = None
    for row in part.itertuples(index=False):
        if previous is not None and row.bin_id != previous + 1:
            centres.append(np.nan)
            means.append(np.nan)
            los.append(np.nan)
            his.append(np.nan)
        centres.append(row.bin_centre)
        means.append(row.equal_run_mean_functional_mae)
        los.append(row.boot_ci_lo)
        his.append(row.boot_ci_hi)
        previous = row.bin_id
    return np.asarray(centres), np.asarray(means), np.asarray(los), np.asarray(his)


def plot_on_axis(ax, summary: pd.DataFrame, policies) -> None:
    for pol in draw_order(policies):
        x, y, lo, hi = _series(summary, pol)
        finite = np.isfinite(y)
        ax.plot(x, y, color=COLORS[pol], lw=LW[pol], ls=LS[pol], zorder=4 if pol == "dne" else 3)
        if finite.any():
            yerr = np.vstack([y[finite] - lo[finite], hi[finite] - y[finite]])
            ax.errorbar(
                x[finite], y[finite], yerr=yerr, fmt="o", ms=3.2, color=COLORS[pol],
                elinewidth=0.7, capsize=1.5, linestyle="none", zorder=5 if pol == "dne" else 4,
            )
    ax.set_xlim(2.6, 9.2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Posterior entropy (nats)")
    ax.set_ylabel("Functional error (MAE)")
    ax.set_title("Functional alignment at matched entropy", loc="left", fontsize=8)


def xy_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Bin centre as X, then each policy's mean, SEM, and n. Blank only when no run entered."""
    centres = np.sort(summary.bin_centre.unique())
    wide = {"entropy": centres}
    for pol in POLICIES:
        part = summary.loc[summary.policy == pol].sort_values("bin_centre")
        show = part["n_runs_entered"].to_numpy() >= 1
        label = LABELS[pol]
        mean = part["equal_run_mean_functional_mae"].to_numpy(float)
        sem = part["sem"].to_numpy(float)
        runs = part["n_runs_entered"].to_numpy(float)
        wide[f"{label} mean"] = np.where(show, mean, np.nan)
        wide[f"{label} sem"] = np.where(show, sem, np.nan)
        wide[f"{label} n"] = np.where(show, runs, np.nan)
    return pd.DataFrame(wide)


def supplementary_figure(summary: pd.DataFrame):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.45, 4.3))
    plot_on_axis(ax, summary, POLICIES)
    panel_label(ax, "S")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.28)
    add_legend(fig, y=0.16, ncol=5)
    return fig


def _save_figure(fig, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def archive_mae_binned_figure() -> None:
    """Keep the MAE-binned Figure 3 the first time the entropy-binned figure replaces it."""
    destination = OUT / f"{ARCHIVE}.pdf"
    if destination.exists():
        return
    source = OUT / "Figure_3_Entropy_vs_Alignment.pdf"
    if not source.exists():
        return
    for ext in ("pdf", "svg", "png"):
        src = OUT / f"Figure_3_Entropy_vs_Alignment.{ext}"
        if src.exists():
            shutil.copy2(src, OUT / f"{ARCHIVE}.{ext}")


def definition_text(support: pd.DataFrame, contrast: pd.DataFrame, environment: str = "Full", seed: int = BOOT_SEED) -> str:
    lines = [
        f"Entropy-binned functional alignment in {environment}.",
        "This analysis bins trials by posterior entropy.",
        "The previous analysis binned trials by functional MAE and then averaged entropy.",
        f"That MAE-binned figure is preserved as {ARCHIVE}.pdf, .svg, and .png.",
        "The workbook sheet F3B_Entropy_Bins remains the MAE-binned table.",
        "",
        "Question: at comparable posterior entropy, is DNE's held-out functional MAE lower than another policy's?",
        "Bin edges, in nats, shared by all ten policies: " + ", ".join(f"{edge:.1f}" for edge in EDGES) + ".",
        "The plotted x value is the centre of the bin, not the mean entropy of the trials in it.",
        "The last bin is closed on the right. Earlier bins are closed on the left and open on the right.",
        "A run contributes its mean functional MAE if it has at least one trial in the bin.",
        "The policy mean gives equal weight to those runs. Trials are not treated as independent.",
        f"Bootstrap: {DRAWS} resamples of the contributing runs, seed {seed}, percentile 2.5 and 97.5.",
        "DNE and a comparator are resampled independently. Seed numbers are not pairs.",
        f"A bin has sufficient run support when both policies have at least {MIN_RUNS} of {N_RUNS} runs in it.",
        "Differences are reported for every shared bin. Bins below that run count are not interpreted.",
        "Intervals are not adjusted for the number of bins.",
        "Visiting the same entropy bin is not a causal effect of entropy.",
        "Curves are not drawn through bins a policy did not enter, and they are not extended past that policy's observed bins.",
        "",
        "Common support with DNE, counting bins in which both have at least one run:",
    ]
    for pol in POLICIES:
        if pol == "dne":
            continue
        sub = contrast.loc[(contrast.comparator == pol) & (contrast.common_support == 1)]
        enough = sub.loc[sub.sufficient_support == 1]
        if sub.empty:
            lines.append(f"  {LABELS[pol]}: no shared bin.")
            continue
        lines.append(
            f"  {LABELS[pol]}: shared centres {sub.bin_centre.min():.2f} to {sub.bin_centre.max():.2f} "
            f"({len(sub)} bins, {len(enough)} with at least {MIN_RUNS} runs each)."
        )
    only = support.loc[support.observed_only_by_this_policy == 1]
    lines.append("")
    lines.append("Bins observed for a comparator and not for DNE:")
    if only.empty:
        lines.append("  none")
    else:
        for row in only.itertuples(index=False):
            lines.append(
                f"  {LABELS[row.policy]} centre {row.bin_centre:.2f}: {row.n_runs} runs, {row.n_trials} trials."
            )
    lines.append("")
    lines.append("Sufficient shared bins whose unadjusted interval excludes zero. A negative difference means lower DNE functional MAE.")
    shown = contrast.loc[(contrast.sufficient_support == 1) & (contrast.ci_excludes_zero == 1)]
    focus = shown.loc[shown.comparator.isin(["eig", "boed_h3"])]
    if focus.empty:
        lines.append("  DNE versus EIG and DNE versus BOED-H3: none.")
    for row in focus.sort_values(["comparator", "bin_centre"]).itertuples(index=False):
        lines.append(
            f"  {LABELS[row.comparator]} centre {row.bin_centre:.2f}: {row.diff_dne_minus_comparator:+.4f} "
            f"[{row.boot_ci_lo:+.4f}, {row.boot_ci_hi:+.4f}], runs {row.n_runs_dne}/{row.n_runs_comparator}."
        )
    return "\n".join(lines) + "\n"


def panel_caption(contrast: pd.DataFrame) -> str:
    def phrase(pol: str) -> str:
        sub = contrast.loc[(contrast.comparator == pol) & (contrast.sufficient_support == 1)]
        below = sub.loc[(sub.ci_excludes_zero == 1) & (sub.diff_dne_minus_comparator < 0)].sort_values("bin_centre")
        covers = sub.loc[sub.ci_excludes_zero == 0].sort_values("bin_centre")
        centres = ", ".join(f"{value:.2f}" for value in below.bin_centre)
        cover_centres = ", ".join(f"{value:.2f}" for value in covers.bin_centre)
        if len(covers) == 0 and len(below) == len(sub) and len(sub) > 0:
            return f"Against {LABELS[pol]}, the unadjusted interval lies below zero in all {len(sub)} sufficient shared bins"
        return (
            f"Against {LABELS[pol]}, the unadjusted interval lies below zero at bin centres {centres or 'none'}"
            f" and includes zero at {cover_centres or 'none'}"
        )
    return (
        "(B) Functional alignment at matched entropy. Full trials are grouped into fixed bins of posterior entropy, "
        "0.5 nat wide, with the same edges for every policy. Each point is the bin centre. The vertical value is the "
        "mean held-out functional MAE, giving equal weight to each run that entered the bin. Bars are unadjusted 95% "
        "bootstrap intervals from 20,000 resamples of those runs. A point is drawn when at least one run entered the bin. "
        "Differences based on fewer than eight runs are reported and are not interpreted. "
        "DNE is not drawn below 3.5 nats, because DNE has no runs there; EIG does. "
        f"{phrase('eig')}. {phrase('boed_h3')}. "
        "Sharing an entropy bin is not a causal effect of entropy. All ten policies are drawn. The copy-ready columns are in entropy_matched_xy.csv."
    )


def run(bundle) -> dict:
    archive_mae_binned_figure()
    result = analyse(bundle)
    OUT.mkdir(parents=True, exist_ok=True)
    result["per_run"].to_csv(OUT / "entropy_matched_per_run.csv", index=False)
    result["summary"].to_csv(OUT / "entropy_matched_summary.csv", index=False)
    result["comparisons"].to_csv(OUT / "entropy_matched_comparisons.csv", index=False)
    result["support"].to_csv(OUT / "entropy_matched_support.csv", index=False)
    xy_table(result["summary"]).to_csv(OUT / "entropy_matched_xy.csv", index=False)
    text = definition_text(result["support"], result["comparisons"])
    (OUT / "entropy_matched_definition.txt").write_text(text, encoding="utf-8")
    (OUT / "entropy_matched_caption.txt").write_text(panel_caption(result["comparisons"]) + "\n", encoding="utf-8")
    _save_figure(supplementary_figure(result["summary"]), "Supplementary_Entropy_Matched")
    print(text)
    return result


def _outside_edges(bundle, environment: str) -> tuple[int, float, float]:
    values = np.concatenate([bundle["raw"][environment][pol]["entropy"].ravel() for pol in POLICIES])
    outside = int(np.sum((values < EDGES[0]) | (values > EDGES[-1])))
    return outside, float(values.min()), float(values.max())


def run_noisy(bundle) -> dict:
    """Same half-nat edges as Full Figure 3B, applied to the noisy trajectories."""
    outside, low, high = _outside_edges(bundle, "noisy")
    result = analyse_environment(bundle, "noisy", NOISY_BOOT_SEED)
    stem = "noisy_entropy_matched"
    result["per_run"].to_csv(OUT / f"{stem}_per_run.csv", index=False)
    result["summary"].to_csv(OUT / f"{stem}_summary.csv", index=False)
    result["comparisons"].to_csv(OUT / f"{stem}_comparisons.csv", index=False)
    result["support"].to_csv(OUT / f"{stem}_support.csv", index=False)
    xy_table(result["summary"]).to_csv(OUT / f"{stem}_xy.csv", index=False)
    text = definition_text(result["support"], result["comparisons"], "Noisy", NOISY_BOOT_SEED)
    text += (
        f"\nObserved posterior entropy in Noisy ranges from {low:.3f} to {high:.3f} nats. "
        f"{outside} trial values fall outside the shared edges [{EDGES[0]:.1f}, {EDGES[-1]:.1f}] and are not binned.\n"
        f"Bootstrap seed {NOISY_BOOT_SEED}. The Full analysis keeps seed {BOOT_SEED}.\n"
    )
    (OUT / f"{stem}_definition.txt").write_text(text, encoding="utf-8")
    apply_style()
    fig, ax = plt.subplots(figsize=(7.45, 4.3))
    plot_on_axis(ax, result["summary"], POLICIES)
    ax.set_title("Functional alignment at matched entropy, Noisy", loc="left", fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.28)
    add_legend(fig, y=0.16, ncol=5)
    _save_figure(fig, "Supplementary_Noisy_Entropy_Matched")
    print(text)
    return result


if __name__ == "__main__":
    from analysis_pipeline.paper_figures_v2.data_bundle import load_or_build
    run_noisy(load_or_build())
