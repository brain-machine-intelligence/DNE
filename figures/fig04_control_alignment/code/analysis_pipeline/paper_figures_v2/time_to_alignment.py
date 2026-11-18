"""Cumulative attainment and restricted mean time to alignment in Full.

The event time is the trial index, with trials numbered 0 through 199.
A seed that never records the event is right-censored at tau, one step past
the last trial on which the event could still have been observed. For a
first hit, tau is 200. For a sustained run of length K, the last trial that
can begin the run is 199 - K + 1, so tau is 200 - K + 1. Restricted mean
time is the mean of min(T, tau). A non-reaching seed therefore contributes
tau. It is not coded as having reached on trial 199 or trial 200.

Differences resample the 20 seeds of each policy independently (20,000
draws, percentile 95% intervals). Matching seed numbers are not pairs.
Intervals are unadjusted. Holm adjustment is applied only to the nine
first-hit contrasts within each pre-specified threshold.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_pipeline.paper_figures.data import first_reach
from analysis_pipeline.paper_figures_v2.data_bundle import OUT, load_or_build
from analysis_pipeline.paper_figures_v2.plot_style import (
    COLORS, LABELS, LS, LW, POLICIES, PRIMARY, add_legend, apply_style, draw_order, figure_title, forest, panel_label,
)

N_TRIALS = 200
BOOT_DRAWS = 20000
BOOT_SEED = 20261027
PERM_SEED = 20261028
PRIMARY_THRESHOLDS = (0.030, 0.075)
SENSITIVITY_THRESHOLDS = (0.02, 0.03, 0.04, 0.05, 0.075)
RUN_LENGTHS = (10, 5, 20)


def _self_check() -> None:
    below = np.ones((1, N_TRIALS))
    below[0, 5:15] = 0.0
    t, tau = sustained_times(below, 0.030, 10)
    if t[0] != 5 or tau != 191:
        raise RuntimeError("sustained-run timing failed the window check")
    always = np.zeros((1, N_TRIALS))
    if first_times(always, 0.030)[0] != 0:
        raise RuntimeError("an immediate hit must have event time 0")
    never = np.ones((1, N_TRIALS))
    if first_times(never, 0.030)[0] != 200:
        raise RuntimeError("a seed with no hit must be censored at 200")


def first_times(func: np.ndarray, threshold: float) -> np.ndarray:
    hit = func <= threshold
    reached = hit.any(axis=1)
    return np.where(reached, hit.argmax(axis=1), N_TRIALS).astype(float)


def sustained_times(func: np.ndarray, threshold: float, run_length: int):
    ok = func <= threshold
    cs = np.cumsum(ok, axis=1)
    n_start = N_TRIALS - run_length + 1
    good = np.zeros((func.shape[0], n_start), dtype=bool)
    good[:, 0] = cs[:, run_length - 1] == run_length
    if n_start > 1:
        good[:, 1:] = (cs[:, run_length:] - cs[:, :-run_length]) == run_length
    tau = float(n_start)
    reached = good.any(axis=1)
    times = np.where(reached, good.argmax(axis=1), tau).astype(float)
    return times, tau


def restricted(times: np.ndarray, tau: float) -> np.ndarray:
    return np.minimum(times, tau)


def attainment_curve(times: np.ndarray) -> np.ndarray:
    trials = np.arange(N_TRIALS)
    return np.array([(times <= t).mean() for t in trials])


def rmst(times: np.ndarray, tau: float) -> float:
    return float(restricted(times, tau).mean())


def _holm(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * float(pvals[idx])))
        adjusted[idx] = running
    return adjusted


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    ia = rng.integers(0, len(a), size=(BOOT_DRAWS, len(a)))
    ib = rng.integers(0, len(b), size=(BOOT_DRAWS, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def _permutation_p(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(float(a.mean() - b.mean()))
    pooled = np.concatenate([a, b])
    keys = rng.random((BOOT_DRAWS, pooled.size))
    chosen = np.argpartition(keys, len(a), axis=1)[:, : len(a)]
    chosen_sum = pooled[chosen].sum(axis=1)
    other_mean = (pooled.sum() - chosen_sum) / len(b)
    diffs = np.abs(chosen_sum / len(a) - other_mean)
    return float((np.sum(diffs >= observed - 1e-12) + 1) / (BOOT_DRAWS + 1))


def _contrast_rows(times, tau, threshold, kind, rng_boot, rng_perm, adjust: bool):
    dne = restricted(times["dne"], tau)
    rows = []
    for pol in POLICIES:
        if pol == "dne":
            continue
        other = restricted(times[pol], tau)
        est, lo, hi = _bootstrap_diff(dne, other, rng_boot)
        rows.append({
            "kind": kind,
            "threshold": threshold,
            "policy": pol,
            "rmst_dne": float(dne.mean()),
            "rmst_other": float(other.mean()),
            "diff_dne_minus_other": est,
            "ci_lo": lo,
            "ci_hi": hi,
            "excludes_zero": bool(hi < 0 or lo > 0),
            "p_permutation": _permutation_p(dne, other, rng_perm),
            "n_reached_dne": int(np.sum(times["dne"] < tau)),
            "n_reached_other": int(np.sum(times[pol] < tau)),
            "tau": tau,
            "run_length": 1 if kind in ("first_hit", "sensitivity") else int(str(kind).split("_")[-1]),
        })
    if adjust:
        holm = _holm(np.array([r["p_permutation"] for r in rows]))
        for row, p_adj in zip(rows, holm):
            row["p_holm"] = float(p_adj)
    else:
        for row in rows:
            row["p_holm"] = np.nan
    return rows


def _step(ax, times, policies):
    trials = np.arange(N_TRIALS)
    for pol in draw_order(policies):
        curve = attainment_curve(times[pol])
        ax.step(trials, curve, where="post", color=COLORS[pol], lw=LW[pol], ls=LS[pol], zorder=4 if pol == "dne" else 3)
    ax.set_xlim(0, 199)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Proportion of seeds reached")


def _forest_panel(ax, rows):
    forest(ax, [{
        "label": LABELS[r["policy"]],
        "estimate": r["diff_dne_minus_other"],
        "lo": r["ci_lo"],
        "hi": r["ci_hi"],
        "color": COLORS[r["policy"]],
    } for r in rows])


def figure_main(times, contrasts):
    fig, axes = plt.subplots(1, 2, figsize=(7.45, 4.15))
    figure_title(fig, "Time to functional alignment in Full")
    _step(axes[0], times[(0.030, "first_hit")], PRIMARY)
    axes[0].set_title("Cumulative attainment, MAE ≤ 0.030", loc="left", fontsize=8)
    panel_label(axes[0], "A")
    rows = [r for r in contrasts if r["kind"] == "first_hit" and r["threshold"] == 0.030]
    _forest_panel(axes[1], rows)
    axes[1].set_title("RMST difference, MAE ≤ 0.030", loc="left", fontsize=8)
    axes[1].set_xlabel("RMST, DNE − comparator (trials)")
    panel_label(axes[1], "B", dx=-0.28)
    add_legend(fig, PRIMARY, y=0.02, ncol=6)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.22, wspace=0.55)
    return fig


def figure_rmst(contrasts):
    fig, axes = plt.subplots(1, 2, figsize=(7.45, 4.15))
    figure_title(fig, "Restricted mean time to alignment")
    for ax, thr, letter in zip(axes, (0.030, 0.075), "AB"):
        rows = [r for r in contrasts if r["kind"] == "first_hit" and np.isclose(r["threshold"], thr)]
        _forest_panel(ax, rows)
        ax.set_title(f"MAE ≤ {thr:.3f}", loc="left", fontsize=8)
        ax.set_xlabel("RMST, DNE − comparator (trials)")
        panel_label(ax, letter, dx=-0.32)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.14, wspace=0.55)
    return fig


def figure_all_policies(times):
    fig, axes = plt.subplots(1, 2, figsize=(7.45, 4.35))
    figure_title(fig, "Cumulative attainment for every policy")
    for ax, thr, letter in zip(axes, (0.030, 0.075), "AB"):
        _step(ax, times[(thr, "first_hit")], POLICIES)
        ax.set_title(f"MAE ≤ {thr:.3f}", loc="left", fontsize=8)
        panel_label(ax, letter)
    add_legend(fig, y=0.02, ncol=5)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.22, wspace=0.28)
    return fig


def figure_sustained(sustained):
    fig, axes = plt.subplots(3, 2, figsize=(7.45, 8.15), sharex=True, sharey=True)
    figure_title(fig, "Sustained functional alignment")
    for row, run_length in enumerate((10, 5, 20)):
        for col, thr in enumerate((0.030, 0.075)):
            ax = axes[row, col]
            _step(ax, sustained[(thr, run_length)], POLICIES)
            role = "Primary run length" if run_length == 10 else "Sensitivity"
            ax.set_title(f"{role}, {run_length} trials, MAE ≤ {thr:.3f}", loc="left", fontsize=7.5)
            panel_label(ax, "ABCDEF"[row * 2 + col], dx=-0.18)
    add_legend(fig, y=0.015, ncol=5)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12, hspace=0.38, wspace=0.22)
    return fig


def figure_sensitivity(summary, contrasts):
    fig, axes = plt.subplots(1, 2, figsize=(7.45, 4.35))
    figure_title(fig, "Threshold sensitivity")
    x = np.arange(len(SENSITIVITY_THRESHOLDS))
    ax = axes[0]
    for pol in draw_order(POLICIES):
        ys = [summary[(thr, pol)]["final_attainment"] for thr in SENSITIVITY_THRESHOLDS]
        ax.plot(x, ys, color=COLORS[pol], lw=LW[pol], ls=LS[pol], marker="o", ms=3.2, zorder=4 if pol == "dne" else 3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{thr:.3f}" for thr in SENSITIVITY_THRESHOLDS])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Functional MAE threshold")
    ax.set_ylabel("Final attainment")
    ax.set_title("Proportion reached by trial 199", loc="left", fontsize=8)
    panel_label(ax, "A")
    ax = axes[1]
    comparators = [p for p in POLICIES if p != "dne"]
    for j, pol in enumerate(comparators):
        rows = [next(r for r in contrasts if r["kind"] == "sensitivity" and r["policy"] == pol and np.isclose(r["threshold"], thr)) for thr in SENSITIVITY_THRESHOLDS]
        ys = np.array([r["diff_dne_minus_other"] for r in rows])
        lo = np.array([r["ci_lo"] for r in rows])
        hi = np.array([r["ci_hi"] for r in rows])
        dodge = (j - (len(comparators) - 1) / 2) * 0.07
        ax.errorbar(x + dodge, ys, yerr=np.vstack([ys - lo, hi - ys]), color=COLORS[pol], lw=0.8, ls="none", marker="o", ms=2.6, capsize=1.2, elinewidth=0.5)
    ax.axhline(0, color="#222222", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{thr:.3f}" for thr in SENSITIVITY_THRESHOLDS])
    ax.set_xlabel("Functional MAE threshold")
    ax.set_ylabel("RMST, DNE − comparator")
    ax.set_title("Unadjusted difference intervals", loc="left", fontsize=8)
    panel_label(ax, "B")
    add_legend(fig, y=0.02, ncol=5)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.22, wspace=0.32)
    return fig


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def _validate(bundle, first) -> list[str]:
    notes = ["First-hit times were recomputed from the stored Full trajectories."]
    for thr in (0.05, 0.03):
        for pol in POLICIES:
            stored_hit = np.asarray(bundle["reach"]["full"][pol][thr])
            fresh = first_reach(bundle["raw"]["full"][pol]["func"], thr)
            if not np.array_equal(stored_hit, fresh):
                notes.append(f"DISCREPANCY stored reach {thr} {pol}")
            mine = first_times(bundle["raw"]["full"][pol]["func"], thr)
            agree = np.where(stored_hit >= 0, stored_hit, N_TRIALS)
            if not np.allclose(agree, mine):
                notes.append(f"DISCREPANCY event time {thr} {pol} max abs {np.max(np.abs(agree - mine))}")
    if not any(line.startswith("DISCREPANCY") for line in notes):
        notes.append("Stored first-hit times at 0.05 and 0.03 match this recomputation for every policy and seed.")
    for thr, expected in ((0.075, {"dne": 5.15, "eig": 17.9, "bald": 6.3, "boed_h2": 6.05}), (0.030, {"dne": 19.3, "eig": 61.12})):
        for pol, prev in expected.items():
            times = first[(thr, "first_hit")][pol]
            got = times[times < N_TRIALS]
            conditional = float(got.mean())
            if abs(conditional - prev) > 0.02:
                notes.append(f"DISCREPANCY conditional mean {thr} {pol}: {conditional:.4f} versus earlier {prev}")
            else:
                notes.append(f"Conditional mean at {thr} for {LABELS[pol]} is {conditional:.2f}, matching the earlier reached-only summary ({prev}).")
    for thr in PRIMARY_THRESHOLDS:
        for pol in POLICIES:
            times = first[(thr, "first_hit")][pol]
            if np.all(times < N_TRIALS):
                gap = abs(rmst(times, N_TRIALS) - float(times.mean()))
                if gap > 1e-9:
                    notes.append(f"DISCREPANCY RMST identity {thr} {pol}")
    notes.append("Where every seed reaches the threshold, RMST equals the mean first-hit trial. Where some do not, RMST is larger because those seeds contribute 200 rather than being dropped.")
    return notes


def _caption(contrasts) -> str:
    def sentence(thr):
        rows = [r for r in contrasts if r["kind"] == "first_hit" and np.isclose(r["threshold"], thr)]
        inside = [LABELS[r["policy"]] for r in rows if not r["excludes_zero"]]
        outside = [LABELS[r["policy"]] for r in rows if r["excludes_zero"]]
        holm_inside = [LABELS[r["policy"]] for r in rows if r["p_holm"] > 0.05]
        excluded = "every comparator" if len(outside) == len(rows) else ", ".join(outside)
        holm = ", ".join(holm_inside) if holm_inside else "no comparator"
        included = ", ".join(inside) if inside else "no comparator"
        return (
            f"At MAE ≤ {thr:.3f}, the unadjusted interval excludes zero for {excluded} and includes zero for {included}. "
            f"After Holm adjustment within these nine comparisons, the p-value remains above 0.05 for {holm}."
        )
    return "\n".join([
        "Time to functional alignment in Full. The denominator of every attainment curve is all 20 seeds.",
        "A first hit is the first trial with functional MAE at or below the threshold. A seed with no such trial is censored at 200.",
        "RMST is the mean of the censored event time, so a non-reaching seed contributes 200. Negative differences mean a shorter restricted mean for DNE.",
        "Intervals are unadjusted percentile bootstrap 95% intervals from 20,000 independent resamples. Holm p-values adjust the nine first-hit contrasts within each threshold.",
        sentence(0.030),
        sentence(0.075),
        "Sustained alignment is the first trial that begins 10 consecutive trials at or below the threshold. Runs of 5 and 20 trials are sensitivity analyses. A run of length K can start only through trial 200-K, and a seed with no such run is censored at 201-K.",
        "The sensitivity panel reports thresholds 0.02, 0.03, 0.04, 0.05, and 0.075. The threshold 0.03 is the same cut as 0.030. No threshold was chosen by inspecting which contrast was largest.",
    ])


def run(bundle=None) -> None:
    _self_check()
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = bundle or load_or_build()
    raw = bundle["raw"]["full"]
    rng_boot = np.random.default_rng(BOOT_SEED)
    rng_perm = np.random.default_rng(PERM_SEED)

    stored = {}
    seed_rows = []
    summary_rows = []
    contrast_rows = []

    def add_endpoint(kind, threshold, run_length, times, tau, adjust):
        stored[(threshold, kind if kind == "first_hit" else (threshold, run_length))] = times
        for pol in POLICIES:
            for seed, event_time in enumerate(times[pol]):
                seed_rows.append({
                    "kind": kind, "threshold": threshold, "run_length": run_length, "policy": pol, "seed": seed,
                    "event_time": event_time, "censored": bool(event_time >= tau), "tau": tau,
                })
            summary_rows.append({
                "kind": kind, "threshold": threshold, "run_length": run_length, "policy": pol, "tau": tau,
                "rmst": rmst(times[pol], tau),
                "final_attainment": float(np.mean(times[pol] < tau)),
                "n_reached": int(np.sum(times[pol] < tau)),
                "conditional_mean_if_reached": float(times[pol][times[pol] < tau].mean()) if np.any(times[pol] < tau) else np.nan,
            })
        contrast_rows.extend(_contrast_rows(times, tau, threshold, kind if kind == "first_hit" else f"sustained_{run_length}", rng_boot, rng_perm, adjust))

    first = {}
    for thr in PRIMARY_THRESHOLDS:
        times = {pol: first_times(raw[pol]["func"], thr) for pol in POLICIES}
        first[(thr, "first_hit")] = times
        add_endpoint("first_hit", thr, 1, times, float(N_TRIALS), True)
    sustained = {}
    for thr in PRIMARY_THRESHOLDS:
        for run_length in RUN_LENGTHS:
            times, tau = {}, None
            for pol in POLICIES:
                times[pol], tau = sustained_times(raw[pol]["func"], thr, run_length)
            sustained[(thr, run_length)] = times
            add_endpoint("sustained", thr, run_length, times, tau, False)
    sensitivity_summary = {}
    for thr in SENSITIVITY_THRESHOLDS:
        times = {pol: first_times(raw[pol]["func"], thr) for pol in POLICIES}
        if np.isclose(thr, 0.03):
            for pol in POLICIES:
                if not np.allclose(times[pol], first[(0.030, "first_hit")][pol]):
                    raise RuntimeError("sensitivity threshold 0.03 disagrees with primary 0.030")
        for pol in POLICIES:
            sensitivity_summary[(thr, pol)] = {
                "final_attainment": float(np.mean(times[pol] < N_TRIALS)),
                "rmst": rmst(times[pol], N_TRIALS),
            }
            summary_rows.append({
                "kind": "sensitivity", "threshold": thr, "run_length": 1, "policy": pol, "tau": N_TRIALS,
                "rmst": sensitivity_summary[(thr, pol)]["rmst"],
                "final_attainment": sensitivity_summary[(thr, pol)]["final_attainment"],
                "n_reached": int(np.sum(times[pol] < N_TRIALS)),
                "conditional_mean_if_reached": float(times[pol][times[pol] < N_TRIALS].mean()) if np.any(times[pol] < N_TRIALS) else np.nan,
            })
        if np.isclose(thr, 0.03):
            for row in list(contrast_rows):
                if row["kind"] == "first_hit" and np.isclose(row["threshold"], 0.030):
                    copied = dict(row)
                    copied["kind"] = "sensitivity"
                    copied["p_holm"] = np.nan
                    contrast_rows.append(copied)
        elif np.isclose(thr, 0.075):
            for row in list(contrast_rows):
                if row["kind"] == "first_hit" and np.isclose(row["threshold"], 0.075):
                    copied = dict(row)
                    copied["kind"] = "sensitivity"
                    copied["p_holm"] = np.nan
                    contrast_rows.append(copied)
        else:
            contrast_rows.extend(_contrast_rows(times, float(N_TRIALS), thr, "sensitivity", rng_boot, rng_perm, False))

    notes = _validate(bundle, first)
    (OUT / "time_to_alignment_validation.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    (OUT / "time_to_alignment_definition.txt").write_text(
        "Trials are indexed 0 to 199. First-hit time is the first trial with functional MAE at or below the threshold. "
        "If that trial does not exist, the seed is censored at 200 and contributes 200 to the restricted mean. "
        "A sustained run of length K is the first trial t such that trials t through t+K-1 are all at or below the threshold. "
        "The last admissible t is 200-K. If no run exists, the seed is censored at 201-K. "
        "RMST differences are DNE minus the comparator. Bootstrap intervals use generator seed 20261027 and 20,000 draws. "
        "Permutation p-values use generator seed 20261028 and 20,000 label shuffles. "
        "Holm adjustment is confined to the nine first-hit contrasts at a single threshold. "
        "The five sensitivity thresholds were specified together; none was selected after seeing the contrasts.\n",
        encoding="utf-8",
    )
    (OUT / "time_to_alignment_caption.txt").write_text(_caption(contrast_rows) + "\n", encoding="utf-8")
    pd.DataFrame(seed_rows).to_csv(OUT / "time_to_alignment_seed_times.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "time_to_alignment_rmst.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(OUT / "time_to_alignment_contrasts.csv", index=False)

    curve_times = {(0.030, "first_hit"): first[(0.030, "first_hit")], (0.075, "first_hit"): first[(0.075, "first_hit")]}
    _save(figure_main(curve_times, contrast_rows), "Figure_Time_to_Alignment")
    _save(figure_rmst(contrast_rows), "Figure_Time_to_Alignment_RMST")
    _save(figure_all_policies(curve_times), "Supplementary_Time_Attainment")
    _save(figure_sustained(sustained), "Supplementary_Time_Sustained")
    _save(figure_sensitivity(sensitivity_summary, contrast_rows), "Supplementary_Time_Sensitivity")
    print("\n".join(notes))
    print(OUT)


if __name__ == "__main__":
    run()
