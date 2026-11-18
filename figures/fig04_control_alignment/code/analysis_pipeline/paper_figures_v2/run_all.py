"""Build Figures 1–4 and supplementary figures S1–S7.

    python -m analysis_pipeline.paper_figures_v2.run_all
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_pipeline.paper_figures_v2.data_bundle import OUT, load_or_build
from analysis_pipeline.paper_figures_v2.figure1 import draw as figure1
from analysis_pipeline.paper_figures_v2.figure2 import draw as figure2
from analysis_pipeline.paper_figures_v2.entropy_matched import run as entropy_matched
from analysis_pipeline.paper_figures_v2.figure3 import draw as figure3
from analysis_pipeline.paper_figures_v2.figure4 import draw as figure4
from analysis_pipeline.paper_figures_v2.figure_speed import draw_common as figure_common
from analysis_pipeline.paper_figures_v2.time_to_alignment import run as time_to_alignment
from analysis_pipeline.paper_figures_v2.plot_style import LABELS, POLICIES, apply_style
from analysis_pipeline.paper_figures_v2.supplementary import all_figures

CHECKPOINTS = [
    ("full", "func", "dne", {20: 0.039, 50: 0.032, 100: 0.022, 199: 0.017}),
    ("full", "func", "eig", {20: 0.068, 50: 0.067, 100: 0.043, 199: 0.032}),
    ("full", "func", "boed_h3", {20: 0.060, 50: 0.050, 100: 0.038, 199: 0.029}),
    ("full", "func", "tiny_rnn_target", {20: 0.061, 50: 0.037, 100: 0.030, 199: 0.020}),
    ("full", "func", "thompson", {20: 0.093, 50: 0.058, 100: 0.048, 199: 0.043}),
    ("full", "func", "random", {20: 0.074, 50: 0.062, 100: 0.048, 199: 0.034}),
    ("full", "struct", "dne", {20: 0.375, 199: 0.168}),
    ("full", "struct", "eig", {20: 0.625, 199: 0.284}),
    ("noisy", "func", "dne", {20: 0.069, 50: 0.068, 100: 0.054, 199: 0.045}),
    ("noisy", "func", "eig", {20: 0.094, 50: 0.088, 100: 0.080, 199: 0.070}),
    ("noisy", "func", "boed_h3", {20: 0.081, 50: 0.078, 100: 0.070, 199: 0.062}),
    ("noisy", "func", "tiny_rnn_target", {20: 0.117, 50: 0.100, 100: 0.092, 199: 0.082}),
    ("noisy", "func", "thompson", {20: 0.107, 50: 0.066, 100: 0.051, 199: 0.042}),
    ("noisy", "func", "random", {20: 0.125, 50: 0.091, 100: 0.085, 199: 0.077}),
    ("noisy", "struct", "dne", {20: 0.426, 199: 0.336}),
    ("noisy", "struct", "eig", {20: 0.753, 199: 0.527}),
]


def save(fig, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def verify(bundle) -> list[str]:
    notes = [f"DNE score replay max abs error {bundle['replay_err']:.3e}"]
    for env, metric, pol, trials in CHECKPOINTS:
        mean = bundle["series"][(env, pol, metric)][0]
        for t, expected in trials.items():
            got = float(mean[t])
            flag = "ok" if abs(got - expected) <= 0.0015 else "MISMATCH"
            notes.append(f"{flag} {env} {pol} {metric} t={t}: {got:.4f} vs {expected:.3f}")
    return notes


def statistics_table(bundle) -> pd.DataFrame:
    rows = []
    for env, metric, pol, trials in CHECKPOINTS:
        mean, lo, hi = bundle["series"][(env, pol, metric)]
        for t in trials:
            rows.append({"kind": "checkpoint", "env": env, "policy": pol, "metric": metric, "trial": t, "estimate": mean[t], "ci_lo": lo[t], "ci_hi": hi[t]})
    for (env, endpoint, pol), (est, lo, hi) in bundle["diffs"].items():
        rows.append({"kind": "dne_minus", "env": env, "policy": pol, "metric": endpoint, "trial": "", "estimate": est, "ci_lo": lo, "ci_hi": hi})
    for (pol, endpoint), (est, lo, hi) in bundle["changes"].items():
        rows.append({"kind": "noisy_minus_full", "env": "noisy_minus_full", "policy": pol, "metric": endpoint, "trial": "", "estimate": est, "ci_lo": lo, "ci_hi": hi})
    for (env, pol, metric), (est, lo, hi) in bundle["levels"].items():
        if env == "full" and (metric.startswith("auc_") or metric.startswith("abs_")):
            rows.append({"kind": "level", "env": env, "policy": pol, "metric": metric, "trial": 199 if metric.startswith("abs_") else "", "estimate": est, "ci_lo": lo, "ci_hi": hi})
    for env in ("full", "full_trap", "noisy"):
        for pol in POLICIES:
            for thr in (0.05, 0.03):
                first = bundle["reach"][env][pol][thr]
                got = first[first >= 0]
                rows.append({
                    "kind": "time_to_threshold", "env": env, "policy": pol, "metric": f"mae_le_{thr}",
                    "trial": "", "estimate": float(np.median(got)) if len(got) else np.nan,
                    "ci_lo": len(got), "ci_hi": 20 - len(got),
                })
    return pd.DataFrame(rows)


def _fmt(triple, digits):
    est, lo, hi = triple
    return f"{est:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def write_captions(bundle) -> None:
    ent = {e: bundle["diffs"][(e, "entropy_auc", "eig")] for e in ("full", "full_trap", "noisy")}
    fun = {e: bundle["diffs"][(e, "func_auc", "eig")] for e in ("full", "full_trap", "noisy")}
    text = f"""# Figure captions

Shading and dot-and-whisker bars are pointwise 95% confidence intervals of the mean. Difference intervals are percentile intervals from 20,000 independent bootstrap resamples of the 20 seeds (generator seed 20261022), read from `analysis_outputs/essential/` after the mean difference was recomputed from the trajectories. Trajectory bands use a separate 4,000-draw bootstrap (seed 20261023). Seed-level mean intervals use 20,000 draws (seed 20261024). Seeds are not paired across policies or conditions. Intervals are not adjusted for multiple comparisons.

Solid lines are DNE, EIG, BOED-H3, RNN-target, Thompson, and Random. Dashed lines are BALD, BOED-H2, Uncertainty, and RNN-info. Every saved policy is shown.

Alignment columns are post-update. Selected discrepancy is pre-update. Functional error is mean absolute predictive error on 200 held-out probes. Structural error is the Euclidean distance of the posterior mean from (1.3, −0.9, 0.4). AUC is the trapezoidal integral over trials 0–199.

DNE receives the reference target. EIG and BOED-H3 do not. The comparison does not isolate the acquisition rule from access to that reference.

## Figure 1

Target-directed control of learning experiences. (A) The DNE controller compares the learner's posterior predictive response with the reference and selects a stimulus from the candidate pool. The environment returns a binary observation, and the shared Bayesian update revises the learner. The controller does not edit the update. In Full the reference coincides with the environment, but it is a separate input. (B) Schematic one-dimensional slice, not experimental data. The vertical bar is the probability gap. (C) Illustrative discrepancy scores. The selection rule is shown beneath the panel. (D) Functional MAE and structural distance are two measurements of one trajectory. DNE does not directly optimise structural distance.

## Figure 2

Alignment in Full, the correctly specified environment. (A) Functional MAE. (B) Structural distance. (C) Trapezoidal AUC. Lower is better. The two panels use different scales. (D) Mean absolute error of each posterior-mean parameter at trial 199. These are absolute errors, not signed biases. In Full, the DNE advantage over EIG is in w_x and b; the absolute error in w_y is the same.

## Figure 3

Posterior concentration and target alignment. (A) Posterior entropy in Full, in nats. {(OUT / "entropy_matched_caption.txt").read_text(encoding="utf-8").strip()} (C) Entropy AUC, DNE minus EIG: Full {_fmt(ent['full'], 0)}, Full with spatial logging {_fmt(ent['full_trap'], 0)}, Noisy {_fmt(ent['noisy'], 0)}. Positive values mean DNE retains higher entropy. (D) Functional AUC, DNE minus EIG: Full {_fmt(fun['full'], 2)}, Full with spatial logging {_fmt(fun['full_trap'], 2)}, Noisy {_fmt(fun['noisy'], 2)}. Negative values mean lower integrated functional error. Full with spatial logging is an independent clean run in which circle and band occupancy were recorded. It is not a trap manipulation.

## Figure 4

Ten per cent label flips. The likelihood, the reference, and the score stay on the clean target. (A) Functional MAE in Noisy. DNE is not the lowest curve at trial 199. (B) Structural distance in Noisy. (C) Functional AUC, DNE minus each other policy. Intervals that include zero do not establish a difference. (D) Change in final functional MAE from Full to Noisy. These are differences between independent runs, not paired effects of the flips. Thompson's interval includes zero.

## Time to functional alignment

Full only. Attainment curves use all 20 seeds in the denominator. A seed that never reaches the threshold is right-censored at trial 200 and contributes 200 to the restricted mean; it is not treated as a hit on trial 199. (A) Cumulative first attainment of functional MAE ≤ 0.030 for the six primary policies. (B) Restricted-mean difference, DNE minus each other policy, at that threshold. A separate forest repeats the contrast at MAE ≤ 0.075. Intervals are unadjusted. Holm adjustment is limited to the nine contrasts within a threshold. Supplementary curves show all ten policies, sustained runs of 10 trials with sensitivity runs of 5 and 20, and the pre-specified thresholds 0.02, 0.03, 0.04, 0.05, and 0.075. The numerical intervals are in time_to_alignment_caption.txt. Comparisons whose intervals include zero are not a demonstrated difference.

## Trials to the error reached by every seed

Full only, all ten policies. The threshold is the lowest functional MAE that every seed reached at least once. It is fixed by the seed with the highest minimum, not by the prior. Every seed reaches it, so each bar is the mean of all 20 seeds. Whiskers are unadjusted bootstrap 95% intervals of that mean. The threshold changes if a policy is removed, and it is different in the other runs.

## Supplementary figures

S1. Functional MAE and structural distance in the spatially logged clean run, for every policy. S2. Selected squared discrepancy before the update, replayed on the shared posterior grid. S3. Chosen stimuli, with the circle and the band marked as analysis regions, plus trial and unique-point fractions. The dashed line is Random's unique-point fraction, an estimate of pool composition because the pools were not stored. S4. Proportion of seeds reaching each functional threshold, and the median trial among seeds that reached it. Seeds that never reach it are not assigned trial 200. S5. The four additional policies in Full, with DNE repeated as a reference. S6. Signed posterior-mean error at trial 199. S6b. Absolute parameter error in Full and Noisy. S7. Seed-level functional AUC in Full, and DNE-minus-comparator functional AUC in each condition.
"""
    (OUT / "figure_captions.md").write_text(text, encoding="utf-8")


def write_readme() -> None:
    text = """# Manuscript figures, version 2

Regenerate from the repository root:

```
python -m analysis_pipeline.paper_figures_v2.run_all
```

The first run builds `analysis_outputs/paper_figures_v2/_bundle.pkl`. Later runs reuse it. Delete that file to recompute bands and discrepancy replay. The script does not rerun the simulations.

## Sources

| Figure | Files | Columns |
|---|---|---|
| 1 | None. Profiles and scores are schematic. | — |
| 2 | `results/trap2d/notrap/` | `alignment_func_abs`, `alignment_struct_l2`, `theta_mean_wx`, `theta_mean_wy`, `theta_mean_b`, after the update |
| 3A–B | `results/trap2d/notrap/` | `posterior_entropy`, `alignment_func_abs` |
| 3C–D | Those trajectories, plus `results/trap2d/trap/` and `results_noisy_obs/trap2d/noisy_obs/` | Trapezoidal AUC; difference intervals from `analysis_outputs/essential/bootstrap_contrasts.csv` |
| 4A–B | `results_noisy_obs/trap2d/noisy_obs/` | `alignment_func_abs`, `alignment_struct_l2` |
| 4C | Noisy AUC | `bootstrap_contrasts.csv`, endpoint `func_auc` |
| 4D | Noisy and Full final functional MAE | `analysis_outputs/essential/degradation.csv` |
| S1, S3 | `results/trap2d/trap/` | alignment columns; `s_x`, `s_y` |
| S2 | All three result directories | discrepancy replayed from `s_x`, `s_y`, `y` before each update |
| S4–S7 | All three result directories | functional trajectories and posterior means |

`longform_trials.csv` is not used. It does not contain the label-flip run. DNE is stored as `dne_l2` in the clean runs and as `dne` in the label-flip run.

## Definitions

- Functional MAE is mean absolute error of the posterior predictive on 200 fixed probes. It is not a squared error.
- Structural distance is the Euclidean norm of the posterior mean minus `(1.3, -0.9, 0.4)`.
- Both are saved after the update on that trial.
- AUC uses `numpy.trapz` over trials 0–199 on this machine (`numpy.trapezoid` when present).
- Selected discrepancy is `(posterior predictive − p*)^2` at the chosen stimulus before the update. Replay of DNE seed 0 matches the logged score.
- Difference intervals resample each policy or condition independently. Matching seed numbers are not pairs.
- A seed that never reaches a threshold is omitted from the median and is not coded as trial 200.
- Full with spatial logging uses the same observation model as Full. The circle and the band are records, not barriers.
- Noisy flips the Bernoulli label with probability 0.10. The update uses the clean likelihood.

## Policies

DNE, EIG, BOED-H3, RNN-target, Thompson, Random, BALD, BOED-H2, Uncertainty, and RNN-info. Dashed lines mark the last four.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_or_build()
    notes = verify(bundle)
    (OUT / "checkpoint_check.txt").write_text("\n".join(notes), encoding="utf-8")
    print("\n".join(notes))
    statistics_table(bundle).to_csv(OUT / "figure_statistics.csv", index=False)
    save(figure1(), "Figure_1_Control")
    save(figure2(bundle), "Figure_2_Alignment")
    entropy_matched(bundle)
    save(figure3(bundle), "Figure_3_Entropy_vs_Alignment")
    save(figure4(bundle), "Figure_4_Noisy_Robustness")
    time_to_alignment(bundle)
    save(figure_common(bundle), "Figure_Time_to_Common_Floor")
    for stem, fig in all_figures(bundle).items():
        save(fig, stem)
    write_captions(bundle)
    write_readme()
    print(OUT)


if __name__ == "__main__":
    main()
