# Fig. 4 — Adversarial alignment control (control module)

Ten stimulus-selection policies are compared in a Bayesian logistic learner
benchmark (fully specified and 10% observation-noise environments).

## Panels

| Panel | Analysis | Results folder |
|-------|----------|----------------|
| a | Functional alignment (full) | `results/panel_a_functional_alignment/` |
| b | Parameter recovery (full) | `results/panel_b_parameter_recovery/` |
| c | Cumulative attainment MAE ≤ 0.020 | `results/panel_c_attainment/` |
| d | Posterior entropy (full) | `results/panel_d_posterior_entropy/` |
| e | Functional MAE at matched entropy | `results/panel_e_entropy_matched/` |
| f | Functional alignment (noisy) | `results/panel_f_noisy_functional/` |
| g | Parameter recovery (noisy) | `results/panel_g_noisy_parameter/` |

## Layout

```
code/src/                 environment, posterior, policies, metrics
code/configs/             experiment yaml
code/experiments/         run_single.py, run_sweep.py
code/analysis_pipeline/   exporters for panels a–g (paper_figures_v2)
code/methods/             SI policy definitions, θ*, stats notes
data/policies.csv
results/panel_*/          frozen AUC / RMST / xy tables
```

## Reproduce (from frozen tables)

Open `results/panel_*/` CSVs — these match the manuscript statistics.

## Reproduce (re-run simulations)

```powershell
cd figures/fig04_control_alignment
$env:PYTHONPATH = (Get-Location).Path + "\code"
python -m pip install -r code/requirements.txt
python code/experiments/run_sweep.py --config code/configs/experiment_base.yaml
python -m analysis_pipeline.paper_figures_v2.export_figure_statistics
python -m analysis_pipeline.paper_figures_v2.time_to_alignment
python -m analysis_pipeline.paper_figures_v2.entropy_matched
```

Reference parameters θ* = (1.3, −0.9, 0.4) in `code/methods/theta_star.txt`.

Large simulation trajectories are not redistributed; point local runs at the
analysis pipeline when regenerating.
