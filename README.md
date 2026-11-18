# Deep Neural Experimenter (DNE)

Code and summary data accompanying:

> An, S. J., Lee, C. H., & Lee, S. W.  
> **A personalized closed-loop neural framework for aligning human reasoning**

## Overview

Human reasoning is dynamic, yet many adaptive experimental designs treat participants as static systems. The **Deep Neural Experimenter (DNE)** is a closed-loop neural framework that continuously updates a participant-specific behavioural model and chooses subsequent stimuli to guide learning toward an experimenter-specified reference model derived from independent human data.

DNE has two coupled modules:

1. **Profile module** — maps sparse stimulus–response histories to a latent participant representation and predicts upcoming actions. Architecture choice (post-attention LSTM) is supported by the benchmarks in Figs. 2–3.
2. **Control module** — uses the current participant model together with a fixed reference model to select stimuli that maximize predicted response divergence (adversarial alignment control). Policy comparisons are reported in Fig. 4.

In the human causal-reasoning experiment (Figs. 5–8), participants completed five sessions (S0–S4) under either a **static** condition (non-adaptive sequences) or a **DNE** condition (closed-loop sequence selection). Analyses evaluate one-shot causal inference, alignment to the reference model, changes in fitted learning dynamics, and task personalization.

This repository is intended for **code and data availability**: it ships experiment software, analysis scripts where self-contained, and frozen summary tables for the main-text figures. Large training checkpoints and full simulation trajectories are not redistributed.

## Repository layout

```
experiment/                      Human causal-reasoning task (Flask + jsPsych)
  with_dne/                      Closed-loop DNE condition
  without_dne/                   Static (non-adaptive) condition

figures/                         One folder per main-text figure
  fig02_profile_benchmark/       Fig. 2 — profile-module architecture benchmark
  fig03_profile_history/         Fig. 3 — history use and strategy-specific probes
  fig04_control_alignment/       Fig. 4 — adversarial alignment-control benchmark
  fig05_causal_behavior/         Fig. 5 — causal ratings and one-shot (OS) index
  fig06_model_alignment/         Fig. 6 — alignment to the reference model M*
  fig07_learning_rate/           Fig. 7 — fitted parameters and LR counterfactuals
  fig08_task_personalization/    Fig. 8 — task personalization score (TPS)

requirements.txt                 Minimal deps for Fig. 5 behavioural reproduction
```

Fig. 1 is a framework schematic and has no associated compute in this repository.

### Figure folder convention

Where content exists for a given figure, folders follow the same pattern:

| Subfolder | Contents |
|-----------|----------|
| `code/` | Scripts used to produce or regenerate that figure’s analyses |
| `data/` | Inputs required for reproduction (e.g. aggregated behavioural tables) |
| `results/` | Frozen manuscript summary tables |
| `README.md` | Panel map, file pointers, and run instructions |

Figs. 5–8 ship `code/reproduce.py` plus intermediates under `data/` so the main reported statistics can be recomputed without external bundles. Figs. 2–3 ship analysis scripts and frozen `results/`, but re-training needs checkpoints that are not redistributed. Fig. 4 ships simulation source and frozen panel tables.

## What each part of the repo covers

| Path | Paper | Role |
|------|-------|------|
| [`experiment/`](experiment/) | Fig. 5 task | End-to-end Flask + jsPsych apps for static and DNE conditions |
| [`figures/fig02_profile_benchmark/`](figures/fig02_profile_benchmark/) | Fig. 2 | Architecture comparison: action prediction, agent identification, composite score |
| [`figures/fig03_profile_history/`](figures/fig03_profile_history/) | Fig. 3 | Attention / channel / order perturbations and stagewise probes |
| [`figures/fig04_control_alignment/`](figures/fig04_control_alignment/) | Fig. 4 | Simulation benchmark of DNE vs alternative stimulus-selection policies |
| [`figures/fig05_causal_behavior/`](figures/fig05_causal_behavior/) | Fig. 5 | Matched-cohort behavioural tables + rating / OS-index stats |
| [`figures/fig06_model_alignment/`](figures/fig06_model_alignment/) | Fig. 6 | Alignment to \(M^*\), trade-off, strategy-proximity forests |
| [`figures/fig07_learning_rate/`](figures/fig07_learning_rate/) | Fig. 7 | Parameter trajectories, LR-reset counterfactuals, ΔLR–alignment coupling |
| [`figures/fig08_task_personalization/`](figures/fig08_task_personalization/) | Fig. 8 | 30D task structure, TPS, behaviour→TPS→parameter coupling |

## Requirements

- **Python 3.8+** recommended
- **Fig. 5–8 statistic reproduction:** each figure’s `code/requirements.txt` (Fig. 5 also mirrored in root [`requirements.txt`](requirements.txt))
- **Fig. 4 control simulations:** `figures/fig04_control_alignment/code/requirements.txt`
- **Human experiment apps:** Flask and the packages imported by each condition’s `app/` package (details in [`experiment/README.md`](experiment/README.md))
- **Figs. 2–3 training / perturbation scripts:** require PyTorch and the original training tree / checkpoints, which are **not** shipped here

## Getting started

### 1. Reproduce Fig. 5 behavioural statistics (fully self-contained)

This is the most complete end-to-end analysis path in the repository. It uses aggregated subject tables and the matched static / DNE cohort IDs shipped under `figures/fig05_causal_behavior/data/`.

```bash
cd figures/fig05_causal_behavior
python -m pip install -r code/requirements.txt
python code/reproduce.py
```

The script recomputes:

- Fig. 5c — cue-type RM-ANOVAs (with Greenhouse–Geisser) and Šidák pairwise contrasts  
- Fig. 5d (left) — Condition × RoundType mixed ANOVA on the normalized OS index  
- Fig. 5d (right) / Supp. Fig. 12 — session-wise Condition × Session analyses and Welch tests  

Matched cohort sizes: **n = 52** static, **n = 54** DNE.

Normalized OS index:

\[
\mathrm{OS\ index}=\frac{R_{\mathrm{novel}}-\tfrac{R_{8\times}+R_{16\times}}{2}}{R_{\mathrm{novel}}+R_{8\times}+R_{16\times}}
\]

Outputs are written to `figures/fig05_causal_behavior/results/` (overwriting the frozen CSVs if re-run).

### 2. Launch the human experiment

Two Flask apps implement the paper conditions. See [`experiment/README.md`](experiment/README.md) for configuration (`app.ini`, `SECRET_KEY`) and launch commands.

Briefly:

```powershell
cd experiment/with_dne          # or experiment/without_dne
# edit app/app.ini — set a real SECRET_KEY
$env:PYTHONPATH = (Get-Location)
$env:FLASK_APP = "app"
python -m flask run --host 127.0.0.1 --port 5000
# open http://localhost/?workerId=TEST001
```

In the DNE arm, online stimulus selection uses the **post-attention LSTM** participant model selected in Figs. 2–3 (`app/os_model/module_RNN/dnns.py`), together with a fixed Bayesian causal-strength reference model (Lee et al., 2015).

Paper behavioural analyses use the curated tables under `figures/fig05_causal_behavior/data/`, not raw local runtime dumps from a test session.

### 3. Fig. 4 control-module benchmark

Frozen panel summaries (AUC comparisons, RMST / attainment curves, entropy-matched alignment, noisy-environment counterparts) live in:

`figures/fig04_control_alignment/results/panel_*/`

The simulation stack (environment, ten policies, sweep runners, and exporters) is under `figures/fig04_control_alignment/code/`. Full multi-seed trajectories are large and are **not** redistributed; to regenerate statistics, run local sweeps and point the analysis pipeline at those results (see that figure’s README).

### 4. Figs. 2–3 (profile module)

Each folder ships:

- **`code/`** — training / evaluation / perturbation / probe scripts used for the manuscript analyses  
- **`results/`** — frozen tables underlying the reported benchmarks  

Re-training or re-running perturbations requires model checkpoints and the original sliding-window training tree, which are not included. For verifying reported numbers, use the files in `results/`.

### 5. Figs. 6–8 (model-based human analyses)

Each folder now includes `code/reproduce.py`, subject-level intermediates under `data/input/`, and frozen panel tables under `results/panel_*/`.

```bash
cd figures/fig06_model_alignment   # or fig07_learning_rate / fig08_task_personalization
python -m pip install -r code/requirements.txt
python code/reproduce.py
```

This recomputes the main reported statistics from the shipped intermediates (no full model re-fit).

## Included vs not included

| Included in this repository | Not redistributed |
|-----------------------------|-------------------|
| Flask / jsPsych experiment software (static + DNE) | Large profile-module checkpoints (`.pt`) |
| Aggregated behavioural tables and matched-cohort subject lists (Fig. 5) | Full control-module simulation trajectory trees |
| Subject-level intermediates + `reproduce.py` for Figs. 5–8 | Full session-wise param-fitting / TPS feature-build pipelines (not shipped) |
| Frozen summary statistics for main figures | |
| Control-module simulation and analysis source code | |
| Profile-module analysis scripts (without weights) | |

## Citation

If you use this code or data, please cite the paper above.

## Code availability (manuscript text)

> Custom code for the Deep Neural Experimenter (DNE)—including profile- and control-module benchmarks (Figs. 2–4), the human causal-reasoning experiment software, and analysis scripts with summary tables for Figs. 5–8—is available in this repository. Aggregated behavioural data and matched-cohort subject lists are included; large simulation trajectories and model checkpoints are not redistributed.

## License

This project is released under the [MIT License](LICENSE).

## Contact

Corresponding author: Sang Wan Lee — sangwan@kaist.ac.kr
