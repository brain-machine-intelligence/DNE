# Deep Neural Experimenter (DNE)

Code and summary data accompanying:

> An, S. J., Lee, C. H., & Lee, S. W.  
> **A personalized closed-loop neural framework for aligning human reasoning**

## Overview

Human reasoning evolves through experience, yet many measurement-oriented adaptive experimental designs primarily aim to estimate or predict behaviour rather than guide how it develops. The **Deep Neural Experimenter (DNE)** is a closed-loop framework that estimates participant-specific response patterns and adaptively selects subsequent stimulus sequences to guide behaviour towards an experimenter-specified reference model. In the human causal-reasoning experiment, this reference model was derived from independent human behavioural data.

DNE couples two processes:

1. ***Neural behavioral profiling*** — uses a neural network to estimate a participant model from accumulated stimulus–response history and predict subsequent responses. Computational architecture benchmarks (Figs. 2–3) supported the use of a post-attention LSTM.
2. ***Adversarial alignment control*** — uses the estimated participant model and a fixed reference model to select candidate stimuli that maximise predicted response discrepancies, with the aim of guiding subsequent learning towards the reference. Independent policy comparisons are reported in Fig. 4.

In the human causal-reasoning experiment (Figs. 5–8), participants completed five sessions (S0–S4) under either a **static** condition (non-adaptive sequence selection) or a **DNE** condition (closed-loop sequence selection). S0 served as a common baseline. During S1–S4, the DNE participant model was updated **between sessions**, while stimulus sequences were selected **for each round**. The analyses examine one-shot (OS) causal inference, model-predicted alignment with the reference model, changes in fitted learning-rate parameters, and task personalization.

This repository provides **code and data supporting the manuscript**: experiment software, self-contained analysis scripts where available, and frozen summary tables for the main-text figures. Large training checkpoints and full simulation trajectories are not redistributed.

## Repository layout

```text
experiment/                      Human causal-reasoning task (Flask + jsPsych)
  with_dne/                      Closed-loop DNE condition
  without_dne/                   Static (non-adaptive) condition

figures/                         One folder per main-text figure
  fig02_profile_benchmark/       Fig. 2 — behavioral profiling architecture benchmark
  fig03_profile_history/         Fig. 3 — history use and strategy-specific probes
  fig04_control_alignment/       Fig. 4 — adversarial alignment control benchmark
  fig05_causal_behavior/         Fig. 5 — causal ratings and one-shot (OS) index
  fig06_model_alignment/         Fig. 6 — alignment with the reference model M*
  fig07_learning_rate/           Fig. 7 — fitted parameters and learning-rate counterfactuals
  fig08_task_personalization/    Fig. 8 — task personalization score (TPS)

requirements.txt                 Minimal dependencies for Fig. 5 reproduction
```

Fig. 1 is a framework schematic and has no associated computation in this repository. The existing directory names are retained so that links and scripts remain compatible.

### Figure folder convention

Where content exists for a given figure, folders follow the same pattern:

| Subfolder | Contents |
|-----------|----------|
| `code/` | Scripts used to produce or regenerate the figure's analyses |
| `data/` | Inputs required for reproduction (for example, aggregated behavioural tables) |
| `results/` | Frozen manuscript summary tables |
| `README.md` | Panel map, file pointers and run instructions |

Figs. 5–8 include `code/reproduce.py` and intermediates under `data/`, allowing the main reported statistics to be recomputed without external bundles. Figs. 2–3 include analysis scripts and frozen `results/` tables, but retraining requires checkpoints that are not redistributed. Fig. 4 includes simulation source code and frozen panel tables.

## What each part of the repository covers

| Path | Paper | Role |
|------|-------|------|
| [`experiment/`](experiment/) | Fig. 5 task | Flask + jsPsych applications for the static and DNE conditions |
| [`figures/fig02_profile_benchmark/`](figures/fig02_profile_benchmark/) | Fig. 2 | *Neural behavioral profiling*: action prediction, agent identification and architecture comparison |
| [`figures/fig03_profile_history/`](figures/fig03_profile_history/) | Fig. 3 | History perturbations, representation-stage probes and strategy-related decoding |
| [`figures/fig04_control_alignment/`](figures/fig04_control_alignment/) | Fig. 4 | *Adversarial alignment control*: simulation comparison with alternative stimulus-selection policies |
| [`figures/fig05_causal_behavior/`](figures/fig05_causal_behavior/) | Fig. 5 | Matched-cohort behavioural tables, causal ratings and OS-index statistics |
| [`figures/fig06_model_alignment/`](figures/fig06_model_alignment/) | Fig. 6 | Alignment with M*, context-dependent changes and alternative-model proximity |
| [`figures/fig07_learning_rate/`](figures/fig07_learning_rate/) | Fig. 7 | Fitted parameter trajectories, learning-rate reset counterfactuals and parameter–alignment associations |
| [`figures/fig08_task_personalization/`](figures/fig08_task_personalization/) | Fig. 8 | Round-level 30-dimensional task structure, TPS and prospective associations |

## Requirements

- **Python 3.8+** recommended
- **Figs. 5–8 statistical reproduction:** each figure's `code/requirements.txt` (Fig. 5 also mirrored in root [`requirements.txt`](requirements.txt))
- **Fig. 4 control simulations:** `figures/fig04_control_alignment/code/requirements.txt`
- **Human experiment applications:** Flask and packages imported by each condition's `app/` package (see [`experiment/README.md`](experiment/README.md))
- **Figs. 2–3 training and perturbation scripts:** PyTorch and the original training tree and checkpoints, which are **not** included here

## Getting started

### 1. Reproduce Fig. 5 behavioural statistics

This is the most complete self-contained analysis path in the repository. It uses aggregated participant tables and matched static/DNE cohort identifiers supplied under `figures/fig05_causal_behavior/data/`.

```bash
cd figures/fig05_causal_behavior
python -m pip install -r code/requirements.txt
python code/reproduce.py
```

The script recomputes:

- Fig. 5c — cue-type repeated-measures ANOVAs (Greenhouse–Geisser correction) and Šidák pairwise comparisons
- Fig. 5d (left) — Condition × Round Type mixed ANOVA of the normalised OS index
- Fig. 5d (right) / Supplementary Fig. 12 — session-wise Condition × Session analyses and Welch's t-tests

Matched cohort sizes: **n = 52** static, **n = 54** DNE.

Outputs are written to `figures/fig05_causal_behavior/results/` (overwriting frozen CSVs if rerun).

### 2. Launch the human experiment

Two Flask applications implement the experimental conditions. See [`experiment/README.md`](experiment/README.md) for configuration (`app.ini`, `SECRET_KEY`) and launch commands.

For example, in PowerShell:

```powershell
cd experiment/with_dne          # or experiment/without_dne
# Edit app/app.ini and set a real SECRET_KEY
$env:PYTHONPATH = (Get-Location)
$env:FLASK_APP = "app"
python -m flask run --host 127.0.0.1 --port 5000
# Open http://localhost/?workerId=TEST001
```

In the DNE condition, *Neural behavioral profiling* uses a **post-attention LSTM** participant model selected through the benchmarks in Figs. 2–3 (`app/os_model/module_RNN/dnns.py`). *Adversarial alignment control* then selects round-level stimulus sequences using this participant model and a fixed Bayesian causal-strength reference model (Lee et al., 2015). The LSTM participant model is updated between sessions, while sequence selection occurs for each round.

The paper's behavioural analyses use the curated tables under `figures/fig05_causal_behavior/data/`, not raw local runtime output from a test session.

### 3. Figs. 2–3: *Neural behavioral profiling* benchmarks

Each folder includes:

- **`code/`** — training, evaluation, perturbation and probe scripts used for the manuscript analyses
- **`results/`** — frozen tables underlying the reported benchmarks

Retraining or rerunning perturbation analyses requires model checkpoints and the original sliding-window training tree, which are not included. To inspect the reported benchmark numbers, use the files in `results/`.

### 4. Fig. 4: *Adversarial alignment control* benchmark

Frozen panel summaries (AUC comparisons, restricted mean survival time (RMST) and attainment curves, entropy-matched alignment, and noisy-environment counterparts) are stored in:

`figures/fig04_control_alignment/results/panel_*/`

The simulation stack (environment, ten policies, sweep runners and exporters) is under `figures/fig04_control_alignment/code/`. Full multi-seed trajectories are large and are **not** redistributed. To regenerate statistics, run the local sweeps and direct the analysis pipeline to those results (see the figure-specific README).

### 5. Figs. 6–8: model-based human analyses

Each folder includes `code/reproduce.py`, participant-level intermediates under `data/input/` and frozen panel tables under `results/panel_*/`.

```bash
cd figures/fig06_model_alignment   # or fig07_learning_rate / fig08_task_personalization
python -m pip install -r code/requirements.txt
python code/reproduce.py
```

These scripts recompute the main reported statistics from the supplied intermediates; they do **not** rerun the complete participant-level model-fitting pipeline. In Figs. 6–7, the offline fitted Bayesian causal-strength models use the same model structure as the fixed reference model (M*), but their parameters are estimated separately for each participant and session.

## Citation

If you use this code or data, please cite the paper listed above.

## License

This project is released under the [MIT License](LICENSE).

## Contact

Corresponding author: Sang Wan Lee — sangwan@kaist.ac.kr
