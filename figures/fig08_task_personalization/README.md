# Fig. 8 — Task personalization under DNE guidance

30D cue–outcome task vectors, Task Personalization Score (TPS), and links
behaviour → TPS → ΔLR / Δ alignment.

**TPS lock:** `dne_score_cv` = all-participant logistic weights on S1–S2 structure,
applied to later sessions (not the GroupKFold balanced decoder in panel b).

## Panels

| Panel | Analysis | Results folder |
|-------|----------|----------------|
| **a** | 30D cue–outcome schematic | `results/panel_a_schematic/` |
| **b** | Sequence-condition decoding AUC | `results/panel_b_sequence_decoding/` |
| **c** | Cross-participant 30D L2 distance | `results/panel_c_task_distance/` |
| **d** | TPS axis construction | `results/panel_d_tps_axis/` |
| **e** | Mean TPS by Control/OS × Static/DNE | `results/panel_e_tps_means/` |
| **f** | Preceding behavioural separation → TPS | `results/panel_f_behav_to_tps/` |
| **g** | TPS → ΔLR / Δ alignment | `results/panel_g_tps_coupling/` |

## Layout

```
code/reproduce.py              rebuild panel b–g stats from data/input/
code/requirements.txt
data/input/                    subject / session intermediates
data/SUMMARY.md
results/panel_*/               frozen manuscript tables
```

## Key reported statistics

**b** — OOF AUC: S0 ≈ 0.53 (ns), S1–S4 ≈ 0.69 (**); permutation vs chance.

**c** — Mean pairwise L2: DNE > Static in Control and OS (perm \(P=0.001\)).

**e** — CV TPS (S1–S4): DNE > Static in Control and OS (Welch ***).

**f** — MixedLM interaction \(\beta \approx -0.105\), \(P_{\mathrm{LMM}}=0.008\), \(P_{\mathrm{perm}}=0.001\).

**g** — Clustered OLS (participant SE); see `tps_to_delta_params_stats_3dp.csv`.

## Reproduce

```powershell
cd figures/fig08_task_personalization
python -m pip install -r code/requirements.txt
python code/reproduce.py
```
