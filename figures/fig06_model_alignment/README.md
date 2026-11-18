# Fig. 6 — Context-dependent alignment to the reference model

Session-wise fitted causal models are compared to the human-derived reference
\(M^*\) in **control** and **OS** rounds (Static vs DNE).

**Fit lock:** Oneshot_sangwan `legacy=True`, `additive=True`, with \(\tau\).

## Panels

| Panel | Analysis | Results folder |
|-------|----------|----------------|
| **a** | L1 alignment distance: S0 vs mean(S1–S4) | `results/panel_a_alignment_distance/` |
| **b** | Trade-off: Δ Control vs Δ OS alignment | `results/panel_b_tradeoff/` |
| **c** | \(M^*\) cue-mass profiles | `results/panel_c_mstar_profiles/` |
| **d** | Control: strategy proximity forest | `results/panel_d_proximity_control/` |
| **e** | OS: strategy proximity forest | `results/panel_e_proximity_os/` |

## Layout

```
code/reproduce.py              rebuild panel a–e stats from data/input/
code/requirements.txt
data/input/                    subject-level intermediates
data/SUMMARY.md
results/panel_*/               frozen manuscript tables
```

## Key reported statistics

**a — Δ = mean(S1–S4) − S0** (`panel_a_welch_delta.csv`)

| Context | DNE Δ | Static Δ | DNE−Static |
|---------|-------|----------|------------|
| Control | +0.083 | −0.003 | +0.086 *** |
| OS | −0.060 | −0.023 | −0.037 * |

**b — Pearson trade-off at S2** (`tradeoff_corr_S2.csv`)

- DNE: \(r = -0.641\), \(P < 0.001\)
- Static: \(r = -0.171\), \(P = 0.225\) (ns)

**c–e** — see `panel_c_mstar_xy_centroids.csv`, `forest_*_mean14.csv`, and `data/SUMMARY.md`.

## Reproduce

```powershell
cd figures/fig06_model_alignment
python -m pip install -r code/requirements.txt
python code/reproduce.py
```

Inputs: `data/input/`. Outputs overwrite `results/panel_*/` CSVs.
