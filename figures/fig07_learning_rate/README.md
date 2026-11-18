# Fig. 7 — Learning-rate dynamics and counterfactual alignment

Session-wise fitted primacy / recency / LR / \(\tau\), single-parameter reset
counterfactuals, and subject-level coupling of \(\Delta\)LR to \(\Delta D^{M*}\).

**Fit lock:** Oneshot_sangwan `legacy=True`, `additive=True`, with \(\tau\).

## Panels

| Panel | Analysis | Results folder |
|-------|----------|----------------|
| **a** | Parameter trajectories (Static vs DNE) | `results/panel_a_param_trajectories/` |
| **b** | Single-parameter reset counterfactuals | `results/panel_b_param_reset/` |
| **c** | \(\Delta\)LR vs \(\Delta\) alignment (with-DNE) | `results/panel_c_lr_vs_alignment/` |

## Layout

```
code/reproduce.py              rebuild panel a–c stats from data/input/
code/requirements.txt
data/input/                    subject-level intermediates
data/SUMMARY.md
results/panel_*/               frozen manuscript tables
```

## Key reported statistics

**a** — Session Welch (Holm within parameter): LR S1–S3 DNE > Static; Recency S1–S2 (see trajectory tables).

**b** — With-DNE paired (Holm): S2 vs LR→S0 significant in Control and OS; other single-parameter resets ns vs S2 (`param_reset_pairwise.csv`).

**c** — mean(S1–4) with-DNE (`panel_c_figure_correlations.csv`):

| Context | \(r\) | \(P\) |
|---------|-------|-------|
| Control | +0.852 | < 0.001 |
| OS | −0.722 | < 0.001 |

## Reproduce

```powershell
cd figures/fig07_learning_rate
python -m pip install -r code/requirements.txt
python code/reproduce.py
```
