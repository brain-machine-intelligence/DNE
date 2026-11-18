# Fig. 5 — Causal reasoning task and behaviour

Human experiment (static vs DNE) causal-strength ratings and one-shot (OS) index.

Task software: [`../../experiment/`](../../experiment/).

## Panels

| Panel | Analysis | Shipped result |
|-------|----------|----------------|
| a–b | Task / round schematics | — |
| c | Cue-type ratings (novel vs non-novel) | `results/fig5c_*.csv` |
| d left | OS index by round type | `results/fig5d_left_*.csv` |
| d right | OS-index session trajectory | `results/fig5d_right_OS_*.csv` |
| Supp. Fig. 12 | Control-round session trajectory | `results/supp_fig12_*.csv` |

## Layout

```
code/reproduce.py          recomputes ANOVA / Welch tables
data/matched_ids/          n=52 static, n=54 DNE
data/behavioural/          aggregated subject tables
results/                   frozen stats tables
```

## Reproduce

```powershell
cd figures/fig05_causal_behavior
python -m pip install -r code/requirements.txt
python code/reproduce.py
```

Outputs are written to `results/` (overwrites frozen tables).
