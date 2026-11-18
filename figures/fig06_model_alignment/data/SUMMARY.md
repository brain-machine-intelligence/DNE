# Fig. 6 — stats digest (legacy + τ additive / MATLAB \(M^*\))

**Fit:** Oneshot_sangwan `legacy=True`, `additive=True` (P / R / LR / τ).  
**Control rounds** in code = Incre trial type.

## (a) Alignment distance Δ = mean(S1–S4) − S0

| Context | Static | DNE | DNE−Static (Holm) |
|---------|--------|-----|-------------------|
| Control | −0.003 ns | +0.083 *** | +0.086 *** |
| OS | −0.023 * | −0.060 *** | −0.037 * |

Source: `results/panel_a_alignment_distance/panel_a_welch_delta.csv` (+ paired follow-up).

## (b) Trade-off (Δ Control vs Δ OS alignment)

| Window | DNE \(r\) | Static \(r\) |
|--------|-----------|--------------|
| **S2 (figure)** | −0.641, \(P < 0.001\) | −0.171, \(P = 0.225\) ns |
| mean14 | −0.656 *** | −0.331 * |

Source: `results/panel_b_tradeoff/tradeoff_corr_S2.csv`.

## (c) \(M^*\) cue-mass centroids

| Context | Non-novel | Novel |
|---------|-----------|-------|
| Control | 0.610 | 0.390 |
| OS | 0.223 | 0.777 |

Source: `results/panel_c_mstar_profiles/panel_c_mstar_xy_centroids.csv`.

## (d)(e) Δ relative proximity to \(M^*\) (DNE, mean14, Holm)

Negative = closer to \(M^*\).

**Control (d)** — closer: Novelty (−0.121), Context-sensitive OS (−0.080), Primacy/Recency (−0.062), Reward association (−0.057); farther: Simple Bayesian (+0.051), Uniform (+0.039); Frequency ns.

**OS (e)** — Novelty closer (−0.038); Primacy/Recency small positive (+0.046/+0.047); farther: Reward association (+0.108), Simple Bayesian / Uniform / Context-sensitive OS / Frequency (~+0.205).

Sources: `forest_incre_mean14.csv`, `forest_os_mean14.csv`.
