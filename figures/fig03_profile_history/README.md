# Fig. 3 — History use and strategy-specific information

Perturbation and probe analyses of recurrent profile architectures
(attention, channel content, temporal order, stagewise readout).

## Panels

| Panel | Analysis | Shipped result |
|-------|----------|----------------|
| a | Research questions (schematic) | — |
| b | Attention perturbation | `results/panel_b_attention/` |
| c | Channel / trial-order perturbation | `results/panel_c_channels/` |
| d | Post-attention LSTM intervention contrast | `results/panel_d_intervention/` |
| e–f | Stagewise history vs final probes | `results/panel_ef_stagewise/` |

## Layout

```
code/attention_perturbation/
code/channel_order_perturbation/
code/postattn_intervention/
code/stagewise_probes/
results/panel_*/          frozen tables used in the manuscript
```

## Reproduce

Scripts expect the shared stack from Fig. 2
(`../fig02_profile_benchmark/code/shared`) and external checkpoints (not
redistributed). Frozen `results/` tables are sufficient for verifying reported
statistics.
