# Fig. 2 — Profile-module architecture benchmark

Candidate history encoders are compared on action prediction and agent-type
identification, then ranked with a domain-balanced composite score.

## Panels

| Panel | Analysis | Shipped result |
|-------|----------|----------------|
| a–c | Architecture / task schematics | (no compute) |
| prediction–ID plane | Operational prediction + within-task agent probe | `results/prediction_identification_*.csv` |
| composite score C | 17-metric domain-balanced score | `results/domain_balanced_composite.xlsx` |

## Layout

```
code/shared/       training + eval stack (inputpad_v2)
code/benchmark/    prediction–identification + metric catalogue
results/           frozen manuscript tables
```

## Reproduce

Full regeneration needs the original training tree and checkpoints (not
redistributed). Manuscript tables in `results/` can be used directly without
retraining.

```text
code/shared/train_inputpad_v2.py
code/shared/eval_inputpad_v2.py
code/benchmark/eval_sliding_suite.py
code/benchmark/metric_catalogue/export_domain_balanced_benchmark.py
```
