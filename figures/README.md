# Figures

One folder per main-text figure. Where present, each folder uses `code/`, `data/`, and `results/`.

| Folder | Paper figure | Self-contained stats? |
|--------|--------------|------------------------|
| [`fig02_profile_benchmark/`](fig02_profile_benchmark/) | Fig. 2 | Frozen `results/` (retrain needs checkpoints) |
| [`fig03_profile_history/`](fig03_profile_history/) | Fig. 3 | Frozen `results/` (retrain needs checkpoints) |
| [`fig04_control_alignment/`](fig04_control_alignment/) | Fig. 4 | Frozen `results/` + simulation source |
| [`fig05_causal_behavior/`](fig05_causal_behavior/) | Fig. 5 | Yes — `code/reproduce.py` + `data/` |
| [`fig06_model_alignment/`](fig06_model_alignment/) | Fig. 6 | Yes — stats from `data/input/` |
| [`fig07_learning_rate/`](fig07_learning_rate/) | Fig. 7 | Yes — stats from `data/input/` |
| [`fig08_task_personalization/`](fig08_task_personalization/) | Fig. 8 | Yes — stats from `data/input/` |

Fig. 1 is a framework schematic (no compute). Human task software: [`../experiment/`](../experiment/).
