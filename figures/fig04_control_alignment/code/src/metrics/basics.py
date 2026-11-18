from __future__ import annotations
import numpy as np
from ..utils.math import bernoulli_entropy

def running_auc(xs: np.ndarray, ys: np.ndarray) -> float:
    # simple trapezoid auc over trial index
    if len(xs) < 2:
        return 0.0
    return float(np.trapz(ys, xs))

def mean_entropy_over_stim(ps: np.ndarray) -> float:
    return float(np.mean([bernoulli_entropy(p) for p in ps]))
