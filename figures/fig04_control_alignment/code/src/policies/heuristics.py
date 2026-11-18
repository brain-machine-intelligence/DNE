from __future__ import annotations
import numpy as np
from .base import BasePolicy, PolicyOutput

class RandomPolicy(BasePolicy):
    def __init__(self, name: str = "random"):
        super().__init__(name)

    def select(self, posterior, candidates: np.ndarray) -> PolicyOutput:
        idx = int(np.random.randint(0, len(candidates)))
        return PolicyOutput(index=idx, score=float("nan"))

class UncertaintyPolicy(BasePolicy):
    """Pick stimulus with posterior predictive closest to 0.5."""

    def __init__(self, name: str = "uncertainty"):
        super().__init__(name)

    def select(self, posterior, candidates: np.ndarray) -> PolicyOutput:
        ps = np.array([posterior.predictive_p(s) for s in candidates], dtype=float)
        idx = int(np.argmin(np.abs(ps - 0.5)))
        return PolicyOutput(index=idx, score=float(-abs(ps[idx]-0.5)))

class ThompsonPolicy(BasePolicy):
    """Posterior sampling: sample theta from posterior grid, then choose p closest to 0.5 under sampled theta."""

    def __init__(self, name: str = "thompson"):
        super().__init__(name)

    def select(self, posterior, candidates: np.ndarray) -> PolicyOutput:
        w = posterior.normalised_weights()
        k = int(np.random.choice(len(w), p=w))
        theta = posterior.theta_grid[k]
        x = candidates[:, 0]*theta[0] + candidates[:, 1]*theta[1] + theta[2]
        p = 1.0/(1.0+np.exp(-x))
        idx = int(np.argmin(np.abs(p-0.5)))
        return PolicyOutput(index=idx, score=float(-abs(p[idx]-0.5)))
