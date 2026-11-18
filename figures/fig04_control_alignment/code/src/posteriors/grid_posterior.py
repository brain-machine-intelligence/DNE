from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from ..utils.math import sigmoid

@dataclass
class GridSpec:
    wx: Tuple[float, float, int]
    wy: Tuple[float, float, int]
    b: Tuple[float, float, int]

def build_theta_grid(spec: Dict[str, Any]) -> np.ndarray:
    wx = spec["wx"]
    wy = spec["wy"]
    bb = spec["b"]
    wxv = np.linspace(float(wx[0]), float(wx[1]), int(wx[2]))
    wyv = np.linspace(float(wy[0]), float(wy[1]), int(wy[2]))
    bv = np.linspace(float(bb[0]), float(bb[1]), int(bb[2]))
    grid = np.array([[x, y, b] for x in wxv for y in wyv for b in bv], dtype=float)
    return grid

class GridPosterior:
    """Discrete posterior over a theta grid. Shared update across all methods."""

    def __init__(self, theta_grid: np.ndarray):
        self.theta_grid = theta_grid  # (K,3)
        self.logw = np.zeros(theta_grid.shape[0], dtype=float)

    def reset_uniform(self):
        self.logw[:] = 0.0

    def normalised_weights(self) -> np.ndarray:
        lw = self.logw - np.max(self.logw)
        w = np.exp(lw)
        w /= np.sum(w)
        return w

    def mean(self) -> np.ndarray:
        w = self.normalised_weights()
        return (w[:, None] * self.theta_grid).sum(axis=0)

    def predictive_p(self, s: np.ndarray) -> float:
        """Posterior predictive p(y=1|s)=E_theta[sigmoid(w·s+b)]."""
        x = self.theta_grid[:, 0] * s[0] + self.theta_grid[:, 1] * s[1] + self.theta_grid[:, 2]
        p = sigmoid(x)
        w = self.normalised_weights()
        return float(np.sum(w * p))

    def entropy(self, eps: float = 1e-12) -> float:
        w = self.normalised_weights()
        w = np.clip(w, eps, 1.0)
        return float(-np.sum(w * np.log(w)))

    def update(self, s: np.ndarray, y: int, eps: float = 1e-12):
        """Bayes update with Bernoulli likelihood."""
        x = self.theta_grid[:, 0] * s[0] + self.theta_grid[:, 1] * s[1] + self.theta_grid[:, 2]
        p = sigmoid(x)
        lik = p if int(y) == 1 else (1.0 - p)
        self.logw += np.log(lik + eps)
        # stabilise
        self.logw -= np.max(self.logw)
