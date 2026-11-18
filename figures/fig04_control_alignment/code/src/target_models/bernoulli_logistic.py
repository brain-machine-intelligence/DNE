from __future__ import annotations
import numpy as np
from typing import Sequence
from ..utils.math import sigmoid

class LogisticBernoulliTarget:
    """p*(y=1|s)=sigmoid((w·s + b)/temp)"""

    def __init__(self, theta_star: Sequence[float], temperature: float = 1.0):
        theta_star = list(theta_star)
        if len(theta_star) != 3:
            raise ValueError("theta_star must be [wx, wy, b]")
        self.theta = np.array(theta_star, dtype=float)
        self.temp = float(temperature)

    def p(self, s: np.ndarray) -> float:
        x = float(self.theta[0] * s[0] + self.theta[1] * s[1] + self.theta[2])
        return float(sigmoid(x / self.temp))

    def sample(self, s: np.ndarray) -> int:
        return int(np.random.rand() < self.p(s))
