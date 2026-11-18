from __future__ import annotations
import numpy as np
from .base import BasePolicy, PolicyOutput
from ..utils.math import bernoulli_entropy

class EIGPolicy(BasePolicy):
    """Expected information gain for discrete theta grid posterior.
    EIG(s) = H[p(y|s)] - E_{theta}[H[p(y|s,theta)]]
    """

    def __init__(self, name: str = "eig"):
        super().__init__(name)

    def select(self, posterior, candidates: np.ndarray) -> PolicyOutput:
        w = posterior.normalised_weights()
        # precompute per-candidate
        scores = np.zeros(len(candidates), dtype=float)

        theta = posterior.theta_grid  # (K,3)
        for i, s in enumerate(candidates):
            x = theta[:, 0]*s[0] + theta[:, 1]*s[1] + theta[:, 2]
            p_theta = 1.0/(1.0+np.exp(-x))
            p_mix = float(np.sum(w * p_theta))
            scores[i] = bernoulli_entropy(p_mix) - float(np.sum(w * bernoulli_entropy(p_theta)))
        idx = int(np.argmax(scores))
        return PolicyOutput(index=idx, score=float(scores[idx]))

class BALDPolicy(BasePolicy):
    """BALD for Bernoulli classification equals mutual information:
    I(y,theta|s) = H[p(y|s)] - E_theta[H[p(y|s,theta)]]
    For this Bernoulli setting it matches EIG above; we keep a separate class for clarity.
    """

    def __init__(self, name: str = "bald"):
        super().__init__(name)
        self._eig = EIGPolicy(name=name)

    def select(self, posterior, candidates: np.ndarray) -> PolicyOutput:
        return self._eig.select(posterior, candidates)
