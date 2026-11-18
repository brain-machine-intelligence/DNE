from __future__ import annotations
import numpy as np
from .base import BasePolicy, PolicyOutput


class DNEPolicy(BasePolicy):
    """DNE: maximise squared difference between learner and target predictive."""

    def __init__(self, target, name: str = "dne"):
        super().__init__(name)
        self.target = target

    def select(self, posterior, candidates: np.ndarray) -> PolicyOutput:
        p_learn = np.array(
            [posterior.predictive_p(s) for s in candidates],
            dtype=float,
        )
        p_tgt = np.array(
            [self.target.reference_p(s) for s in candidates],
            dtype=float,
        )
        scores = (p_learn - p_tgt) ** 2
        idx = int(np.argmax(scores))
        return PolicyOutput(index=idx, score=float(scores[idx]))
