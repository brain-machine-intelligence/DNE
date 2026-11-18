from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional  # noqa: F401
from ..utils.math import sigmoid

@dataclass
class WrapperSpec:
    mode: str = "full"  # full | noisy_theta | noisy_obs | misspecified
    sigma: float = 0.35
    drop_dim: int = 1
    flip_prob: float = 0.0

class TargetWrapper:
    """World model plus the reference shown to a controller.

    noisy_theta keeps data generation and alignment on the fixed base target.
    The controller instead sees one perturbed parameter vector, drawn once per
    episode. noisy_obs keeps that same target as the reference and as the
    alignment score, but flips the Bernoulli label with probability flip_prob.
    full and misspecified show the controller the same function that generates
    the observations.
    """

    def __init__(self, base_target, spec: WrapperSpec):
        self.base = base_target
        self.spec = spec
        self.reference_theta: Optional[np.ndarray] = None

    @property
    def mode(self) -> str:
        return self.spec.mode

    def set_reference_seed(self, seed: int) -> None:
        """Draw the controller reference for this episode. Same seed gives the same reference."""
        if self.spec.mode != "noisy_theta":
            self.reference_theta = None
            return
        rng = np.random.default_rng(int(seed) + 10_000_019)
        self.reference_theta = self.base.theta + rng.normal(0.0, self.spec.sigma, size=3)

    def p(self, s: np.ndarray) -> float:
        """Observation probability and alignment target."""
        mode = self.spec.mode
        if mode in ("full", "noisy_theta", "noisy_obs"):
            return self.base.p(s)

        if mode == "misspecified":
            drop = int(self.spec.drop_dim)
            if drop not in (0, 1):
                raise ValueError("drop_dim must be 0 or 1 for 2D stimuli")
            s2 = np.delete(s, drop)
            w2 = np.delete(self.base.theta[:2], drop)
            x = float(w2[0] * s2[0] + self.base.theta[2])
            return float(sigmoid(x / self.base.temp))

        raise ValueError(f"Unknown wrapper mode: {mode}")

    def reference_p(self, s: np.ndarray) -> float:
        """Reference probability available to the controller."""
        if self.spec.mode != "noisy_theta":
            return self.p(s)
        if self.reference_theta is None:
            raise RuntimeError("Call set_reference_seed before using a noisy reference")
        theta = self.reference_theta
        x = float(theta[0] * s[0] + theta[1] * s[1] + theta[2])
        return float(sigmoid(x / self.base.temp))

    def sample(self, s: np.ndarray) -> int:
        y = int(np.random.rand() < self.p(s))
        if self.spec.mode == "noisy_obs" and np.random.rand() < float(self.spec.flip_prob):
            y = 1 - y
        return y
