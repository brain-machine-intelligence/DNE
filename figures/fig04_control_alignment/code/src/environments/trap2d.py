from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from ..utils.seeding import set_seed

def anisotropic_pool(n_info: int, n_disc: int, theta_star: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fixed pool that separates information from target discrepancy.

    Most candidates sit on the large-norm ends of the level set theta·s = 0,
    where a uniform prior has high EIG and p* is near 1/2. A minority sit at
    moderate radius along theta's steepest direction, where p* is already
    saturated and the prior EIG is lower.
    """
    wx, wy, b = (float(v) for v in theta_star)
    # y = -(wx x + b) / wy  stays inside [-2, 2]
    x_lo = (-2.0 * wy - b) / wx
    x_hi = (2.0 * wy - b) / wx
    if x_lo > x_hi:
        x_lo, x_hi = x_hi, x_lo
    n1 = n_info // 2
    xs = np.concatenate([
        rng.normal(x_lo + 0.05 * (x_hi - x_lo), 0.08, size=n1),
        rng.normal(x_hi - 0.05 * (x_hi - x_lo), 0.08, size=n_info - n1),
    ])
    xs = np.clip(xs, x_lo, x_hi)
    ys = np.clip(-(wx * xs + b) / wy + rng.normal(0.0, 0.03, size=n_info), -2.0, 2.0)
    info = np.stack([xs, ys], axis=1)

    direction = np.array([wx, wy], dtype=float)
    direction /= np.linalg.norm(direction)
    radii = np.linspace(0.75, 1.35, n_disc)
    disc = radii[:, None] * direction[None, :]
    return np.vstack([info, disc]).astype(float)


def latin_hypercube(n: int, d: int, lo: float, hi: float) -> np.ndarray:
    """Simple Latin hypercube sampling in [lo, hi]^d."""
    rng = np.random.default_rng()
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.uniform(size=(n, d))
    a = cut[:n]
    b = cut[1:n+1]
    rdpoints = u * (b - a)[:, None] + a[:, None]  # (n,1) broadcast
    # permute each dimension independently
    H = np.zeros_like(rdpoints)
    for j in range(d):
        order = rng.permutation(n)
        H[:, j] = rdpoints[order, j]
    return lo + H * (hi - lo)

@dataclass
class TrapSpec:
    enabled: bool = True
    low_norm_radius: float = 0.5
    orth_band_width: float = 0.25
    orth_band_axis: str = "y"  # x or y

class Trap2DEnv:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.name = cfg.get("name", "trap2d")
        set_seed(int(cfg.get("seed", 0)))
        sp = cfg["stimulus_pool"]
        n = int(sp["n_candidates"])
        lo, hi = map(float, sp["bounds"])
        self._bounds = (lo, hi)

        method = str(sp.get("method", "latin")).lower()

        if method == "latin":
            pts = latin_hypercube(n=n, d=2, lo=lo, hi=hi)
        elif method == "anisotropic":
            n_info = int(sp.get("n_info", 40))
            n_disc = int(sp.get("n_disc", max(n - n_info, 1)))
            theta = np.array(sp.get("theta_star", [1.3, -0.9, 0.4]), dtype=float)
            rng = np.random.default_rng(int(cfg.get("seed", 0)) + 10_000_103)
            pts = anisotropic_pool(n_info, n_disc, theta, rng)
        else:
            pts = np.random.uniform(lo, hi, size=(n, 2))
        self._cands = pts.astype(float)

        tcfg = cfg.get("trap", {}) or {}
        self.trap = TrapSpec(
            enabled=bool(tcfg.get("enabled", True)),
            low_norm_radius=float(tcfg.get("low_norm_radius", 0.5)),
            orth_band_width=float(tcfg.get("orth_band_width", 0.25)),
            orth_band_axis=str(tcfg.get("orth_band_axis", "y")).lower(),
        )

    def candidates(self) -> np.ndarray:
        return self._cands

    @property
    def bounds(self) -> Tuple[float, float]:
        return self._bounds

    def is_in_trap(self, s: np.ndarray) -> Tuple[bool, bool]:
        """Return (in_low_norm, in_orth_band)."""
        if not self.trap.enabled:
            return (False, False)
        r = float(np.linalg.norm(s))
        in_low = r <= self.trap.low_norm_radius
        axis = 1 if self.trap.orth_band_axis == "y" else 0
        in_band = abs(float(s[axis])) <= self.trap.orth_band_width
        return (in_low, in_band)
