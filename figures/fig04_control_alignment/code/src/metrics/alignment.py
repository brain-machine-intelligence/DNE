from __future__ import annotations
import numpy as np

def theta_l2(theta_hat: np.ndarray, theta_star: np.ndarray) -> float:
    return float(np.linalg.norm(theta_hat - theta_star))

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na*nb + eps))
