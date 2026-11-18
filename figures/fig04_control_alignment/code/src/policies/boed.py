# src/policies/boed.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .base import BasePolicy, PolicyOutput


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _posterior_entropy_from_logw(logw: np.ndarray, eps: float = 1e-12) -> float:
    lw = logw - np.max(logw)
    w = np.exp(lw)
    w = w / np.sum(w)
    w = np.clip(w, eps, 1.0)
    return float(-np.sum(w * np.log(w)))


@dataclass
class _Precomp:
    # For each candidate i: loglik1[i, k] = log p(y=1|theta_k, s_i)
    #                      loglik0[i, k] = log p(y=0|theta_k, s_i)
    loglik1: np.ndarray  # (N, K)
    loglik0: np.ndarray  # (N, K)


def _precompute_loglik(theta_grid: np.ndarray, candidates: np.ndarray, eps: float = 1e-12) -> _Precomp:
    """
    theta_grid: (K,3)
    candidates: (N,2)
    """
    K = theta_grid.shape[0]
    N = candidates.shape[0]
    loglik1 = np.empty((N, K), dtype=float)
    loglik0 = np.empty((N, K), dtype=float)

    wx = theta_grid[:, 0]
    wy = theta_grid[:, 1]
    b = theta_grid[:, 2]

    for i, s in enumerate(candidates):
        x = wx * s[0] + wy * s[1] + b
        p = _sigmoid(x)
        p = np.clip(p, eps, 1.0 - eps)
        loglik1[i, :] = np.log(p)
        loglik0[i, :] = np.log(1.0 - p)

    return _Precomp(loglik1=loglik1, loglik0=loglik0)


def _mix_p_y1(logw: np.ndarray, loglik1: np.ndarray) -> float:
    """
    Compute p(y=1|s) under current posterior weights:
      p = sum_k w_k * p_k
    where loglik1 = log p_k (k over theta grid).
    """
    lw = logw - np.max(logw)
    w = np.exp(lw)
    w = w / np.sum(w)
    p_k = np.exp(loglik1)  # p_k
    return float(np.sum(w * p_k))


def _update_logw(logw: np.ndarray, loglik_y: np.ndarray) -> np.ndarray:
    new = logw + loglik_y
    new = new - np.max(new)
    return new


class BOEDH2Policy(BasePolicy):
    """
    BOED-H2: 2-step lookahead with receding horizon, Bernoulli outcomes.

    Utility: - E[ H(p(theta | data after 2 steps)) ]
    First-step choice accounts for optimal second-step choice after observing y1.
    """
    def __init__(self, name: str = "boed_h2"):
        super().__init__(name)
        self._cache_key = None
        self._pre: _Precomp | None = None

    def _ensure_precomp(self, posterior, candidates: np.ndarray):
        key = (id(posterior.theta_grid), id(candidates), candidates.shape[0], posterior.theta_grid.shape[0])
        if self._cache_key != key:
            self._pre = _precompute_loglik(posterior.theta_grid, candidates)
            self._cache_key = key

    def select(self, posterior, candidates: np.ndarray) -> PolicyOutput:
        self._ensure_precomp(posterior, candidates)
        assert self._pre is not None

        logw0 = posterior.logw.copy()
        N = candidates.shape[0]

        best_i = 0
        best_score = -np.inf

        for i in range(N):
            # First-step outcome probabilities under current posterior
            p_y1 = _mix_p_y1(logw0, self._pre.loglik1[i])
            p_y0 = 1.0 - p_y1

            # Posteriors after observing y1=1 or y1=0 for stimulus i
            logw_y1 = _update_logw(logw0, self._pre.loglik1[i])
            logw_y0 = _update_logw(logw0, self._pre.loglik0[i])

            # Second-step: choose stimulus j that minimises expected entropy after step2
            # for each branch separately (i.e., optimal contingent action).
            exp_ent_branch_1 = np.inf
            exp_ent_branch_0 = np.inf

            for j in range(N):
                # branch y1=1
                p2_y1 = _mix_p_y1(logw_y1, self._pre.loglik1[j])
                # expected entropy after observing y2
                logw_11 = _update_logw(logw_y1, self._pre.loglik1[j])
                logw_10 = _update_logw(logw_y1, self._pre.loglik0[j])
                ent_11 = _posterior_entropy_from_logw(logw_11)
                ent_10 = _posterior_entropy_from_logw(logw_10)
                exp_ent_1 = p2_y1 * ent_11 + (1.0 - p2_y1) * ent_10
                if exp_ent_1 < exp_ent_branch_1:
                    exp_ent_branch_1 = exp_ent_1

                # branch y1=0
                p2_y1_b0 = _mix_p_y1(logw_y0, self._pre.loglik1[j])
                logw_01 = _update_logw(logw_y0, self._pre.loglik1[j])
                logw_00 = _update_logw(logw_y0, self._pre.loglik0[j])
                ent_01 = _posterior_entropy_from_logw(logw_01)
                ent_00 = _posterior_entropy_from_logw(logw_00)
                exp_ent_0 = p2_y1_b0 * ent_01 + (1.0 - p2_y1_b0) * ent_00
                if exp_ent_0 < exp_ent_branch_0:
                    exp_ent_branch_0 = exp_ent_0

            # Total expected entropy after 2 steps with optimal second action per branch
            exp_ent_total = p_y1 * exp_ent_branch_1 + p_y0 * exp_ent_branch_0

            # Utility = - expected entropy (maximise)
            score = -exp_ent_total
            if score > best_score:
                best_score = score
                best_i = i

        return PolicyOutput(index=int(best_i), score=float(best_score))


class BOEDH3Policy(BasePolicy):
    """
    BOED-H3: 3-step lookahead (approximate dynamic programming).
    For tractability, we use:
      - exact branching over y at each step (binary)
      - but we restrict step2 and step3 actions to greedy minimisation of expected entropy
        within each branch (receding horizon inside horizon).

    This is a strong planning baseline while remaining feasible for N=50.
    """
    def __init__(self, name: str = "boed_h3"):
        super().__init__(name)
        self._cache_key = None
        self._pre: _Precomp | None = None

    def _ensure_precomp(self, posterior, candidates: np.ndarray):
        key = (id(posterior.theta_grid), id(candidates), candidates.shape[0], posterior.theta_grid.shape[0])
        if self._cache_key != key:
            self._pre = _precompute_loglik(posterior.theta_grid, candidates)
            self._cache_key = key

    def _best_step_min_exp_entropy(self, logw: np.ndarray, N: int) -> Tuple[int, float]:
        """
        Choose j minimising expected entropy after one more step.
        Returns (best_j, min_expected_entropy).
        """
        assert self._pre is not None
        best_j = 0
        best_val = np.inf
        for j in range(N):
            p_y1 = _mix_p_y1(logw, self._pre.loglik1[j])
            logw1 = _update_logw(logw, self._pre.loglik1[j])
            logw0 = _update_logw(logw, self._pre.loglik0[j])
            ent1 = _posterior_entropy_from_logw(logw1)
            ent0 = _posterior_entropy_from_logw(logw0)
            exp_ent = p_y1 * ent1 + (1.0 - p_y1) * ent0
            if exp_ent < best_val:
                best_val = exp_ent
                best_j = j
        return best_j, float(best_val)

    def select(self, posterior, candidates: np.ndarray) -> PolicyOutput:
        self._ensure_precomp(posterior, candidates)
        assert self._pre is not None

        logw0 = posterior.logw.copy()
        N = candidates.shape[0]

        best_i = 0
        best_score = -np.inf

        for i in range(N):
            # step1
            p1_y1 = _mix_p_y1(logw0, self._pre.loglik1[i])
            p1_y0 = 1.0 - p1_y1
            logw_1 = _update_logw(logw0, self._pre.loglik1[i])
            logw_0 = _update_logw(logw0, self._pre.loglik0[i])

            # step2: optimal (greedy wrt expected entropy after step2) per branch
            j1, _ = self._best_step_min_exp_entropy(logw_1, N)
            j0, _ = self._best_step_min_exp_entropy(logw_0, N)

            # After step2, we branch again on y2 and then choose step3 greedily
            # branch y1=1
            p2_y1_b1 = _mix_p_y1(logw_1, self._pre.loglik1[j1])
            logw_11 = _update_logw(logw_1, self._pre.loglik1[j1])
            logw_10 = _update_logw(logw_1, self._pre.loglik0[j1])

            # step3 greedy for each sub-branch
            _, exp_ent_11 = self._best_step_min_exp_entropy(logw_11, N)
            _, exp_ent_10 = self._best_step_min_exp_entropy(logw_10, N)
            exp_ent_b1 = p2_y1_b1 * exp_ent_11 + (1.0 - p2_y1_b1) * exp_ent_10

            # branch y1=0
            p2_y1_b0 = _mix_p_y1(logw_0, self._pre.loglik1[j0])
            logw_01 = _update_logw(logw_0, self._pre.loglik1[j0])
            logw_00 = _update_logw(logw_0, self._pre.loglik0[j0])

            _, exp_ent_01 = self._best_step_min_exp_entropy(logw_01, N)
            _, exp_ent_00 = self._best_step_min_exp_entropy(logw_00, N)
            exp_ent_b0 = p2_y1_b0 * exp_ent_01 + (1.0 - p2_y1_b0) * exp_ent_00

            exp_ent_total = p1_y1 * exp_ent_b1 + p1_y0 * exp_ent_b0

            score = -exp_ent_total
            if score > best_score:
                best_score = score
                best_i = i

        return PolicyOutput(index=int(best_i), score=float(best_score))
