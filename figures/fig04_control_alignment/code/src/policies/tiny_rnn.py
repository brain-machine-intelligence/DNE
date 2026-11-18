# src/policies/tiny_rnn.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from .base import BasePolicy, PolicyOutput

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    torch = None
    nn = None
    optim = None


def _safe_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for TinyRNN policies. Install torch and retry.")


@dataclass
class TinyRNNConfig:
    hidden: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    steps_per_trial: int = 5
    device: str = "cpu"


class _TinyRNNNet(nn.Module):
    """
    GRUCell over time. For each candidate we compute a score using (h_t, phi(candidate)).
    """
    def __init__(self, feat_dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRUCell(feat_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: (B, feat_dim)
        return self.gru(x, h)

    def score(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, hidden)
        return self.head(h).squeeze(-1)


class TinyRNNInfoPolicy(BasePolicy):
    """
    Neural baseline (target-unaware): learns online to imitate EIG scores.

    Inputs per candidate at trial t:
      [s_x, s_y, theta_mean_wx, theta_mean_wy, theta_mean_b, posterior_entropy, p_pred]
    Teacher label:
      EIG(s) computed exactly from grid posterior.

    Selection:
      argmax predicted score (after a few gradient steps per trial).
    """
    def __init__(self, name: str = "tiny_rnn_info", cfg: Optional[TinyRNNConfig] = None):
        _safe_torch()
        super().__init__(name)
        self.cfg = cfg or TinyRNNConfig()
        self.device = torch.device(self.cfg.device)

        self.feat_dim = 7
        self.net = _TinyRNNNet(self.feat_dim, self.cfg.hidden).to(self.device)
        self.opt = optim.AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.h = None  # (N, hidden) per-candidate hidden is not needed; we keep a single global hidden.

        self._last_obs = None  # store (s_x, s_y, y)

    def reset(self):
        self.h = torch.zeros((1, self.cfg.hidden), device=self.device)
        self._last_obs = None

    def _teacher_eig_scores(self, posterior, candidates: np.ndarray) -> np.ndarray:
        # exact EIG = H[p_mix] - E_theta[H[p_theta]]
        w = posterior.normalised_weights()
        theta = posterior.theta_grid
        scores = np.zeros(len(candidates), dtype=float)
        for i, s in enumerate(candidates):
            x = theta[:, 0]*s[0] + theta[:, 1]*s[1] + theta[:, 2]
            p_theta = 1.0/(1.0+np.exp(-x))
            p_mix = float(np.sum(w * p_theta))
            # Bernoulli entropy
            eps = 1e-12
            pm = np.clip(p_mix, eps, 1-eps)
            H_mix = -(pm*np.log(pm) + (1-pm)*np.log(1-pm))
            pt = np.clip(p_theta, eps, 1-eps)
            H_theta = -(pt*np.log(pt) + (1-pt)*np.log(1-pt))
            scores[i] = H_mix - float(np.sum(w * H_theta))
        return scores

    def _build_features(self, posterior, candidates: np.ndarray) -> np.ndarray:
        theta_mean = posterior.mean()
        post_ent = posterior.entropy()
        feats = np.zeros((len(candidates), self.feat_dim), dtype=float)
        for i, s in enumerate(candidates):
            p_pred = posterior.predictive_p(s)
            feats[i, :] = np.array(
                [s[0], s[1], theta_mean[0], theta_mean[1], theta_mean[2], post_ent, p_pred],
                dtype=float
            )
        return feats

    def select(self, posterior, candidates: np.ndarray) -> PolicyOutput:
        if self.h is None:
            self.reset()

        feats_np = self._build_features(posterior, candidates)
        y_teacher = self._teacher_eig_scores(posterior, candidates)

        X = torch.tensor(feats_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_teacher, dtype=torch.float32, device=self.device)

        # --- FIX: update hidden as pure state (no graph) ---
        with torch.no_grad():
            x_bar = X.mean(dim=0, keepdim=True)
            self.h = self.net.step(self.h.detach(), x_bar).detach()

        # --- train a few steps per trial ---
        self.net.train()
        for _ in range(int(self.cfg.steps_per_trial)):
            self.opt.zero_grad(set_to_none=True)

            h0 = self.h.detach()
            h_cand = self.net.step(h0.repeat(X.shape[0], 1), X)  # (N, hidden)
            pred = self.net.score(h_cand)  # (N,)

            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

            self.opt.step()

        # --- selection (no grad) ---
        self.net.eval()
        with torch.no_grad():
            h0 = self.h.detach()
            h_cand = self.net.step(h0.repeat(X.shape[0], 1), X)
            pred = self.net.score(h_cand)
            idx = int(torch.argmax(pred).item())
            score = float(pred[idx].item())

        return PolicyOutput(index=idx, score=score)


class TinyRNNTargetAwarePolicy(BasePolicy):
    """
    Neural baseline (target-aware): learns online to imitate DNE scores.
    Uses target access via p_target(s) as an input feature.

    Inputs per candidate:
      [s_x, s_y, theta_mean_wx, theta_mean_wy, theta_mean_b, posterior_entropy, p_pred, p_target]
    Teacher:
      JS( p_pred, p_target )
    """
    def __init__(self, target, name: str = "tiny_rnn_target", cfg: Optional[TinyRNNConfig] = None):
        _safe_torch()
        super().__init__(name)
        self.target = target
        self.cfg = cfg or TinyRNNConfig()
        self.device = torch.device(self.cfg.device)

        self.feat_dim = 8
        self.net = _TinyRNNNet(self.feat_dim, self.cfg.hidden).to(self.device)
        self.opt = optim.AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.h = None

    def reset(self):
        self.h = torch.zeros((1, self.cfg.hidden), device=self.device)

    @staticmethod
    def _js_bernoulli(p: float, q: float, eps: float = 1e-12) -> float:
        p = float(np.clip(p, eps, 1-eps))
        q = float(np.clip(q, eps, 1-eps))
        m = 0.5*(p+q)

        def kl(a, b):
            return a*np.log(a/b) + (1-a)*np.log((1-a)/(1-b))

        return 0.5*kl(p, m) + 0.5*kl(q, m)

    def _teacher_js_scores(self, posterior, candidates: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(candidates), dtype=float)
        for i, s in enumerate(candidates):
            p_pred = posterior.predictive_p(s)
            p_tgt = self.target.reference_p(s)
            scores[i] = self._js_bernoulli(p_pred, p_tgt)
        return scores

    def _build_features(self, posterior, candidates: np.ndarray) -> np.ndarray:
        theta_mean = posterior.mean()
        post_ent = posterior.entropy()
        feats = np.zeros((len(candidates), self.feat_dim), dtype=float)
        for i, s in enumerate(candidates):
            p_pred = posterior.predictive_p(s)
            p_tgt = self.target.reference_p(s)
            feats[i, :] = np.array(
                [s[0], s[1], theta_mean[0], theta_mean[1], theta_mean[2], post_ent, p_pred, p_tgt],
                dtype=float
            )
        return feats

    def select(self, posterior, candidates: np.ndarray) -> PolicyOutput:
        if self.h is None:
            self.reset()

        feats_np = self._build_features(posterior, candidates)
        y_teacher = self._teacher_js_scores(posterior, candidates)

        X = torch.tensor(feats_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_teacher, dtype=torch.float32, device=self.device)

        # --- FIX: update hidden as pure state (no graph) ---
        with torch.no_grad():
            x_bar = X.mean(dim=0, keepdim=True)
            self.h = self.net.step(self.h.detach(), x_bar).detach()

        self.net.train()
        for _ in range(int(self.cfg.steps_per_trial)):
            self.opt.zero_grad(set_to_none=True)

            h0 = self.h.detach()
            h_cand = self.net.step(h0.repeat(X.shape[0], 1), X)
            pred = self.net.score(h_cand)

            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

            self.opt.step()

        self.net.eval()
        with torch.no_grad():
            h0 = self.h.detach()
            h_cand = self.net.step(h0.repeat(X.shape[0], 1), X)
            pred = self.net.score(h_cand)
            idx = int(torch.argmax(pred).item())
            score = float(pred[idx].item())

        return PolicyOutput(index=idx, score=score)

