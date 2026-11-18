from __future__ import annotations
from typing import Dict, Any
import numpy as np

from .environments.trap2d import Trap2DEnv
from .target_models.bernoulli_logistic import LogisticBernoulliTarget
from .target_models.wrappers import TargetWrapper, WrapperSpec
from .posteriors.grid_posterior import GridPosterior, build_theta_grid
from .policies.dne import DNEPolicy
from .policies.info import EIGPolicy, BALDPolicy
from .policies.heuristics import RandomPolicy, UncertaintyPolicy, ThompsonPolicy
from .policies.boed import BOEDH2Policy, BOEDH3Policy
from .policies.tiny_rnn import TinyRNNInfoPolicy, TinyRNNTargetAwarePolicy, TinyRNNConfig



def make_env(env_cfg: Dict[str, Any]):
    name = str(env_cfg.get("name", "trap2d")).lower()
    if name in ("trap2d", "trap2d_env", "trap"):
        return Trap2DEnv(env_cfg)
    raise ValueError(f"Unknown env name: {name}")

def make_target(target_cfg: Dict[str, Any]):
    if str(target_cfg.get("type", "")).lower() != "logistic_bernoulli":
        raise ValueError("Only logistic_bernoulli is implemented")
    base = LogisticBernoulliTarget(theta_star=target_cfg["theta_star"], temperature=float(target_cfg.get("temperature", 1.0)))
    wcfg = target_cfg.get("wrapper", {}) or {}
    spec = WrapperSpec(
        mode=str(wcfg.get("mode", "full")).lower(),
        sigma=float(wcfg.get("sigma", 0.35)),
        drop_dim=int(wcfg.get("drop_dim", 1)),
        flip_prob=float(wcfg.get("flip_prob", 0.0)),
    )
    return TargetWrapper(base, spec), np.array(target_cfg["theta_star"], dtype=float), spec.mode

def make_posterior(exp_cfg: Dict[str, Any]):
    grid = build_theta_grid(exp_cfg["posterior"]["theta_grid"])
    post = GridPosterior(grid)
    post.reset_uniform()
    return post

def make_policy(policy_item: Dict[str, Any], target):
    kind = str(policy_item["kind"]).lower()
    name = str(policy_item.get("name", kind))
    if kind == "dne":
        return DNEPolicy(target=target, name=name)
    if kind == "eig":
        return EIGPolicy(name=name)
    if kind == "bald":
        return BALDPolicy(name=name)
    if kind == "random":
        return RandomPolicy(name=name)
    if kind == "uncertainty":
        return UncertaintyPolicy(name=name)
    if kind == "thompson":
        return ThompsonPolicy(name=name)
    if kind == "boed_h2":
        return BOEDH2Policy(name=name)
    if kind == "boed_h3":
        return BOEDH3Policy(name=name)
    if kind == "tiny_rnn_info":
        # optional config fields
        cfg = TinyRNNConfig(
            hidden=int(policy_item.get("hidden", 32)),
            lr=float(policy_item.get("lr", 1e-3)),
            weight_decay=float(policy_item.get("weight_decay", 0.0)),
            steps_per_trial=int(policy_item.get("steps_per_trial", 5)),
            device=str(policy_item.get("device", "cpu")),
        )
        return TinyRNNInfoPolicy(name=name, cfg=cfg)

    if kind == "tiny_rnn_target":
        cfg = TinyRNNConfig(
            hidden=int(policy_item.get("hidden", 32)),
            lr=float(policy_item.get("lr", 1e-3)),
            weight_decay=float(policy_item.get("weight_decay", 0.0)),
            steps_per_trial=int(policy_item.get("steps_per_trial", 5)),
            device=str(policy_item.get("device", "cpu")),
        )
        return TinyRNNTargetAwarePolicy(target=target, name=name, cfg=cfg)

    raise ValueError(f"Unknown policy kind: {kind}")
