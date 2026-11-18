from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from .utils.config import ensure_dir
from .metrics.alignment import theta_l2
from .utils.math import bernoulli_entropy

def _trapz(y, x) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))

def run_episode(
    env,
    target,
    posterior,
    policy,
    trials: int,
    theta_star: np.ndarray,
    compute_trap: bool = True,
    probe_size: int = 200,
    seed: Optional[int] = None,
):
    if seed is not None and hasattr(target, "set_reference_seed"):
        target.set_reference_seed(int(seed))

    # reset stateful policies (e.g., TinyRNN)
    if hasattr(policy, "reset") and callable(getattr(policy, "reset")):
        policy.reset()

    cands = env.candidates()
    rows = []

    # ---- probe set for functional alignment (fixed per episode)
    rng = np.random.default_rng(0)
    low, high = env.bounds
    probe_set = rng.uniform(low=low, high=high, size=(probe_size, 2))

    ent0 = posterior.entropy()

    for t in range(int(trials)):
        out = policy.select(posterior, cands)
        s = cands[out.index]
        y = target.sample(s)

        posterior.update(s, y)

        # ===== structural alignment =====
        theta_mean = posterior.mean()
        align_struct_l2 = theta_l2(theta_mean, theta_star)

        # ===== behavioural alignment (on-policy) =====
        p_pred = posterior.predictive_p(s)
        p_tgt = target.p(s)
        align_beh_abs = abs(p_pred - p_tgt)
        align_beh_sq = (p_pred - p_tgt) ** 2

        # ===== functional alignment (off-policy, sparse) =====
        errs = []
        for sp in probe_set:
            pp = posterior.predictive_p(sp)
            pt = target.p(sp)
            errs.append(abs(pp - pt))
        align_func_abs = float(np.mean(errs))

        pred_ent = bernoulli_entropy(p_pred)
        post_ent = posterior.entropy()

        in_low, in_band = env.is_in_trap(s) if compute_trap else (False, False)

        rows.append({
            "t": t,
            "s_x": float(s[0]),
            "s_y": float(s[1]),
            "y": int(y),
            "policy_score": float(out.score),

            # posterior state
            "theta_mean_wx": float(theta_mean[0]),
            "theta_mean_wy": float(theta_mean[1]),
            "theta_mean_b": float(theta_mean[2]),
            "posterior_entropy": float(post_ent),

            # --- alignment decomposition ---
            "alignment_struct_l2": float(align_struct_l2),
            "alignment_beh_abs": float(align_beh_abs),
            "alignment_beh_sq": float(align_beh_sq),
            "alignment_func_abs": float(align_func_abs),

            # predictive diagnostics
            "pred_p_at_s": float(p_pred),
            "target_p_at_s": float(p_tgt),
            "pred_entropy_at_s": float(pred_ent),

            # trap
            "trap_low_norm": int(in_low),
            "trap_orth_band": int(in_band),
        })

    df = pd.DataFrame(rows)

    # ===== metrics (seed-level summaries) =====
    metrics = {
        "trials": int(trials),

        # final
        "final_alignment_struct_l2": float(df["alignment_struct_l2"].iloc[-1]),
        "final_alignment_beh_abs": float(df["alignment_beh_abs"].iloc[-1]),
        "final_alignment_func_abs": float(df["alignment_func_abs"].iloc[-1]),


        # auc
        "auc_alignment_struct_l2": _trapz(df["alignment_struct_l2"].values, df["t"].values),
        "auc_alignment_beh_abs": _trapz(df["alignment_beh_abs"].values, df["t"].values),
        "auc_alignment_func_abs": _trapz(df["alignment_func_abs"].values, df["t"].values),

        # entropy
        "final_posterior_entropy": float(df["posterior_entropy"].iloc[-1]),
        "auc_posterior_entropy": _trapz(df["posterior_entropy"].values, df["t"].values),
        "entropy_drop": float(ent0 - df["posterior_entropy"].iloc[-1]),

        # trap
        "trap_low_norm_rate": float(df["trap_low_norm"].mean()),
        "trap_orth_band_rate": float(df["trap_orth_band"].mean()),
        "trap_any_rate": float(((df["trap_low_norm"] + df["trap_orth_band"]) > 0).mean()),
    }

    return df, metrics


def save_episode(out_dir: Path, df, metrics: Dict[str, Any]):
    ensure_dir(out_dir)
    df.to_csv(out_dir / "trajectory.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
