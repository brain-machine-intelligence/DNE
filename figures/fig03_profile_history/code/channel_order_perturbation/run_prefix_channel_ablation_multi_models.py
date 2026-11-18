#!/usr/bin/env python3
"""
Multi-model ablation:
1) Prefix order shuffle
2) Channel masking (action/reward removal)

Models (default):
  lstm_attn, post_bilstm, rnn, bilstm, transformer, mlp_pool

Outputs:
  results/ablation_multi_models/
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    "lstm_attn",
    "post_bilstm",
    "rnn",
    "bilstm",
    "pre_lstm",
    "transformer",
    "pre_bilstm",
    "cnn",
    "mlp_pool",
    "tinyrnn",
    "tinyrnn_paper",
]
MASK_MODES = ["full", "no_past_action", "no_past_reward", "no_action_no_reward", "stimulus_only"]
PREFIX_MODES = ["original", "prefix_shuffled"]


def _setup_runtime(handoff: Path) -> None:
    import sys

    # Prefer live repo (has tinyrnn_paper). Handoff runtime_code may lag MODEL_REGISTRY.
    # handoff = <repo>/profiled6/experiments/sliding/handoff_profiled6_sliding
    repo = handoff.parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    rt = handoff / "runtime_code"
    slide = rt / "profiled6_experiments_sliding"
    for p in (rt, slide):
        if str(p) not in sys.path:
            sys.path.append(str(p))  # after repo so live package wins


def _within_task_probe(z: np.ndarray, agents: np.ndarray, tasks: np.ndarray, seed: int) -> float:
    from profile_benchmark.eval.analyse_sliding_geometry import within_task_agent_probe

    return within_task_agent_probe(z, agents, tasks, seed=seed)


def _z(v: np.ndarray) -> np.ndarray:
    s = np.std(v, ddof=0)
    if s < 1e-12 or np.isnan(s):
        return np.zeros_like(v, dtype=float)
    return (v - np.mean(v)) / s


def _maybe_shuffle_prefix(ps, pa, pr, mode: str, rng: random.Random):
    import torch

    if mode == "original":
        return ps, pa, pr
    B, L, _ = ps.shape
    idx_all = []
    for _ in range(B):
        idx = list(range(L))
        rng.shuffle(idx)
        idx_all.append(idx)
    idx_t = torch.tensor(idx_all, device=ps.device, dtype=torch.long)
    ps2 = torch.gather(ps, 1, idx_t.unsqueeze(-1).expand(-1, -1, ps.shape[-1]))
    pa2 = torch.gather(pa, 1, idx_t)
    pr2 = torch.gather(pr, 1, idx_t)
    return ps2, pa2, pr2


def _apply_mask(pa, pr, mode: str):
    import torch

    if mode == "full":
        return pa, pr
    if mode == "no_past_action":
        return torch.zeros_like(pa), pr
    if mode == "no_past_reward":
        return pa, torch.zeros_like(pr)
    if mode in ("no_action_no_reward", "stimulus_only"):
        return torch.zeros_like(pa), torch.zeros_like(pr)
    return pa, pr


def run_model_mode(
    model_name: str,
    ckpt: Path,
    data_csv: Path,
    *,
    prefix_mode: str | None,
    mask_mode: str | None,
    use_prefix_mask: bool = False,
) -> Dict:
    import torch
    import torch.nn.functional as F
    from profile_benchmark.engine.checkpoint import load_checkpoint
    from profile_benchmark.engine.forward import forward_model, logits_from_outputs
    from profile_benchmark.data import load_episodes
    from profile_benchmark.data_profile import filter_episodes_by_tasks, split_episodes
    from profile_benchmark.data_profile_sliding import make_episode_window_loader, operational_window, WindowSpec

    model, meta = load_checkpoint(ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    model.eval()
    seed = int(meta.get("seed") or 0)
    rng = random.Random(1000 + seed)

    eps = load_episodes(str(data_csv))
    eps = filter_episodes_by_tasks(eps, exclude_tasks=["switching_easy"])
    _, _, test_eps = split_episodes(eps, split_seed=123)

    op_loader = make_episode_window_loader(test_eps, [operational_window()], batch_size=32)
    op_acc_num, op_acc_den = 0, 0
    op_nlls = []
    z80s, task80, agent80 = [], [], []

    with torch.no_grad():
        for batch in op_loader:
            ps = batch["prefix_stim"].to(device)
            pa = batch["prefix_action"].to(device)
            pr = batch["prefix_reward"].to(device)
            fs = batch["future_stim"].to(device)
            fa = batch["future_action"].to(device)

            if prefix_mode is not None:
                ps, pa, pr = _maybe_shuffle_prefix(ps, pa, pr, prefix_mode, rng)
            if mask_mode is not None:
                pa, pr = _apply_mask(pa, pr, mask_mode)

            pmask = batch.get("prefix_mask")
            if use_prefix_mask and pmask is not None:
                pmask = pmask.to(device)
            else:
                pmask = None
            out = forward_model(model, ps, pa, pr, fs, return_outputs=True, prefix_mask=pmask)
            logits = logits_from_outputs(out)
            pred = logits.argmax(dim=-1)
            op_acc_num += int((pred == fa).sum().item())
            op_acc_den += int(np.prod(fa.shape))
            logp = F.log_softmax(logits, dim=-1)
            nll = -logp.gather(dim=-1, index=fa.unsqueeze(-1)).squeeze(-1)
            op_nlls.append(float(nll.mean().item()))
            z = out.get("z_profile")
            if z is not None:
                z80s.append(z.detach().cpu().numpy())
                task80.extend(batch["task"])
                agent80.extend(batch["agent"])

    wt80 = np.nan
    if z80s:
        z80 = np.concatenate(z80s, axis=0)
        wt80 = _within_task_probe(z80, np.asarray(agent80, str), np.asarray(task80, str), seed + 1)

    # L5 for profile gain
    l5_loader = make_episode_window_loader(test_eps, [WindowSpec(0, 5, 5, 20)], batch_size=32)
    z5s, task5, agent5 = [], [], []
    with torch.no_grad():
        for batch in l5_loader:
            ps = batch["prefix_stim"].to(device)
            pa = batch["prefix_action"].to(device)
            pr = batch["prefix_reward"].to(device)
            fs = batch["future_stim"].to(device)

            if prefix_mode is not None:
                ps, pa, pr = _maybe_shuffle_prefix(ps, pa, pr, prefix_mode, rng)
            if mask_mode is not None:
                pa, pr = _apply_mask(pa, pr, mask_mode)

            pmask = batch.get("prefix_mask")
            if use_prefix_mask and pmask is not None:
                pmask = pmask.to(device)
            else:
                pmask = None
            out = forward_model(model, ps, pa, pr, fs, return_outputs=True, prefix_mask=pmask)
            z = out.get("z_profile")
            if z is not None:
                z5s.append(z.detach().cpu().numpy())
                task5.extend(batch["task"])
                agent5.extend(batch["agent"])

    wt5 = np.nan
    if z5s:
        z5 = np.concatenate(z5s, axis=0)
        wt5 = _within_task_probe(z5, np.asarray(agent5, str), np.asarray(task5, str), seed + 2)

    condition = prefix_mode if prefix_mode is not None else mask_mode
    ablation_type = "prefix_shuffle" if prefix_mode is not None else "channel_masking"
    return {
        "model": model_name,
        "seed": seed,
        "ablation_type": ablation_type,
        "condition": condition,
        "operational_acc": op_acc_num / max(op_acc_den, 1),
        "operational_nll": float(np.mean(op_nlls)) if op_nlls else np.nan,
        "within_task_agent_probe_L80": wt80,
        "within_task_agent_probe_L5": wt5,
        "profile_gain_probe": wt80 - wt5 if (not np.isnan(wt80) and not np.isnan(wt5)) else np.nan,
    }


def add_p_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("operational_acc", "operational_nll", "within_task_agent_probe_L80", "profile_gain_probe"):
        out[f"z_{c}"] = np.nan
        for (model, ab), g in out.groupby(["model", "ablation_type"], observed=True):
            idx = g.index
            vals = g[c].to_numpy(dtype=float)
            out.loc[idx, f"z_{c}"] = _z(vals)
    out["P_composite_proxy"] = (
        out["z_operational_acc"]
        - out["z_operational_nll"]
        + out["z_within_task_agent_probe_L80"]
        + out["z_profile_gain_probe"]
    ) / 4.0
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, ab, cond), g in df.groupby(["model", "ablation_type", "condition"], observed=True):
        for metric in ("operational_acc", "within_task_agent_probe_L80", "profile_gain_probe", "P_composite_proxy"):
            v = g[metric].dropna().to_numpy(dtype=float)
            mean = float(np.mean(v)) if len(v) else np.nan
            sem = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            rows.append(
                {
                    "model": model,
                    "ablation_type": ab,
                    "condition": cond,
                    "metric": metric,
                    "mean": mean,
                    "sem": sem,
                    "n_seeds": len(v),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: results/ablation_multi_models (canonical).",
    )
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Checkpoint directory. Default: handoff_profiled6_sliding/checkpoints.",
    )
    p.add_argument(
        "--data-csv",
        type=str,
        default=None,
        help="Episode CSV. Default: handoff_profiled6_sliding/data/simulated_profiled6.csv.",
    )
    p.add_argument("--use-prefix-mask", action="store_true")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    workspace = Path(__file__).resolve().parents[1]
    handoff = workspace.parent / "handoff_profiled6_sliding"
    _setup_runtime(handoff)
    out_dir = Path(args.out_dir) if args.out_dir else workspace / "results" / "ablation_multi_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else handoff / "checkpoints"
    data_csv = Path(args.data_csv) if args.data_csv else handoff / "data" / "simulated_profiled6.csv"

    rows: List[Dict] = []
    for model_name in models:
        ckpts = sorted(ckpt_dir.glob(f"{model_name}_seed*.pt"))
        if not ckpts:
            print(f"[WARN] no checkpoints for {model_name}")
            continue
        for ckpt in ckpts:
            # prefix shuffle: original vs shuffled
            for cond in PREFIX_MODES:
                r = run_model_mode(
                    model_name,
                    ckpt,
                    data_csv,
                    prefix_mode=cond,
                    mask_mode=None,
                    use_prefix_mask=args.use_prefix_mask,
                )
                rows.append(r)
                print(f"[OK] {ckpt.stem} prefix:{cond}")
            # channel masking
            for cond in MASK_MODES:
                r = run_model_mode(
                    model_name,
                    ckpt,
                    data_csv,
                    prefix_mode=None,
                    mask_mode=cond,
                    use_prefix_mask=args.use_prefix_mask,
                )
                rows.append(r)
                print(f"[OK] {ckpt.stem} mask:{cond}")

    df = pd.DataFrame(rows)
    df = add_p_proxy(df)
    df.to_csv(out_dir / "ablation_multi_per_seed.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out_dir / "ablation_multi_summary_mean_sem.csv", index=False)

    # delta vs baseline for quick inspection
    delta_rows = []
    for model_name, g in df.groupby("model", observed=True):
        # prefix baseline
        gp = g[g["ablation_type"] == "prefix_shuffle"]
        basep = gp[gp["condition"] == "original"].set_index("seed")
        for cond in [c for c in PREFIX_MODES if c != "original"]:
            cur = gp[gp["condition"] == cond].set_index("seed")
            if basep.empty or cur.empty:
                continue
            d = (cur[["operational_acc", "within_task_agent_probe_L80", "profile_gain_probe"]] -
                 basep[["operational_acc", "within_task_agent_probe_L80", "profile_gain_probe"]]).mean()
            delta_rows.append(
                {
                    "model": model_name,
                    "ablation_type": "prefix_shuffle",
                    "condition": cond,
                    "delta_operational_acc": float(d["operational_acc"]),
                    "delta_within_task_probe_L80": float(d["within_task_agent_probe_L80"]),
                    "delta_profile_gain_probe": float(d["profile_gain_probe"]),
                }
            )
        # channel baseline
        gc = g[g["ablation_type"] == "channel_masking"]
        basec = gc[gc["condition"] == "full"].set_index("seed")
        for cond in [c for c in MASK_MODES if c != "full"]:
            cur = gc[gc["condition"] == cond].set_index("seed")
            if basec.empty or cur.empty:
                continue
            d = (cur[["operational_acc", "within_task_agent_probe_L80", "profile_gain_probe"]] -
                 basec[["operational_acc", "within_task_agent_probe_L80", "profile_gain_probe"]]).mean()
            delta_rows.append(
                {
                    "model": model_name,
                    "ablation_type": "channel_masking",
                    "condition": cond,
                    "delta_operational_acc": float(d["operational_acc"]),
                    "delta_within_task_probe_L80": float(d["within_task_agent_probe_L80"]),
                    "delta_profile_gain_probe": float(d["profile_gain_probe"]),
                }
            )
    pd.DataFrame(delta_rows).to_csv(out_dir / "ablation_multi_delta_vs_baseline.csv", index=False)
    print(f"[OK] multi-model ablation summary -> {out_dir}")


if __name__ == "__main__":
    main()

