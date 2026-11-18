#!/usr/bin/env python3
"""
Channel masking ablation for LSTM-attn prefix inputs.

Conditions:
- full
- no_past_action
- no_past_reward
- no_action_no_reward
- stimulus_only (same as no_action_no_reward in current input design)

Metrics:
- operational_acc
- within_task_agent_probe_L80
- profile_gain_probe
- P_composite_proxy
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


MASK_MODES = ["full", "no_past_action", "no_past_reward", "no_action_no_reward", "stimulus_only"]


def _setup_runtime(handoff: Path) -> None:
    import sys

    rt = handoff / "runtime_code"
    slide = rt / "profiled6_experiments_sliding"
    for p in (rt, slide):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _prefix_x(model, prefix_stim, prefix_action, prefix_reward):
    import torch

    ar = torch.cat(
        [prefix_action.float().unsqueeze(-1), prefix_reward.unsqueeze(-1)],
        dim=-1,
    )
    x = torch.cat([prefix_stim, ar], dim=-1)
    if hasattr(model, "embed"):
        x = torch.relu(model.embed(x))
    return x


def _forward_lstm_attn(model, prefix_stim, prefix_action, prefix_reward, future_stim):
    import torch

    x = _prefix_x(model, prefix_stim, prefix_action, prefix_reward)
    h_seq, _ = model.lstm(x)
    queries = model.query_proj(future_stim)
    logits_attn = torch.bmm(queries, h_seq.transpose(1, 2)) / (h_seq.size(-1) ** 0.5)
    attn = torch.softmax(logits_attn, dim=-1)
    context = torch.bmm(attn, h_seq)
    out = torch.cat([context, future_stim], dim=-1)
    logits = model.out_proj(out)
    z = context.mean(dim=1)
    return logits, z


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


def _within_task_probe(z: np.ndarray, agents: np.ndarray, tasks: np.ndarray, seed: int) -> float:
    from profile_benchmark.eval.analyse_sliding_geometry import within_task_agent_probe

    return within_task_agent_probe(z, agents, tasks, seed=seed)


def _z(v: np.ndarray) -> np.ndarray:
    s = np.std(v, ddof=0)
    if s < 1e-12 or np.isnan(s):
        return np.zeros_like(v, dtype=float)
    return (v - np.mean(v)) / s


def run_ckpt_mode(ckpt: Path, data_csv: Path, mode: str) -> Dict:
    import torch
    from profile_benchmark.engine.checkpoint import load_checkpoint
    from profile_benchmark.data import load_episodes
    from profile_benchmark.data_profile import filter_episodes_by_tasks, split_episodes
    from profile_benchmark.data_profile_sliding import make_episode_window_loader, operational_window, WindowSpec

    model, meta = load_checkpoint(ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    model.eval()
    seed = int(meta.get("seed") or 0)

    eps = load_episodes(str(data_csv))
    eps = filter_episodes_by_tasks(eps, exclude_tasks=["switching_easy"])
    _, _, test_eps = split_episodes(eps, split_seed=123)

    op_loader = make_episode_window_loader(test_eps, [operational_window()], batch_size=32)
    op_acc_num, op_acc_den = 0, 0
    op_nlls = []
    z80s, task80, agent80 = [], [], []
    with torch.no_grad():
        for batch in op_loader:
            fs = batch["future_stim"].to(device)
            fa = batch["future_action"].to(device)
            ps = batch["prefix_stim"].to(device)
            pa = batch["prefix_action"].to(device)
            pr = batch["prefix_reward"].to(device)
            pa, pr = _apply_mask(pa, pr, mode)
            logits, z = _forward_lstm_attn(model, ps, pa, pr, fs)
            pred = logits.argmax(dim=-1)
            op_acc_num += int((pred == fa).sum().item())
            op_acc_den += int(np.prod(fa.shape))
            logp = torch.log_softmax(logits, dim=-1)
            nll = -logp.gather(dim=-1, index=fa.unsqueeze(-1)).squeeze(-1)
            op_nlls.append(float(nll.mean().item()))
            z80s.append(z.detach().cpu().numpy())
            task80.extend(batch["task"])
            agent80.extend(batch["agent"])

    z80 = np.concatenate(z80s, axis=0)
    wt80 = _within_task_probe(z80, np.asarray(agent80, str), np.asarray(task80, str), seed + 1)

    # L5
    l5_loader = make_episode_window_loader(test_eps, [WindowSpec(0, 5, 5, 20)], batch_size=32)
    z5s, task5, agent5 = [], [], []
    with torch.no_grad():
        for batch in l5_loader:
            fs = batch["future_stim"].to(device)
            ps = batch["prefix_stim"].to(device)
            pa = batch["prefix_action"].to(device)
            pr = batch["prefix_reward"].to(device)
            pa, pr = _apply_mask(pa, pr, mode)
            _, z = _forward_lstm_attn(model, ps, pa, pr, fs)
            z5s.append(z.detach().cpu().numpy())
            task5.extend(batch["task"])
            agent5.extend(batch["agent"])
    z5 = np.concatenate(z5s, axis=0)
    wt5 = _within_task_probe(z5, np.asarray(agent5, str), np.asarray(task5, str), seed + 2)

    return {
        "model": "lstm_attn",
        "seed": seed,
        "mode": mode,
        "operational_acc": op_acc_num / max(op_acc_den, 1),
        "operational_nll": float(np.mean(op_nlls)) if op_nlls else np.nan,
        "within_task_agent_probe_L80": wt80,
        "within_task_agent_probe_L5": wt5,
        "profile_gain_probe": wt80 - wt5 if (not np.isnan(wt80) and not np.isnan(wt5)) else np.nan,
    }


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    handoff = workspace.parent / "handoff_profiled6_sliding"
    _setup_runtime(handoff)
    out_dir = workspace / "results" / "channel_masking_lstm_attn"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted((handoff / "checkpoints").glob("lstm_attn_seed*.pt"))
    data_csv = handoff / "data" / "simulated_profiled6.csv"
    if not ckpts:
        raise FileNotFoundError("No lstm_attn checkpoints found")

    rows: List[Dict] = []
    for ckpt in ckpts:
        for mode in MASK_MODES:
            r = run_ckpt_mode(ckpt, data_csv, mode)
            rows.append(r)
            print(f"[OK] {ckpt.stem} {mode}")
    df = pd.DataFrame(rows)

    for c in ("operational_acc", "operational_nll", "within_task_agent_probe_L80", "profile_gain_probe"):
        df[f"z_{c}"] = np.nan
        for mode, g in df.groupby("mode", observed=True):
            idx = g.index
            vals = g[c].to_numpy(dtype=float)
            df.loc[idx, f"z_{c}"] = _z(vals)
    df["P_composite_proxy"] = (
        df["z_operational_acc"]
        - df["z_operational_nll"]
        + df["z_within_task_agent_probe_L80"]
        + df["z_profile_gain_probe"]
    ) / 4.0
    df.to_csv(out_dir / "channel_masking_per_seed.csv", index=False)

    rows2 = []
    for mode, g in df.groupby("mode", observed=True):
        for metric in ("operational_acc", "within_task_agent_probe_L80", "profile_gain_probe", "P_composite_proxy"):
            v = g[metric].dropna().to_numpy(dtype=float)
            mean = float(np.mean(v)) if len(v) else np.nan
            sem = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            rows2.append({"mode": mode, "metric": metric, "mean": mean, "sem": sem, "n_seeds": len(v)})
    pd.DataFrame(rows2).to_csv(out_dir / "channel_masking_summary_mean_sem.csv", index=False)
    print(f"[OK] channel masking summary -> {out_dir}")


if __name__ == "__main__":
    main()

