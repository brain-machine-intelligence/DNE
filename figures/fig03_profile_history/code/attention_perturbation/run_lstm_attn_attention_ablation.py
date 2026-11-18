#!/usr/bin/env python3
"""
Attention ablation for LSTM-attn on operational / L5 windows.

Modes:
- original
- uniform attention
- shuffled attention

Metrics:
- operational_acc
- within_task_agent_probe_L80
- profile_gain (within_task_probe_L80 - within_task_probe_L5)

Note:
This script requires PyTorch + runtime_code in handoff package.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _setup_runtime(handoff: Path) -> None:
    import sys

    runtime_root = handoff / "runtime_code"
    slide_rt = runtime_root / "profiled6_experiments_sliding"
    for p in (runtime_root, slide_rt):
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


def forward_lstm_attn_ablation(model, prefix_stim, prefix_action, prefix_reward, future_stim, mode: str, rng):
    import torch

    x = _prefix_x(model, prefix_stim, prefix_action, prefix_reward)
    h_seq, _ = model.lstm(x)  # [B, L, H]
    queries = model.query_proj(future_stim)  # [B, K, H]
    logits = torch.bmm(queries, h_seq.transpose(1, 2)) / (h_seq.size(-1) ** 0.5)
    attn = torch.softmax(logits, dim=-1)  # [B, K, L]

    B, K, L = attn.shape
    if mode == "uniform":
        attn = torch.full_like(attn, 1.0 / L)
    elif mode == "shuffled":
        idx = np.arange(L)
        shuffled = []
        for _ in range(B):
            p = idx.copy()
            rng.shuffle(p)
            shuffled.append(torch.tensor(p, device=attn.device))
        out = []
        for b in range(B):
            out.append(attn[b : b + 1, :, shuffled[b]])
        attn = torch.cat(out, dim=0)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    context = torch.bmm(attn, h_seq)
    out_inp = torch.cat([context, future_stim], dim=-1)
    pred_logits = model.out_proj(out_inp)
    z = context.mean(dim=1)
    return pred_logits, z


def within_task_agent_probe(z: np.ndarray, agents: np.ndarray, tasks: np.ndarray, seed: int) -> float:
    from profile_benchmark.eval.analyse_sliding_geometry import within_task_agent_probe as wt_probe

    return wt_probe(z, agents, tasks, seed=seed)


def run_mode_for_ckpt(ckpt_path: Path, data_csv: Path, mode: str, seed_offset: int = 0) -> Dict:
    import torch
    from profile_benchmark.engine.checkpoint import load_checkpoint
    from profile_benchmark.data_profile import filter_episodes_by_tasks, split_episodes
    from profile_benchmark.data_profile_sliding import WindowSpec, make_episode_window_loader, operational_window

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(ckpt_path, device)
    model.eval()
    seed = int(meta.get("seed") or 0)

    # test episodes
    from profile_benchmark.data import load_episodes

    episodes = load_episodes(str(data_csv))
    episodes = filter_episodes_by_tasks(episodes, exclude_tasks=["switching_easy"])
    _, _, test_eps = split_episodes(episodes, split_seed=123)

    # operational L80
    op_loader = make_episode_window_loader(test_eps, [operational_window()], batch_size=32)

    op_correct = 0
    op_total = 0
    z80_list = []
    task80 = []
    agent80 = []
    rng = random.Random(1234 + seed + seed_offset)

    with torch.no_grad():
        for batch in op_loader:
            ps = batch["prefix_stim"].to(device)
            pa = batch["prefix_action"].to(device)
            pr = batch["prefix_reward"].to(device)
            fs = batch["future_stim"].to(device)
            fa = batch["future_action"].to(device)

            if mode == "original":
                out = model(ps, pa, pr, fs)
                logits = out if not isinstance(out, dict) else out["logits"]
                # z from same manual path for consistency
                _, z = forward_lstm_attn_ablation(model, ps, pa, pr, fs, mode="original", rng=rng)
            else:
                logits, z = forward_lstm_attn_ablation(model, ps, pa, pr, fs, mode=mode, rng=rng)

            pred = logits.argmax(dim=-1)
            op_correct += int((pred == fa).sum().item())
            op_total += int(np.prod(fa.shape))
            z80_list.append(z.detach().cpu().numpy())
            task80.extend(batch["task"])
            agent80.extend(batch["agent"])

    z80 = np.concatenate(z80_list, axis=0)
    wt80 = within_task_agent_probe(
        z80, np.asarray(agent80, dtype=str), np.asarray(task80, dtype=str), seed=seed + 1
    )

    # L5 for profile gain
    l5_loader = make_episode_window_loader(test_eps, [WindowSpec(0, 5, 5, 20)], batch_size=32)
    z5_list = []
    task5 = []
    agent5 = []
    with torch.no_grad():
        for batch in l5_loader:
            ps = batch["prefix_stim"].to(device)
            pa = batch["prefix_action"].to(device)
            pr = batch["prefix_reward"].to(device)
            fs = batch["future_stim"].to(device)
            if mode == "original":
                _, z = forward_lstm_attn_ablation(model, ps, pa, pr, fs, mode="original", rng=rng)
            else:
                _, z = forward_lstm_attn_ablation(model, ps, pa, pr, fs, mode=mode, rng=rng)
            z5_list.append(z.detach().cpu().numpy())
            task5.extend(batch["task"])
            agent5.extend(batch["agent"])
    z5 = np.concatenate(z5_list, axis=0)
    wt5 = within_task_agent_probe(
        z5, np.asarray(agent5, dtype=str), np.asarray(task5, dtype=str), seed=seed + 2
    )

    return {
        "model": meta["model"],
        "seed": seed,
        "mode": mode,
        "operational_acc": op_correct / max(op_total, 1),
        "within_task_agent_probe_L80": wt80,
        "within_task_agent_probe_L5": wt5,
        "profile_gain": wt80 - wt5 if (not np.isnan(wt80) and not np.isnan(wt5)) else np.nan,
    }


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    handoff = workspace.parent / "handoff_profiled6_sliding"
    _setup_runtime(handoff)

    out_dir = workspace / "results" / "attention_ablation_lstm_attn"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_csv = handoff / "data" / "simulated_profiled6.csv"
    ckpt_dir = handoff / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("lstm_attn_seed*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No lstm_attn checkpoints found in {ckpt_dir}")

    rows: List[Dict] = []
    modes = ["original", "uniform", "shuffled"]
    for ckpt in ckpts:
        for mode in modes:
            row = run_mode_for_ckpt(ckpt, data_csv, mode)
            rows.append(row)
            print(f"[OK] {ckpt.stem} {mode}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_per_seed.csv", index=False)

    # model-level mean ± sem
    summ_rows = []
    for mode, g in df.groupby("mode", observed=True):
        for metric in ["operational_acc", "within_task_agent_probe_L80", "profile_gain"]:
            v = g[metric].dropna().to_numpy(dtype=float)
            if len(v) == 0:
                mean, sem, n = np.nan, np.nan, 0
            elif len(v) == 1:
                mean, sem, n = float(v[0]), 0.0, 1
            else:
                mean = float(np.mean(v))
                sem = float(np.std(v, ddof=1) / np.sqrt(len(v)))
                n = int(len(v))
            summ_rows.append({"mode": mode, "metric": metric, "mean": mean, "sem": sem, "n_seeds": n})
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(out_dir / "ablation_summary_mean_sem.csv", index=False)

    # quick bar figure
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    metrics = ["operational_acc", "within_task_agent_probe_L80", "profile_gain"]
    for ax, metric in zip(axes, metrics):
        g = summ[summ["metric"] == metric].set_index("mode").reindex(modes)
        y = g["mean"].to_numpy(dtype=float)
        e = g["sem"].to_numpy(dtype=float)
        ax.bar(modes, y, yerr=e, capsize=3)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_attention_ablation_lstm_attn.png", dpi=220)
    plt.close(fig)

    (out_dir / "note.txt").write_text(
        "If ablation degrades probe/profile_gain despite modest enrichment, attention can still be functionally important.\n",
        encoding="utf-8",
    )
    print(f"[OK] attention ablation (lstm_attn) -> {out_dir}")


if __name__ == "__main__":
    main()

