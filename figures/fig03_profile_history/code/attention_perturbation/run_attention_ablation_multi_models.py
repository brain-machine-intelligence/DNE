#!/usr/bin/env python3
"""
Attention ablation (uniform / shuffled) for all attention-capable profile models.

Models:
  lstm_attn, post_bilstm          — post-hoc query×prefix attention
  pre_lstm, pre_bilstm            — prefix self-attention (MultiheadAttention)
  transformer                     — encoder self-attention bypass / shuffle

Modes: original, uniform, shuffled

Outputs:
  results/attention_ablation_multi/ablation_per_seed.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_MODELS = ["lstm_attn", "post_bilstm", "pre_lstm", "pre_bilstm", "transformer"]
MODES = ["original", "uniform", "shuffled"]


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


def _shuffle_seq(x, rng: random.Random):
    import torch

    b, l, _ = x.shape
    idx_all = []
    for _ in range(b):
        idx = list(range(l))
        rng.shuffle(idx)
        idx_all.append(idx)
    idx_t = torch.tensor(idx_all, device=x.device, dtype=torch.long)
    return torch.gather(x, 1, idx_t.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _modify_posthoc_attn(attn, mode: str, rng: random.Random):
    import torch

    if mode == "original":
        return attn
    b, _, l = attn.shape
    if mode == "uniform":
        return torch.full_like(attn, 1.0 / l)
    out = []
    for bi in range(b):
        idx = list(range(l))
        rng.shuffle(idx)
        out.append(attn[bi : bi + 1, :, idx])
    attn_s = torch.cat(out, dim=0)
    return attn_s / attn_s.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _model_family(model) -> str:
    name = model.__class__.__name__
    if hasattr(model, "query_proj") and hasattr(model, "lstm"):
        return "posthoc"
    if hasattr(model, "pre_attn") and hasattr(model, "lstm"):
        return "pre_attn"
    if hasattr(model, "encoder"):
        return "transformer"
    raise ValueError(f"Unsupported attention model: {name}")


def forward_with_ablation(model, prefix_stim, prefix_action, prefix_reward, future_stim, mode: str, rng: random.Random):
    import torch

    family = _model_family(model)

    if family == "posthoc":
        x = _prefix_x(model, prefix_stim, prefix_action, prefix_reward)
        h_seq, _ = model.lstm(x)
        queries = model.query_proj(future_stim)
        attn_logits = torch.bmm(queries, h_seq.transpose(1, 2)) / (h_seq.size(-1) ** 0.5)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = _modify_posthoc_attn(attn, mode, rng)
        context = torch.bmm(attn, h_seq)
        out_inp = torch.cat([context, future_stim], dim=-1)
        logits = model.out_proj(out_inp)
        z = context.mean(dim=1)
        return logits, z

    if family == "pre_attn":
        x = _prefix_x(model, prefix_stim, prefix_action, prefix_reward)
        if mode == "original":
            x_attn, _ = model.pre_attn(x, x, x)
        elif mode == "uniform":
            x_attn = x.mean(dim=1, keepdim=True).expand_as(x)
        else:
            x_attn = _shuffle_seq(x, rng)
        h_seq, (h_last, _) = model.lstm(x_attn)
        if getattr(model.lstm, "bidirectional", False):
            h = torch.cat([h_last[-2], h_last[-1]], dim=-1)
        else:
            h = h_last[-1]
        b, k, _ = future_stim.shape
        h_rep = h.unsqueeze(1).expand(-1, k, -1)
        inp = torch.cat([h_rep, future_stim], dim=-1)
        logits = model.head(inp)
        return logits, h

    # transformer
    b, l, _ = prefix_stim.shape
    x = _prefix_x(model, prefix_stim, prefix_action, prefix_reward)
    if getattr(model, "pos_embedding", None) is not None:
        x = x + model.pos_embedding[:l].unsqueeze(0)
    if mode == "original":
        h_seq = model.encoder(x)
    elif mode == "uniform":
        h_seq = x.mean(dim=1, keepdim=True).expand_as(x)
    else:
        h_seq = model.encoder(_shuffle_seq(x, rng))
    h = model._pool_prefix(h_seq) if hasattr(model, "_pool_prefix") else h_seq.mean(dim=1)
    b, k, _ = future_stim.shape
    h_rep = h.unsqueeze(1).expand(-1, k, -1)
    inp = torch.cat([h_rep, future_stim], dim=-1)
    logits = model.head(inp)
    return logits, h


def within_task_agent_probe(z: np.ndarray, agents: np.ndarray, tasks: np.ndarray, seed: int) -> float:
    from profile_benchmark.eval.analyse_sliding_geometry import within_task_agent_probe as wt_probe

    return wt_probe(z, agents, tasks, seed=seed)


def run_mode_for_ckpt(ckpt_path: Path, data_csv: Path, mode: str) -> Dict:
    import torch
    from profile_benchmark.engine.checkpoint import load_checkpoint
    from profile_benchmark.data import load_episodes
    from profile_benchmark.data_profile import filter_episodes_by_tasks, split_episodes
    from profile_benchmark.data_profile_sliding import WindowSpec, make_episode_window_loader, operational_window

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(ckpt_path, device)
    model.eval()
    seed = int(meta.get("seed") or 0)
    rng = random.Random(1234 + seed)

    episodes = load_episodes(str(data_csv))
    episodes = filter_episodes_by_tasks(episodes, exclude_tasks=["switching_easy"])
    _, _, test_eps = split_episodes(episodes, split_seed=123)

    op_loader = make_episode_window_loader(test_eps, [operational_window()], batch_size=32)
    op_correct = 0
    op_total = 0
    z80_list, task80, agent80 = [], [], []

    with torch.no_grad():
        for batch in op_loader:
            ps = batch["prefix_stim"].to(device)
            pa = batch["prefix_action"].to(device)
            pr = batch["prefix_reward"].to(device)
            fs = batch["future_stim"].to(device)
            fa = batch["future_action"].to(device)
            logits, z = forward_with_ablation(model, ps, pa, pr, fs, mode, rng)
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

    l5_loader = make_episode_window_loader(test_eps, [WindowSpec(0, 5, 5, 20)], batch_size=32)
    z5_list, task5, agent5 = [], [], []
    with torch.no_grad():
        for batch in l5_loader:
            ps = batch["prefix_stim"].to(device)
            pa = batch["prefix_action"].to(device)
            pr = batch["prefix_reward"].to(device)
            fs = batch["future_stim"].to(device)
            _, z = forward_with_ablation(model, ps, pa, pr, fs, mode, rng)
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
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    workspace = Path(__file__).resolve().parents[1]
    handoff = workspace.parent / "handoff_profiled6_sliding"
    _setup_runtime(handoff)

    out_dir = workspace / "results" / "attention_ablation_multi"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_csv = handoff / "data" / "simulated_profiled6.csv"
    ckpt_dir = handoff / "checkpoints"

    rows: List[Dict] = []
    for model_name in models:
        ckpts = sorted(ckpt_dir.glob(f"{model_name}_seed*.pt"))
        if not ckpts:
            print(f"[WARN] no checkpoints for {model_name}")
            continue
        for ckpt in ckpts:
            for mode in MODES:
                row = run_mode_for_ckpt(ckpt, data_csv, mode)
                rows.append(row)
                print(f"[OK] {ckpt.stem} {mode}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_per_seed.csv", index=False)

    summ_rows = []
    for (model, mode), g in df.groupby(["model", "mode"], observed=True):
        for metric in ["operational_acc", "within_task_agent_probe_L80", "profile_gain"]:
            v = g[metric].dropna().to_numpy(dtype=float)
            mean = float(np.mean(v)) if len(v) else np.nan
            sem = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            summ_rows.append(
                {"model": model, "mode": mode, "metric": metric, "mean": mean, "sem": sem, "n_seeds": len(v)}
            )
    pd.DataFrame(summ_rows).to_csv(out_dir / "ablation_summary_mean_sem.csv", index=False)
    print(f"[OK] attention ablation multi -> {out_dir}")


if __name__ == "__main__":
    main()
