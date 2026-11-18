#!/usr/bin/env python3
"""Attention uniform/shuffled ablation for inputpad_v2 (one-hot + prefix mask)."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

SLIDING = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
for p in (str(REPO), str(SLIDING)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_profiled6_sliding import CKPT_DIR_INPUTPAD_V2, DATA_CSV  # noqa: E402
from profile_benchmark.data import load_episodes  # noqa: E402
from profile_benchmark.data_profile import filter_episodes_by_tasks, split_episodes  # noqa: E402
from profile_benchmark.data_profile_sliding import (  # noqa: E402
    WindowSpec,
    make_episode_window_loader,
    operational_window,
)
from profile_benchmark.engine.checkpoint import load_checkpoint  # noqa: E402
from profile_benchmark.engine.prefix_packing import (  # noqa: E402
    key_padding_mask,
    last_hidden,
    masked_dot_attention,
    run_packed_rnn,
)
from profile_benchmark.eval.analyse_sliding_geometry import within_task_agent_probe  # noqa: E402

ATTN_MODELS = ["lstm_attn", "post_bilstm", "pre_lstm", "pre_bilstm", "transformer"]
MODES = ["original", "uniform", "shuffled"]
OUT = (
    SLIDING
    / "handoff_analysis_workspace"
    / "results"
    / "attention_ablation_inputpad_v2"
)


def _embed(model, ps, pa, pr):
    relu = model.__class__.__name__ != "TransformerProfile"
    return model._embed_prefix(ps, pa, pr, relu=relu)


def _shuffle_seq(x, rng: random.Random):
    b, l, _ = x.shape
    idx = torch.tensor([rng.sample(range(l), l) for _ in range(b)], device=x.device, dtype=torch.long)
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _modify_posthoc_attn(attn, mode: str, rng: random.Random, prefix_mask):
    if mode == "original":
        return attn
    b, k, l = attn.shape
    if mode == "uniform":
        if prefix_mask is None:
            return torch.full_like(attn, 1.0 / l)
        m = prefix_mask.to(dtype=attn.dtype).unsqueeze(1)
        w = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return w.expand(-1, k, -1)
    out = []
    for bi in range(b):
        perm = rng.sample(range(l), l)
        out.append(attn[bi : bi + 1, :, perm])
    attn_s = torch.cat(out, dim=0)
    if prefix_mask is not None:
        attn_s = attn_s.masked_fill(~prefix_mask.unsqueeze(1), 0)
    return attn_s / attn_s.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def forward_with_ablation(model, ps, pa, pr, fs, mode: str, rng: random.Random, prefix_mask):
    name = model.__class__.__name__
    x = _embed(model, ps, pa, pr)

    if hasattr(model, "query_proj") and hasattr(model, "lstm"):
        if prefix_mask is None:
            h_seq, _ = model.lstm(x)
        else:
            h_seq, _ = run_packed_rnn(model.lstm, x, prefix_mask)
        queries = model.query_proj(fs)
        attn, context = masked_dot_attention(queries, h_seq, prefix_mask)
        attn = _modify_posthoc_attn(attn, mode, rng, prefix_mask)
        context = torch.bmm(attn, h_seq)
        logits = model.out_proj(torch.cat([context, fs], dim=-1))
        return logits, context.mean(dim=1)

    if hasattr(model, "pre_attn") and hasattr(model, "lstm"):
        pad_k = key_padding_mask(prefix_mask)
        if mode == "original":
            x_attn, _ = model.pre_attn(x, x, x, key_padding_mask=pad_k)
        elif mode == "uniform":
            if prefix_mask is None:
                x_attn = x.mean(dim=1, keepdim=True).expand_as(x)
            else:
                from profile_benchmark.engine.prefix_packing import masked_mean
                x_attn = masked_mean(x, prefix_mask).unsqueeze(1).expand_as(x)
        else:
            x_attn = _shuffle_seq(x, rng)
        bidir = bool(getattr(model.lstm, "bidirectional", False))
        if prefix_mask is None:
            h_seq, hidden = model.lstm(x_attn)
            h = last_hidden(hidden, bidirectional=bidir)
        else:
            h_seq, hidden = run_packed_rnn(model.lstm, x_attn, prefix_mask)
            h = last_hidden(hidden, bidirectional=bidir)
        h_rep = h.unsqueeze(1).expand(-1, fs.size(1), -1)
        logits = model.head(torch.cat([h_rep, fs], dim=-1))
        return logits, h

    # transformer
    L = x.size(1)
    if getattr(model, "pos_embedding", None) is not None:
        x = x + model.pos_embedding[:L].unsqueeze(0)
    pad_k = key_padding_mask(prefix_mask)
    if mode == "original":
        h_seq = model.encoder(x, src_key_padding_mask=pad_k)
    elif mode == "uniform":
        if prefix_mask is None:
            h_seq = x.mean(dim=1, keepdim=True).expand_as(x)
        else:
            from profile_benchmark.engine.prefix_packing import masked_mean
            h_seq = masked_mean(x, prefix_mask).unsqueeze(1).expand_as(x)
    else:
        h_seq = model.encoder(_shuffle_seq(x, rng), src_key_padding_mask=pad_k)
    h = model._pool_prefix(h_seq, prefix_mask)
    h_rep = h.unsqueeze(1).expand(-1, fs.size(1), -1)
    logits = model.head(torch.cat([h_rep, fs], dim=-1))
    return logits, h


@torch.no_grad()
def run_mode(ckpt: Path, test_eps, mode: str) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(ckpt, device)
    model.eval()
    seed = int(meta.get("seed") or 0)
    rng = random.Random(1234 + seed)
    loader = make_episode_window_loader(test_eps, [operational_window()], batch_size=32)
    n_ok = n_all = 0
    zs, agents, tasks = [], [], []
    for batch in loader:
        ps = batch["prefix_stim"].to(device)
        pa = batch["prefix_action"].to(device)
        pr = batch["prefix_reward"].to(device)
        fs = batch["future_stim"].to(device)
        fa = batch["future_action"].to(device)
        pmask = batch.get("prefix_mask")
        if pmask is not None:
            pmask = pmask.to(device)
        logits, z = forward_with_ablation(model, ps, pa, pr, fs, mode, rng, pmask)
        n_ok += int((logits.argmax(dim=-1) == fa).sum().item())
        n_all += int(fa.numel())
        zs.append(z.detach().cpu().numpy())
        agents.extend(batch["agent"])
        tasks.extend(batch["task"])
    z80 = np.concatenate(zs, axis=0)
    wt80 = within_task_agent_probe(z80, np.asarray(agents, str), np.asarray(tasks, str), seed=seed + 1)

    l5 = make_episode_window_loader(test_eps, [WindowSpec(0, 5, 5, 20)], batch_size=32)
    z5s, a5, t5 = [], [], []
    for batch in l5:
        ps = batch["prefix_stim"].to(device)
        pa = batch["prefix_action"].to(device)
        pr = batch["prefix_reward"].to(device)
        fs = batch["future_stim"].to(device)
        pmask = batch.get("prefix_mask")
        if pmask is not None:
            pmask = pmask.to(device)
        _, z = forward_with_ablation(model, ps, pa, pr, fs, mode, rng, pmask)
        z5s.append(z.detach().cpu().numpy())
        a5.extend(batch["agent"])
        t5.extend(batch["task"])
    z5 = np.concatenate(z5s, axis=0)
    wt5 = within_task_agent_probe(z5, np.asarray(a5, str), np.asarray(t5, str), seed=seed + 2)
    return {
        "model": str(meta["model"]),
        "seed": seed,
        "mode": mode,
        "operational_acc": n_ok / max(n_all, 1),
        "within_task_agent_probe_L80": wt80,
        "within_task_agent_probe_L5": wt5,
        "profile_gain": wt80 - wt5 if np.isfinite(wt80) and np.isfinite(wt5) else np.nan,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default=",".join(ATTN_MODELS))
    p.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR_INPUTPAD_V2))
    p.add_argument("--out-dir", type=str, default=str(OUT))
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = load_episodes(str(DATA_CSV))
    eps = filter_episodes_by_tasks(eps, include_tasks=["causal", "bandit", "field2d", "switching_medium"])
    _, _, test_eps = split_episodes(eps, split_seed=123)

    path = out_dir / "ablation_per_seed.csv"
    done = set()
    rows: List[Dict] = []
    if path.is_file():
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = {(r["model"], int(r["seed"]), r["mode"]) for r in rows}
        print(f"[resume] {len(done)} rows", flush=True)

    for model_name in models:
        ckpts = sorted(ckpt_dir.glob(f"{model_name}_seed*.pt"))
        for ckpt in ckpts:
            for mode in MODES:
                key = (model_name, int(ckpt.stem.split("seed")[-1]), mode)
                if key in done:
                    continue
                row = run_mode(ckpt, test_eps, mode)
                rows.append(row)
                done.add(key)
                print(f"[OK] {ckpt.stem} {mode} A={row['operational_acc']:.4f}", flush=True)
                pd.DataFrame(rows).to_csv(path, index=False)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    summ = []
    for (model, mode), g in df.groupby(["model", "mode"], observed=True):
        for metric in ["operational_acc", "within_task_agent_probe_L80", "profile_gain"]:
            v = g[metric].dropna().to_numpy(float)
            summ.append(
                {
                    "model": model,
                    "mode": mode,
                    "metric": metric,
                    "mean": float(np.mean(v)) if len(v) else np.nan,
                    "sem": float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                    "n_seeds": len(v),
                }
            )
    pd.DataFrame(summ).to_csv(out_dir / "ablation_summary_mean_sem.csv", index=False)
    print(f"[OK] attention ablation -> {out_dir} n={len(df)}")


if __name__ == "__main__":
    main()
