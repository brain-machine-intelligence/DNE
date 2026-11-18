#!/usr/bin/env python3
"""Profiled6 sliding-window evaluation suite (Eval A–E)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import torch

from profile_benchmark.data_profile import DEFAULT_SPLIT_SEED, filter_episodes_by_tasks, split_episodes
from profile_benchmark.data_profile_sliding import (
    DEFAULT_EVAL_L,
    DEFAULT_EPISODE_TRIALS,
    DEFAULT_SLIDING_STRIDE,
    DEFAULT_TARGET_LEN,
    WindowSpec,
    generate_sliding_windows,
    make_episode_window_loader,
    operational_window,
)
from profile_benchmark.engine.checkpoint import load_checkpoint
from profile_benchmark.eval.common import (
    PROFILED4_EXCLUDE,
    episode_summary_rows,
    predict_batch,
    trial_rows_from_batch,
)
from profile_benchmark.utils import set_seed


def _load_test_episodes(data: str, split_seed: int):
    from profile_benchmark.data import load_episodes

    episodes = load_episodes(data)
    episodes = filter_episodes_by_tasks(episodes, exclude_tasks=PROFILED4_EXCLUDE)
    _, _, test_eps = split_episodes(episodes, split_seed=split_seed)
    return test_eps


def _prefix_lens(batch) -> List[int]:
    pl = batch["prefix_len"]
    if isinstance(pl, torch.Tensor):
        return [int(x) for x in pl.tolist()]
    return [int(x) for x in pl]


def _run_loader(
    loader,
    model,
    device: str,
    model_name: str,
    seed: int,
    eval_tag: str,
    use_prefix_mask: bool = False,
) -> tuple[list, list]:
    ep_rows, tr_rows = [], []
    for batch in loader:
        plens = _prefix_lens(batch)
        pred = predict_batch(
            model, batch, device, prefix_len=None, use_prefix_mask=use_prefix_mask
        )
        for i, row in enumerate(episode_summary_rows(batch, pred, model_name, seed, 0)):
            row["prefix_L"] = plens[i]
            row["eval_tag"] = eval_tag
            row["prefix_start"] = int(batch["prefix_start"][i])
            ep_rows.append(row)
        B = len(batch["episode_id"])
        for i in range(B):
            mini = {k: (v[i : i + 1] if isinstance(v, torch.Tensor) else [v[i]]) for k, v in batch.items()}
            pmini = {
                k: (v[i : i + 1] if isinstance(v, torch.Tensor) else v)
                for k, v in pred.items()
                if isinstance(v, torch.Tensor)
            }
            tr_rows.extend(trial_rows_from_batch(mini, pmini, model_name, seed, plens[i]))
    return ep_rows, tr_rows


def eval_operational(test_eps, model, device, model_name, seed, use_prefix_mask: bool = False):
    loader = make_episode_window_loader(test_eps, [operational_window()], batch_size=32)
    return _run_loader(
        loader, model, device, model_name, seed, "operational_L80", use_prefix_mask=use_prefix_mask
    )


def eval_sliding_l80(
    test_eps, model, device, model_name, seed, stride: int, episode_trials: int, use_prefix_mask: bool = False
):
    specs = generate_sliding_windows(
        episode_trials=episode_trials,
        L_values=[80],
        target_len=DEFAULT_TARGET_LEN,
        stride=stride,
    )
    loader = make_episode_window_loader(test_eps, specs, batch_size=32)
    return _run_loader(
        loader, model, device, model_name, seed, "sliding_L80", use_prefix_mask=use_prefix_mask
    )


def eval_prefix_curve_next(
    test_eps, model, device, model_name, seed, L_values: Sequence[int], use_prefix_mask: bool = False
):
    specs = [WindowSpec(0, L, L, DEFAULT_TARGET_LEN) for L in L_values if L + DEFAULT_TARGET_LEN <= DEFAULT_EPISODE_TRIALS]
    loader = make_episode_window_loader(test_eps, specs, batch_size=32)
    return _run_loader(
        loader, model, device, model_name, seed, "prefix_curve_next", use_prefix_mask=use_prefix_mask
    )


def eval_prefix_curve_fixed_target(
    test_eps, model, device, model_name, seed, L_values: Sequence[int], target_start: int = 80, use_prefix_mask: bool = False
):
    specs = []
    for L in L_values:
        if target_start + DEFAULT_TARGET_LEN <= DEFAULT_EPISODE_TRIALS:
            specs.append(WindowSpec(0, L, target_start, DEFAULT_TARGET_LEN))
    loader = make_episode_window_loader(test_eps, specs, batch_size=32)
    return _run_loader(
        loader, model, device, model_name, seed, "prefix_curve_fixed80", use_prefix_mask=use_prefix_mask
    )


def eval_horizon_stability(
    test_eps, model, device, model_name, seed, target_starts: Sequence[int], use_prefix_mask: bool = False
):
    specs = [
        WindowSpec(0, 80, ts, DEFAULT_TARGET_LEN)
        for ts in target_starts
        if ts + DEFAULT_TARGET_LEN <= DEFAULT_EPISODE_TRIALS
    ]
    loader = make_episode_window_loader(test_eps, specs, batch_size=32)
    return _run_loader(
        loader, model, device, model_name, seed, "horizon_from_L80", use_prefix_mask=use_prefix_mask
    )


def _z_rows_from_batch(batch, pred, model_name: str, seed: int) -> List[dict]:
    z = pred.get("z_profile")
    if z is None:
        return []
    z_np = z.detach().cpu().numpy()
    rows: List[dict] = []
    for i in range(z_np.shape[0]):
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "episode_id": batch["episode_id"][i],
                "subject_id": int(batch["subject_id"][i]),
                "task": batch["task"][i],
                "agent": batch["agent"][i],
                "prefix_start": int(batch["prefix_start"][i]),
                "prefix_len": int(batch["prefix_len"][i]),
                "target_start": int(batch["target_start"][i]),
                "z": z_np[i].tolist(),
            }
        )
    return rows


def eval_z_prefix_curve(
    test_eps, model, device, model_name, seed, L_values: Sequence[int], use_prefix_mask: bool = False
):
    """z_profile at prefix 0..L-1 for each L (Eval 5: profile gain across L)."""
    specs = [
        WindowSpec(0, L, L, DEFAULT_TARGET_LEN)
        for L in L_values
        if L + DEFAULT_TARGET_LEN <= DEFAULT_EPISODE_TRIALS
    ]
    loader = make_episode_window_loader(test_eps, specs, batch_size=32)
    rows: List[dict] = []
    for batch in loader:
        pred = predict_batch(
            model, batch, device, prefix_len=None, return_outputs=True, use_prefix_mask=use_prefix_mask
        )
        rows.extend(_z_rows_from_batch(batch, pred, model_name, seed))
    return rows


def eval_profile_stability(
    test_eps, model, device, model_name, seed, stride: int, episode_trials: int, use_prefix_mask: bool = False
):
    """Extract z_profile for sliding L=80 windows (Eval 5: stability)."""
    specs = generate_sliding_windows(
        episode_trials=episode_trials,
        L_values=[80],
        target_len=DEFAULT_TARGET_LEN,
        stride=stride,
    )
    loader = make_episode_window_loader(test_eps, specs, batch_size=16)
    rows: List[dict] = []
    for batch in loader:
        pred = predict_batch(
            model, batch, device, prefix_len=None, return_outputs=True, use_prefix_mask=use_prefix_mask
        )
        rows.extend(_z_rows_from_batch(batch, pred, model_name, seed))
    return rows


HORIZON_TARGET_STARTS = (80, 100, 140, 180, 220, 260)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--mode",
        default="all",
        choices=["all", "operational", "sliding", "prefix", "horizon", "repr", "repr_prefix"],
    )
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument("--stride", type=int, default=DEFAULT_SLIDING_STRIDE)
    p.add_argument("--episode-trials", type=int, default=DEFAULT_EPISODE_TRIALS)
    p.add_argument(
        "--use-prefix-mask",
        action="store_true",
        help="Pass collate prefix_mask. Also auto-on for inputpad_v2 / padding-corrected ckpts.",
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(args.ckpt, device)
    seed = int(meta.get("seed") or 0)
    model_name = meta["model"]
    set_seed(seed)
    cfg = meta.get("config") or {}
    use_mask = (
        bool(args.use_prefix_mask)
        or str(cfg.get("implementation", "legacy")) in ("inputpad_v2", "padding_corrected")
        or bool(cfg.get("padding_corrected"))
    )

    test_eps = _load_test_episodes(args.data, args.split_seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.ckpt).stem
    mode = args.mode

    ep_all, tr_all = [], []

    if mode in ("all", "operational"):
        e, t = eval_operational(test_eps, model, device, model_name, seed, use_prefix_mask=use_mask)
        ep_all.extend(e)
        tr_all.extend(t)

    if mode in ("all", "sliding"):
        e, t = eval_sliding_l80(
            test_eps, model, device, model_name, seed, args.stride, args.episode_trials, use_prefix_mask=use_mask
        )
        ep_all.extend(e)
        tr_all.extend(t)

    if mode in ("all", "prefix"):
        e, t = eval_prefix_curve_next(
            test_eps, model, device, model_name, seed, DEFAULT_EVAL_L, use_prefix_mask=use_mask
        )
        ep_all.extend(e)
        tr_all.extend(t)
        e2, t2 = eval_prefix_curve_fixed_target(
            test_eps, model, device, model_name, seed, DEFAULT_EVAL_L, use_prefix_mask=use_mask
        )
        ep_all.extend(e2)
        tr_all.extend(t2)

    if mode in ("all", "horizon"):
        e, t = eval_horizon_stability(
            test_eps, model, device, model_name, seed, HORIZON_TARGET_STARTS, use_prefix_mask=use_mask
        )
        ep_all.extend(e)
        tr_all.extend(t)

    if ep_all:
        pd.DataFrame(ep_all).to_csv(out / f"{stem}_sliding_episode.csv", index=False)
        pd.DataFrame(tr_all).to_csv(out / f"{stem}_sliding_trial.csv", index=False)
        print(f"[OK] episode/trial -> {out / f'{stem}_sliding_episode.csv'}")

    if mode in ("all", "repr"):
        import json

        zrows = eval_profile_stability(
            test_eps, model, device, model_name, seed, args.stride, args.episode_trials, use_prefix_mask=use_mask
        )
        zpath = out / f"{stem}_z_profile_windows.jsonl"
        with zpath.open("w", encoding="utf-8") as f:
            for row in zrows:
                f.write(json.dumps(row) + "\n")
        print(f"[OK] z_profile sliding -> {zpath} ({len(zrows)} rows)")

    if mode in ("all", "repr", "repr_prefix"):
        import json

        zpref = eval_z_prefix_curve(
            test_eps, model, device, model_name, seed, DEFAULT_EVAL_L, use_prefix_mask=use_mask
        )
        zppath = out / f"{stem}_z_prefix_curve.jsonl"
        with zppath.open("w", encoding="utf-8") as f:
            for row in zpref:
                f.write(json.dumps(row) + "\n")
        print(f"[OK] z_profile prefix-curve -> {zppath} ({len(zpref)} rows)")


if __name__ == "__main__":
    main()
