"""
Profile benchmark dataset with flexible prefix / target windows and task filters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .data import Episode, load_episodes

DEFAULT_SPLIT_SEED = 123
DEFAULT_TRAIN_PREFIX_L = 80
DEFAULT_TARGET_START = 80
DEFAULT_TARGET_LEN = 20


def filter_episodes_by_tasks(
    episodes: Sequence[Episode],
    include_tasks: Optional[Sequence[str]] = None,
    exclude_tasks: Optional[Sequence[str]] = None,
) -> List[Episode]:
    out: List[Episode] = []
    include = set(include_tasks) if include_tasks else None
    exclude = set(exclude_tasks) if exclude_tasks else set()
    for ep in episodes:
        task = str(ep.df["task"].iloc[0])
        if include is not None and task not in include:
            continue
        if task in exclude:
            continue
        out.append(ep)
    return out


def split_episodes(
    episodes: Sequence[Episode],
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[List[Episode], List[Episode], List[Episode]]:
    eps = list(episodes)
    rng = np.random.default_rng(seed=split_seed)
    rng.shuffle(eps)
    n = len(eps)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_eps = eps[:n_train]
    val_eps = eps[n_train : n_train + n_val]
    test_eps = eps[n_train + n_val :]
    return train_eps, val_eps, test_eps


def episode_row_id(ep: Episode) -> str:
    df = ep.df
    return f"{int(df['subject_id'].iloc[0])}|{df['task'].iloc[0]}|{df['agent'].iloc[0]}"


def save_split_manifest(
    train_eps: Sequence[Episode],
    val_eps: Sequence[Episode],
    test_eps: Sequence[Episode],
    path: Path,
    split_seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "split_seed": split_seed,
        "train": [episode_row_id(e) for e in train_eps],
        "val": [episode_row_id(e) for e in val_eps],
        "test": [episode_row_id(e) for e in test_eps],
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(path: Path) -> Dict[str, List[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


class ProfileDataset(Dataset):
    """
    prefix: trials [0, prefix_len)
    target: trials [target_start, target_start + target_len)
    """

    def __init__(
        self,
        episodes: Sequence[Episode],
        prefix_len: int = DEFAULT_TRAIN_PREFIX_L,
        target_start: int = DEFAULT_TARGET_START,
        target_len: int = DEFAULT_TARGET_LEN,
    ):
        self.episodes = list(episodes)
        self.prefix_len = int(prefix_len)
        self.target_start = int(target_start)
        self.target_len = int(target_len)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep = self.episodes[idx]
        stim, action, reward = ep.to_arrays()
        T = ep.length
        L = min(self.prefix_len, self.target_start)
        if L < 1:
            raise ValueError(f"Invalid prefix_len={self.prefix_len} for target_start={self.target_start}")

        start_f = self.target_start
        end_f = min(T, start_f + self.target_len)
        if end_f <= start_f:
            raise ValueError(f"Episode too short: T={T}, target_start={start_f}")

        df = ep.df
        return {
            "prefix_stim": torch.from_numpy(stim[:L]),
            "prefix_action": torch.from_numpy(action[:L]),
            "prefix_reward": torch.from_numpy(reward[:L]),
            "future_stim": torch.from_numpy(stim[start_f:end_f]),
            "future_action": torch.from_numpy(action[start_f:end_f]),
            "task": str(df["task"].iloc[0]),
            "agent": str(df["agent"].iloc[0]),
            "subject_id": int(df["subject_id"].iloc[0]),
            "episode_id": episode_row_id(ep),
            "prefix_len": L,
            "target_start": start_f,
            "target_end": end_f,
        }


def profile_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length prefix/future within a batch (for prefix_len < max in batch)."""
    max_l = max(int(b["prefix_stim"].shape[0]) for b in batch)
    max_k = max(int(b["future_stim"].shape[0]) for b in batch)
    stim_dim = batch[0]["prefix_stim"].shape[1]

    bs = len(batch)
    prefix_stim = torch.zeros(bs, max_l, stim_dim, dtype=torch.float32)
    prefix_action = torch.zeros(bs, max_l, dtype=torch.int64)
    prefix_reward = torch.zeros(bs, max_l, dtype=torch.float32)
    future_stim = torch.zeros(bs, max_k, stim_dim, dtype=torch.float32)
    future_action = torch.zeros(bs, max_k, dtype=torch.int64)
    prefix_mask = torch.zeros(bs, max_l, dtype=torch.bool)
    future_mask = torch.zeros(bs, max_k, dtype=torch.bool)

    meta_keys = [
        "task",
        "agent",
        "subject_id",
        "episode_id",
        "prefix_len",
        "target_start",
        "target_end",
    ]
    if batch and "prefix_start" in batch[0]:
        meta_keys.append("prefix_start")
    meta = {k: [] for k in meta_keys}

    for i, b in enumerate(batch):
        l = b["prefix_stim"].shape[0]
        k = b["future_stim"].shape[0]
        prefix_stim[i, :l] = b["prefix_stim"]
        prefix_action[i, :l] = b["prefix_action"]
        prefix_reward[i, :l] = b["prefix_reward"]
        future_stim[i, :k] = b["future_stim"]
        future_action[i, :k] = b["future_action"]
        prefix_mask[i, :l] = True
        future_mask[i, :k] = True
        for key in meta:
            meta[key].append(b[key])

    out = {
        "prefix_stim": prefix_stim,
        "prefix_action": prefix_action,
        "prefix_reward": prefix_reward,
        "future_stim": future_stim,
        "future_action": future_action,
        "prefix_mask": prefix_mask,
        "future_mask": future_mask,
    }
    out.update(meta)
    return out


def make_profile_loaders(
    csv_path: str,
    batch_size: int = 64,
    prefix_len: int = DEFAULT_TRAIN_PREFIX_L,
    target_start: int = DEFAULT_TARGET_START,
    target_len: int = DEFAULT_TARGET_LEN,
    split_seed: int = DEFAULT_SPLIT_SEED,
    include_tasks: Optional[Sequence[str]] = None,
    exclude_tasks: Optional[Sequence[str]] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, Dict[str, List[Episode]]]:
    episodes = load_episodes(csv_path)
    episodes = filter_episodes_by_tasks(episodes, include_tasks, exclude_tasks)
    train_eps, val_eps, test_eps = split_episodes(episodes, split_seed=split_seed)

    train_ds = ProfileDataset(train_eps, prefix_len, target_start, target_len)
    val_ds = ProfileDataset(val_eps, prefix_len, target_start, target_len)
    test_ds = ProfileDataset(test_eps, prefix_len, target_start, target_len)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=profile_collate_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    stim_dim = train_eps[0].to_arrays()[0].shape[1]
    df_actions = pd.read_csv(csv_path, usecols=["action"])
    n_actions = int(df_actions["action"].max()) + 1
    splits = {"train": train_eps, "val": val_eps, "test": test_eps}
    return train_loader, val_loader, test_loader, stim_dim, n_actions, splits


def make_test_profile_loader(
    csv_path: str,
    batch_size: int = 32,
    prefix_len: int = DEFAULT_TRAIN_PREFIX_L,
    target_start: int = DEFAULT_TARGET_START,
    target_len: int = DEFAULT_TARGET_LEN,
    split_seed: int = DEFAULT_SPLIT_SEED,
    include_tasks: Optional[Sequence[str]] = None,
    exclude_tasks: Optional[Sequence[str]] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, int, int, List[Episode]]:
    """Test split only (same episode split as training)."""
    episodes = load_episodes(csv_path)
    episodes = filter_episodes_by_tasks(episodes, include_tasks, exclude_tasks)
    _, _, test_eps = split_episodes(episodes, split_seed=split_seed)
    test_ds = ProfileDataset(test_eps, prefix_len, target_start, target_len)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=profile_collate_fn,
    )
    stim_dim = test_eps[0].to_arrays()[0].shape[1]
    df_actions = pd.read_csv(csv_path, usecols=["action"])
    n_actions = int(df_actions["action"].max()) + 1
    return test_loader, stim_dim, n_actions, test_eps


def build_profiled4_csv(
    src: str = "profiled5/data/simulated_profiled_5tasks.csv",
    dst: str = "profiled4/data/simulated_profiled_4tasks_no_easy.csv",
) -> Path:
    """Exclude switching_easy; keep causal, bandit, field2d, switching_medium."""
    df = pd.read_csv(src)
    out = df[df["task"] != "switching_easy"].copy()
    path = Path(dst)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"[OK] wrote {len(out)} rows, {out.groupby(['task','agent']).ngroups} episodes -> {path}")
    return path


if __name__ == "__main__":
    build_profiled4_csv()
