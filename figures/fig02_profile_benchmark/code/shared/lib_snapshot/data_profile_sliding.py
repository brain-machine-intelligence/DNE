"""
Sliding / variable-prefix windows for long episodes (e.g. 300 trials).

Training (default): L in {10,20,40,80}, K=20, prefix_start stride 20, next-window targets.
Split is always at episode level (subject|task|agent); windows stay within one episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from profile_benchmark.data import Episode, load_episodes
from profile_benchmark.data_profile import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TARGET_LEN,
    episode_row_id,
    filter_episodes_by_tasks,
    profile_collate_fn,
    save_split_manifest,
    split_episodes,
)

DEFAULT_EPISODE_TRIALS = 300
DEFAULT_TRAIN_L = (10, 20, 40, 80)
DEFAULT_EVAL_L = (5, 10, 20, 40, 80)
DEFAULT_SLIDING_STRIDE = 20
# Legacy operational window (profiled4 comparison)
DEFAULT_OP_PREFIX_START = 0
DEFAULT_OP_PREFIX_L = 80
DEFAULT_OP_TARGET_START = 80


@dataclass(frozen=True)
class WindowSpec:
    prefix_start: int
    prefix_len: int
    target_start: int
    target_len: int = DEFAULT_TARGET_LEN

    @property
    def prefix_end(self) -> int:
        return self.prefix_start + self.prefix_len

    @property
    def target_end(self) -> int:
        return self.target_start + self.target_len

    def to_dict(self) -> Dict[str, int]:
        return {
            "prefix_start": self.prefix_start,
            "prefix_len": self.prefix_len,
            "target_start": self.target_start,
            "target_len": self.target_len,
        }


def generate_sliding_windows(
    *,
    episode_trials: int = DEFAULT_EPISODE_TRIALS,
    L_values: Sequence[int] = DEFAULT_TRAIN_L,
    target_len: int = DEFAULT_TARGET_LEN,
    stride: int = DEFAULT_SLIDING_STRIDE,
    min_prefix_start: int = 0,
) -> List[WindowSpec]:
    """Next-window samples: prefix [ps, ps+L) -> target [ps+L, ps+L+K)."""
    specs: List[WindowSpec] = []
    for L in L_values:
        max_ps = episode_trials - L - target_len
        for ps in range(min_prefix_start, max_ps + 1, stride):
            ts = ps + L
            te = ts + target_len
            if te <= episode_trials:
                specs.append(WindowSpec(ps, L, ts, target_len))
    return specs


class SlidingProfileDataset(Dataset):
    """Flattened (episode, window_spec) samples."""

    def __init__(
        self,
        episodes: Sequence[Episode],
        window_specs: Sequence[WindowSpec],
    ):
        self.episodes = list(episodes)
        self.window_specs = list(window_specs)
        self._index: List[Tuple[int, WindowSpec]] = []
        for ei, _ep in enumerate(self.episodes):
            for ws in self.window_specs:
                self._index.append((ei, ws))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ei, ws = self._index[idx]
        ep = self.episodes[ei]
        stim, action, reward = ep.to_arrays()
        T = ep.length
        ps, L, ts, K = ws.prefix_start, ws.prefix_len, ws.target_start, ws.target_len
        pe = ps + L
        te = ts + K
        if pe > T or te > T:
            raise ValueError(f"Window out of range T={T}: {ws}")

        df = ep.df
        return {
            "prefix_stim": torch.from_numpy(stim[ps:pe]),
            "prefix_action": torch.from_numpy(action[ps:pe]),
            "prefix_reward": torch.from_numpy(reward[ps:pe]),
            "future_stim": torch.from_numpy(stim[ts:te]),
            "future_action": torch.from_numpy(action[ts:te]),
            "task": str(df["task"].iloc[0]),
            "agent": str(df["agent"].iloc[0]),
            "subject_id": int(df["subject_id"].iloc[0]),
            "episode_id": episode_row_id(ep),
            "prefix_start": ps,
            "prefix_len": L,
            "target_start": ts,
            "target_end": te,
        }


def make_sliding_loaders(
    csv_path: str,
    *,
    batch_size: int = 64,
    split_seed: int = DEFAULT_SPLIT_SEED,
    include_tasks: Optional[Sequence[str]] = None,
    exclude_tasks: Optional[Sequence[str]] = None,
    train_L_values: Sequence[int] = DEFAULT_TRAIN_L,
    target_len: int = DEFAULT_TARGET_LEN,
    stride: int = DEFAULT_SLIDING_STRIDE,
    episode_trials: int = DEFAULT_EPISODE_TRIALS,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, Dict[str, List[Episode]], List[WindowSpec]]:
    episodes = load_episodes(csv_path)
    episodes = filter_episodes_by_tasks(episodes, include_tasks, exclude_tasks)
    train_eps, val_eps, test_eps = split_episodes(episodes, split_seed=split_seed)

    train_specs = generate_sliding_windows(
        episode_trials=episode_trials,
        L_values=train_L_values,
        target_len=target_len,
        stride=stride,
    )
    # Val uses same window grid (no shuffle leakage across episodes)
    val_specs = train_specs

    train_ds = SlidingProfileDataset(train_eps, train_specs)
    val_ds = SlidingProfileDataset(val_eps, val_specs)
    test_ds = SlidingProfileDataset(test_eps, train_specs)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=profile_collate_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    stim_dim = train_eps[0].to_arrays()[0].shape[1]
    df_actions = pd.read_csv(csv_path, usecols=["action"])
    n_actions = int(df_actions["action"].max()) + 1
    splits = {"train": train_eps, "val": val_eps, "test": test_eps}
    return train_loader, val_loader, test_loader, stim_dim, n_actions, splits, train_specs


def make_episode_window_loader(
    episodes: Sequence[Episode],
    window_specs: Sequence[WindowSpec],
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    ds = SlidingProfileDataset(episodes, window_specs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=profile_collate_fn,
    )


def operational_window() -> WindowSpec:
    return WindowSpec(
        prefix_start=DEFAULT_OP_PREFIX_START,
        prefix_len=DEFAULT_OP_PREFIX_L,
        target_start=DEFAULT_OP_TARGET_START,
        target_len=DEFAULT_TARGET_LEN,
    )
